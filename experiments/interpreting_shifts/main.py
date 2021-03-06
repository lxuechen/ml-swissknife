"""
First test run of learning to map using mini-batch unbalanced OT.

The tricky bit of marginalization is tracking indices of examples!

To run:
    python -m interpreting_shifts.main
    python -m interpreting_shifts.main --task subpop_discovery

w/ or w/o domain adapation:
    MNIST (lost digits) -> MNIST: It seems randomly initialized network could already do pretty well.
"""
import argparse
import collections
import itertools
import logging
from typing import Dict
from typing import Sequence

import numpy as np
from numpy import testing
import ot
from sklearn.manifold import TSNE
import torch
from torch import optim, nn
import torch.nn.functional as F
import tqdm

from swissknife import utils
from . import models
from . import solvers
from .custom_datasets import get_loaders


class OptimalTransportDomainAdapter(object):
    def __init__(
        self,
        model_g, model_f,
        n_class=10, eta1=0.001, eta2=0.0001, eta_src=1.,
        reg_target=0.1, reg_source=10., reg_entropy=0.1,
        normalize_embeddings=True,
    ):
        self.model_g: nn.Module = model_g
        self.model_f: nn.Module = model_f
        self.n_class = n_class
        self.eta_src = eta_src
        self.eta1 = eta1  # Weight for feature cost.
        self.eta2 = eta2  # Weight for label cost.
        self.reg_target = reg_target
        self.reg_source = reg_source
        self.reg_entropy = reg_entropy
        self.normalize_embeddings = normalize_embeddings

    def fit_source(
        self,
        source_train_loader, source_test_loader=None,
        epochs=10, criterion=F.cross_entropy, learning_rate=1e-4,
        eval_steps=25,
    ):
        global_step = 0
        record = []
        global_steps = []

        params = tuple(self.model_g.parameters()) + tuple(self.model_f.parameters())
        optimizer = optim.Adam(params=params, lr=learning_rate)
        for epoch in tqdm.tqdm(range(epochs), desc="fit source"):
            for i, data in enumerate(source_train_loader):
                self.model_g.train()
                self.model_f.train()
                optimizer.zero_grad()

                data = tuple(t.to(device) for t in data)
                x, y = data[:2]
                loss = criterion(self._model(x), y)
                loss.backward()
                optimizer.step()

                global_step += 1
                if global_step % eval_steps == 0:
                    result = self._evaluate(loader=source_test_loader, criterion=criterion)
                    avg_xent, avg_zeon = result['xent'], result['zeon']
                    logging.warning(
                        f'fit_source -- source test -- '
                        f'epoch: {epoch}, global_step: {global_step}, avg_xent: {avg_xent}, avg_zeon: {avg_zeon}'
                    )
                    record.append(result)
                    global_steps.append(global_step)

        return {'global_steps': global_steps, 'record': record}

    def fit_joint(
        self,
        source_train_loader, target_train_loader, target_test_loader,
        epochs=100, criterion=F.cross_entropy, learning_rate=1e-4,
        balanced_op=False,
        eval_steps=25,
    ):
        global_step = 0
        record = []
        global_steps = []

        target_train_loader_cycled = itertools.cycle(target_train_loader)
        params = tuple(self.model_g.parameters()) + tuple(self.model_f.parameters())
        optimizer = optim.Adam(params=params, lr=learning_rate)

        for epoch in tqdm.tqdm(range(epochs), desc=f"fit joint"):
            for i, source_train_data in enumerate(source_train_loader):
                self.model_g.train()
                self.model_f.train()
                optimizer.zero_grad()

                source_train_data = tuple(t.to(device) for t in source_train_data)
                source_x, source_y = source_train_data[:2]

                target_train_data = next(target_train_loader_cycled)
                target_train_data = tuple(t.to(device) for t in target_train_data)
                target_x = target_train_data[0]

                source_gx, target_gx = tuple(self._model_g_apply(t) for t in (source_x, target_x))
                source_fgx, target_fgx = tuple(self.model_f(t) for t in (source_gx, target_gx))

                # Source classification loss.
                source_cls_loss = criterion(source_fgx, source_y)

                # JDOT loss.
                pairwise_diff = (source_gx[..., None] - target_gx.permute(1, 0)[None, ...])
                feature_cost = torch.sum(pairwise_diff * pairwise_diff, dim=1)

                source_y_oh = F.one_hot(source_y, num_classes=self.n_class).to(source_x.dtype)
                label_cost = source_y_oh @ (- torch.log_softmax(target_fgx, dim=1).permute(1, 0))

                cost = self.eta1 * feature_cost + self.eta2 * label_cost
                cost_numpy = cost.detach().cpu().numpy()

                # Compute alignment.
                a, b = ot.unif(source_x.size(0)), ot.unif(target_x.size(0))
                if balanced_op:
                    pi = ot.emd(a, b, cost_numpy)
                else:  # Unbalanced optimal transport.
                    pi = solvers.sinkhorn_knopp_unbalanced(
                        cost_numpy,
                        reg_a=self.reg_source, reg_b=self.reg_target, reg=self.reg_entropy,
                        a=a, b=b,
                    )
                pi = torch.tensor(pi, device=device, dtype=torch.get_default_dtype())

                da_loss = torch.sum(pi * cost)
                loss = self.eta_src * source_cls_loss + da_loss
                loss.backward()

                optimizer.step()

                global_step += 1
                if global_step % eval_steps == 0:
                    result = self._evaluate(target_test_loader, criterion)
                    avg_xent, avg_zeon = result['xent'], result['zeon']
                    logging.warning(
                        f'fit_joint -- target test -- '
                        f'epoch: {epoch}, global_step: {global_step}, avg_xent: {avg_xent}, avg_zeon: {avg_zeon}'
                    )
                    global_steps.append(global_step)
                    record.append(result)

        return {"global_steps": global_steps, "record": record}

    @torch.no_grad()
    def tsne(self, loader, maxsize=3000, class_offset=0):
        self.model_g.eval()
        features = []
        labels = []
        for batch in loader:
            batch_features = self._model_g_apply(batch[0].to(device)).cpu().numpy()
            batch_labels = batch[1].numpy() + class_offset
            features.append(batch_features)
            labels.append(batch_labels)
            if sum(len(b) for b in features) > maxsize:
                break
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        embeddeds = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(features)
        return embeddeds, labels

    @torch.no_grad()
    def _evaluate(self, loader, criterion) -> Dict[str, float]:
        if loader is None:
            return {"xent": 0.0, "zeon": 0.0}

        xents, zeons = [], []
        self.model_g.eval()
        self.model_f.eval()

        for data in loader:
            data = tuple(t.to(device) for t in data)
            x, y = data[:2]
            y_hat = self._model(x)

            xent = criterion(y_hat, y, reduction="none")
            zeon = torch.eq(y_hat.argmax(dim=1), y)

            xents.extend(xent.cpu().tolist())
            zeons.extend(zeon.cpu().tolist())

        return {
            "xent": float(np.mean(np.array(xents))),
            "zeon": float(np.mean(np.array(zeons))),
        }

    def _model_g_apply(self, x):
        """`model_g.forward` wrapper that optionally enforces normalized outputs."""
        features = self.model_g(x)
        if self.normalize_embeddings:
            features = features / features.norm(2, dim=1, keepdim=True)
        return features

    def _model(self, x):
        features = self._model_g_apply(x)
        return self.model_f(features)

    @torch.no_grad()
    def target_marginal(
        self,
        source_train_loader, target_train_loader_unshuffled,
        source_size, target_size,
        epochs=1, balanced_op=False,
        return_joint=False,
    ):
        assert sum(pack[0].size(0) for pack in target_train_loader_unshuffled) == target_size

        # Logic:
        #   Sequentially loop over target data.
        #   For each target batch, randomly fetch a source batch and compute approximate mapping.
        #   "Broadcast" local mapping to be a global mapping, then do online averaging.
        global_step = 0

        avg_marginal = np.zeros((target_size,))
        avg_joint = np.zeros((source_size, target_size))
        source_train_loader_cycled = itertools.cycle(source_train_loader)

        for _ in tqdm.tqdm(range(epochs), desc="target marginal"):
            # Sequential to avoid some examples not assigned.
            for target_train_data in tqdm.tqdm(target_train_loader_unshuffled):
                target_train_data = tuple(t.to(device) for t in target_train_data)
                target_x, _, target_indices = target_train_data
                target_gx = self._model_g_apply(target_x)
                target_fgx = self.model_f(target_gx)

                source_train_data = next(source_train_loader_cycled)
                source_train_data = tuple(t.to(device) for t in source_train_data)
                source_x, source_y, source_indices = source_train_data
                source_gx = self._model_g_apply(source_x)

                if self.normalize_embeddings:
                    target_gx = target_gx / target_gx.norm(2, dim=1, keepdim=True)
                    source_gx = source_gx / source_gx.norm(2, dim=1, keepdim=True)

                # JDOT loss.
                pairwise_diff = (source_gx[..., None] - target_gx.permute(1, 0)[None, ...])
                # (source bsz, target bsz). norm-squared.
                feature_cost = torch.sum(pairwise_diff * pairwise_diff, dim=1)

                source_y_oh = F.one_hot(source_y, num_classes=self.n_class).to(source_x.dtype)
                label_cost = source_y_oh @ (- torch.log_softmax(target_fgx, dim=1).permute(1, 0))

                assert feature_cost.size() == label_cost.size()
                cost = self.eta1 * feature_cost + self.eta2 * label_cost
                cost_numpy = cost.detach().cpu().numpy()

                a, b = ot.unif(source_x.size(0)), ot.unif(target_x.size(0))
                if balanced_op:
                    joint, log = ot.emd(a, b, cost_numpy, log=True)
                else:  # Unbalanced optimal transport.
                    joint, log = solvers.sinkhorn_knopp_unbalanced(
                        cost_numpy,
                        reg_a=self.reg_source, reg_b=self.reg_target, reg=self.reg_entropy,
                        a=a, b=b, log=True
                    )

                # `joint`: (source bsz, target bsz).
                marginal = np.sum(joint, axis=0)  # p(t_j) = \sum_i p(s_i, t_j).
                target_indices = target_indices.cpu().numpy()
                marginal_full = np.zeros_like(avg_marginal)
                np.put(marginal_full, target_indices, marginal)

                joint_full = _broadcast_joint(
                    end_shape=avg_joint.shape, joint=joint, source_ids=source_indices, target_ids=target_indices,
                )

                # Online average.
                global_step += 1
                avg_marginal = avg_marginal * (global_step - 1) / global_step + marginal_full / global_step
                avg_joint = avg_joint * (global_step - 1) / global_step + joint_full / global_step

        if return_joint:
            return avg_marginal, avg_joint
        else:
            return avg_marginal


def _broadcast_joint(end_shape: Sequence[int], joint: np.ndarray, source_ids: Sequence[int], target_ids: Sequence[int]):
    joint_full = np.zeros(end_shape)
    for source_idx, row in utils.zip_(source_ids, joint):
        np.put(joint_full[source_idx], target_ids, row)
    return joint_full


def test_broadcast_joint():
    end_shape = (5, 4)
    joint = np.array([[1, -1], [2, -2.]])
    source_ids = [0, 2]
    target_ids = [3, 1]
    joint_full = _broadcast_joint(end_shape=end_shape, joint=joint, source_ids=source_ids, target_ids=target_ids)
    desired = np.array(
        [[0., -1., 0., 1.],
         [0., 0., 0., 0.],
         [0., -2., 0., 2.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],
    )
    testing.assert_allclose(joint_full, desired)


def _get_feature_extractor_and_classifier(feature_extractor, n_class):
    if feature_extractor == 'cnn':
        model_g = models.Cnn_generator().to(device).apply(models.weights_init)
        model_f = models.Classifier2().to(device).apply(models.weights_init)
    elif feature_extractor == 'id':
        model_g = nn.Flatten()
        model_f = nn.Linear(3072, n_class).to(device)
    elif feature_extractor == "fc":
        model_g = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
        ).to(device)
        model_f = nn.Linear(200, n_class).to(device)
    elif feature_extractor == "resnet":
        # PyCharm gives a stupid syntax error highlight.
        from simclrv2 import resnet  # noqa

        # TODO: Write a separate script to validate this checkpoint!
        simclr_ckpt = "/nlp/scr/lxuechen/simclr-ckpts/r50_1x_sk0_ema.pth"
        depth, width, sk_ratio = resnet.name_to_params(simclr_ckpt)
        model, _ = resnet.get_resnet(depth, width, sk_ratio)
        model.load_state_dict(torch.load(simclr_ckpt)['resnet'])

        n_channel = width * 2048
        model_g = model.to(device)
        model_f = nn.Linear(n_channel, n_class).to(device)
    else:
        raise ValueError(f"Unknown feature_extractor: {feature_extractor}")
    return model_g, model_f


def subpop_discovery(
    # --- core hparams ---
    eta_src=1., eta1=0.1, eta2=0.1,
    reg_target=0.1, reg_source=10., reg_entropy=0.01,
    fit_source_lr=1e-4,
    fit_joint_lr=1e-4,
    # --------------------

    data_name="mnist",
    train_batch_size=500,
    eval_batch_size=500,
    source_classes=(1, 2, 3, 9, 0,),
    target_classes=tuple(range(10)),
    train_source_epochs=3,
    train_joint_epochs=3,
    match_epochs=5,
    balanced_op=False,
    feature_extractor="cnn",
    train_dir="/nlp/scr/lxuechen/interpreting_shifts/test",
    bottom_percentages=(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    normalize_embeddings=True,
    eval_steps=50,
    **unused_kwargs,
):
    utils.handle_unused_kwargs(unused_kwargs)
    logging.warning(f'source classes:{source_classes}, target_classes: {target_classes}')

    pile = get_loaders(
        train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
        source_data_name=data_name, target_data_name=data_name,
        source_classes=source_classes,
        target_classes=target_classes,
        return_data=True
    )
    (
        source_train_loader, source_test_loader,
        target_train_loader, target_test_loader,
        target_train_loader_unshuffled, target_test_loader_unshuffled,
    ) = pile["loaders"]
    (source_train_data, source_test_data, target_train_data, target_test_data) = pile["data"]

    # TODO: Expand imagenet-dogs later.
    n_class = {
        'mnist': 10,
        'cifar-10': 10,
        'imagenet-dogs': 10,
    }[data_name]
    class_offset = {
        'mnist': 0,
        'cifar-10': 0,
        'imagenet-dogs': 151
    }[data_name]

    model_g, model_f = _get_feature_extractor_and_classifier(feature_extractor, n_class)

    domain_adapter = OptimalTransportDomainAdapter(
        model_g, model_f,
        eta_src=eta_src, eta1=eta1, eta2=eta2,
        reg_target=reg_target, reg_source=reg_source, reg_entropy=reg_entropy,
        n_class=n_class,
        normalize_embeddings=normalize_embeddings
    )
    source_results = domain_adapter.fit_source(
        source_train_loader, source_test_loader=source_test_loader,
        epochs=train_source_epochs,
        eval_steps=eval_steps,
        learning_rate=fit_source_lr,
    )
    joint_results = domain_adapter.fit_joint(
        source_train_loader, target_train_loader, target_test_loader,
        epochs=train_joint_epochs, balanced_op=balanced_op,
        eval_steps=eval_steps,
        learning_rate=fit_joint_lr,
    )
    utils.jdump(source_results, utils.join(train_dir, 'source_results.json'))
    utils.jdump(joint_results, utils.join(train_dir, 'joint_results.json'))

    if train_dir is not None:
        # Plot1: t-SNE.

        scatters = []
        for loader, tag, marker in utils.zip_((target_train_loader, source_train_loader), ('tgt', 'src'), ('x', 'o')):
            embeddeds, labels = domain_adapter.tsne(loader, class_offset=class_offset)  # Must include offset here!
            class2embedded = collections.defaultdict(list)
            for embedded, label in utils.zip_(embeddeds, labels):
                class2embedded[int(label)].append(embedded)

            for target_class in target_classes:
                embedded = class2embedded.get(target_class, None)
                if embedded is not None:
                    embedded = np.stack(embedded, axis=0)
                    scatters.append(
                        dict(
                            x=embedded[:, 0], y=embedded[:, 1],
                            label=f"{target_class} ({tag})", s=5, marker=marker,
                        )
                    )

        img_path = utils.join(train_dir, 'tsne')
        utils.plot_wrapper(
            img_path=img_path,
            suffixes=('.png', '.pdf'),
            scatters=scatters,
            options=dict(
                title=f"S: {source_classes}, "
                      f"\nT: {target_classes}"
            ),
            legend_options=dict(fontsize=8, framealpha=0.6),
        )

        # Plot2: Marginalize over source to get the target distribution.
        marginal, joint = domain_adapter.target_marginal(
            source_train_loader, target_train_loader_unshuffled,
            source_size=len(source_train_data), target_size=len(target_train_data),
            epochs=match_epochs, balanced_op=balanced_op,
            return_joint=True
        )

        # Bar plot full class marginals.
        img_path = utils.join(train_dir, 'class_marginals')
        class_marginals = collections.defaultdict(int)
        for marginal_i, label_i in utils.zip_(marginal, target_train_data.targets):
            label_i = int(label_i) + class_offset  # Labels always start from 0.
            class_marginals[label_i] = class_marginals[label_i] + marginal_i
        bar = dict(
            x=target_classes,
            height=[class_marginals[target_class] for target_class in target_classes]
        )
        sum_prob = sum(class_marginals.values())
        top_marginal_classes = tuple(k for k, _ in sorted(class_marginals.items(), key=lambda item: -item[1]))[:5]
        utils.plot_wrapper(
            img_path=img_path,
            suffixes=('.png', '.pdf'),
            bars=(bar,),
            options=dict(
                title=f"S: {source_classes}, "
                      f"\nT: {target_classes}, "
                      f"\ntop marginal: {top_marginal_classes}, "
                      f"\nsum_prob: {sum_prob:.4f}",
                ylabel="transport map marginal prob.",
                xlabel="class label",
            ),
            legend_options=dict(fontsize=8, framealpha=0.6),
        )
        del bar

        # Plot3: Bar plot class counts of bottom of the marginal.
        marginal = torch.tensor(marginal, dtype=torch.get_default_dtype())
        marginal_len = len(marginal)

        for bottom_percentage in bottom_percentages:
            bottom_percentage_int = int(bottom_percentage * 100)
            bot_values, bot_indices = (-marginal).topk(int(bottom_percentage * marginal_len))
            bot_class_counts = collections.defaultdict(int)
            for bot_index in bot_indices:
                label = int(target_train_data.targets[bot_index]) + class_offset  # Labels always start from 0.
                bot_class_counts[label] = bot_class_counts[label] + 1
            top_count_classes = tuple(k for k, _ in sorted(bot_class_counts.items(), key=lambda item: item[1]))[5:]
            bar = dict(
                x=target_classes,
                height=[bot_class_counts[target_class] for target_class in target_classes]
            )
            img_path = utils.join(train_dir, f'bottom_class_counts_{bottom_percentage_int:03d}')
            utils.plot_wrapper(
                img_path=img_path,
                suffixes=('.png', '.pdf'),
                bars=(bar,),
                options=dict(
                    title=f"S: {source_classes}, "
                          f"\nT: {target_classes}, "
                          f"\ntop classes in tail: {top_count_classes} (expect target)",
                    ylabel=f"bottom {bottom_percentage_int}% class counts",
                    xlabel="class label",
                ),
                legend_options=dict(fontsize=8, framealpha=0.6),
            )
            del bar

        # TODO: Plot 4: connections!
        #   High target marginal => source images
        #   Low target marginal => source images


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="subpop_discovery",
                        choices=('subpop_discovery', 'test_broadcast_joint'))
    parser.add_argument('--seed', type=int, default=42)

    # --- core hparams ---
    parser.add_argument('--eta_src', type=float, default=1., help="In fit_joint, trades source cls loss.")
    parser.add_argument('--eta1', type=float, default=0.1, help="In fit_joint/target_marginal, trades feature cost.")
    parser.add_argument('--eta2', type=float, default=0.1, help="In fit_joint/target_marginal, trades label cost.")

    parser.add_argument('--reg_target', type=float, default=0.1, help="OT target marginal penalty strength.")
    parser.add_argument('--reg_source', type=float, default=10., help="OT source marginal penalty strength.")
    parser.add_argument('--reg_entropy', type=float, default=0.01, help="OT joint entropy penalty.")
    # ---

    # --- training hparams ---
    parser.add_argument('--train_batch_size', type=int, default=500)
    parser.add_argument('--eval_batch_size', type=int, default=500, help="In target_marginal; larger better for OT.")
    parser.add_argument('--train_source_epochs', type=int, default=3)
    parser.add_argument('--train_joint_epochs', type=int, default=3)
    parser.add_argument('--match_epochs', type=int, default=3)
    parser.add_argument('--balanced_op', type=utils.str2bool, default=False)
    parser.add_argument('--feature_extractor', type=str, default='cnn', choices=('cnn', 'id', 'fc', 'resnet'))
    parser.add_argument('--eval_steps', type=int, default=50, help="Steps between evaluation.")

    # ---
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--source_classes', type=int, nargs="+", default=(1, 2, 3, 9, 0))
    parser.add_argument('--target_classes', type=int, nargs="+", default=tuple(range(10)))
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--bottom_percentages', type=float, nargs="+", default=(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5))
    parser.add_argument('--normalize_embeddings', type=utils.str2bool, default=True)

    args = parser.parse_args()

    if args.task == "subpop_discovery":
        assert args.train_dir is not None
        utils.manual_seed(args)
        utils.write_argparse(args)

        subpop_discovery(**args.__dict__)
    elif args.task == "test_broadcast_joint":
        # python -m interpreting_shifts.main --task "test_broadcast_joint"
        test_broadcast_joint()
