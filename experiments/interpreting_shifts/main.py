"""
First test run of learning to map using mini-batch unbalanced OT.

The tricky bit of marginalization is tracking indices of examples!

To run:
    python -m interpreting_shifts.main
    python -m interpreting_shifts.main --task subpop_discovery

w/ or w/o domain adapation:
    MNIST (lost digits) -> MNIST: It seems randomly initialized network could already do pretty well.
"""
import collections
import itertools

import fire
import numpy as np
import ot
from sklearn.manifold import TSNE
import torch
from torch import optim, nn
import torch.nn.functional as F
import tqdm

from swissknife import utils
from . import models
from . import solvers
from .custom_datasets import get_data, get_loaders


class OptimalTransportDomainAdapter(object):
    def __init__(
        self,
        model_g, model_f,
        n_class=10, eta1=0.001, eta2=0.0001, eta_src=1.,
        reg_target=0.1, reg_source=10., reg_entropy=0.1,
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

    def fit_source(
        self,
        source_train_loader, source_test_loader=None,
        epochs=10, criterion=F.cross_entropy, learning_rate=2e-4,
    ):
        params = tuple(self.model_g.parameters()) + tuple(self.model_f.parameters())
        optimizer = optim.Adam(params=params, lr=learning_rate)
        for _ in tqdm.tqdm(range(epochs), desc="fit source"):
            for i, data in enumerate(source_train_loader):
                self.model_g.train()
                self.model_f.train()
                optimizer.zero_grad()

                data = tuple(t.to(device) for t in data)
                x, y = data[:2]
                loss = criterion(self._model(x), y)
                loss.backward()
                optimizer.step()
        self._evaluate(loader=source_test_loader, criterion=criterion)

    def fit_joint(
        self,
        source_train_loader, target_train_loader, target_test_loader,
        epochs=100, criterion=F.cross_entropy, learning_rate=2e-4,
        balanced_op=False,
        eval_steps=25,
    ):
        target_train_loader_cycled = itertools.cycle(target_train_loader)
        params = tuple(self.model_g.parameters()) + tuple(self.model_f.parameters())
        optimizer = optim.Adam(params=params, lr=learning_rate)

        global_step = 0
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

                source_gx, target_gx = tuple(self.model_g(t) for t in (source_x, target_x))
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
                    avg_xent, avg_zeon = self._evaluate(target_test_loader, criterion)
                    print(f"epoch: {epoch}, global_step: {global_step}, avg_xent: {avg_xent}, avg_zeon: {avg_zeon}")

    @torch.no_grad()
    def tsne(self, loader, maxsize=3000):
        self.model_g.eval()
        features = []
        labels = []
        for batch in loader:
            batch_features = self.model_g(batch[0].to(device)).cpu().numpy()
            batch_labels = batch[1].numpy()
            features.append(batch_features)
            labels.append(batch_labels)
            if sum(len(b) for b in features) > maxsize:
                break
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        embeddeds = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(features)
        return embeddeds, labels

    @torch.no_grad()
    def _evaluate(self, loader, criterion):
        if loader is None:
            return

        self.model_g.eval()
        self.model_f.eval()

        xents, zeons = [], []
        for data in loader:
            data = tuple(t.to(device) for t in data)
            x, y = data[:2]
            y_hat = self._model(x)

            xent = criterion(y_hat, y, reduction="none")
            zeon = torch.eq(y_hat.argmax(dim=1), y)

            xents.extend(xent.cpu().tolist())
            zeons.extend(zeon.cpu().tolist())
        return tuple(np.mean(np.array(t)) for t in (xents, zeons))

    def _model(self, x):
        return self.model_f(self.model_g(x))

    @torch.no_grad()
    def target_marginal(
        self,
        source_train_loader, target_train_loader_unshuffled,
        epochs=1, balanced_op=False,
    ):
        # Logic:
        #   Sequentially loop over target data.
        #   For each target batch, randomly fetch a source batch and compute approximate mapping.
        #   "Broadcast" local mapping to be a global mapping, then do online averaging.
        source_train_loader_cycled = itertools.cycle(source_train_loader)

        global_step = 0
        target_train_size = sum(packed[0].size(0) for packed in target_train_loader_unshuffled)
        avg = np.zeros((target_train_size,))
        for _ in tqdm.tqdm(range(epochs), desc="target marginal"):
            for target_train_data in target_train_loader_unshuffled:  # Sequential to avoid some examples not assigned.
                target_train_data = tuple(t.to(device) for t in target_train_data)
                target_x, _, target_indices = target_train_data
                target_gx = self.model_g(target_x)
                target_fgx = self.model_f(target_gx)

                source_train_data = next(source_train_loader_cycled)
                source_train_data = tuple(t.to(device) for t in source_train_data)
                source_x, source_y = source_train_data[:2]
                source_gx = self.model_g(source_x)

                # JDOT loss.
                pairwise_diff = (source_gx[..., None] - target_gx.permute(1, 0)[None, ...])
                feature_cost = torch.sum(pairwise_diff * pairwise_diff, dim=1)  # (source bsz, target bsz).

                source_y_oh = F.one_hot(source_y, num_classes=self.n_class).to(source_x.dtype)
                label_cost = source_y_oh @ (- torch.log_softmax(target_fgx, dim=1).permute(1, 0))

                cost = self.eta1 * feature_cost + self.eta2 * label_cost
                cost_numpy = cost.detach().cpu().numpy()

                a, b = ot.unif(source_x.size(0)), ot.unif(target_x.size(0))
                if balanced_op:
                    joint = ot.emd(a, b, cost_numpy)
                else:  # Unbalanced optimal transport.
                    joint = solvers.sinkhorn_knopp_unbalanced(
                        cost_numpy,
                        reg_a=self.reg_source, reg_b=self.reg_target, reg=self.reg_entropy,
                        a=a, b=b,
                    )

                marginal = np.sum(joint, axis=0)
                target_indices = target_indices.cpu().numpy()
                marginal_full = np.zeros_like(avg)
                np.put(marginal_full, target_indices, marginal)

                # Online average.
                global_step += 1
                avg = avg * (global_step - 1) / global_step + marginal_full / global_step

        return avg


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
    # --------------------

    # --- imagenet-dogs starts with class 151, but labels start with 0 ---
    class_offset=151,
    # ------

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
    **unused_kwargs,
):
    utils.handle_unused_kwargs(unused_kwargs)
    print(f'source classes:{source_classes}, target_classes: {target_classes}')

    (source_train_loader, source_test_loader,
     target_train_loader, target_test_loader,
     target_train_loader_unshuffled, target_test_loader_unshuffled,) = get_loaders(
        train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
        source_data_name=data_name, target_data_name=data_name,
        source_classes=source_classes,
        target_classes=target_classes,
    )

    # TODO: Expand imagenet-dogs later.
    n_class = {
        'mnist': 10,
        'cifar-10': 10,
        'imagenet-dogs': 10,
    }[data_name]

    model_g, model_f = _get_feature_extractor_and_classifier(feature_extractor, n_class)

    domain_adapter = OptimalTransportDomainAdapter(
        model_g, model_f,
        eta_src=eta_src, eta1=eta1, eta2=eta2,
        reg_target=reg_target, reg_source=reg_source, reg_entropy=reg_entropy,
        n_class=n_class,
    )
    domain_adapter.fit_source(
        source_train_loader,
        epochs=train_source_epochs,
    )
    domain_adapter.fit_joint(
        source_train_loader, target_train_loader, target_test_loader,
        epochs=train_joint_epochs, balanced_op=balanced_op,
    )

    if train_dir is not None:
        # Plot1: t-SNE.
        embeddeds, labels = domain_adapter.tsne(target_train_loader)  # Shuffled!
        class2embedded = collections.defaultdict(list)
        for embedded, label in utils.zip_(embeddeds, labels):
            class2embedded[int(label)].append(embedded)

        scatters = []
        for target_class in target_classes:
            embedded = class2embedded[target_class - class_offset]
            embedded = np.stack(embedded, axis=0)
            scatters.append(
                dict(x=embedded[:, 0], y=embedded[:, 1], label=target_class, s=10)  # s: marker size.
            )

        img_path = utils.join(train_dir, 'tsne')
        utils.plot_wrapper(
            img_path=img_path,
            suffixes=('.png', '.pdf'),
            scatters=scatters,
            options=dict(
                title=f"S: {source_classes}, "
                      f"\nT: {target_classes}"
            )
        )

        # Plot2: Marginalize over source to get the target distribution.
        marginal = domain_adapter.target_marginal(
            source_train_loader, target_train_loader_unshuffled,
            epochs=match_epochs, balanced_op=balanced_op,
        )

        # Retrieve the ordered target dataset. Must match up with `target_train_loader_unshuffled`.
        target_train_data = get_data(name=data_name, split='train', classes=target_classes)

        # Bar plot full class marginals.
        img_path = utils.join(train_dir, 'class_marginals')
        class_marginals = collections.defaultdict(int)
        for marginal_i, label_i in utils.zip_(marginal, target_train_data.targets):
            class_marginals[int(label_i)] += marginal_i
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
            )
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
                label = int(target_train_data.targets[bot_index])
                bot_class_counts[label] = bot_class_counts[label] + 1
            top_count_classes = tuple(k for k, _ in sorted(bot_class_counts.items(), key=lambda item: item[1]))[:5]
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
                    title=f"S: {source_classes}, \nT: {target_classes}, \ntop count classes: {top_count_classes}",
                    ylabel=f"bottom {bottom_percentage_int}% class counts",
                    xlabel="class label",
                )
            )
            del bar


def main(
    task="domain_adaptation",
    seed=0,
    **kwargs
):
    torch.manual_seed(seed)

    if task == "subpop_discovery":
        subpop_discovery(**kwargs)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fire.Fire(main)
