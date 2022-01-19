"""
First test run of learning mapping using mini-batch unbalanced OT.

To run:
    python -m interpreting_shifts.main
    python -m interpreting_shifts.main --task subpop_discovery

w/ or w/o domain adapation:
    MNIST (lost digits) -> MNIST: It seems randomly initialized network could already do pretty well.
"""
import collections
import itertools
import os
from typing import Optional, Callable, Tuple, Any

import fire
import numpy as np
import ot
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm

from swissknife import utils
from . import models


class SVHN(datasets.SVHN):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        classes=tuple(range(10)),
    ) -> None:
        super(SVHN, self).__init__(
            root=root, split=split, transform=transform, target_transform=target_transform, download=download,
        )
        new_data, new_labels = [], []
        for x, y in zip(self.data, self.labels):
            if y in classes:
                new_data.append(x)
                new_labels.append(y)
        self.data = np.stack(new_data, axis=0)
        self.labels = np.stack(new_labels, axis=0)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        # Return index of example in addition to image and label. Helps accumulate OT mapping.
        img, target = super(SVHN, self).__getitem__(index)
        return img, target, index


class MNIST(datasets.MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        classes=tuple(range(10)),
    ) -> None:
        super(MNIST, self).__init__(
            root=root, train=train, transform=transform, target_transform=target_transform, download=download
        )
        new_data, new_labels = [], []
        for x, y in zip(self.data, self.targets):
            if y in classes:
                new_data.append(x)
                new_labels.append(y)
        self.data = torch.stack(new_data, dim=0)
        self.targets = torch.stack(new_labels, dim=0)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = super(MNIST, self).__getitem__(index=index)
        return img, target, index


def get_data(
    name, split, classes, download=True,
    root=os.path.join(os.path.expanduser('~'), 'data'),
):
    transform_svhn = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_mnist = transforms.Compose([
        transforms.Resize(32),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if name == "svhn":
        return SVHN(
            root=root, split=split, download=download, transform=transform_svhn, classes=classes
        )
    elif name == "mnist":
        return MNIST(
            root=root, train=(split == 'train'), download=download, transform=transform_mnist, classes=classes
        )
    else:
        raise ValueError(f'Unknown name: {name}')


def get_loaders(
    root=os.path.join(os.path.expanduser('~'), 'data'),
    source_classes=tuple(range(10)),
    target_classes=tuple(range(10)),
    source_data_name="svhn",
    target_data_name="mnist",
    pin_memory=False,
    num_workers=0,
    train_batch_size=500,
    eval_batch_size=500,
):
    source_train = get_data(root=root, name=source_data_name, split='train', classes=source_classes)
    source_test = get_data(root=root, name=source_data_name, split='test', classes=target_classes)
    target_train = get_data(root=root, name=target_data_name, split='train', classes=target_classes)
    target_test = get_data(root=root, name=target_data_name, split='test', classes=target_classes)

    source_train_loader = DataLoader(
        source_train, batch_size=train_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
    )
    source_test_loader = DataLoader(
        source_test, batch_size=eval_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
    )
    target_train_loader = DataLoader(
        target_train, batch_size=train_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
    )
    target_test_loader = DataLoader(
        target_test, batch_size=eval_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
    )

    target_train_loader_unshuffled = DataLoader(
        target_train, batch_size=train_batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers,
        drop_last=False,
    )
    target_test_loader_unshuffled = DataLoader(
        target_test, batch_size=eval_batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers,
        drop_last=False,
    )

    return (
        source_train_loader, source_test_loader,
        target_train_loader, target_test_loader,
        target_train_loader_unshuffled, target_test_loader_unshuffled,
    )


class OptimalTransportDomainAdapter(object):
    def __init__(
        self,
        model_g, model_f,
        n_class=10, eta1=0.001, eta2=0.0001, tau=1., epsilon=0.1
    ):
        self.model_g: nn.Module = model_g
        self.model_f: nn.Module = model_f
        self.n_class = n_class
        self.eta1 = eta1  # Weight for feature cost.
        self.eta2 = eta2  # Weight for label cost.
        self.tau = tau
        self.epsilon = epsilon

    def fit_source(
        self,
        source_train_loader, source_test_loader=None,
        epochs=10, criterion=F.cross_entropy, learning_rate=2e-4,
    ):
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
        self._evaluate(loader=source_test_loader, criterion=F.softmax)

    def fit_joint(
        self,
        source_train_loader, target_train_loader, target_test_loader,
        epochs=100, criterion=F.cross_entropy, learning_rate=2e-4,
        balanced_op=False,
        eval_steps=100,
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
                    pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, cost_numpy, self.epsilon, self.tau)
                pi = torch.tensor(pi, device=device)

                da_loss = torch.sum(pi * cost)
                loss = source_cls_loss + da_loss
                loss.backward()

                optimizer.step()

                global_step += 1
                if global_step % eval_steps == 0:
                    avg_xent, avg_zeon = self._evaluate(target_test_loader, criterion)
                    print(f"epoch: {epoch}, global_step: {global_step}, avg_xent: {avg_xent}, avg_zeon: {avg_zeon}")

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
        target_train_size = sum(img.size(0) for img, _, _ in target_train_loader_unshuffled)
        avg = np.zeros((target_train_size,))
        for epoch in tqdm.tqdm(range(epochs), desc="target marginal"):
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
                    joint = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, cost_numpy, self.epsilon, self.tau)
                marginal = np.sum(joint, axis=0)
                target_indices = target_indices.cpu().numpy()
                marginal_full = np.zeros_like(avg)
                np.put(marginal_full, target_indices, marginal)

                # Online average.
                global_step += 1
                avg = avg * (global_step - 1) / global_step + marginal_full / global_step

        return avg


def domain_adaptation(
    eta1=0.1,
    eta2=0.1,
    tau=1.0,
    epsilon=0.1,

    train_batch_size=500,
    eval_batch_size=500,
    balanced_op=False,
    feature_extractor="cnn",
    classifier='linear',
    **kwargs,
):
    (source_train_loader, source_test_loader,
     target_train_loader, target_test_loader,
     target_train_loader_unshuffled, target_test_loader_unshuffled,) = get_loaders(
        train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
    )

    model_g = _get_feature_extractor(feature_extractor)
    model_f = _get_classifier(classifier)

    domain_adapter = OptimalTransportDomainAdapter(
        model_g, model_f, eta1=eta1, eta2=eta2, tau=tau, epsilon=epsilon
    )
    domain_adapter.fit_source(source_train_loader)
    domain_adapter.fit_joint(
        source_train_loader, target_train_loader, target_test_loader,
        balanced_op=balanced_op
    )


def _get_feature_extractor(feature_extractor):
    if feature_extractor == 'cnn':
        model_g = models.Cnn_generator().to(device).apply(models.weights_init)
    elif feature_extractor == 'id':
        model_g = nn.Identity()
    else:
        raise ValueError(f"Unknown feature_extractor: {feature_extractor}")
    return model_g


def _get_classifier(classifier):
    if classifier == "linear":
        model_f = models.Classifier2().to(device).apply(models.weights_init)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    return model_f


def subpop_discovery(
    eta1=0.1,
    eta2=0.1,
    tau=1.0,
    epsilon=0.1,

    train_batch_size=500,
    eval_batch_size=500,
    source_classes=(1, 2, 3, 9, 0,),
    target_classes=tuple(range(10)),
    train_epochs=3,
    match_epochs=5,
    balanced_op=False,
    feature_extractor="cnn",
    classifier='linear',
    img_path="/nlp/scr/lxuechen/interpreting_shifts/test",
    **kwargs,
):
    (source_train_loader, source_test_loader,
     target_train_loader, target_test_loader,
     target_train_loader_unshuffled, target_test_loader_unshuffled,) = get_loaders(
        train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
        source_data_name="mnist", target_data_name="mnist",
        source_classes=source_classes,
        target_classes=target_classes,
    )

    model_g = _get_feature_extractor(feature_extractor)
    model_f = _get_classifier(classifier)

    domain_adapter = OptimalTransportDomainAdapter(
        model_g, model_f, eta1=eta1, eta2=eta2, tau=tau, epsilon=epsilon,
    )
    domain_adapter.fit_source(
        source_train_loader, epochs=train_epochs,
    )
    domain_adapter.fit_joint(
        source_train_loader, target_train_loader, target_test_loader, epochs=train_epochs,
        balanced_op=balanced_op,
    )

    # Marginalize over source to get the target distribution.
    marginal = domain_adapter.target_marginal(
        source_train_loader, target_train_loader_unshuffled,
        epochs=match_epochs, balanced_op=balanced_op,
    )

    # Retrieve the ordered target dataset. Must match up with `target_train_loader_unshuffled`.
    target_train_data = get_data(name="mnist", split='train', classes=target_classes)

    # Get class marginals.
    class_marginals = collections.defaultdict(int)
    for marginal_i, label_i in utils.zip_(marginal, target_train_data.targets):
        class_marginals[int(label_i)] += marginal_i

    if img_path is not None:
        bar = dict(
            x=target_classes,
            height=[class_marginals[target_class] for target_class in target_classes]
        )
        sum_prob = sum(class_marginals)
        utils.plot_wrapper(
            img_path=img_path,
            suffixes=('.png', '.pdf'),
            bars=(bar,),
            options=dict(
                title=f"S: {source_classes}, \nT: {target_classes}, \nsum_prob: {sum_prob:.4f}",
                ylabel="transport map marginal prob.",
                xlabel="class label",
            )
        )


def main(
    task="domain_adaptation",
    seed=0,
    **kwargs
):
    torch.manual_seed(seed)

    if task == "domain_adaptation":
        domain_adaptation(**kwargs)
    elif task == "subpop_discovery":
        subpop_discovery(**kwargs)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fire.Fire(main)
