"""
First test run of learning mapping using mini-batch unbalanced OT.

TODO:
    2) function/method for accumulating matching
"""

from typing import Optional, Callable, Tuple, Any

import fire
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
        self.data = np.stack(new_data, axis=0)
        self.targets = np.stack(new_labels, axis=0)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = super(MNIST, self).__getitem__(index=index)
        return img, target, index


def get_da_loaders(
    source_classes=tuple(range(10)),
    target_classes=tuple(range(10)),
    pin_memory=False,
    num_workers=0,
    train_batch_size=500,
    eval_batch_size=500,
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

    source_train = SVHN(
        root='./data', split='train', download=True, transform=transform_svhn, classes=source_classes
    )
    target_train = MNIST(
        root='./data', train=True, download=True, transform=transform_mnist, classes=target_classes
    )
    target_test = MNIST(
        root='./data', train=False, download=True, transform=transform_mnist, classes=target_classes
    )

    source_train_loader = DataLoader(
        source_train, batch_size=train_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
    )
    target_train_loader = DataLoader(
        target_train, batch_size=train_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
    )
    target_test_loader = DataLoader(
        target_test, batch_size=eval_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
    )

    return source_train_loader, target_train_loader, target_test_loader


class OptimalTransportDomainAdaptation(object):
    def __init__(self, model_g, model_f, n_class, eta1=0.001, eta2=0.0001, tau=1., epsilon=0.1):
        self.model_g = model_g
        self.model_f = model_f
        self.n_class = n_class
        self.eta1 = eta1  # Weight for feature cost.
        self.eta2 = eta2  # Weight for label cost.
        self.tau = tau
        self.epsilon = epsilon

    def fit_source(self, source_train_loader, epochs=10, criterion=F.cross_entropy, device=None):
        optimizer = optim.Adam(
            params=tuple(self.model_g.parameters()) + tuple(self.model_f.parameters()), lr=2e-4
        )
        for epoch in range(epochs):
            self.model_g.train()
            self.model_f.train()
            for i, data in enumerate(source_train_loader):
                optimizer.zero_grad()
                x, y = tuple(t.to(device) for t in data)
                loss = criterion(self.model_f(self.model_g(x)), y)
                loss.backward()
                optimizer.step()

    def fit_joint(self):
        pass


def main():
    l1, l2, l3 = get_da_loaders(train_batch_size=10, eval_batch_size=4)


if __name__ == "__main__":
    fire.Fire(main)
