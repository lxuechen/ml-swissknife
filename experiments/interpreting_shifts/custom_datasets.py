"""
Custom datasets and loaders which could exclude certain classes.
"""

import os
from typing import Optional, Callable, Tuple, Any

import fire
import numpy as np
import torch
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


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
