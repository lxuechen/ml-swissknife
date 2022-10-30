"""
Custom datasets and loaders which could exclude certain classes.

Also enables returning the indices of selected instances.
"""

import os
from typing import Optional, Callable, Tuple, Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml_swissknife import utils
from . import folder


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
    enable_data_aug=False,
    is_valid_file=None,
):
    # TODO: data aug for svhn and mnist!
    # TODO: mnist svhn need overlap removal!
    if name == "svhn":
        transform_svhn = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return SVHN(
            root=root, split=split, download=download, transform=transform_svhn, classes=classes
        )
    elif name == "mnist":
        transform_mnist = transforms.Compose([
            transforms.Resize(32),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return MNIST(
            root=root, train=(split == 'train'), download=download, transform=transform_mnist, classes=classes
        )
    elif name == "imagenet-dogs":
        ImageNet = _get_imagenet_data(
            split=split,
            classes=classes,
            imagenet_path=utils.join(root, 'imagenet-dogs'),
            enable_data_aug=enable_data_aug,
            is_valid_file=is_valid_file
        )
        print(f'spilt: {split}, classes: {classes}')
        print(ImageNet.class_to_idx)
        return ImageNet
    else:
        raise ValueError(f'Unknown name: {name}')


def _get_imagenet_data(
    classes: Sequence[int],
    imagenet_path: str,
    split: str,
    enable_data_aug: bool,
    normalize=False,
    is_valid_file=None,  # Not quite safe, but quick to get working.
):
    if split not in ('train', 'val'):
        raise ValueError(f"Unknown split for imagenet: {split}")

    # SimCLR pre-trained models don't normalize to z-score; just / 255.!
    normalization_ops = []
    if normalize:
        normalization_ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if enable_data_aug:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ] + normalization_ops
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ] + normalization_ops
        )

    split_path = utils.join(imagenet_path, split)
    metadata_path = utils.join(imagenet_path, 'metadata.json')  # Maps class id in int to folder id.
    meta_data = utils.jload(metadata_path)
    if isinstance(meta_data, list):
        # Folder ids, not int class ids! E.g., n02113978.
        classes_set = set(classes)
        folder_ids = [
            class_triplet[1] for class_triplet in meta_data
            if class_triplet[0] in classes_set
        ]

    def _is_valid_file(path):
        """Check if an image is in one of the designated folders and `is_valid_file`."""
        for folder_id in folder_ids:
            if folder_id in path:
                if callable(is_valid_file):
                    return is_valid_file(path)
                else:
                    return True
        return False

    return folder.ImageFolder(
        root=split_path, transform=transform, is_valid_file=_is_valid_file,
    )


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
    return_data=False,
):
    held_out_split_name = 'val' if 'imagenet' in source_data_name else 'test'

    def make_file_id_parity_check_fn(odd):
        def check_fn(path):
            file_name = os.path.basename(path)
            file_id = file_name.split('_')[1].split('.')[0]
            file_id = int(file_id)
            return file_id % 2 == (1 if odd else 0)

        return check_fn

    odd_file_is_valid = make_file_id_parity_check_fn(odd=True)
    eve_file_is_valid = make_file_id_parity_check_fn(odd=False)

    source_train = get_data(
        root=root, name=source_data_name, split='train', classes=source_classes,
        is_valid_file=odd_file_is_valid,
    )
    target_train = get_data(
        root=root, name=target_data_name, split='train', classes=target_classes,
        is_valid_file=eve_file_is_valid,
    )

    source_test = get_data(root=root, name=source_data_name, split=held_out_split_name, classes=source_classes)
    target_test = get_data(root=root, name=target_data_name, split=held_out_split_name, classes=target_classes)

    print(
        f"example counts: \n"
        f"\tsource_train: {len(source_train)}, source_test: {len(source_test)}, \n"
        f"\ttarget_train: {len(target_train)}, target_test: {len(target_test)}"
    )

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

    if return_data:
        return {
            "loaders": (
                source_train_loader, source_test_loader,
                target_train_loader, target_test_loader,
                target_train_loader_unshuffled, target_test_loader_unshuffled,
            ),
            "data": (
                source_train, source_test,
                target_train, target_test,
            )
        }
    else:
        return (
            source_train_loader, source_test_loader,
            target_train_loader, target_test_loader,
            target_train_loader_unshuffled, target_test_loader_unshuffled,
        )
