"""
Image classification.

Goals:
    How much is the disparate impact when pre-trained models are used???
        - CLIP
        - Pretrained resnet
        - How does this vary with scale???
"""

from collections import defaultdict, OrderedDict
import os
import random

import fire
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision as tv


def exponential_decay(cls_id, base_size, alpha=0.8):
    # Assume cls_id start from 0.
    return int(alpha ** (cls_id + 1) * base_size)


def power_law_decay(cls_id, base_size, alpha=0.8):
    return int((cls_id + 1) ** (-alpha) * base_size)


def make_loaders(root=None, decay_type="power_law", base_size=5000, train_batch_size=1024, test_batch_size=1024):
    if root is None:
        root = os.path.join(os.path.expanduser("~"), 'data')
        os.makedirs(root, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = tv.datasets.CIFAR10(root, transform=transform_train, train=True, download=True)
    test_data = tv.datasets.CIFAR10(root, transform=transform_test, train=False, download=True)

    # Collect indices by class.
    per_class_list = defaultdict(list)
    for ind, x in enumerate(train_data):
        _, label = x
        per_class_list[int(label)].append(ind)
    per_class_list = OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))

    decay_fn = {"power_law": power_law_decay, "exponential": exponential_decay}[decay_type]
    cls_ids = per_class_list.keys()
    cls_sizes = [decay_fn(cls_id=cls_id, base_size=base_size) for cls_id in cls_ids]

    # Allocate indices by class.
    example_ids = []
    for cls_size, (cls_id, indices) in zip(cls_sizes, per_class_list.items()):
        random.shuffle(indices)
        example_ids.extend(indices[:cls_size])

    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(example_ids),
        drop_last=True,
    )
    test_loader = DataLoader(test_data, batch_size=test_batch_size, drop_last=False)
    return train_loader, test_loader, sum(cls_sizes)


def main():
    train_loader, test_loader, sample_size = make_loaders()


if __name__ == "__main__":
    fire.Fire(main)
