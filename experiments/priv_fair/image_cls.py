"""
Image classification.

Goals:
    How much is the disparate impact when pre-trained models are used???
        - CLIP
        - Pretrained resnet
        - How does this vary with scale???

Run from root:
    python -m experiments.priv_fair.image_cls
"""

from collections import defaultdict, OrderedDict
import os
import random

import fire
from opacus import PrivacyEngine
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision as tv

from swissknife import utils
from ..simclrv2.resnet import get_resnet

base_dir = "/home/lxuechen_stanford_edu/software/swissknife/experiments/simclrv2"


def exponential_decay(cls_id, base_size, alpha=0.8):
    # Assume cls_id starts from 0.
    return int(alpha ** (cls_id + 1) * base_size)


def power_law_decay(cls_id, base_size, alpha=0.8):
    return int((cls_id + 1) ** (-alpha) * base_size)


def make_loaders(
    root=None, decay_type="power_law", base_size=5000, train_batch_size=1024, test_batch_size=1024,
    alpha=0.9
):
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
    cls_sizes = [decay_fn(cls_id=cls_id, base_size=base_size, alpha=alpha) for cls_id in cls_ids]

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


class SimCLRv2(nn.Module):
    def __init__(self, depth=50, width_multiplier=1, sk_ratio=0, n_classes=10):
        super(SimCLRv2, self).__init__()
        resnet, original_head = get_resnet(depth=depth, width_multiplier=width_multiplier, sk_ratio=sk_ratio)

        checkpoint_path = os.path.join(base_dir, f'r{depth}_{width_multiplier}x_sk{sk_ratio}_ema.pth')
        state_dicts = torch.load(checkpoint_path)
        resnet.load_state_dict(state_dicts["resnet"])
        original_head.load_state_dict(state_dicts["head"])

        self.resnet = resnet.requires_grad_(False)
        self.new_head = nn.Linear(2048, n_classes)

    def forward(self, x):
        features = self.resnet(x)
        return self.new_head(features)


def test(model, loader, device, max_batches=100):
    zeons, xents = [], []
    for i, tensors in loader:
        if i >= max_batches:
            break
        x, y = tuple(t.to(device) for t in tensors)
        y_hat = model(x)
        zeons.extend(torch.eq(y_hat.argmax(dim=1), y).cpu().tolist())
        xents.extend(F.cross_entropy(y_hat, y, reduction="none").cpu().tolist())
    return sum(zeons) / len(zeons), sum(xents) / len(xents)


def train(model, optimizer, num_epoch, train_loader, test_loader, device):
    for epoch in range(num_epoch):
        for tensors in train_loader:
            optimizer.zero_grad()
            x, y = tuple(t.to(device) for t in tensors)
            y_hat = model(x)
            loss = F.cross_entropy(y_hat, y, reduction="none")
            loss = loss.sum(dim=0)
            loss.backward()
            optimizer.step()
        avg_zeon, avg_xent = test(model, test_loader, device=device)
        print(f"Epoch {epoch}, avg_zeon: {avg_zeon}, avg_xent: {avg_xent}")


def main(lr=1e-1, momentum=0.9, num_epoch=10, target_epsilon=3, target_delta=1e-5,
         train_batch_size=1024, test_batch_size=1024, max_grad_norm=1, alpha=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, sample_size = make_loaders(
        train_batch_size=train_batch_size, test_batch_size=test_batch_size, alpha=alpha,
    )
    model = SimCLRv2().to(device)
    optimizer = optim.SGD(utils.trainable_parameters(model), lr=lr, momentum=momentum)
    privacy_engine = PrivacyEngine(
        module=model,
        batch_size=train_batch_size,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_size=sample_size,
        loss_reduction="sum",
        max_grad_norm=max_grad_norm,
        epochs=num_epoch,
    )
    privacy_engine.attach(optimizer=optimizer)

    train(
        model=model, optimizer=optimizer, num_epoch=num_epoch,
        train_loader=train_loader, test_loader=test_loader, device=device
    )


if __name__ == "__main__":
    fire.Fire(main)
