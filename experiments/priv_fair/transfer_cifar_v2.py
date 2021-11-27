"""Another shameless copy from Florian's codebase.

Goals:
    How much is the disparate impact when pre-trained models are used???
        - CLIP
        - Pretrained resnet
        - SimCLR features
        - How does this vary with scale

@formatter:off
Run from root:
    python -m experiments.priv_fair.transfer_cifar_v2 --feature_path "simclr_r101_2x_sk0" --batch_size=1024 --lr=4 --noise_multiplier=2.40
@formatter:on

Put all the converted features in `base_dir`
"""
# TODO:
#   3) disparate impact vs scale,
#   4) increasing majority group help (positive transfer)?

import argparse
import collections
import os
import random

import numpy as np
from opacus import PrivacyEngine
import torch
import torch.nn as nn

from swissknife import utils
from .misc.data import get_data
from .misc.dp_utils import ORDERS, get_privacy_spent, get_renyi_divergence
from .misc.log import Logger
from .misc.train_utils import train, test, test_by_groups
from .shared import exponential_decay, power_law_decay


def get_cifar10_imbalanced_tensor_dataset(base_dir, feature_path, decay_type="exponential", alpha=0.9, base_size=5000):
    # base_size: Sample size for the largest class.
    x_train = np.load(os.path.join(base_dir, f"{feature_path}_train.npy"))
    x_test = np.load(os.path.join(base_dir, f"{feature_path}_test.npy"))

    train_data, test_data = get_data("cifar10", augment=False)
    y_train = np.asarray(train_data.targets)
    y_test = np.asarray(test_data.targets)

    # Imbalance training set.
    per_class_list = collections.defaultdict(list)
    for ind, x in enumerate(train_data):
        _, label = x
        per_class_list[int(label)].append(ind)
    per_class_list = collections.OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))

    decay_fn = {"power_law": power_law_decay, "exponential": exponential_decay}[decay_type]
    cls_ids = per_class_list.keys()
    cls_sizes = [decay_fn(cls_id=cls_id, base_size=base_size, alpha=alpha) for cls_id in cls_ids]
    print(f'class sizes: {cls_sizes}')

    # Allocate indices by class.
    example_ids = []
    for cls_size, (cls_id, indices) in zip(cls_sizes, per_class_list.items()):
        random.shuffle(indices)
        example_ids.extend(indices[:cls_size])

    x_train, y_train = x_train[example_ids], y_train[example_ids]
    trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return trainset, testset


def get_cifar10_tensor_dataset(base_dir, feature_path):
    x_train = np.load(os.path.join(base_dir, f"{feature_path}_train.npy"))
    x_test = np.load(os.path.join(base_dir, f"{feature_path}_test.npy"))

    train_data, test_data = get_data("cifar10", augment=False)
    y_train = np.asarray(train_data.targets)
    y_test = np.asarray(test_data.targets)

    trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return trainset, testset


def non_private_training(
    feature_path=None, batch_size=2048, mini_batch_size=256,
    lr=1, optim="SGD", momentum=0.9, nesterov=False, epochs=100, logdir=None,
    base_dir="/nlp/scr/lxuechen/features", train_dir=None, seed=0, imba=False, alpha=0.9,
    **kwargs
):
    utils.manual_seed(seed)
    logger = Logger(logdir)

    if imba:
        trainset, testset = get_cifar10_imbalanced_tensor_dataset(
            base_dir=base_dir, feature_path=feature_path, alpha=alpha
        )
    else:
        trainset, testset = get_cifar10_tensor_dataset(base_dir=base_dir, feature_path=feature_path)

    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=mini_batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=mini_batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    # Get the shape of features.
    x_batch, y_batch = next(iter(train_loader))
    n_features = x_batch.size(-1)
    model = nn.Sequential(nn.Linear(n_features, 10)).to(device)

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(0, epochs):
        print(f"\nEpoch: {epoch}")

        train_loss, train_acc = train(model, train_loader, optimizer, n_acc_steps=n_acc_steps)
        test_loss, test_acc = test(model, test_loader)
        logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc)

        train_zeon_by_groups, train_xent_by_groups = test_by_groups(model, train_loader)
        test_zeon_by_groups, test_xent_by_groups = test_by_groups(model, test_loader)

        history.append(
            dict(
                epoch=epoch, train_xent=train_loss, train_zeon=train_acc, test_xent=test_loss, test_zeon=test_acc,
                train_zeon_by_groups=train_zeon_by_groups, train_xent_by_groups=train_xent_by_groups,
                test_zeon_by_groups=test_zeon_by_groups, test_xent_by_groups=test_xent_by_groups
            )
        )

    if train_dir is not None:
        dump_path = os.path.join(train_dir, 'log_history.json')
        utils.jdump(history, dump_path)


def private_training(
    feature_path=None, batch_size=2048, mini_batch_size=256,
    lr=1, optim="SGD", momentum=0.9, nesterov=False, noise_multiplier=1,
    max_grad_norm=0.1, max_epsilon=None, epochs=100, logdir=None,
    base_dir="/nlp/scr/lxuechen/features", train_dir=None, seed=0,
    imba=False, alpha=0.9,
):
    utils.manual_seed(seed)
    logger = Logger(logdir)

    if imba:
        trainset, testset = get_cifar10_imbalanced_tensor_dataset(
            base_dir=base_dir, feature_path=feature_path, alpha=alpha
        )
    else:
        trainset, testset = get_cifar10_tensor_dataset(base_dir=base_dir, feature_path=feature_path)

    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=mini_batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=mini_batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    # Get the shape of features.
    x_batch, y_batch = next(iter(train_loader))
    n_features = x_batch.size(-1)
    model = nn.Sequential(nn.Linear(n_features, 10)).to(device)

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine(
        model,
        sample_rate=bs / len(trainset),
        alphas=ORDERS,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    privacy_engine.attach(optimizer)
    print(f'sample size: {len(trainset)}')

    history = []
    for epoch in range(0, epochs):
        print(f"\nEpoch: {epoch}")

        train_loss, train_acc = train(model, train_loader, optimizer, n_acc_steps=n_acc_steps)
        test_loss, test_acc = test(model, test_loader)

        if noise_multiplier > 0:
            rdp_sgd = get_renyi_divergence(
                privacy_engine.sample_rate, privacy_engine.noise_multiplier
            ) * privacy_engine.steps
            epsilon, _ = get_privacy_spent(rdp_sgd)
            print(f"ε = {epsilon:.3f}")

            if max_epsilon is not None and epsilon >= max_epsilon:
                return
        else:
            epsilon = None

        logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc, epsilon)

        train_zeon_by_groups, train_xent_by_groups = test_by_groups(model, train_loader)
        test_zeon_by_groups, test_xent_by_groups = test_by_groups(model, test_loader)

        history.append(
            dict(
                epoch=epoch, train_xent=train_loss, train_zeon=train_acc, test_xent=test_loss, test_zeon=test_acc,
                train_zeon_by_groups=train_zeon_by_groups, train_xent_by_groups=train_xent_by_groups,
                test_zeon_by_groups=test_zeon_by_groups, test_xent_by_groups=test_xent_by_groups
            )
        )

    if train_dir is not None:
        dump_path = os.path.join(train_dir, 'log_history.json')
        utils.jdump(history, dump_path)


def main(task="private_training", **kwargs):
    if task == "private":
        private_training(**kwargs)
    elif task == "non_private":
        # Hyperparameters for private training seems to also work.
        non_private_training(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optim', type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action="store_true")
    parser.add_argument('--noise_multiplier', type=float, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--feature_path', default=None)
    parser.add_argument('--max_epsilon', type=float, default=None)
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--base_dir', default="/nlp/scr/lxuechen/features", type=str)
    parser.add_argument('--train_dir', default=None, type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=str, default="private", choices=("private", "non_private"))
    parser.add_argument('--imba', type=utils.str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--alpha', type=float, default=0.9, help="Decay rate for power law or exponential.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(**vars(args))
