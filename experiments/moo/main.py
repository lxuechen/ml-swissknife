"""
python -m moo.main
"""

import fire
import numpy as np
import torch
from torch import nn, optim
import tqdm

from swissknife import utils


def make_data(n_train, n_test, d, obs_noise_std=1):
    beta = torch.randn(d, 1).abs() * torch.randn(d, 1).sign()

    # TODO: Is mean shift really a problem?
    mu1 = torch.full(size=(d,), fill_value=-0.5)
    mu2 = torch.full(size=(d,), fill_value=0.5)
    std1 = torch.cat([torch.randn(d // 2) * .3, torch.randn(d // 2) * 3.])
    std2 = torch.cat([torch.randn(d // 2) * 3., torch.randn(d // 2) * .3])
    x1_train = torch.randn(n_train // 2, d) * std1[None, :] + mu1[None, :]
    x2_train = torch.randn(n_train // 2, d) * std2[None, :] + mu2[None, :]
    y1_train = x1_train @ beta
    y2_train = x2_train @ beta

    x1_test = torch.randn(n_test // 2, d) * std1[None, :] + mu1[None, :]
    x2_test = torch.randn(n_test // 2, d) * std2[None, :] + mu2[None, :]
    y1_test = x1_test @ beta
    y2_test = x2_test @ beta

    y1_train.add_(torch.randn_like(y1_train) * obs_noise_std)
    y2_train.add_(torch.randn_like(y2_train) * obs_noise_std)

    return x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test


def train_and_eval(
    x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
    eta, lr, train_steps,
):
    model = nn.Linear(x1_train.size(1), 1)
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    for global_step in range(train_steps):
        optimizer.zero_grad()
        loss1 = ((model(x1_train) - y1_train) ** 2.).sum(dim=1).mean(dim=0)
        loss2 = ((model(x2_train) - y2_train) ** 2.).sum(dim=1).mean(dim=0)
        loss = (torch.stack([loss1, loss2]) * eta).sum()
        loss.backward()
        optimizer.step()
        print(loss1, loss2, loss, eta)  # Sanity check loss stabilizes.

    with torch.no_grad():
        loss1 = ((model(x1_test) - y1_test) ** 2.).sum(dim=1).mean(dim=0)
        loss2 = ((model(x2_test) - y2_test) ** 2.).sum(dim=1).mean(dim=0)
    return loss1.item(), loss2.item()


def brute_force(
    x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
    etas, lr, train_steps
):
    losses1 = []
    losses2 = []
    for eta in tqdm.tqdm(etas):
        eta = float(eta)
        eta = torch.tensor([eta, 1. - eta])
        loss1, loss2 = train_and_eval(
            x1_train, y1_train, x2_train, y2_train,
            x1_test, y1_test, x2_test, y2_test,
            eta=eta, lr=lr, train_steps=train_steps,
        )
        losses1.append(loss1)
        losses2.append(loss2)
    return losses1, losses2


def first_order():
    pass


def main(n_train=40, n_test=300, d=20, lr=1e-2, train_steps=10000):
    utils.manual_seed(62)

    (x1_train, y1_train, x2_train, y2_train,
     x1_test, y1_test, x2_test, y2_test) = make_data(n_train=n_train, n_test=n_test, d=d)

    losses1, losses2 = brute_force(
        x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
        etas=np.linspace(0.2, 0.8, num=31),
        lr=lr, train_steps=train_steps,
    )
    plots = [dict(x=losses1, y=losses2, marker='x')]
    utils.plot_wrapper(
        plots=plots,
        options=dict(xlabel='group1 loss', ylabel='group2 loss')
    )


if __name__ == "__main__":
    fire.Fire(main)
