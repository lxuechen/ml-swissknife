"""
python -m moo.main
"""

import fire
import torch
from torch import nn, optim

from swissknife import utils


def make_data(n_train, n_test, d, obs_noise_std=1):
    beta = torch.randn(d, 1).exp() * torch.randn(d, 1).sign()
    std1 = torch.cat([torch.randn(d // 2) * .3, torch.randn(d // 2) * 3.])
    std2 = torch.cat([torch.randn(d // 2) * 3., torch.randn(d // 2) * .3])
    x1_train = torch.randn(n_train // 2, d) * std1[None, :]
    x2_train = torch.randn(n_train // 2, d) * std2[None, :]
    y1_train = x1_train @ beta
    y2_train = x2_train @ beta

    x1_test = torch.randn(n_test // 2, d) * std1[None, :]
    x2_test = torch.randn(n_test // 2, d) * std2[None, :]
    y1_test = x1_test @ beta
    y2_test = x2_test @ beta

    y1_train.add_(torch.randn_like(y1_train) * obs_noise_std)
    y2_train.add_(torch.randn_like(y2_train) * obs_noise_std)
    y1_test.add_(torch.randn_like(y1_test) * obs_noise_std)
    y2_test.add_(torch.randn_like(y2_test) * obs_noise_std)

    return x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test


def train_and_eval(
    x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
    d, eta, lr, train_steps,
):
    model = nn.Linear(d, 1)
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    for global_step in range(train_steps):
        optimizer.zero_grad()
        loss1 = ((model(x1_train) - y1_train) ** 2.).sum(dim=1).mean(dim=0)
        loss2 = ((model(x2_train) - y2_train) ** 2.).sum(dim=1).mean(dim=0)
        loss = (torch.stack([loss1, loss2]) * eta).sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        loss1 = ((model(x1_test) - y1_test) ** 2.).sum(dim=1).mean(dim=0)
        loss2 = ((model(x2_test) - y2_test) ** 2.).sum(dim=1).mean(dim=0)
    return torch.stack([loss1, loss2])


def main(n_train=40, n_test=300, d=10, lr=1e-1, train_steps=10000):
    utils.manual_seed(42)

    (x1_train, y1_train, x2_train, y2_train,
     x1_test, y1_test, x2_test, y2_test) = make_data(n_train=n_train, n_test=n_test, d=d)

    res = train_and_eval(
        x1_train, y1_train, x2_train, y2_train,
        x1_test, y1_test, x2_test, y2_test,
        d=d, eta=torch.tensor([0.5, 0.5]), lr=lr, train_steps=train_steps,
    )
    print(res)


if __name__ == "__main__":
    fire.Fire(main)
