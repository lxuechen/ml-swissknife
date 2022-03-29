"""
python -m moo.main
"""

import fire
import torch
from torch import nn, optim
import tqdm

from swissknife import utils


def make_data(n_train, n_test, d, obs_noise_std=1):
    """Mixture of Gaussian features; two clusters have different mean and covariance."""
    beta = torch.randn(d, 1).abs() * torch.randn(d, 1).sign()

    mu1 = torch.full(size=(d,), fill_value=-0.4)
    mu2 = torch.full(size=(d,), fill_value=0.4)
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


def compute_mse(x, y, model):
    return .5 * ((model(x) - y) ** 2.).sum(1).mean(0)


def compute_covar(x, bias=False):
    if bias:
        x = torch.cat([x, torch.ones(size=(x.size(0), 1), dtype=x.dtype, device=x.device)], dim=1)
    return x.t() @ x / x.size(0)


def train(
    x1_train, y1_train, x2_train, y2_train,
    eta, lr, train_steps,
    model=None, optimizer=None,
):
    if model is None:
        model = nn.Linear(x1_train.size(1), 1, bias=False)
    if optimizer is None:
        optimizer = optim.SGD(params=model.parameters(), lr=lr)
    for global_step in range(train_steps):
        model.train()
        model.zero_grad()
        loss1 = compute_mse(x1_train, y1_train, model)
        loss2 = compute_mse(x2_train, y2_train, model)
        loss = (torch.stack([loss1, loss2]) * eta).sum()
        loss.backward()
        optimizer.step()
        print(loss1, loss2, loss, eta)  # Sanity check loss stabilizes.
    return model, optimizer


@torch.no_grad()
def evaluate(
    model,
    x1_test, y1_test, x2_test, y2_test,
):
    model.eval()
    loss1 = compute_mse(x1_test, y1_test, model)
    loss2 = compute_mse(x2_test, y2_test, model)
    return loss1.item(), loss2.item()


def train_and_evaluate(
    x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
    eta, lr, train_steps,
):
    model, _ = train(
        x1_train, y1_train, x2_train, y2_train,
        eta, lr, train_steps,
    )
    return evaluate(
        model,
        x1_test, y1_test, x2_test, y2_test,
    )


def brute_force(
    x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
    etas, lr, train_steps,
    show_plots=False,
):
    losses1 = []
    losses2 = []
    for eta in tqdm.tqdm(etas):
        loss1, loss2 = train_and_evaluate(
            x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
            eta=eta, lr=lr, train_steps=train_steps,
        )
        losses1.append(loss1)
        losses2.append(loss2)
    plots = [dict(x=losses1, y=losses2, marker='x')]
    if show_plots:
        utils.plot_wrapper(
            plots=plots,
            options=dict(xlabel='group1 loss', ylabel='group2 loss')
        )
    return plots


def _first_order_helper(
    model: nn.Module,  # Trained model at eta.
    x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
    eta, query_etas,
):
    """Approximate left and right function values at eta_l and eta_r with first-order expansion."""
    outs = []
    for query_eta in query_etas:
        model.zero_grad()
        tr_loss1 = compute_mse(x1_train, y1_train, model)
        tr_loss2 = compute_mse(x2_train, y2_train, model)
        tr_loss = torch.stack([tr_loss1, tr_loss2])
        vjp = utils.vjp(
            outputs=tr_loss, inputs=tuple(model.parameters()), grad_outputs=query_eta - eta,
        )
        vjp = utils.flatten(vjp)
        # Hessian is weighted uncentered empirical covariances.
        h1 = compute_covar(x1_train)
        h2 = compute_covar(x2_train)
        h = eta[0] * h1 + eta[1] * h2
        h_inv = torch.inverse(h.to(torch.float64)).float()
        h_inv_vjp = (h_inv @ vjp).detach()
        h_inv_vjp = (h_inv_vjp.reshape(1, 20),)

        model.zero_grad()
        te_loss1 = compute_mse(x1_test, y1_test, model)
        delta1, = utils.jvp(te_loss1, tuple(model.parameters()), grad_inputs=h_inv_vjp, create_graph=True)
        delta1 = delta1.detach()

        model.zero_grad()
        te_loss2 = compute_mse(x2_test, y2_test, model)
        delta2, = utils.jvp(te_loss2, tuple(model.parameters()), grad_inputs=h_inv_vjp, create_graph=True)
        delta2 = delta2.detach()
        delta = torch.stack([delta1, delta2])

        te_loss = torch.stack([te_loss1, te_loss2])
        new_te_loss = te_loss - delta

        outs.append(new_te_loss.tolist())
    return [out[0] for out in outs], [out[1] for out in outs]


def first_order(
    x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
    etas: torch.Tensor, lr, train_steps,
    show_plots=False,
    num_pts=5
):
    delta = torch.mean(etas[1:, 0] - etas[:-1, 0]) / 2.  # How far to interpolate.

    centroids = dict(x=[], y=[], marker='^', label='centroids')
    expansions = []
    for eta in etas:
        model, _ = train(
            x1_train, y1_train, x2_train, y2_train,
            eta, lr, train_steps,
        )
        loss1, loss2 = evaluate(
            model,
            x1_test, y1_test, x2_test, y2_test
        )

        query_etas_left = []
        for idx in range(num_pts):
            query_eta = eta.clone()
            query_eta[0] -= (1 - idx / num_pts) * delta
            query_eta[1] += (1 - idx / num_pts) * delta
            query_etas_left.append(query_eta)
        interps_left = _first_order_helper(
            model,
            x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
            eta=eta, query_etas=query_etas_left,
        )
        del query_etas_left

        query_etas_right = []
        for idx in range(num_pts):
            query_eta = eta.clone()
            query_eta[0] += (1 - idx / num_pts) * delta
            query_eta[1] -= (1 - idx / num_pts) * delta
            query_etas_right.append(query_eta)
        interps_right = _first_order_helper(
            model,
            x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
            eta=eta, query_etas=query_etas_right,
        )
        del query_etas_right

        expansions.append(
            dict(
                x=interps_left[0] + [loss1] + interps_right[0],
                y=interps_left[1] + [loss2] + interps_left[1],
                marker='+',
                markersize=0.5,
            )
        )
        centroids['x'].append(loss1)
        centroids['y'].append(loss2)

    # Mark the original points.
    plots = [centroids] + expansions
    if show_plots:
        utils.plot_wrapper(
            plots=plots,
            options=dict(xlabel='group1 loss', ylabel='group2 loss')
        )
    return plots


def main(n_train=40, n_test=300, d=20, lr=1e-2, train_steps=20000):
    utils.manual_seed(62)

    (x1_train, y1_train, x2_train, y2_train,
     x1_test, y1_test, x2_test, y2_test) = make_data(n_train=n_train, n_test=n_test, d=d)

    etas = torch.linspace(0.3, 0.7, steps=5)
    etas = torch.stack([etas, 1 - etas], dim=1)
    first_order_plots = first_order(
        x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
        etas=etas, lr=lr, train_steps=train_steps,
    )

    etas = torch.linspace(0.3, 0.7, steps=31)
    etas = torch.stack([etas, 1 - etas], dim=1)
    brute_force_plots = brute_force(
        x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test,
        etas=etas,
        lr=lr, train_steps=train_steps,
    )
    utils.plot_wrapper(
        plots=first_order_plots + brute_force_plots,
        options=dict(xlabel='group1 loss', ylabel='group2 loss'),
    )


if __name__ == "__main__":
    fire.Fire(main)
