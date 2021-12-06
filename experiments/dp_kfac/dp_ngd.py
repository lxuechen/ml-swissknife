# Knobs: dampening, power
import os

import fire
import numpy as np
from private_transformers.privacy_utils.accounting import rdp_accounting
import torch

from swissknife import utils

DEFAULT_ALPHAS = tuple(1 + x / 10.0 for x in range(1, 100)) + tuple(range(12, 64))


def make_data(
    d,
    n_train, n_test, n_unlabeled,
    offdiag_scale=2, diag_scale=10, noise_std=1e-3, beta_std=3, beta_mean=0.
):
    beta = beta_mean + torch.randn(d) * beta_std  # True coefficients.

    n = n_train + n_test
    mean = torch.randn(d)
    # Covariance is AA^t + diag_embed(1/i); highly anisotropic covariance.
    A = torch.randn(d, d) * offdiag_scale
    covariance = A @ A.t() + torch.diag_embed(torch.tensor([1 / i for i in range(1, d + 1)])) * diag_scale
    inverse_covariance = torch.inverse(covariance)
    x = mean[None, :] + torch.randn(n, d) @ torch.cholesky(covariance).t()
    y = x @ beta + torch.randn(size=(n,)) * noise_std

    x_train, x_test = x.split((n_train, n_test), dim=0)
    y_train, y_test = y.split((n_train, n_test), dim=0)

    x_unlabeled = mean[None, :] + torch.randn(n_unlabeled, d) @ torch.cholesky(covariance).t()
    sample_covariance = torch.tensor(np.cov(x_unlabeled.t().numpy()), dtype=torch.get_default_dtype())

    return {
        'mean': mean,
        'covariance': covariance,
        'sample_covariance': sample_covariance,
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'beta': beta,
        'inverse_covariance': inverse_covariance,
        'n_train': n_train,
        'n_test': n_test,
    }


def squared_loss(x, y, w):
    return .5 * ((x @ w - y) ** 2).mean(dim=0)


@torch.no_grad()
def dp_gd(
    state, x, y,
    clipping_norm, noise_multiplier, lr,
    steps,  # The iteration counter before the update.
    momentum=0,
    **_,
):
    w = state["w"]
    v = state["v"]

    grad_sample = (x @ w - y)[:, None] * x  # (n, d)

    norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
    coef_sample = torch.clamp_max(clipping_norm / (norm_sample + 1e-7), 1.)
    summed_grad = torch.einsum("i,id->d", coef_sample, grad_sample)

    noise = torch.randn_like(w) * noise_multiplier * clipping_norm
    grad = (summed_grad + noise) / y.size(0)

    v_next = momentum * v + grad
    w_next = w - lr * v_next

    loss = squared_loss(x, y, w_next)

    return {
        "loss": loss,
        "state": {"w": w_next, "v": v_next},
        "grad": grad,
        "noise": noise,
        "summed_grad": summed_grad,
        "steps": steps,
    }


@torch.no_grad()
def dp_pgd(
    state, x, y, P,
    clipping_norm, noise_multiplier, lr,
    steps,  # The iteration counter before the update.
    **_,
):
    """Precondition before clipping and noising."""
    w = state["w"]

    grad_sample = (x @ w - y)[:, None] * x  # (n, d)
    grad_sample = grad_sample @ P.t()  # (n, d)

    norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
    coef_sample = torch.clamp_max(clipping_norm / (norm_sample + 1e-7), 1.)
    summed_grad = torch.einsum("i,id->d", coef_sample, grad_sample)

    noise = torch.randn_like(w) * noise_multiplier * clipping_norm
    grad = (summed_grad + noise) / y.size(0)
    w_next = w - lr * grad

    loss = squared_loss(x, y, w)

    return {
        "loss": loss,
        "state": {"w": w_next},
        "grad": grad,
        "noise": noise,
        "summed_grad": summed_grad,
        "steps": steps,
    }


@torch.no_grad()
def dp_pgd2(
    state, x, y, P,
    clipping_norm, noise_multiplier, lr,
    steps,  # The iteration counter before the update.
    **_,
):
    """Precondition after clipping and noising.

    Privacy follows from post-processing.
    """
    w = state["w"]

    grad_sample = (x @ w - y)[:, None] * x  # (n, d)

    norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
    coef_sample = torch.clamp_max(clipping_norm / (norm_sample + 1e-7), 1.)
    summed_grad = torch.einsum("i,id->d", coef_sample, grad_sample)

    noise = torch.randn_like(w) * noise_multiplier * clipping_norm
    grad = (summed_grad + noise) / y.size(0)

    grad = grad @ P.t()
    w_next = w - lr * grad

    loss = squared_loss(x, y, w)
    return {
        "loss": loss,
        "state": {"w": w_next},
        "grad": grad,
        "noise": noise,
        "summed_grad": summed_grad,
        "steps": steps,
    }


def train(
    data,
    algo,
    clipping_norm,
    noise_multiplier,
    delta,
    T,
    lr,
    damping,
    momentum,
    batch_size,
    verbose=False,
):
    results = dict(global_step=[], train_loss=[], test_loss=[], dist2opt=[])

    state = dict(w=torch.zeros_like(data["beta"]), v=torch.zeros_like(data["beta"]))
    if verbose:
        train_loss = squared_loss(data["x_train"], data["y_train"], state["w"])
        test_loss = squared_loss(data["x_test"], data["y_test"], state["w"])
        dist2opt = torch.norm(data["beta"] - state["w"])
        print(
            f"Before training: "
            f"train_loss: {train_loss:.4f}, "
            f"test_loss: {test_loss:.4f}, "
            f"distance to optimum: {dist2opt:.4f}"
        )

    # Shared keyword argument for the algorithm.
    kwargs = dict(lr=lr, clipping_norm=clipping_norm, noise_multiplier=noise_multiplier, momentum=momentum)

    # Preconditioner.
    P_ng_oracle = torch.inverse(torch.eye(data["beta"].size(0)) * damping + data["covariance"])
    P_ng = torch.inverse(torch.eye(data["beta"].size(0)) * damping + data["sample_covariance"])

    for global_step in range(T):
        # Sample mini-batch; slightly different from epoch-based.
        permutation = torch.randperm(data["n_train"])
        indices = permutation[:batch_size]
        x_train, y_train = data["x_train"][indices], data["y_train"][indices]

        # Run optimizer.
        if algo == "ng":
            result = dp_pgd(x=x_train, y=y_train, state=state, P=P_ng, steps=global_step, **kwargs)
        elif algo == "ng_oracle":
            result = dp_pgd(x=x_train, y=y_train, state=state, P=P_ng_oracle, steps=global_step, **kwargs)
        elif algo == "ng2":
            result = dp_pgd2(x=x_train, y=y_train, state=state, P=P_ng, steps=global_step, **kwargs)
        elif algo == "ng2_oracle":
            result = dp_pgd2(x=x_train, y=y_train, state=state, P=P_ng_oracle, steps=global_step, **kwargs)
        elif algo == "gd":
            result = dp_gd(x=x_train, y=y_train, state=state, steps=global_step, **kwargs)
        else:
            raise ValueError(f"Unknown algo: {algo}")

        state = result["state"]
        train_loss = result["loss"]
        test_loss = squared_loss(data["x_test"], data["y_test"], state["w"])
        dist2opt = torch.norm(data["beta"] - state["w"])

        results['global_step'].append(global_step)
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['dist2opt'].append(dist2opt)

        if verbose:
            print(
                f"global_step: {global_step}, "
                f"train_loss: {train_loss:.4f}, "
                f"test_loss: {test_loss:.4f}, "
                f"distance to optimum: {dist2opt:.4f}"
            )

    train_loss = squared_loss(data["x_train"], data["y_train"], state["w"])
    test_loss = squared_loss(data["x_test"], data["y_test"], state["w"])
    dist2opt = torch.norm(data["beta"] - state["w"])
    rdp = rdp_accounting.compute_rdp(
        q=batch_size / data["n_train"],
        noise_multiplier=noise_multiplier,
        steps=T,
        orders=DEFAULT_ALPHAS
    )
    epsilon, _ = rdp_accounting.get_privacy_spent(orders=DEFAULT_ALPHAS, rdp=rdp, delta=delta)

    print(
        f"After training: algo: {algo}, "
        f"train_loss: {train_loss:.4f}, "
        f"test_loss: {test_loss:.4f}, "
        f"dist2opt: {dist2opt:.4f}, "
        f"epsilon: {epsilon:.2f}"
    )

    return results


def _main(
    d=10,
    n_train=50000,
    n_test=1000,
    n_unlabeled=10000,
    clipping_norm=0.1,
    noise_multiplier=1.,
    T=100,
    lr=0.1,
    algo="gd",
    delta=2e-4,
    damping=1e-4,
    noise_std=1e-3,
    batch_size=1024,
    momentum=0,
    data=None,
):
    if data is None:
        data = make_data(d, n_train, n_test, n_unlabeled, noise_std=noise_std)
    return train(
        data,
        algo=algo,
        clipping_norm=clipping_norm,
        noise_multiplier=noise_multiplier,
        T=T,
        lr=lr,
        delta=delta,
        damping=damping,
        momentum=momentum,
        batch_size=batch_size,
    )


def default(
    T=500,
    n_train=50000,
    n_test=10000,
    n_unlabeled=10000,
    label_noise_std=1e-1,
    noise_multiplier=20,
    damping=0,
    momentum=0,
    batch_size=2048,
    seed=42,
    ng_lr=10,
    gd_lr=1,
    ng_clipping_norm=1.,
    gd_clipping_norm=1.,
    d=10,
    plot_ng2=False
):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)

    shared_kwargs = dict(n_train=n_train, n_test=n_test, n_unlabeled=n_unlabeled)
    ng_kwargs = {
        **dict(
            lr=ng_lr,
            noise_multiplier=noise_multiplier,
            clipping_norm=ng_clipping_norm,
            T=T,
            damping=damping,
            noise_std=label_noise_std,
            batch_size=batch_size,
            d=d
        ),
        **shared_kwargs,
    }
    ng_oracle_results = _main(algo="ng_oracle", **ng_kwargs)
    ng_results = _main(algo="ng", **ng_kwargs)
    ng2_oracle_results = _main(algo="ng2_oracle", **ng_kwargs)
    ng2_results = _main(algo="ng2", **ng_kwargs)

    gd_kwargs = {
        **dict(
            lr=gd_lr,
            noise_multiplier=noise_multiplier,
            clipping_norm=gd_clipping_norm,
            T=T,
            algo="gd",
            noise_std=label_noise_std,
            batch_size=batch_size,
            d=d,
            momentum=momentum,
        ),
        **shared_kwargs,
    }
    gd_results = _main(**gd_kwargs)

    plots = (
        {'x': ng_results['global_step'], 'y': ng_results['test_loss'], 'label': 'ng'},
        {'x': ng_oracle_results["global_step"], 'y': ng_oracle_results['test_loss'], 'label': 'ng (oracle)'},
        {'x': gd_results['global_step'], 'y': gd_results['test_loss'], 'label': 'gd'},
    )
    if plot_ng2:
        plots = plots + (
            {'x': ng2_results['global_step'], 'y': ng2_results['test_loss'], 'label': 'ng2'},
            {'x': ng2_oracle_results["global_step"], 'y': ng2_oracle_results['test_loss'], 'label': 'ng2 (oracle)'},
        )
    options = {'xlabel': 'Iteration', 'ylabel': 'Test Loss', 'yscale': 'log'}
    img_path = os.path.join('.', 'plots', 'dp_ng')
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=('.png', '.pdf'),
        plots=plots,
        options=options,
    )

    plots = (
        {'x': ng_results['global_step'], 'y': ng_results['dist2opt'], 'label': 'ng'},
        {'x': ng_oracle_results["global_step"], 'y': ng_oracle_results['dist2opt'], 'label': 'ng (oracle)'},
        {'x': gd_results['global_step'], 'y': gd_results['dist2opt'], 'label': 'gd'},
    )
    if plot_ng2:
        plots = plots + (
            {'x': ng2_results['global_step'], 'y': ng2_results['dist2opt'], 'label': 'ng2'},
            {'x': ng2_oracle_results["global_step"], 'y': ng2_oracle_results['dist2opt'], 'label': 'ng2 (oracle)'},
        )
    options = {'xlabel': 'Iteration', 'ylabel': '$\|\| \hat{\\beta} - \\beta \|\|_2 $', 'yscale': 'log'}
    img_path = os.path.join('.', 'plots', 'dp_ng_beta')
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=('.png', '.pdf'),
        plots=plots,
        options=options,
    )


def sweep_gd(
    seeds=(0, 1, 2),
    lrs=(0.5, 1, 2),
    clipping_norms=(0.1, 0.3, 1, 3, 10),

    T=500,
    n_train=50000,
    n_test=10000,
    n_unlabeled=10000,
    label_noise_std=1e-1,
    noise_multiplier=20,
    momentum=0,
    batch_size=2048,
    d=10,
    data=None,
):
    if data is None:
        data = make_data(d, n_train, n_test, n_unlabeled, noise_std=label_noise_std)

    shared_kwargs = dict(n_train=n_train, n_test=n_test, n_unlabeled=n_unlabeled)

    errorbars = []
    for lr in lrs:
        for clipping_norm in clipping_norms:
            x, y = [], []
            for seed in seeds:
                torch.manual_seed(seed)
                gd_kwargs = {
                    **dict(
                        lr=lr,
                        noise_multiplier=noise_multiplier,
                        clipping_norm=clipping_norm,
                        T=T,
                        algo="gd",
                        noise_std=label_noise_std,
                        batch_size=batch_size,
                        d=d,
                        momentum=momentum,
                        data=data,
                    ),
                    **shared_kwargs,
                }
                gd_results = _main(**gd_kwargs)
                x.append(gd_results["global_step"])
                y.append(gd_results['dist2opt'])

            x, _ = utils.average_over_seed(x)
            y_mean, y_std = utils.average_over_seed(y)
            errorbars.append({"x": x, "y": y_mean, "yerr": y_std, "label": f"C={clipping_norm}, lr={lr}"})

    img_path = os.path.join('.', 'plots', 'dp_gd_sweep')
    utils.plot(img_path=img_path, errorbars=errorbars,
               options={"title": "gd", "yscale": "log", 'xlabel': 'Iteration', 'ylabel': '$\| \\beta - \\beta_*\| $'})

    best_errorbar = sorted(errorbars, key=lambda this_dict: this_dict["y"][-1])[0]
    best_errorbar["label"] = "gd"
    return best_errorbar


def sweep_ng(
    seeds=(0, 1, 2),
    lrs=(0.5, 1, 10),
    clipping_norms=(0.1, 0.3, 1, 3, 10),

    T=500,
    n_train=50000,
    n_test=10000,
    n_unlabeled=10000,
    label_noise_std=1e-1,
    noise_multiplier=20,
    batch_size=2048,
    damping=1e-4,
    d=10,
    data=None,
):
    if data is None:
        data = make_data(d, n_train, n_test, n_unlabeled, noise_std=label_noise_std)

    shared_kwargs = dict(n_train=n_train, n_test=n_test, n_unlabeled=n_unlabeled)
    errorbars = []
    for lr in lrs:
        for clipping_norm in clipping_norms:
            x, y = [], []
            for seed in seeds:
                torch.manual_seed(seed)
                ng_kwargs = {
                    **dict(
                        lr=lr,
                        noise_multiplier=noise_multiplier,
                        clipping_norm=clipping_norm,
                        T=T,
                        damping=damping,
                        noise_std=label_noise_std,
                        batch_size=batch_size,
                        d=d,
                        data=data,
                        algo="ng"
                    ),
                    **shared_kwargs,
                }
                results = _main(**ng_kwargs)
                x.append(results["global_step"])
                y.append(results['dist2opt'])

            x, _ = utils.average_over_seed(x)
            y_mean, y_std = utils.average_over_seed(y)
            errorbars.append({"x": x, "y": y_mean, "yerr": y_std, "label": f"C={clipping_norm}, lr={lr}"})

    img_path = os.path.join('.', 'plots', 'dp_ng_sweep')
    utils.plot(img_path=img_path, errorbars=errorbars,
               options={"title": "ng", "yscale": "log", 'xlabel': 'Iteration', 'ylabel': '$\| \\beta - \\beta_*\| $'})

    best_errorbar = sorted(errorbars, key=lambda this_dict: this_dict["y"][-1])[0]
    best_errorbar["label"] = "ng"
    return best_errorbar


def sweep_ng2(
    seeds=(0, 1, 2),
    lrs=(0.5, 1, 10),
    clipping_norms=(0.1, 0.3, 1, 3, 10),

    T=500,
    n_train=50000,
    n_test=10000,
    n_unlabeled=10000,
    label_noise_std=1e-1,
    noise_multiplier=20,
    momentum=0,
    batch_size=2048,
    damping=1e-4,
    d=10,
    data=None,
):
    if data is None:
        data = make_data(d, n_train, n_test, n_unlabeled, noise_std=label_noise_std)

    shared_kwargs = dict(n_train=n_train, n_test=n_test, n_unlabeled=n_unlabeled)

    errorbars = []
    for lr in lrs:
        for clipping_norm in clipping_norms:
            x, y = [], []
            for seed in seeds:
                torch.manual_seed(seed)
                kwargs = {
                    **dict(
                        lr=lr,
                        noise_multiplier=noise_multiplier,
                        clipping_norm=clipping_norm,
                        T=T,
                        damping=damping,
                        noise_std=label_noise_std,
                        batch_size=batch_size,
                        d=d,
                        momentum=momentum,
                        data=data,
                        algo="ng2",
                    ),
                    **shared_kwargs,
                }
                results = _main(**kwargs)
                x.append(results["global_step"])
                y.append(results['dist2opt'])

            x, _ = utils.average_over_seed(x)
            y_mean, y_std = utils.average_over_seed(y)
            errorbars.append({"x": x, "y": y_mean, "yerr": y_std, "label": f"C={clipping_norm}, lr={lr}"})

    img_path = os.path.join('.', 'plots', 'dp_ng2_sweep')
    utils.plot(img_path=img_path, errorbars=errorbars,
               options={"title": "ng2", "yscale": "log", 'xlabel': 'Iteration', 'ylabel': '$\| \\beta - \\beta_*\| $'})

    best_errorbar = sorted(errorbars, key=lambda this_dict: this_dict["y"][-1])[0]
    best_errorbar["label"] = "ng2"
    return best_errorbar


def main(task, **kwargs):
    if task == "default":
        default(**kwargs)
    elif task == "sweep_gd":
        sweep_gd(**kwargs)
    elif task == "sweep_ng":
        sweep_ng(**kwargs)
    elif task == "sweep_ng2":
        sweep_ng2(**kwargs)
    elif task == "sweep_all":
        # d, n_train, n_test, n_unlabeled = 10, 50000, 1000, 10000,
        # data = make_data(d, n_train, n_test, n_unlabeled)
        # seeds = (0, 1, 2, 3, 4)

        # Also average over data sampling.
        data = None
        seeds = list(range(20))
        gd_errorbar = sweep_gd(data=data, seeds=seeds)
        ng_errorbar = sweep_ng(data=data, seeds=seeds)
        ng2_errorbar = sweep_ng2(data=data, seeds=seeds)

        img_path = os.path.join('.', 'plots', 'compare_best')
        utils.plot(
            img_path=img_path,
            errorbars=(gd_errorbar, ng2_errorbar, ng_errorbar),
            options={"yscale": 'log', "title": "comparing algos with best respective hparams",
                     'xlabel': 'Iteration', 'ylabel': '$\| \\beta - \\beta_*\| $'},
        )
    else:
        raise ValueError()


if __name__ == "__main__":
    # This example only seems to work so far as the data covariance is highly non-trivial.

    # The following command's unlabeled data is reasonable, and learning rate for gradient descent is optimized.
    # Slightly tuning momentum more gives better results.
    # @formatter:off
    # python3  dp_ngd.py --noise_multiplier 5 --ng_lr 1e0 --T 1000 --ng_clipping_norm 0.1 --damping 1e-6 --n_unlabeled 5000 --d 10 --gd_lr 1e0 --gd_clipping_norm 0.1 --task default
    # @formatter:on

    # dp_pgd1 (precondition -> clip + noise) needs small clipping norms
    # dp_pgd2 (clip + noise -> precondition) needs large clipping norms
    fire.Fire(main)
