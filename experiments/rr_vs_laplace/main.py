"""
python main.py

exercise 6.1 from ECE377
"""

import math

import fire
import numpy as np
import torch
from torch import distributions
import tqdm

from swissknife import utils

torch.set_default_dtype(torch.float64)


def rr(x, epsilon):
    """Randomized response for continuous-valued inputs.

    First round probabilistically, then flip according to the usual RR.
    TODO: Does the first step provide any privacy amplification?
    """
    xt = torch.bernoulli(x)
    p = torch.full_like(x, fill_value=(np.e ** epsilon / (1 + np.e ** epsilon)))
    mask = torch.bernoulli(p)
    # Conditional on tilde x.
    z = mask * xt + (1 - mask) * (1 - xt)

    a = (np.e ** epsilon + 1) / (np.e ** epsilon - 1)
    b = 1 / (np.e ** epsilon - 1)
    return (a * z - b).mean().item()


def laplace(x, epsilon):
    loc = torch.zeros_like(x)
    scale = torch.full_like(x, fill_value=1 / epsilon)
    dist = distributions.laplace.Laplace(loc=loc, scale=scale)
    w = dist.sample()
    assert x.size() == w.size()
    return (x + w).mean().item()


def non_private(x):
    return x.mean().item()


def _squared_error(x, mean):
    return (x - mean) ** 2


def _make_estimates(ns, t, epsilon, sample_x):
    mse_rr = []
    mse_lp = []
    mse_np = []

    mean_rr = []
    mean_lp = []
    mean_np = []

    ci_rr = []
    ci_lp = []
    ci_np = []

    for n in ns:
        err_rr = []
        err_lp = []
        err_np = []

        x_rr_l = []
        x_lp_l = []
        x_np_l = []
        for _ in tqdm.tqdm(range(t), desc="T"):
            x, true_mean = sample_x(n)

            x_rr = rr(x, epsilon=epsilon)
            err_rr.append(_squared_error(x_rr, mean=true_mean))

            x_lp = laplace(x, epsilon=epsilon)
            err_lp.append(_squared_error(x_lp, mean=true_mean))

            x_np = non_private(x)
            err_np.append(_squared_error(x_np, mean=true_mean))

            x_rr_l.append(x_rr)
            x_lp_l.append(x_lp)
            x_np_l.append(x_np)

        mse_rr.append(torch.tensor(err_rr).mean().item())
        mse_lp.append(torch.tensor(err_lp).mean().item())
        mse_np.append(torch.tensor(err_np).mean().item())

        ci_rr.append(_ci(x_rr_l))
        ci_lp.append(_ci(x_lp_l))
        ci_np.append(_ci(x_np_l))

        mean_rr.append(np.mean(x_rr_l))
        mean_lp.append(np.mean(x_lp_l))
        mean_np.append(np.mean(x_np_l))

    err_rr = [(hi - lo) for lo, hi in ci_rr]
    err_lp = [(hi - lo) for lo, hi in ci_lp]
    err_np = [(hi - lo) for lo, hi in ci_np]

    return mse_rr, mse_lp, mse_np, ci_rr, ci_lp, ci_np, err_rr, err_lp, err_np, mean_rr, mean_lp, mean_np, true_mean


def _ci(sample, alpha=0.05):  # (Asymptotic) Confidence interval.
    alpha2zscore = {
        0.05: 1.960,
        0.1: 1.645,
        0.15: 1.440,
    }

    if isinstance(sample, (list, tuple)):
        sample = torch.tensor(sample)
    sample: torch.Tensor
    assert sample.dim() == 1
    sample_size = len(sample)
    sample_mean = sample.mean()
    sample_std = sample.std(unbiased=False)
    zscore = alpha2zscore[alpha]

    lo = sample_mean - zscore * sample_std / math.sqrt(sample_size)
    hi = sample_mean + zscore * sample_std / math.sqrt(sample_size)
    return lo, hi


def _uniform(epsilon=1, t=1000, ns=(10, 100, 1000, 10000)):
    def sample_x(sample_size):
        return torch.rand(size=(sample_size,)), 0.5

    mse_rr, mse_lp, ci_rr, ci_lp, err_rr, err_lp, mean_rr, mean_lp, true_mean = _make_estimates(
        ns=ns, t=t, epsilon=epsilon, sample_x=sample_x
    )
    plots = [dict(x=ns, y=mse_rr, label="RR"), dict(x=ns, y=mse_lp, label="Laplace")]
    utils.plot_wrapper(
        img_path="./uniform_mse",
        suffixes=('.png', '.pdf'),
        plots=plots,
        options=dict(ylabel="$\mathrm{Unif}[0, 1] \quad \mathrm{MSE}$", xscale="log", xlabel="$N$")
    )

    errorbars = [
        dict(x=ns, y=mean_rr, yerr=err_rr, label="RR", alpha=0.8),
        dict(x=ns, y=mean_lp, yerr=err_lp, label="Laplace", alpha=0.8),
    ]
    utils.plot_wrapper(
        img_path="./uniform_ci",
        suffixes=('.png', '.pdf'),
        errorbars=errorbars,
        options=dict(ylabel="$\mathrm{Unif}[0, 1] \quad \theta$", xscale="log", xlabel="$N$")
    )


def _bernoulli(epsilon=1, t=1000, ns=(10, 100, 1000, 10000), pfloat=0.1):
    def sample_x(sample_size):
        return torch.bernoulli(torch.full(size=(sample_size,), fill_value=pfloat)), pfloat

    mse_rr, mse_lp, ci_rr, ci_lp, err_rr, err_lp, mean_rr, mean_lp, true_mean = _make_estimates(
        ns=ns, t=t, epsilon=epsilon, sample_x=sample_x
    )
    plots = [dict(x=ns, y=mse_rr, label="RR"), dict(x=ns, y=mse_lp, label="Laplace")]
    utils.plot_wrapper(
        img_path="./bernoulli_mse",
        suffixes=('.png', '.pdf'),
        plots=plots,
        options=dict(ylabel="$\mathrm{Bern}(p=0.1) \quad \mathrm{MSE}$", xscale="log", xlabel="$N$")
    )

    errorbars = [
        dict(x=ns, y=mean_rr, yerr=err_rr, label="RR", alpha=0.8),
        dict(x=ns, y=mean_lp, yerr=err_lp, label="Laplace", alpha=0.8),
    ]
    utils.plot_wrapper(
        img_path="./bernoulli_ci",
        suffixes=('.png', '.pdf'),
        errorbars=errorbars,
        options=dict(ylabel="$\mathrm{Bern}(p=0.1) \quad \theta$", xscale="log", xlabel="$N$")
    )


def _tight_uniform(epsilon=1, t=1000, ns=(10, 100, 1000, 10000)):
    def sample_x(sample_size):
        return torch.rand(size=(sample_size,)) * 0.02 + 0.49, 0.50

    mse_rr, mse_lp, ci_rr, ci_lp, err_rr, err_lp, mean_rr, mean_lp, true_mean = _make_estimates(
        ns=ns, t=t, epsilon=epsilon, sample_x=sample_x
    )
    plots = [dict(x=ns, y=mse_rr, label="RR"), dict(x=ns, y=mse_lp, label="Laplace")]
    utils.plot_wrapper(
        img_path="./tight_uniform_mse",
        suffixes=('.png', '.pdf'),
        plots=plots,
        options=dict(ylabel="$\mathrm{Unif}[0.49, 0.51] \quad \mathrm{MSE}$", xscale="log", xlabel="$N$")
    )

    errorbars = [
        dict(x=ns, y=mean_rr, yerr=err_rr, label="RR", alpha=0.8),
        dict(x=ns, y=mean_lp, yerr=err_lp, label="Laplace", alpha=0.8),
    ]
    utils.plot_wrapper(
        img_path="./tight_uniform_ci",
        suffixes=('.png', '.pdf'),
        errorbars=errorbars,
        options=dict(ylabel="$\mathrm{Unif}[0.49, 0.51] \quad \theta$", xscale="log", xlabel="$N$")
    )


def main():
    _uniform()
    _bernoulli()
    _tight_uniform()


if __name__ == "__main__":
    fire.Fire(main)
