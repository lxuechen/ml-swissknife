"""
python main.py

exercise 6.1 from ECE377
"""

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
    return (a * z - b).mean()


def laplace(x, epsilon):
    loc = torch.zeros_like(x)
    scale = torch.full_like(x, fill_value=1 / epsilon)
    dist = distributions.laplace.Laplace(loc=loc, scale=scale)
    w = dist.sample()
    assert x.size() == w.size()
    return (x + w).mean()


def _squared_error(x, mean):
    return (x - mean) ** 2


def _make_mse_estimates(ns, t, epsilon, sample_x):
    mse_rr = []
    mse_lp = []
    for n in ns:
        err_rr = []
        err_lp = []
        for _ in tqdm.tqdm(range(t), desc="T"):
            x, true_mean = sample_x(n)

            x_rr = rr(x, epsilon=epsilon)
            assert x_rr.dim() == 0
            err_rr.append(_squared_error(x_rr, mean=true_mean))

            x_lp = laplace(x, epsilon=epsilon)
            assert x_lp.dim() == 0
            err_lp.append(_squared_error(x_lp, mean=true_mean))

        mse_rr.append(torch.stack(err_rr).mean())
        mse_lp.append(torch.stack(err_lp).mean())
    return mse_rr, mse_lp


def _uniform(epsilon=1, t=1000, ns=(10, 100, 1000, 10000)):
    def sample_x(sample_size):
        return torch.rand(size=(sample_size,)), 0.5

    mse_rr, mse_lp = _make_mse_estimates(ns=ns, t=t, epsilon=epsilon, sample_x=sample_x)
    plots = [dict(x=ns, y=mse_rr, label="RR"), dict(x=ns, y=mse_lp, label="Laplace")]
    utils.plot_wrapper(
        img_path="./uniform_mse",
        suffixes=('.png', '.pdf'),
        plots=plots,
        options=dict(ylabel="$\mathrm{Unif}[0, 1] \quad \mathrm{MSE}$", xscale="log", xlabel="$N$")
    )


def _bernoulli(epsilon=1, t=1000, ns=(10, 100, 1000, 10000), pfloat=0.1):
    def sample_x(sample_size):
        return torch.bernoulli(torch.full(size=(sample_size,), fill_value=pfloat)), pfloat

    mse_rr, mse_lp = _make_mse_estimates(ns=ns, t=t, epsilon=epsilon, sample_x=sample_x)
    plots = [dict(x=ns, y=mse_rr, label="RR"), dict(x=ns, y=mse_lp, label="Laplace")]
    utils.plot_wrapper(
        img_path="./bernoulli_mse",
        suffixes=('.png', '.pdf'),
        plots=plots,
        options=dict(ylabel="$\mathrm{Bern}(p=0.1) \quad \mathrm{MSE}$", xscale="log", xlabel="$N$")
    )


def _tight_uniform(epsilon=1, t=1000, ns=(10, 100, 1000, 10000)):
    def sample_x(sample_size):
        return torch.rand(size=(sample_size,)) * 0.02 + 0.49, 0.50

    mse_rr, mse_lp = _make_mse_estimates(ns=ns, t=t, epsilon=epsilon, sample_x=sample_x)
    plots = [dict(x=ns, y=mse_rr, label="RR"), dict(x=ns, y=mse_lp, label="Laplace")]
    utils.plot_wrapper(
        img_path="./tight_uniform_mse",
        suffixes=('.png', '.pdf'),
        plots=plots,
        options=dict(ylabel="$\mathrm{Unif}[0.49, 0.51] \quad \mathrm{MSE}$", xscale="log", xlabel="$N$")
    )


def main():
    _uniform()
    _bernoulli()
    _tight_uniform()


if __name__ == "__main__":
    fire.Fire(main)
