"""
python -m experiments.rr_vs_laplace

exercise 6.1 from ECE377
"""

import numpy as np
import torch
from torch import distributions
import tqdm

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


def _uniform(n=1000, epsilon=1, t=1000):
    mse_rr = []
    mse_lp = []
    for _ in tqdm.tqdm(range(t), desc="T"):
        x = torch.rand(size=(n,))

        x_rr = rr(x, epsilon=epsilon)
        mse_rr.append(_squared_error(x_rr, mean=0.5))

        x_lp = laplace(x, epsilon=epsilon)
        mse_lp.append(_squared_error(x_lp, mean=0.5))

    mse_rr = torch.stack(mse_rr).mean()
    mse_lp = torch.stack(mse_lp).mean()
    print('rr', mse_rr, 'lp', mse_lp)


def _bernoulli(n=1000, epsilon=1, t=1000, p=0.1):
    p = torch.full(size=(n,), fill_value=p)
    mse_rr = []
    mse_lp = []
    for _ in tqdm.tqdm(range(t), desc="T"):
        x = torch.bernoulli(p)

        x_rr = rr(x, epsilon=epsilon)
        mse_rr.append(_squared_error(x_rr, mean=0.5))

        x_lp = laplace(x, epsilon=epsilon)
        mse_lp.append(_squared_error(x_lp, mean=0.5))

    mse_rr = torch.stack(mse_rr).mean()
    mse_lp = torch.stack(mse_lp).mean()
    print('rr', mse_rr, 'lp', mse_lp)


def _tight_uniform(n=1000, epsilon=1, t=1000):
    mse_rr = []
    mse_lp = []
    for _ in tqdm.tqdm(range(t), desc="T"):
        x = torch.rand(size=(n,)) * 0.02 + 0.49

        x_rr = rr(x, epsilon=epsilon)
        mse_rr.append(_squared_error(x_rr, mean=0.5))

        x_lp = laplace(x, epsilon=epsilon)
        mse_lp.append(_squared_error(x_lp, mean=0.5))

    mse_rr = torch.stack(mse_rr).mean()
    mse_lp = torch.stack(mse_lp).mean()
    print('rr', mse_rr, 'lp', mse_lp)


# Vary n.
def main(n=1000, epsilon=1):
    _uniform(n=n, epsilon=epsilon, )
    _bernoulli(n=n, epsilon=epsilon, )
    _tight_uniform(n=n, epsilon=epsilon, )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
