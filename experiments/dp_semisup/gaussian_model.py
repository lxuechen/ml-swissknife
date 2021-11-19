"""
Two questions:
    1. (fairness) disparate impact in terms of convergence rate? epsilon, d, n dependence?
    2. (self-training) does self-training post-processing boost performance?

Where does randomness come from?
    a. sampled data (both labeled and unlabeled)
    b. dp noise
    c. test data
"""

import math

import fire
import numpy as np
import torch
import tqdm

from swissknife import utils


def make_labeled_data(n, d, sigma, prob, mu, group_id=2):
    # mu: tensor of size (1, d,)
    if group_id == 2:  # Sample both groups.
        coin_flips = torch.bernoulli(torch.full(size=(n, 1), fill_value=prob))
        y = 2 * coin_flips - 1.  # Convert to Rademacher.
    elif group_id == 1:  # y=1 (bernoulli=1).
        y = torch.full(size=(n, 1), fill_value=1.)
    elif group_id == 0:  # y=-1 (bernoulli=0).
        y = torch.full(size=(n, 1), fill_value=-1.)
    else:
        raise ValueError(f"Unknown group_id: {group_id}")
    x = mu * y + sigma * torch.randn(size=(n, d))  # Get features.
    # x: (sample_size, d).
    # y: (sample_size, 1).
    return x, y


def make_unlabeled_data(n, d, sigma, mu, prob, group_id=2):
    x, _ = make_labeled_data(n=n, d=d, sigma=sigma, mu=mu, prob=prob, group_id=group_id)  # Throw away labels.
    return x


def dp_estimator(x, y, clipping_norm, epsilon, delta=None):
    # Clip features.
    coefficient = torch.clamp_max(clipping_norm / x.norm(dim=1, keepdim=True), 1.)
    x = x * coefficient
    estimator = (x * y).mean(dim=0)

    # Usual Gaussian mechanism.
    sample_size = x.size(0)
    if delta is None:
        delta = 1 / (2 * sample_size)
    sensitivity = 2 / sample_size * clipping_norm
    var = 2 * math.log(1.25 / delta) * sensitivity ** 2 / (epsilon ** 2)

    return estimator + torch.randn_like(estimator) * math.sqrt(var)


def usual_estimator(x, y):
    return (x * y).mean(dim=0)


def classify(estimator, x):
    y_hat = torch.where((estimator * x).sum(dim=1) > 0, 1., -1.)[:, None]
    return y_hat


def compute_error_rate(estimator, x, y):
    y_hat = torch.where((estimator * x).sum(dim=1) > 0, 1., -1.)[:, None]
    return torch.eq(y, y_hat).float().mean(dim=0).item()


def fairness(**kwargs):
    """Compare majority vs minority group's error.

    Plot error as a function of d, n, p (imbalance ratio), epsilon.

    Plot1: Check rate dependence on epsilon (same plot multiple p).
    Plot2: Check rate dependence on d.
    Plot3: Check rate dependence on p (same plot multiple epsilon).

    Maybe also plot ratio between accuracies? Need to reduce variance!
    Expectation:
        - O(1/p) vs O(1/(1-p)); not very surprising.
        - example too simple; doing well on majority => doing well on minority.

    Conclusion: This model is unsuited for fairness.
    """
    epsilons = (0.001, 0.01, 0.1, 0.5, 1., 2., 5.,)
    probs = (0.999, 0.9999)
    d = 20
    mu = torch.randn((1, d)) * 2
    sigma = 0.3
    n_labeled = 100
    n_test = 10000
    clipping_norm = 3
    seeds = list(range(100))

    errorbars = []
    for prob in probs:
        errbar1 = dict(x=epsilons, y=[], yerr=[], label=f"group 1 (p={prob:.4f})", marker='o')
        errbar0 = dict(x=epsilons, y=[], yerr=[], label=f"group 0 (p={1 - prob:.4f})", marker="^")
        errorbars.extend([errbar1, errbar0])
        for epsilon in epsilons:
            errs1, errs0 = [], []
            for seed in tqdm.tqdm(seeds, desc="seeds"):
                x, y = make_labeled_data(n=n_labeled, d=d, mu=mu, prob=prob, sigma=sigma)

                theta_hat = dp_estimator(x=x, y=y, clipping_norm=clipping_norm, epsilon=epsilon)

                x1, y1 = make_labeled_data(n=n_test, d=d, mu=mu, prob=prob, sigma=sigma, group_id=1)
                err1 = compute_error_rate(estimator=theta_hat, x=x1, y=y1)

                # Minority.
                x0, y0 = make_labeled_data(n=n_test, d=d, mu=mu, prob=prob, sigma=sigma, group_id=0)
                err0 = compute_error_rate(estimator=theta_hat, x=x0, y=y0)

                errs1.append(err1)
                errs0.append(err0)

            avg1, std1 = np.mean(errs1), np.std(errs1)
            avg0, std0 = np.mean(errs0), np.std(errs0)
            errbar1['y'].append(avg1)
            errbar1['yerr'].append(std1)
            errbar0['y'].append(avg0)
            errbar0['yerr'].append(std0)

    utils.plot_wrapper(
        errorbars=errorbars,
        options=dict(xlabel="$\epsilon$", xscale="log", yscale='log')
    )


def self_training(alpha=0, beta=1, **kwargs):
    """Compare error with and without unlabeled data.

    Semi-sup estimator = alpha * private estimator + beta * PL estimator.

    Questions:
        - separation between overparam vs underparam?
        - phase transition between low to high privacy?
        - how does noise variance in the data model affect things?
    """
    epsilons = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3.,)
    probs = (0.5,)
    d = 100  # This seems already overparameterized!
    mu = torch.randn((1, d)) * 2
    sigma = 2
    n_labeled = 50
    n_unlabeled = 50000  # x100 factor.
    n_test = 10000
    clipping_norm = 3
    seeds = list(range(100))

    errorbars = []
    for prob in probs:
        errbar1 = dict(x=epsilons, y=[], yerr=[], label="w/o unlabeled", marker="o")
        errbar0 = dict(x=epsilons, y=[], yerr=[], label="w/  unlabeled", marker="^")
        errorbars.extend([errbar1, errbar0])
        for epsilon in epsilons:
            errs1, errs0 = [], []
            for seed in tqdm.tqdm(seeds, desc="seeds"):
                x, y = make_labeled_data(n=n_labeled, d=d, mu=mu, prob=prob, sigma=sigma)
                x_test, y_test = make_labeled_data(n=n_test, d=d, mu=mu, prob=prob, sigma=sigma)

                theta_hat = dp_estimator(x=x, y=y, clipping_norm=clipping_norm, epsilon=epsilon)

                # Without.
                err1 = compute_error_rate(estimator=theta_hat, x=x_test, y=y_test)
                errs1.append(err1)

                # With unlabeled.
                x_unlabeled = make_unlabeled_data(n=n_unlabeled, d=d, mu=mu, prob=prob, sigma=sigma)
                y_unlabeled = classify(estimator=theta_hat, x=x_unlabeled)  # Pseudo-label.
                theta_tilde = usual_estimator(x=x_unlabeled, y=y_unlabeled)

                theta_bar = alpha * theta_hat + beta * theta_tilde
                err0 = compute_error_rate(estimator=theta_bar, x=x_test, y=y_test)
                errs0.append(err0)

            avg1, std1 = np.mean(errs1), np.std(errs1)
            avg0, std0 = np.mean(errs0), np.std(errs0)
            errbar1['y'].append(avg1)
            errbar1['yerr'].append(std1)
            errbar0['y'].append(avg0)
            errbar0['yerr'].append(std0)

    utils.plot_wrapper(
        errorbars=errorbars,
        options=dict(xlabel="$\epsilon$", xscale="log", yscale='log')
    )


def main(task="self_training", **kwargs):
    if task == "fairness":
        fairness(**kwargs)
    elif task == "self_training":
        self_training(**kwargs)
    else:
        raise ValueError


if __name__ == "__main__":
    fire.Fire(main)
