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
import torch


def make_labeled_data(n, d, sigma, prob, mu, group_id=2):
    # mu: tensor of size (d,)
    if group_id == 2:  # Sample both groups.
        coin_flips = torch.bernoulli(torch.full(size=(n, 1), fill_value=prob))
        y = 2 * coin_flips - 1.  # Convert to Rademacher.
    elif group_id == 1:  # y=1 (bernoulli=1).
        y = torch.full(size=(n, 1), fill_value=1)
    elif group_id == 0:  # y=-1 (bernoulli=0).
        y = torch.full(size=(n, 1), fill_value=-1)
    else:
        raise ValueError(f"Unknown group_id: {group_id}")
    x = mu[None, :] * y + sigma * torch.randn(size=(n, d))  # Get features.
    return x, y


def make_unlabeled_data(n, d, sigma, mu, prob, group_id=2):
    x, _ = make_labeled_data(n=n, d=d, sigma=sigma, mu=mu, prob=prob, group_id=group_id)  # Throw away labels.
    return x


def dp_estimator(x, y, clipping_norm, epsilon, delta):
    # Clip features.
    coefficient = torch.clamp_max(clipping_norm / x.norm(dim=1, keepdim=True), 1.)
    x = x * coefficient
    estimator = (x * y[:, None]).mean(dim=0)

    # Usual Gaussian mechanism.
    sample_size = x.size(0)
    sensitivity = 2 / sample_size * clipping_norm
    var = 2 * math.log(1.25 / delta) * sensitivity ** 2 / (epsilon ** 2)

    return estimator + torch.randn_like(estimator) * torch.sqrt(var)


def usual_estimator(x, y):
    return (x * y[:, None]).mean(dim=0)


def classify(estimator, x):
    y_hat = torch.ge((estimator * x).sum(dim=1), 0).float()
    return y_hat


def fairness():
    """Compare majority vs minority group's error.

    Plot error as a function of d, n, p (imbalance ratio), epsilon.

    Plot1: Check rate dependence on epsilon (same plot multiple p).
    Plot2: Check rate dependence on d.
    Plot3: Check rate dependence on p (same plot multiple epsilon).

    Maybe also plot ratio between accuracies? Need to reduce variance!
    """
    pass


def self_training():
    pass


def main(task="fairness"):
    if task == "fairness":
        fairness()
    elif task == "self_training":
        self_training()
    else:
        raise ValueError


if __name__ == "__main__":
    fire.Fire(main)
