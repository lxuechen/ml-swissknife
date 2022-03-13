import fire
import numpy as np
import torch


def density(x, mu, sigma):
    return (
        torch.linalg.det(2 * np.pi * sigma) ** (-0.5) *
        torch.exp(-.5 * (x - mu) @ (torch.inverse(sigma) @ (x - mu)))
    )


def grad_psi(x, mu, sigma):
    return torch.inverse(sigma) @ (x - mu)


def main():
    torch.set_printoptions(precision=20)
    torch.set_default_dtype(torch.float64)

    a = torch.tensor([3., 3.])
    mu = torch.tensor([0., 0.])
    rho = 0.5
    sigma = torch.tensor([
        [1., rho],
        [rho, 1.]
    ])

    p_a = density(a, mu, sigma)
    grad_a = grad_psi(a, mu, sigma)
    bound = p_a * 1. / torch.prod(grad_a)
    print(bound)


if __name__ == "__main__":
    fire.Fire(main)
