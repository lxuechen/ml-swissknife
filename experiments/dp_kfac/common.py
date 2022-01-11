import numpy as np
import torch


def make_data(
    d,
    n_train, n_test, n_unlabeled,
    offdiag_scale=2, diag_scale=10, noise_std=1e-3, beta_std=3, beta_mean=0.,
    zero_mean=True,
):
    torch.set_default_dtype(torch.float64)
    beta = beta_mean + torch.randn(d) * beta_std  # True coefficients.

    n = n_train + n_test
    mean = torch.randn(d) if not zero_mean else torch.zeros(size=(d,))
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

    evals, evecs = torch.symeig(covariance, eigenvectors=True)
    covariance_m1h = evecs @ torch.diag(evals ** (-0.5)) @ evecs.T

    x_train_whitened = x_train @ covariance_m1h
    x_test_whitened = x_test @ covariance_m1h

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

        'x_test_whitened': x_test_whitened,
        'x_train_whitened': x_train_whitened,
    }


def squared_loss(x, y, w):
    return .5 * ((x @ w - y) ** 2).mean(dim=0)


def predict(x, w):
    return x @ w


@torch.no_grad()
def gd(
    state, x, y, lr,
    steps,  # The iteration counter before the update.
    momentum=0,
    **_,
):
    w = state["w"]
    v = state["v"]

    grad_sample = (x @ w - y)[:, None] * x  # (n, d)
    grad = grad_sample.mean(dim=0)

    v_next = momentum * v + grad
    w_next = w - lr * v_next

    loss = squared_loss(x, y, w_next)

    return {
        "loss": loss,
        "state": {"w": w_next, "v": v_next},
        "grad": grad,
        "steps": steps,
    }


@torch.no_grad()
def pg(
    state, x, y, P, lr,
    steps,  # The iteration counter before the update.
    **_,
):
    w = state["w"]

    grad_sample = (x @ w - y)[:, None] * x  # (n, d)
    grad_sample = grad_sample @ P.t()  # (n, d)
    grad = grad_sample.mean(dim=0)

    w_next = w - lr * grad
    loss = squared_loss(x, y, w)

    return {
        "loss": loss,
        "state": {"w": w_next},
        "grad": grad,
        "steps": steps,
    }
