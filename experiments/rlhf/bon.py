import fire
import numpy as np
import torch
from scipy.special import softmax

from ml_swissknife import utils


def bon(
    p,
    r,
    n,  # number of draws
    k,  # total samples
):
    choices = len(p)

    samples = np.random.choice(choices, p=p, size=(k, n), replace=True)
    rewards = r[samples]
    best_indices = rewards.argmax(1)

    ret = [
        sample[index] for sample, index in utils.zip_(samples, best_indices)
    ]
    return np.array(ret)


def bon_binning(
    # core
    p=np.array([1 / 3, 2 / 3]),
    r=np.array([1, 2]),
    n=5,

    # control accuracy of binning.
    k=200000,
):
    # python -m bon --task bon_binning
    # draw k samples
    out = bon(p=p, r=r, n=n, k=k)
    choices = len(p)
    p_hat = np.array(
        [(out == choice).mean() for choice in range(choices)]
    )
    return p_hat


def bon_analytical(
    p=np.array([1 / 3, 2 / 3]),
    r=np.array([1, 2]),
    n=5,
):
    # python -m bon --task bon_analytical
    choices = len(p)
    ascending = r.argsort()
    p = p[ascending]
    c = torch.tensor([0.] + p.cumsum(0).tolist())
    q = torch.tensor([
        c[i] ** n - c[i - 1] ** n
        for i in range(1, choices + 1)
    ])
    return q.numpy()


def bon_compare(
    reps=10,
    choices=5,  # size of sample space.
    n=5,
):
    # python -m bon --task bon_compare
    for _ in range(reps):
        p = softmax(np.random.rand(choices), axis=0)
        r = np.random.rand(choices)
        ans1 = bon_analytical(p=p, r=r, n=n)
        ans2 = bon_binning(p=p, r=r, n=n)
        print(ans1)
        print(ans2)
        breakpoint()


def kl(p, q):
    return (p * torch.log(p / q)).sum()


def assert_prob(p):
    assert torch.allclose(p.sum(), torch.ones_like(p.sum()))


def kl_bon(k=3, n=2):
    # p = torch.randn(k).softmax(0)
    p = torch.tensor([1 / 3, 1 / 3, 1 / 3])
    p = torch.tensor([1 / 2, 1 / 3, 1 / 6])

    c = torch.tensor([0.] + p.cumsum(0).tolist())
    q = torch.tensor([
        c[i] ** n - c[i - 1] ** n
        for i in range(1, k + 1)
    ])
    assert_prob(p)
    assert_prob(q)

    print(f'kl(q||p): {kl(q, p)}')
    print(f'kl(p||q): {kl(p, q)}')

    print('analytical')
    print(np.log(n) - (n - 1) / n)


def main(task="kl_bon", **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
