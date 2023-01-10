import fire
import numpy as np
import torch


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
