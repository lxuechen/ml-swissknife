"""
"""

import cvxpy as cp
import fire
import numpy as np

from swissknife import utils

N = 10
n = 100

q = [(np.exp(-(i - 30) ** 2 / 100) + 2 * np.exp(-(i - 68) ** 2 / 100)) for i in range(1, 101)]
q = np.array(q)
r = np.array([np.exp(-(i - 50) ** 2 / 100) for i in range(1, 101)])

q = q / sum(q)
r = r / sum(r)


def plot(p):
    utils.plot_wrapper(
        plots=[
            dict(x=np.arange(n), y=p.value[i], label=f'{i}')
            for i in range(N)
        ]
    )


def parta():
    p = cp.Variable(shape=(N, n))
    cons = [
        p[0] == q,
        p[N - 1] == r,
    ]
    cons += [p[i] >= 0. for i in range(N)]
    cons += [sum(p[i]) == 1. for i in range(N)]

    obj = cp.Minimize(
        sum([
            cp.norm2(_next - _this) ** 2 for _next, _this in zip(p[1:], p[:-1])
        ])
    )
    cp.Problem(objective=obj, constraints=cons).solve()

    plot(p)


def partb():
    p = cp.Variable(shape=(N, n))
    cons = [
        p[0] == q,
        p[N - 1] == r,
    ]
    cons += [p[i] >= 0. for i in range(N)]
    cons += [sum(p[i]) == 1. for i in range(N)]

    loss = 0
    for _next, _this in zip(p[1:], p[:-1]):
        loss = loss + sum(_next) + sum(_this)

        for ui, vi in zip(_next, _this):
            loss = loss - 2. * cp.geo_mean(
                cp.vstack([ui, vi])
            )

    obj = cp.Minimize(loss)
    cp.Problem(objective=obj, constraints=cons).solve(solver=cp.ECOS)
    for i in range(N):
        print(sum(p[i].value))

    plot(p)


def partc():
    p = cp.Variable(shape=(N, n))
    cons = [
        p[0] == q,
        p[N - 1] == r,
    ]
    cons += [p[i] >= 0. for i in range(N)]
    cons += [sum(p[i]) == 1. for i in range(N)]

    pcum = cp.cumsum(p, axis=1)
    obj = cp.Minimize(
        sum([
            cp.norm_inf(_next - _this) for _next, _this in zip(pcum[1:], pcum[:-1])
        ])
    )
    cp.Problem(objective=obj, constraints=cons).solve()

    plot(p)


def main():
    # parta()
    partb()
    # partc()


if __name__ == "__main__":
    fire.Fire(main)
