"""
"""

import cvxpy as cp
import fire
import numpy as np

from ml_swissknife import utils


def main():
    n = 4
    m = 2
    T = 100

    A = np.array(
        [
            [0.95, 0.16, 0.12, 0.01],
            [-0.12, 0.98, -0.11, -0.03],
            [-0.16, 0.02, 0.98, 0.03],
            [-0.0, 0.02, -0.04, 1.03],
        ]
    )

    B = np.array(
        [
            [0.8, 0.0],
            [0.1, 0.2],
            [0.0, 0.8],
            [-0.2, 0.1],
        ]
    )

    x_init = np.ones(n)

    for loss_fn in (
        lambda var: sum(cp.norm2(var[i]) ** 2.0 for i in range(T)),
        lambda var: sum(cp.norm2(var[i]) for i in range(T)),
        lambda var: cp.max(cp.vstack([cp.norm2(var[i]) for i in range(T)])),
        lambda var: sum(cp.norm1(var[i]) for i in range(T)),
    ):
        u = cp.Variable(shape=(T, m))
        x = cp.Variable(shape=(T + 1, n))
        unorm = cp.norm2(u, axis=1)

        cons = [
            x[0] == x_init,
            x[T] == 0.0,
        ]
        cons += [x[t + 1] == A @ x[t] + B @ u[t] for t in range(T)]

        loss = loss_fn(u)
        obj = cp.Minimize(loss)
        cp.Problem(constraints=cons, objective=obj).solve()

        uval = u.value

        utils.plot_wrapper(
            plots=[
                dict(x=np.arange(T), y=uval[:, 0], label="u1"),
                dict(x=np.arange(T), y=uval[:, 1], label="u2"),
                dict(x=np.arange(T), y=unorm.value, label="$\| u \|_2$"),
            ]
        )


if __name__ == "__main__":
    fire.Fire(main)
