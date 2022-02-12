import cvxpy as cp
import fire
import numpy as np


def main():
    r = 1.05
    m = 200
    n = 7
    F = 0.9
    C = 1.15
    S0 = 1
    S1 = 0.5
    S200 = 2

    V = np.zeros((m, n))
    V[:, 0] = r
    V[:, 1] = np.linspace(S1, S200, m)
    V[:, 2] = np.maximum(V[:, 1] - 1.1, 0)
    V[:, 3] = np.maximum(V[:, 1] - 1.2, 0)
    V[:, 4] = np.maximum(0.8 - V[:, 1], 0)
    V[:, 5] = np.maximum(0.7 - V[:, 1], 0)
    V[:, 6] = np.minimum(np.maximum(V[:, 1] - S0, F - S0), C - S0)

    for obj_fn in (
        lambda _p: cp.Maximize(_p[6]), lambda _p: cp.Minimize(_p[6]),
    ):
        p, y = cp.Variable(7), cp.Variable(200)
        objective = obj_fn(p)
        constraints = [
            p[:6] == np.array([1., 1., 0.06, 0.03, 0.02, 0.01]),
            y >= 0,
            cp.matmul(V.T, y) == p
        ]
        cp.Problem(objective, constraints).solve()
        print(p.value[-1])


if __name__ == "__main__":
    fire.Fire(main)
