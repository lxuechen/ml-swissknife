"""
"""

import cvxpy as cp
import fire


def main():
    x = cp.Variable(shape=(2, 2, 2, 2))
    for obj_func in (
        lambda t: cp.Minimize(cp.sum(t[:, :, :, 1])),
        lambda t: cp.Maximize(cp.sum(t[:, :, :, 1])),
    ):
        objective = obj_func(x)

        constraints = [
            cp.sum(x[1, :, :, :]) == 0.9,
            cp.sum(x[:, 1, :, :]) == 0.9,
            cp.sum(x[:, :, 1, :]) == 0.1,
            cp.sum(x[1, :, 1, 0]) / cp.sum(x[:, :, 1, :]) == 0.7,
            cp.sum(x[:, 1, 0, 1]) / cp.sum(x[:, 1, 0, :]) == 0.6,

            cp.sum(x) == 1,
            x >= 0,
            x <= 1,
        ]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        print(result, x.value)


if __name__ == "__main__":
    fire.Fire(main)
