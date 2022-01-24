import cvxpy as cp
import fire
import numpy as np


def main():
    for obj_func in (
        lambda t: cp.Minimize(cp.sum(t * np.array([1., 1.]))),
        lambda t: cp.Minimize(cp.sum(t * np.array([-1., -1.]))),
        lambda t: cp.Minimize(cp.sum(t * np.array([1., 0.]))),
        lambda t: cp.Minimize(cp.max(t)),
        lambda t: cp.Minimize(t[0] ** 2 + 9. * t[1] ** 2.),
    ):
        x = cp.Variable(2)
        objective = obj_func(x)
        constraints = [
            2 * x[0] + x[1] >= 1,
            x[0] + 3 * x[1] >= 1,
            x >= np.zeros_like(x),
        ]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        print(result, x.value)


if __name__ == "__main__":
    fire.Fire(main)
