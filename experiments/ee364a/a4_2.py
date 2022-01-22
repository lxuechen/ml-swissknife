import cvxpy as cp
import fire
import numpy as np


def obj_a(x):
    return cp.Minimize(cp.sum(x * np.array([1., 1.])))


def obj_b(x):
    return cp.Minimize(cp.sum(x * np.array([-1., -1.])))


def obj_c(x):
    return cp.Minimize(cp.sum(x * np.array([1., 0.])))


def obj_d(x):
    return cp.Minimize(cp.max(x))


def obj_e(x):
    return cp.Minimize(x[0] ** 2 + 9. * x[1] ** 2.)


def main():
    d = 2

    for obj_func in (
        obj_a,
        obj_b,
        obj_c,
        obj_d,
        obj_e,
    ):
        x = cp.Variable(d)
        objective = obj_func(x)

        constraints = [
            2 * x[0] + x[1] >= 1,
            x[0] + 3 * x[1] >= 1,
            x >= np.zeros_like(x),
        ]
        prob = cp.Problem(objective, constraints)

        result = prob.solve()
        print(result)
        print(x.value)


if __name__ == "__main__":
    fire.Fire(main)
