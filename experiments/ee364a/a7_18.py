import cvxpy as cp
import fire
import numpy as np

from swissknife import utils


def main():
    n = 13
    m = np.array([
        1, 5, 6, 15,
        18, 20, 22, 11,
        22, 8, 9, 4, 2
    ])
    s = cp.Variable(n)
    obj = cp.Maximize(m @ s)
    constraints = [
        cp.log_sum_exp(s, axis=0) <= 0.,
    ]
    constraints += [
        2 * s[i] >= s[i - 1] + s[i + 1]
        for i in range(1, n - 1)
    ]
    prob = cp.Problem(obj, constraints).solve()

    p = np.exp(s.value)
    f = m / sum(m)

    x = np.arange(1, n + 1)
    img_path = utils.join('.', 'plots', 'a7_18')
    utils.plot_wrapper(
        img_path=img_path,
        plots=[
            dict(x=x, y=p, label='hat m'),
            dict(x=x, y=f, label='f')
        ]
    )


if __name__ == "__main__":
    fire.Fire(main)
