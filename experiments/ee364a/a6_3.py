import cvxpy as cp
import fire
import numpy as np

from swissknife import utils


def main():
    K = 10
    N = 100

    a = cp.Variable(K + 1)
    b = cp.Variable(K + 1)

    def f(t, a, b):
        cos = np.cos(np.arange(K + 1) * t)
        sin = np.sin(np.arange(K + 1) * t)
        return a @ cos + b @ sin

    def y(t):
        return 1 if abs(t) <= np.pi / 2 else 0

    def err(t):
        yt = y(t)
        ft = f(t, a, b)
        return yt - ft

    for obj_fn, tag in zip(
        (
            lambda aa: cp.abs(aa),
            lambda aa: cp.square(aa)
        ),
        ('l1', 'l2')
    ):
        obj = sum(
            obj_fn(err(-np.pi + i * np.pi / N))
            for i in range(1, 2 * N + 1)
        )
        obj = cp.Minimize(obj)
        constraints = []
        prob = cp.Problem(obj, constraints).solve()

        ts = [
            -np.pi + i * np.pi / N
            for i in range(1, 2 * N + 1)
        ]
        ys = []
        fs = []
        for t in ts:
            yt = y(t)
            ft = f(t, a=a.value, b=b.value)
            ys.append(yt)
            fs.append(ft)

        img_path = utils.join('.', 'plots', f'a6_3_{tag}')
        utils.plot_wrapper(
            img_path=img_path,
            plots=[
                dict(x=ts, y=ys, label='y'),
                dict(x=ts, y=fs, label='f'),
            ]
        )


if __name__ == "__main__":
    fire.Fire(main)
