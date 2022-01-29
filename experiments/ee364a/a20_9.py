# Problem data given in https://web.stanford.edu/~boyd/cvxbook/cvxbook_additional_exercises/storage_tradeoff_data.py

import cvxpy as cp
import fire
import numpy as np
import tqdm

from swissknife import utils


def run(Q=35, C=3, D=3, seed=1):
    np.random.seed(seed)

    T = 96
    t = np.linspace(1, T, num=T)
    p = np.exp(-np.cos((t - 15) * 2 * np.pi / T) + 0.01 * np.random.randn(T))
    u = 2 * np.exp(-0.6 * np.cos((t + 40) * np.pi / T) - 0.7 * np.cos(t * 4 * np.pi / T) + 0.01 * np.random.randn(T))

    c = cp.Variable(shape=(T,))
    q = cp.Variable(shape=(T,))

    objective = cp.Minimize(p @ (u + c))
    constraints = [
        q <= np.full(shape=(T,), fill_value=float(Q)),
        c <= np.full(shape=(T,), fill_value=float(C)),
        c >= np.full(shape=(T,), fill_value=float(-D)),
    ]
    constraints += [
        q[1:T - 1] == q[0:T - 2] + u[0:T - 2]
    ]
    constraints += [
        q[0] == q[T - 1] + c[T - 1]
    ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return dict(result=result, c=c.value, q=q.value, p=p, u=u, t=t)


def main():
    # (b)
    results = run(Q=35, C=3, D=3)
    t = results.get('t')

    p = results.get('p')
    u = results.get('u')
    c = results.get('c')
    q = results.get('q')

    quarter_t = t / 4
    plots = [
        dict(x=quarter_t, y=p, label='p'),
        dict(x=quarter_t, y=u, label='u'),
        dict(x=quarter_t, y=c, label='c'),
        dict(x=quarter_t, y=q, label='q'),
    ]
    utils.plot_wrapper(
        img_path=utils.join('.', 'plots', 'a20_9_b'),
        suffixes=('.png', '.pdf'),
        plots=plots,
    )

    # (c)
    for bound in (3, 1):
        plot = dict(x=[], y=[])
        for Q in tqdm.tqdm(np.linspace(1, 50, 100)):
            results = run(Q=Q, C=bound, D=bound)
            plot['x'].append(Q)
            plot['y'].append(results['result'])

        utils.plot_wrapper(
            img_path=utils.join('.', 'plots', f'a20_9_c_{bound}'),
            plots=(plot,),
            options=dict(xlabel='Q', ylabel='Minimum total cost'),
        )


if __name__ == "__main__":
    fire.Fire(main)
