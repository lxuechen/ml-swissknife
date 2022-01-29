# Problem data given in https://web.stanford.edu/~boyd/cvxbook/cvxbook_additional_exercises/storage_tradeoff_data.py

import cvxpy as cp
import fire
import numpy as np
import tqdm

from swissknife import utils


def run(Q, C, D, seed=1):
    np.random.seed(seed)

    T = 96
    t = np.linspace(1, T, num=T).reshape(T, 1)
    p = np.exp(-np.cos((t - 15) * 2 * np.pi / T) + 0.01 * np.random.randn(T, 1))
    u = 2 * np.exp(-0.6 * np.cos((t + 40) * np.pi / T) - 0.7 * np.cos(t * 4 * np.pi / T) + 0.01 * np.random.randn(T, 1))

    t = t.squeeze()
    p = p.squeeze()
    u = u.squeeze()

    c = cp.Variable(shape=(T,))
    q = cp.Variable(shape=(T,))

    objective = cp.Minimize(p @ (u + c))

    constraints = [
        q >= 0,
        q[0] == q[T - 1] + c[T - 1],
        u + c >= 0,
        q <= Q,
        c <= C,
        c >= -D,
    ]
    # TODO: There's likely a bug in the library???
    # Using `q[1:T - 1] == q[0:T - 2] + c[0:T - 2]` doesn't seem to work!?
    constraints += [q[i + 1] == q[i] + c[i] for i in range(T - 1)]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return dict(result=result, c=c.value, q=q.value, p=p, u=u, t=t,
                probval=prob.value)


def main():
    # (b)
    results = run(Q=35, C=3, D=3)
    print(results['result'])

    t = results.get('t')
    p = results.get('p')
    u = results.get('u')
    c = results.get('c')
    q = results.get('q')

    plots = [
        dict(x=t, y=p, label='p'),
        dict(x=t, y=u, label='u'),
        dict(x=t, y=c, label='c'),
        dict(x=t, y=q, label='q'),
    ]
    utils.plot_wrapper(
        img_path=utils.join('.', 'plots', 'a20_9_b'),
        suffixes=('.png', '.pdf'),
        plots=plots,
        options=dict(xlabel='t'),
    )

    # (c)
    plots = []
    for bound in (3, 1,):
        plot = dict(x=[], y=[], label=f'C=D={bound}')
        for Q in tqdm.tqdm(np.linspace(1, 300, 300)):
            results = run(Q=Q, C=bound, D=bound)
            plot['x'].append(Q)
            plot['y'].append(results['result'])
        plots.append(plot)
    utils.plot_wrapper(
        img_path=utils.join('.', 'plots', f'a20_9_c'),
        suffixes=('.png', '.pdf'),
        plots=plots,
        options=dict(xlabel='Q', ylabel='Minimum total cost'),
    )


if __name__ == "__main__":
    fire.Fire(main)
