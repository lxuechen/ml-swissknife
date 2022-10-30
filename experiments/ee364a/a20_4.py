import cvxpy as cp
import fire

import numpy as np
import tqdm

prec = np.matrix(
    '1 16; 1 36; 1 44; 1 56; 1 58; 2 5; 2 41; 2 44; 2 45; 3 18; 3 45; 4 20; 4 23; 4 26; 4 29; 4 58; 4 59; 5 6; 6 13; '
    '6 18; 7 25; 7 60; 8 13; 9 26; 9 38; 12 22; 12 36; 13 27; 13 40; 13 43; 13 59; 14 40; 14 44; 14 55; 15 39; 15 46; '
    '16 42; 17 48; 18 27; 18 47; 19 21; 19 54; 20 28; 21 22; 21 35; 22 23; 22 26; 22 29; 22 40; 22 46; 23 32; 23 50; '
    '23 59; 24 30; 24 54; 25 29; 25 37; 25 38; 25 55; 26 33; 26 51; 26 58; 28 59; 29 42; 29 51; 31 34; 31 39; 31 41; '
    '31 56; 33 36; 33 47; 34 58; 35 59; 36 51; 39 46; 40 53; 40 59; 41 60; 43 53; 44 57; 45 47; 45 58; 45 60; 46 51; '
    '46 55; 48 50; 49 58; 49 59; 53 57; 55 56; 58 60')

alpha = np.matrix(
    '3.2412; 1.8411; 2.9417; 3.903; 2.929; 0.6524; 1.3959; 3.2555; 1.2509; 3.2912; 0.9591; 1.6417; 3.831; 4.0341; '
    '3.1048; 2.5788; 0.3225; 1.4541; 1.059; 2.2371; 2.808; 0.9798; 2.7515; 0.5774; 3.5149; 0.7717; 2.5578; 2.6463; '
    '3.1775; 2.2835; 1.7302; 0.9499; 2.6101; 3.4274; 3.1508; 3.1533; 2.9054; 3.3718; 0.8511; 0.8876; 0.9465; 3.1647; '
    '3.7559; 1.0235; 1.4882; 2.8957; 0.1675; 2.334; 2.5249; 1.9003; 0.2216; 4.0567; 0.2641; 0.2905; 0.556; 1.3382; '
    '2.9388; 0.5932; 2.1499; 0.4848')

alpha = np.array(alpha).squeeze()
prec = np.array(prec)

n = 60
m = 91
s_min = 1
s_max = 5


def per_proc_cost(a, t):  # tf(a/t)
    return (
        t +
        a +
        cp.multiply(a ** 2, cp.inv_pos(t)) +
        cp.multiply(a ** 3, cp.square(cp.inv_pos(t)))
    )


def main():
    x = np.arange(1, 200, 1)
    y = []
    for E_max in tqdm.tqdm(x):
        b = cp.Variable(shape=(n,))  # begin
        e = cp.Variable(shape=(n,))  # end
        t = cp.Variable(shape=(n,))  # elapse.
        terminal = cp.max(e)

        E = per_proc_cost(alpha, t)
        obj = cp.Minimize(terminal)
        con = [
            t * s_min <= alpha,
            alpha <= t * s_max,
            e >= b + t,
            b >= 0,
            E <= E_max,
        ]
        # order constraints 2
        for i, (this, that) in enumerate(prec):
            this, that = int(this) - 1, int(that) - 1
            assert this >= 0 and that >= 0
            con.append(b[that] >= e[this])

        prob = cp.Problem(obj, con)
        prob.solve()
        y.append(terminal.value)

    plots = [dict(x=x, y=y, label='diff s')]

    x = np.arange(1, 200, 1)
    y = []
    for E_max in tqdm.tqdm(x):
        b = cp.Variable(shape=(n,))  # begin
        e = cp.Variable(shape=(n,))  # end
        t = cp.Variable(shape=(n,))  # elapse.
        terminal = cp.max(e)

        E = per_proc_cost(alpha, t)
        obj = cp.Minimize(terminal)
        con = [
            t * s_min <= alpha,
            alpha <= t * s_max,
            e >= b + t,
            b >= 0,
            E <= E_max,
        ]
        # order constraints 2
        for i, (this, that) in enumerate(prec):
            this, that = int(this) - 1, int(that) - 1
            assert this >= 0 and that >= 0
            con.append(b[that] >= e[this])
        con += [
            t[i] / alpha[i] == t[0] / alpha[0]
            for i in range(1, n)
        ]

        prob = cp.Problem(obj, con)
        prob.solve()
        y.append(terminal.value)

    plots.append(dict(x=x, y=y, label='same s'))

    from ml_swissknife import utils
    utils.plot_wrapper(
        plots=plots,
        options=dict(xlabel="E", ylabel="T"),
    )


if __name__ == "__main__":
    fire.Fire(main)
