import cvxpy as cp
import fire
import matplotlib.pyplot as plt
import numpy as np

T = 40
r = 0.04
gamma_call = 0.23
gamma_dist = 0.15
c_max = 4.0
p_max = 3.0
B = 85.0
n_des = 15.0
lamb = 5.0


def main():
    def compute_rms(_n):
        return np.sqrt(np.mean((_n - n_des) ** 2.0))

    css = (gamma_dist - r) * n_des
    c, d, p = tuple(cp.Variable(T) for _ in range(3))
    n, u = tuple(cp.Variable(T + 1) for _ in range(2))
    obj = cp.Minimize(sum(cp.square(n - n_des)) / (T + 1) + lamb * sum(cp.square(c[1:] - c[:-1])) / (T - 1))
    con = [
        n[1:] == (1 + r) * n[:-1] + p - d,
        u[1:] == u[:-1] - p + c,
        n[0] == 0.0,
        u[0] == 0.0,
        p == gamma_call * u[:-1],
        d == gamma_dist * n[:-1],
        c <= c_max,
        p <= p_max,
        sum(c) <= B,
        c == css,
        c >= 0.0,
        d >= 0.0,
        p >= 0.0,
        n >= 0.0,
        u >= 0.0,
    ]
    cp.Problem(constraints=con, objective=obj).solve()
    print(c.value)
    print(compute_rms(n.value))
    plt.figure()
    for vec, lab in zip((c, d, p), ("c", "d", "p")):
        plt.plot(range(T), vec.value, label=lab)
    for vec, lab in zip((n, u), ("n", "u")):
        plt.plot(range(T + 1), vec.value, label=lab)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
