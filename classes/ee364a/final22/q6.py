import cvxpy as cp
import fire
import numpy as np

rng = np.random.Generator(np.random.MT19937(seed=12345))
n = 20
# _A and _C are internal - you don't need them.
_A = rng.standard_normal((2 * n, n))
_C = np.diag(0.5 * np.exp(rng.standard_normal((n,))))

Sigma = _C @ _A.T @ _A @ _C
Sigma = 0.5 * (Sigma + Sigma.T)
M = np.ones(n) * 0.2
sigma = np.sqrt(np.diag(Sigma))


def main():
    def compute_dr(_x):
        return sigma @ _x / ((Sigma @ _x @ _x) ** 0.5)

    x = cp.Variable(n)
    cons = [
        x >= 0.0,
        x - M * cp.sum(x) <= 0.0,
        sigma @ x == 1.0,
    ]
    obj = cp.Minimize(cp.quad_form(x, Sigma))
    cp.Problem(objective=obj, constraints=cons).solve()

    x_star = x.value / (np.sum(x.value))
    d_x_star = compute_dr(x_star)
    print(f"x_star: {x_star}, d_x_star: {d_x_star:.6f}")

    x = cp.Variable(n)
    cons = [x >= 0.0, x <= M, cp.sum(x) == 1]
    obj = cp.Minimize(cp.quad_form(x, Sigma))
    cp.Problem(objective=obj, constraints=cons).solve()
    x_mv = x.value
    d_x_mv = compute_dr(x_mv)
    print(f"x_mv: {x_mv}, d_x_mv: {d_x_mv:.6f}")

    import matplotlib.pyplot as plt

    plt.bar(np.arange(0, 20), x_star, width=0.5, label="Max diversification")
    plt.bar((np.arange(0, 20) * 2 + 1) / 2, x_mv, width=0.5, label="Min variance")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
