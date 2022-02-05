import cvxpy as cp
import fire
import numpy as np

h = 1.
g = 0.1
m = 10.
Fmax = 10.
p0 = np.array([50, 50, 100])
v0 = np.array([-10, 0, -10])
alpha = 0.5
gamma = 1.
K = 35


def main():
    f = cp.Variable(shape=(K + 1, 3))
    v = cp.Variable(shape=(K + 2, 3))
    p = cp.Variable(shape=(K + 2, 3))
    e3 = np.array([0, 0, 1])

    objective = cp.Minimize(
        gamma * h * sum(
            cp.norm2(f[k]) for k in range(1, K + 1)
        )
    )
    constraints = [
        v[k + 1] == v[k] + h / m * f[k] - h * g * e3
        for k in range(1, K + 1)
    ]
    constraints += [
        p[k + 1] == p[k] + (h / 2) * (v[k] + v[k + 1])
        for k in range(1, K + 1)
    ]
    constraints += [
        v[1] == v0,
        p[1] == p0.squeeze(),
    ]
    constraints += [
        p[k][2] >= alpha * cp.norm2(p[k][:2]) for k in range(1, K + 1)
    ]
    constraints += [
        cp.norm2(f[k]) <= Fmax
        for k in range(1, K + 1)
    ]
    constraints += [
        v[K + 1] == 0,
        p[K + 1] == 0
    ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print(result)


if __name__ == "__main__":
    fire.Fire(main)
