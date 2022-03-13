import cvxpy as cp
import fire
import matplotlib.pyplot as plt
import numpy as np

N = 4
T = 90
Cmax = 3.
qinit = np.array([20., 0., 30., 25.])
gamma = np.array([0.5, 0.3, 2.0, 0.6])
qdes = np.array([60., 100., 75., 125.])
qtar = np.stack([(t / (T + 1)) ** gamma * qdes for t in range(1, T + 2)])


def main():
    def compute_RMS(inp):
        return np.sqrt(np.mean(inp ** 2.))

    def plot1(_q, label: str):
        plt.figure()
        for i, color in zip(range(N), ('red', 'green', 'blue', 'cyan')):
            plt.plot(range(T + 1), qtar[:, i], linestyle='dashed', color=color, label=f'target ($i={i + 1}$)')
            plt.plot(range(T + 1), _q[:, i], linestyle='solid', color=color, label=f'{label} ($i={i + 1}$)')
        plt.legend()
        plt.show()

    def plot2(_c):
        plt.figure()
        plt.stackplot(range(T), _c.T)
        plt.show()

    c = cp.Variable(shape=(T, N))
    q = cp.Variable(shape=(T + 1, N))
    s = cp.Variable(shape=(T + 1, N))
    con = [c >= 0., sum(c[:, i] for i in range(N)) <= Cmax, q >= 0., q[0] == qinit]
    con += [q[t + 1] == q[t] + c[t] for t in range(T)]
    con += [s[t] >= cp.maximum(qtar[t], q[t]) - q[t] for t in range(T + 1)]
    cp.Problem(objective=cp.Minimize(cp.sum(cp.square(s))), constraints=con).solve()
    print(compute_RMS(s.value))
    plot1(q.value, label="optimal")
    plot2(c.value)

    theta = (qdes - qinit) / np.sum(qdes - qinit)
    c_const = np.stack([theta * Cmax] * T)
    q = np.zeros((T + 1, N))
    q[0] = qinit
    for t in range(0, T):
        q[t + 1] = q[t] + c_const[t]
    s = np.maximum(qtar - q, 0)
    print(compute_RMS(s))
    plot1(q, label="constant")
    plot2(c_const)


if __name__ == "__main__":
    fire.Fire(main)
