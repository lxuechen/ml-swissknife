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


def plot(p, f):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    X = np.linspace(-40, 55, num=30)
    Y = np.linspace(0, 55, num=30)
    X, Y = np.meshgrid(X, Y)
    Z = alpha * np.sqrt(X ** 2 + Y ** 2)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.autumn, linewidth=0.1, alpha=0.7, edgecolors="k")
    ax = plt.gca();
    ax.view_init(azim=225)

    # Have your solution be stored in p
    ax.plot(xs=p.value[0, :], ys=p.value[1, :], zs=p.value[2, :], c='b', lw=2, zorder=5)
    ax.quiver(p.value[0, :-1], p.value[1, :-1], p.value[2, :-1],
              f.value[0, :], f.value[1, :], f.value[2, :], zorder=5, color="black")

    ax.set_xlabel("x");
    ax.set_ylabel("y");
    ax.set_zlabel("z")
    plt.show()


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

    plot(p=cp.transpose(p), f=cp.transpose(f))


if __name__ == "__main__":
    fire.Fire(main)
