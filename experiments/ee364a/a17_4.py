import cvxpy as cp
import fire
import numpy as np


def main():
    Sigma = cp.Variable(shape=(4, 4), symmetric=True)
    x = np.array(
        [[0.1], [0.2], [-0.05], [0.1]],
    )  # (4, 1)
    objective = cp.Maximize(
        cp.transpose(x) @ Sigma @ x
    )
    constraints = [
        Sigma >> 0,

        Sigma[0, 0] == 0.2,
        Sigma[0, 1] >= 0,
        Sigma[0, 2] >= 0,

        Sigma[1, 0] >= 0,
        Sigma[1, 1] == 0.1,
        Sigma[1, 2] <= 0,
        Sigma[1, 3] <= 0,

        Sigma[2, 0] >= 0,
        Sigma[2, 1] <= 0,
        Sigma[2, 2] == 0.3,
        Sigma[2, 3] >= 0,

        Sigma[3, 1] <= 0,
        Sigma[3, 2] >= 0,
        Sigma[3, 3] == 0.1,
    ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print(result)
    print(Sigma.value)

    Sigma_diag = np.array(
        [
            [0.2, 0, 0, 0,],
            [0, 0.1, 0, 0,],
            [0, 0, 0.3, 0],
            [0, 0, 0, 0.1],
        ]
    )
    print(x.T @ Sigma_diag @ x)


if __name__ == "__main__":
    fire.Fire(main)
