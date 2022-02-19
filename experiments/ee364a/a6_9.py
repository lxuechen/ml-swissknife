"""
"""

import fire


def main():
    import numpy as np
    import cvxpy as cvx
    A_bar = np.array(np.mat(
        '60 45 -8;\
        90 30 -30;\
        0 -8 -4;\
        30 10 -10')
    )
    d = .05
    R = d * np.ones((4, 3))
    b = np.array([[-6],
                  [-3],
                  [18],
                  [-9]])
    # least-squares solution
    x_ls = np.linalg.lstsq(A_bar, b)[0]
    print(x_ls)

    # robust least-squares solution
    x = cvx.Variable(3)
    y = cvx.Variable(4)
    z = cvx.Variable(3)
    objective = cvx.Minimize(cvx.norm(y))
    constraints = [
        A_bar * x + R * z - b.squeeze() <= y,
        A_bar * x - R * z - b.squeeze() >= -y,
        x <= z,
        x + z >= 0
    ]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    print(x.value)


if __name__ == "__main__":
    fire.Fire(main)
