"""
"""

from typing import List

import cvxpy as cp
import fire
import numpy as np


def main():
    p = np.array([4.0, 2.0, 2.0, 1.0])
    d = np.array([20.0, 5.0, 10.0, 15.0])
    s = np.array([30.0, 10.0, 5.0, 0.0])
    dt = np.array([10.0, 25.0, 5.0, 15.0])
    st = np.array([5.0, 20.0, 15.0, 20.0])
    kappa = 0.5
    ones = np.ones((4,))

    B = cp.Variable(shape=(4, 4))
    Bt = cp.Variable(shape=(4, 4))
    t = cp.Variable(shape=(4,))

    def yield_sparsity_constraints(mat) -> List:
        out = [
            mat[0, 1] <= 0,
            mat[0, 2] <= 0,
            mat[0, 3] <= 0,
            mat[1, 2] <= 0,
            mat[1, 3] <= 0,
            mat[2, 1] <= 0,
            mat[2, 3] <= 0,
        ]
        return out

    sp = s - t
    stp = st + t

    constraints = [B >= 0, Bt >= 0]
    constraints += yield_sparsity_constraints(B)
    constraints += yield_sparsity_constraints(Bt)
    constraints += [B @ ones == d, Bt @ ones == dt, cp.transpose(B) @ ones <= sp, cp.transpose(Bt) @ ones <= stp]

    objective = cp.Minimize(kappa * cp.norm(t, 1) + p @ cp.transpose(B) @ ones + p @ cp.transpose(Bt) @ ones)
    cp.Problem(objective, constraints).solve()
    print("B")
    print(B.value)
    print("Bt")
    print(Bt.value)
    print("t")
    print(t.value)

    print("objective")
    print(objective.value)


if __name__ == "__main__":
    fire.Fire(main)
