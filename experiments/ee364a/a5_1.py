import cvxpy as cp
import fire
import numpy as np


def rep(u1, u2):
    x = cp.Variable(shape=(2,))
    P = np.array([
        [1., -0.5],
        [-0.5, 2],
    ])

    x1 = x[0]
    x2 = x[1]

    objective = cp.Minimize(
        cp.quad_form(x, P) - x1
    )
    constraints = [
        x1 + 2 * x2 <= u1,
        x1 - 4 * x2 <= u2,
        5 * x1 + 76 * x2 <= 1,
    ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return result


def main(
    u1=-2, u2=-3,
):
    grad1 = 2.747741246681633
    grad2 = 2.885233448461934
    base = 8.222222222222221
    for d1 in (0, -0.1, 0.1,):
        for d2 in (0, -0.1, 0.1,):
            result = rep(
                u1=u1 + d1,
                u2=u2 + d2,
            )
            print(d1, d2,)
            print('actual', result)
            pred = base - grad1 * d1 - grad2 * d2
            print('pred', pred)
            import pdb; pdb.set_trace()


if __name__ == "__main__":
    fire.Fire(main)
