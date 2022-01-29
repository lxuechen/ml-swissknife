from typing import Optional

import cvxpy
import cvxpy as cp
import fire


def sum4(
    p,  # List of probabilities as a cvxpy Variable; (16,).
    x1: Optional[int] = None,
    x2: Optional[int] = None,
    x3: Optional[int] = None,
    x4: Optional[int] = None,
):
    """Summation function for "4-dimensional" tensor.

    When `xi` is not None, only take entries with prescribed values.
    A value of `None` means marginalization over the given variable.
    """
    res = cp.Constant(0.)
    for _x1 in (0, 1):
        for _x2 in (0, 1):
            for _x3 in (0, 1):
                for _x4 in (0, 1):
                    # Has prescribed value and is not equal to current index.
                    if x1 is not None and _x1 != x1:
                        continue
                    if x2 is not None and _x2 != x2:
                        continue
                    if x3 is not None and _x3 != x3:
                        continue
                    if x4 is not None and _x4 != x4:
                        continue

                    # 2^0, 2^1, 2^2, 2^3.
                    index = _x1 + 2 * _x2 + 4 * _x3 + 8 * _x4
                    res += p[index]
    return res


def main():
    p = cp.Variable(shape=(16,))
    for obj_func in (
        lambda t: cp.Minimize(sum4(p, x4=1)),
        lambda t: cp.Maximize(sum4(p, x4=1)),
    ):
        objective = obj_func(p)
        constraints = [
            sum4(p, x1=1) == 0.9,
            sum4(p, x2=1) == 0.9,
            sum4(p, x3=1) == 0.1,
            sum4(p, x1=1, x3=1, x4=0) == cvxpy.Constant(0.7) * sum4(p, x3=1),
            sum4(p, x2=1, x3=0, x4=1) == cvxpy.Constant(0.6) * sum4(p, x2=1, x3=0),

            cp.sum(p) == 1.,
            p >= 0.,
            p <= 1.,
        ]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        print(result, p.value)


if __name__ == "__main__":
    fire.Fire(main)
