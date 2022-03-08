"""

[x] Plot residual norms to verify quadratic convergence (semilog).
[x] Stop when norm residual <= 10 ** -6 or num iterations reach 50.
[ ] Vary algo. parameters alpha, beta
[x] Generating the problem instance requires care
[ ] Feasible vs non-feasible

- backtracking line search has to backtrack until the new point, x + t * dx, is in the domain of the objective function
-  the sufficient decrease condition, is that you iterate on reducing t until the 2 norm of the residual at the new
iterate (after taking the step corresponding to that value of t) achieves some sufficient decrease from the 2 norm of
the residual at the current iterate

Common mistakes:
    - nan due to division
    - not checking if iterate still in domain (use `in_domain`)
"""

from dataclasses import dataclass
from typing import Optional, Callable

import fire
import numpy as np
import torch


@dataclass
class Soln:
    x: torch.Tensor
    nu: Optional[torch.Tensor] = None

    def __add__(self, other):
        return Soln(x=self.x + other.x, nu=self.nu + other.nu)

    def __mul__(self, other: float):
        return Soln(x=self.x * other, nu=self.nu * other)

    def norm(self):
        return torch.cat([self.x.flatten(), self.nu.flatten()]).norm(p=2)


@dataclass
class LPCenteringProb:
    A: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    in_domain: Optional[Callable] = None

    def loss(self, soln: Soln):
        return self.c @ soln.x - torch.log(soln.x).sum()


def _solve_residual(soln: Soln, prob: LPCenteringProb):
    # correct
    # gradf + A^t nu, A x - b
    x, nu = soln.x, soln.nu
    A = prob.A
    g = prob.c - 1. / x + A.t() @ nu
    h = A @ x - prob.b
    return Soln(x=g, nu=h)


def _solve_kkt(soln: Soln, prob: LPCenteringProb):
    # correct
    # textbook 10.21 + 10.33s
    x, nu = soln.x, soln.nu
    Hinv = torch.diag(x ** 2.)
    A = prob.A
    residual = _solve_residual(soln, prob)
    g, h = residual.x, residual.nu

    HinvAt = Hinv @ A.t()
    Hinvg = Hinv @ g
    S = -A @ HinvAt
    w = torch.inverse(S) @ (A @ Hinvg - h)
    v = Hinv @ (- A.t() @ w - g)
    return Soln(x=v, nu=w)  # Descent direction.


def solve(soln: Soln, prob: LPCenteringProb, alpha, beta, max_steps=50, epsilon=10 ** -6):
    if not torch.all(soln.x > 0):
        raise ValueError("Initial iterate not in domain")
    if not (0 < alpha < .5) or not (0 < beta < 1.):
        raise ValueError("Invalid solver params")

    this_step = 0
    residual_norms = []
    losses = []
    steps = []
    while True:
        res = _solve_residual(soln=soln, prob=prob)
        direction = _solve_kkt(soln, prob)

        steps.append(this_step)
        residual_norms.append(res.norm().item())

        # === backtracking line search
        t = 1.
        while True:
            proposal = soln + direction * t
            new_res = _solve_residual(soln=proposal, prob=prob)

            factor = (1. - alpha * t)
            if new_res.norm() <= factor * res.norm():
                if prob.in_domain is None:
                    break
                elif prob.in_domain(proposal):
                    break
            t = beta * t  # Reduce step size.
        # ===
        soln = proposal
        res = new_res
        this_step += 1
        if this_step >= max_steps or res.norm() <= epsilon:
            break

    return soln, steps, residual_norms, losses


def _generate_prob():
    m = 5
    n = 10
    A = torch.randn(m, n)
    A[0].abs_()
    rank = torch.linalg.matrix_rank(A)
    assert rank == m

    p = torch.randn(n).abs()  # Make positive.
    b = A @ p
    c = torch.randn(n)
    in_domain = lambda soln: torch.all(soln.x > 0)

    x = torch.randn(n).exp() * 2  # Make positive.
    nu = torch.randn(m)

    return Soln(x=x, nu=nu), LPCenteringProb(A=A, b=b, c=c, in_domain=in_domain)


def _load_example_prob():
    """Problem for one-step sanity check."""
    m = 4
    n = 8
    np.random.seed(364)
    A = np.round(np.random.rand(m, n), 2)
    A = np.vstack((A, np.ones(n)))
    p = np.random.rand(n)
    b = np.round(A.dot(p), 2)
    c = np.round(np.random.rand(n), 2)
    x0 = np.random.rand(n)
    nu = np.zeros((m + 1,))
    A, b, c, x0, nu = tuple(torch.from_numpy(a) for a in (A, b, c, x0, nu))
    in_domain = lambda soln: torch.all(soln.x > 0)
    return Soln(x=x0, nu=nu), LPCenteringProb(A=A, b=b, c=c, in_domain=in_domain)

    # Use alpha = 0.1
    #     beta = 0.5
    #     epsilon = 1e-5

    # RESULTS OF FIRST ITERATION
    # dual_residual   = np.array([-9.45445041e-01, -8.68890819e-01, -6.19225322e-01, -1.36394692e-01,
    # -2.80209340e-01, -2.29080275e+02, -1.74922026e+01, -9.77238004e-01,])
    # primal_residual = np.array([0.76155693, 1.26350154, 1.59255689, 0.74703098, 2.0962205,])
    # residuals       = 229.774887
    # direction_x     = np.array([-0.20992729,  0.01027243, -0.52409603, -0.33186397, -0.62116449,  0.00432129,
    # 0.05484001, -0.47860245])
    # direction_nu    = np.array([ 1.46632973, -1.64664555,  2.18458155,  1.37453686, -0.23884889,])
    # DO NOT USE == TO COMPARE - CHECK norm(your_value, our_value) IS SMALL

    # LINE SEARCH HINT: If the step length is too long, the tentative point can leave the domain.
    # Make sure to check your point is in the domain before testing whether you should terminate the line search.


@torch.no_grad()
def main(seed=0):
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    soln, prob = _generate_prob()
    soln, steps, residual_norms, losses = solve(
        soln=soln, prob=prob,
        alpha=0.1, beta=0.5, epsilon=1e-5,
    )
    print(soln)
    print(steps)
    print(residual_norms)
    print(losses)


if __name__ == "__main__":
    fire.Fire(main)
