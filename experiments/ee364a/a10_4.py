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
"""

from dataclasses import dataclass
from typing import Optional

import fire
import torch


@dataclass
class LPCenteringProb:
    A: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor


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


def _solve_residual(soln: Soln, prob: LPCenteringProb):
    x, nu = soln.x, soln.nu
    A = prob.A
    g = prob.c - 1. / x + A.t() @ nu
    h = A @ x - prob.b
    return Soln(x=g, nu=h)


def _solve_kkt(soln: Soln, prob: LPCenteringProb):
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


def solve(soln: Soln, prob: LPCenteringProb, alpha, beta, max_steps=50, epsilon=10 ** -6, return_residual_norms=False):
    if not torch.all(soln.x > 0):
        raise ValueError("Initial iterate not in domain")
    if not (0 < alpha < .5) or not (0 < beta < 1):
        raise ValueError("Invalid solver params")

    this_step = 0
    residual_norms = []
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
                break
            t = beta * t  # Reduce step size.
        # ===
        print(res.norm())
        soln = proposal
        res = new_res
        this_step += 1
        if this_step >= max_steps or res.norm() <= epsilon:
            break

    if return_residual_norms:
        return soln, steps, residual_norms
    else:
        return soln, steps


def _generate_prob(m=5, n=10):
    A = torch.randn(m, n).abs()
    p = torch.randn(n)
    b = A @ p
    c = torch.randn(n).exp()

    x = torch.randn(n).exp()  # Make positive.
    nu = torch.zeros(m)
    return Soln(x=x, nu=nu), LPCenteringProb(A=A, b=b, c=c)


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64)

    soln, prob = _generate_prob()
    soln, steps, residual_norms = solve(
        soln=soln, prob=prob, alpha=0.45, beta=0.9, return_residual_norms=True
    )
    print(soln, steps)


if __name__ == "__main__":
    fire.Fire(main)
