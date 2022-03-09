"""

[x] Plot residual norms to verify quadratic convergence (semilog).
[x] Stop when norm residual <= 10 ** -6 or num iterations reach 50.
[x] Generating the problem instance requires care
[x] Vary algo. parameters alpha, beta
[x] infeasible => KKT linear system cannot be solved if A isn't full rank
[x] unbounded below

- backtracking line search has to backtrack until the new point, x + t * dx, is in the domain of the objective function
-  the sufficient decrease condition, is that you iterate on reducing t until the 2 norm of the residual at the new
iterate (after taking the step corresponding to that value of t) achieves some sufficient decrease from the 2 norm of
the residual at the current iterate

Common mistakes:
    - nan due to division
    - not checking if iterate still in domain (use `in_domain`)

python -m ee364a.a10_4
"""

from dataclasses import dataclass
import logging
from typing import Optional, Callable

import fire
import numpy as np
import torch
import tqdm

from swissknife import utils


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
    t: Optional[float] = None  # To reuse in barrier method.
    in_domain: Optional[Callable] = None

    @property
    def m(self):
        return self.A.size(0)

    @property
    def n(self):
        return self.A.size(1)

    def loss(self, soln: Soln):
        return self.c @ soln.x - torch.log(soln.x).sum()

    def _grad(self, soln: Soln):  # grad objective
        t = 1. if self.t is None else self.t
        return t * self.c - 1. / soln.x

    def _hinv(self, soln: Soln):  # Hessian inverse.
        return torch.diag(soln.x ** 2.)

    def solve_residual(self, soln: Soln):
        # gradf + A^t nu, A x - b
        x, nu = soln.x, soln.nu
        g = self._grad(soln) + self.A.t() @ nu
        h = self.A @ x - self.b
        return Soln(x=g, nu=h)

    def solve_kkt(self, soln: Soln):
        # Solve for descent direction.
        # textbook 10.21 + 10.33
        Hinv = self._hinv(soln)
        residual = self.solve_residual(soln)
        g, h = residual.x, residual.nu

        HinvAt = Hinv @ self.A.t()
        Hinvg = Hinv @ g
        S = -self.A @ HinvAt
        w = torch.inverse(S) @ (self.A @ Hinvg - h)
        v = Hinv @ (- self.A.t() @ w - g)
        return Soln(x=v, nu=w)


def infeasible_start_newton_solve(
    soln: Soln, prob: LPCenteringProb, alpha=0.4, beta=0.9, max_steps=100, epsilon=1e-8,
):
    if not torch.all(soln.x > 0):
        raise ValueError("Initial iterate not in domain")
    if not (0 < alpha < .5) or not (0 < beta < 1.):
        raise ValueError("Invalid solver params")

    this_step = 0
    residual_norms = []
    losses = []
    steps = []
    ls_steps = []  # line search steps
    while True:
        res = prob.solve_residual(soln)
        direction = prob.solve_kkt(soln)

        steps.append(this_step)
        residual_norms.append(res.norm().item())

        # === backtracking line search
        t = 1.
        cnt = 0
        while True:
            proposal = soln + direction * t
            new_res = prob.solve_residual(soln=proposal)

            factor = (1. - alpha * t)
            if new_res.norm() <= factor * res.norm():
                if prob.in_domain is None:
                    break
                elif prob.in_domain(proposal):
                    break
            t = beta * t  # Reduce step size.
            cnt += 1
        ls_steps.append(cnt)
        # ===
        soln = proposal
        res = new_res
        this_step += 1
        if this_step >= max_steps or res.norm() <= epsilon:
            if this_step >= max_steps:
                logging.warning("Hit max steps!")
            steps.append(this_step)
            residual_norms.append(res.norm().item())
            break

    return soln, steps, residual_norms, losses, ls_steps


def _generate_prob(infeasible=False, unbounded=False):
    m = 100
    n = 600

    if infeasible:
        A = torch.zeros(m, n)
        b = torch.ones(m)

        c = torch.randn(n)

        x = torch.randn(n).exp() * 3  # Make positive.
        nu = torch.randn(m)

    elif unbounded:
        A = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ],
            dtype=torch.get_default_dtype()
        )
        b = torch.tensor([0, 0], dtype=torch.get_default_dtype())
        c = torch.zeros(4)

        x = torch.randn(4).exp()
        nu = torch.zeros(2)
    else:
        A = torch.randn(m, n)
        A[0].abs_()
        rank = torch.linalg.matrix_rank(A)
        assert rank == m

        p = torch.randn(n).abs()  # Make positive.
        b = A @ p

        c = torch.randn(n)

        x = torch.randn(n).exp() * 3  # Make positive.
        nu = torch.randn(m)

    in_domain = lambda soln: torch.all(soln.x > 0)
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

    # Quad conv.
    soln_init, prob = _generate_prob()
    soln, steps, residual_norms, losses, _ = infeasible_start_newton_solve(
        soln=soln_init, prob=prob,
        alpha=0.4, beta=0.9, epsilon=1e-7, max_steps=100,
    )
    utils.plot_wrapper(
        img_path=utils.join('.', 'plots', 'a10_4'),
        suffixes=(".png", '.pdf'),
        plots=[dict(x=steps, y=residual_norms, label='infeasible start Newton')],
        options=dict(xlabel='step count', ylabel='norm of residual', yscale='log')
    )

    soln_init, prob = _generate_prob()
    alphas = np.linspace(0.01, 0.49, num=10)
    plots = []
    for beta in tqdm.tqdm(np.linspace(0.1, 0.9, num=5), desc="beta"):
        this_x = alphas
        this_y = []
        for alpha in tqdm.tqdm(alphas, desc="alpha"):
            soln, steps, residual_norms, losses, ls_steps = infeasible_start_newton_solve(
                soln=soln_init, prob=prob, alpha=alpha, beta=beta
            )
            this_y.append(steps[-1])
        plots.append(
            dict(x=this_x, y=this_y, label=f'$\\beta={beta:.2f}$', marker='x', alpha=0.8)
        )
    utils.plot_wrapper(
        img_path=utils.join('.', 'plots', 'a10_4_2'),
        suffixes=(".png", '.pdf'),
        plots=plots,
        options=dict(xlabel='alpha', ylabel='total Newton steps')
    )

    # infeasible.
    # soln_init, prob = _generate_prob(infeasible=True)
    # soln, steps, residual_norms, losses, _ = infeasible_start_newton_solve(
    #     soln=soln_init, prob=prob,
    #     alpha=0.4, beta=0.9, epsilon=1e-7, max_steps=100,
    # )
    # utils.plot_wrapper(
    #     img_path=utils.join('.', 'plots', 'a10_4_3'),
    #     suffixes=(".png", '.pdf'),
    #     plots=[dict(x=steps, y=residual_norms, label='infeasible start Newton')],
    #     options=dict(xlabel='step count', ylabel='norm of residual', yscale='log', title='infeasible problem')
    # )

    # unbounded.
    soln_init, prob = _generate_prob(unbounded=True)
    soln, steps, residual_norms, losses, _ = infeasible_start_newton_solve(
        soln=soln_init, prob=prob,
        alpha=0.4, beta=0.9, epsilon=1e-7, max_steps=500,
    )
    utils.plot_wrapper(
        img_path=utils.join('.', 'plots', 'a10_4_4'),
        suffixes=(".png", '.pdf'),
        plots=[dict(x=steps, y=residual_norms, label='infeasible start Newton')],
        options=dict(xlabel='step count', ylabel='norm of residual', yscale='log', title='infeasible problem')
    )


if __name__ == "__main__":
    fire.Fire(main)
