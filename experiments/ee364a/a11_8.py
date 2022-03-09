"""
[x] terminate based on duality gap at 1e-3
[x] check against cvxpy
[x] experiment with mu; inner newton steps and total newton steps
[x] plot log duality gap vs total newton steps
    - textbook format step plot

python -m ee364a.a11_8
"""

import fire
import numpy as np
import torch
import tqdm

from .a10_4 import Soln, LPCenteringProb, infeasible_start_newton_solve


def barrier_solve(soln: Soln, prob: LPCenteringProb, t: float, mu: float, epsilon=1e-3, verbose=False):
    this_step = 0
    steps = []
    newton_steps = []
    gaps = []
    while True:
        prob.t = t  # Solve the right problem.
        soln.nu = torch.zeros_like(soln.nu)
        soln, this_newton_steps, _, _, _ = infeasible_start_newton_solve(
            soln=soln, prob=prob, max_steps=2000, epsilon=1e-5,
        )

        this_step += 1
        this_gap = prob.m / t
        this_newton_steps = this_newton_steps[-1]

        steps.append(this_step)
        gaps.append(this_gap)
        newton_steps.append(this_newton_steps)

        if verbose:
            print(f'this_step: {this_step}, this_gap: {this_gap}, newton_steps: {this_newton_steps}')

        if this_gap < epsilon:
            break
        t = mu * t

    return soln, steps, gaps, newton_steps


def _generate_prob():
    m = 100
    n = 500

    A = torch.randn(m, n)
    A[0].abs_()
    rank = torch.linalg.matrix_rank(A)
    assert rank == m

    p = torch.randn(n).abs()  # Make positive.
    b = A @ p
    c = torch.randn(n)
    in_domain = lambda soln: torch.all(soln.x > 0)

    x = torch.randn(n).exp()  # Make positive.
    nu = torch.zeros(m)

    return Soln(x=x, nu=nu), LPCenteringProb(A=A, b=b, c=c, in_domain=in_domain)


@torch.no_grad()
def main(seed=0, t=0.5, mu=8):
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)  # Single precision makes this fail.

    soln_init, prob = _generate_prob()

    soln, steps, gaps, newton_steps = barrier_solve(
        soln=soln_init, prob=prob, t=t, mu=mu, verbose=True
    )
    # duality gap vs cumulative Newton steps
    from swissknife import utils
    utils.plot_wrapper(
        img_path=utils.join('.', 'ee364a', 'plots', 'a11_8'),
        suffixes=(".png", '.pdf'),
        steps=[dict(x=np.cumsum(newton_steps), y=gaps)],
        options=dict(yscale='log', xlabel='cumulative Newton steps', ylabel='duality gap')
    )

    # compare against CVXPY
    import cvxpy as cp

    x = cp.Variable(prob.n)
    obj = cp.Minimize(prob.c.numpy() @ x)
    con = [
        x >= 0.,
        prob.A.numpy() @ x == prob.b.numpy()
    ]
    cp.Problem(obj, con).solve()

    # diff should be small
    print(np.sum(prob.c.numpy() * soln.x.numpy()))  # -762.7124775791468
    print(np.sum(prob.c.numpy() * x.value))  # -762.7143847548298

    # vary mu
    avg_steps = []
    tot_steps = []
    mus = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    for mu in tqdm.tqdm(mus, desc="mu"):
        _, steps, gaps, newton_steps = barrier_solve(
            soln=soln_init, prob=prob, t=t, mu=mu, verbose=False
        )
        avg_steps.append(np.mean(newton_steps))
        tot_steps.append(np.sum(newton_steps))
    utils.plot_wrapper(
        img_path=utils.join('.', 'ee364a', 'plots', 'a11_8_2'),
        suffixes=(".png", '.pdf'),
        plots=[
            dict(x=mus, y=avg_steps, label='average steps per centering'),
            dict(x=mus, y=tot_steps, label='total steps'),
        ],
        options=dict(ylabel='Newton steps', xlabel='mu'),
    )


if __name__ == "__main__":
    fire.Fire(main)
