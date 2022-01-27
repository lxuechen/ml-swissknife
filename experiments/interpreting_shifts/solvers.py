import warnings

import fire
import numpy as np
from scipy.special import logsumexp

from swissknife import utils


def sinkhorn_knopp_unbalanced(M, reg, reg_a, reg_b, numItermax=1000,
                              stopThr=1e-6, verbose=False, log=False,
                              a=np.array([]), b=np.array([]),
                              eps_div=1e-7, **unused_kwargs):
    """Allows different regularization weights on source and target domains."""
    utils.handle_unused_kwargs(unused_kwargs)

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = np.ones(dim_a, dtype=np.float64) / dim_a
    if len(b) == 0:
        b = np.ones(dim_b, dtype=np.float64) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = np.ones((dim_a, 1)) / dim_a
        v = np.ones((dim_b, n_hists)) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = np.ones(dim_a) / dim_a
        v = np.ones(dim_b) / dim_b

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    f_a = reg_a / (reg_a + reg)
    f_b = reg_b / (reg_b + reg)

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = K.dot(v)
        # Kv = np.where(Kv == 0, eps_div, Kv)
        u = (a / Kv) ** f_a
        Ktu = K.T.dot(u)
        # Ktu = np.where(Ktu == 0, eps_div, Ktu)
        v = (b / Ktu) ** f_b

        if (np.any(Ktu == 0.)
            or np.any(np.isnan(u)) or np.any(np.isnan(v))
            or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        err_u = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(), 1.)
        err_v = abs(v - vprev).max() / max(abs(v).max(), abs(vprev).max(), 1.)
        err = 0.5 * (err_u + err_v)
        if log:
            log['err'].append(err)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['logu'] = np.log(u + 1e-300)
        log['logv'] = np.log(v + 1e-300)

    if n_hists:  # return only loss
        res = np.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u[:, None] * K * v[None, :], log
        else:
            return u[:, None] * K * v[None, :]


def sinkhorn_stabilized_unbalanced(M, reg, reg_a, reg_b, tau=1e5, numItermax=1000,
                                   stopThr=1e-6, verbose=False, log=False,
                                   a=np.array([]), b=np.array([]), **kwargs):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = np.ones(dim_a, dtype=np.float64) / dim_a
    if len(b) == 0:
        b = np.ones(dim_b, dtype=np.float64) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = np.ones((dim_a, n_hists)) / dim_a
        v = np.ones((dim_b, n_hists)) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = np.ones(dim_a) / dim_a
        v = np.ones(dim_b) / dim_b

    # print(reg)
    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    f_a = reg_a / (reg_a + reg)
    f_b = reg_b / (reg_b + reg)

    cpt = 0
    err = 1.
    alpha = np.zeros(dim_a)
    beta = np.zeros(dim_b)
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v

        Kv = K.dot(v)
        f_alpha = np.exp(- alpha / (reg + reg_a))
        f_beta = np.exp(- beta / (reg + reg_b))

        if n_hists:
            f_alpha = f_alpha[:, None]
            f_beta = f_beta[:, None]
        u = ((a / (Kv + 1e-16)) ** f_a) * f_alpha
        Ktu = K.T.dot(u)
        v = ((b / (Ktu + 1e-16)) ** f_b) * f_beta
        absorbing = False
        if (u > tau).any() or (v > tau).any():
            absorbing = True
            if n_hists:
                alpha = alpha + reg * np.log(np.max(u, 1))
                beta = beta + reg * np.log(np.max(v, 1))
            else:
                alpha = alpha + reg * np.log(np.max(u))
                beta = beta + reg * np.log(np.max(v))
            K = np.exp((alpha[:, None] + beta[None, :] -
                        M) / reg)
            v = np.ones_like(v)

        if (np.any(Ktu == 0.)
            or np.any(np.isnan(u)) or np.any(np.isnan(v))
            or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % cpt)
            u = uprev
            v = vprev
            break
        if (cpt % 10 == 0 and not absorbing) or cpt == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = abs(u - uprev).max() / max(abs(u).max(), abs(uprev).max(),
                                             1.)
            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1

    if err > stopThr:
        warnings.warn("Stabilized Unbalanced Sinkhorn did not converge." +
                      "Try a larger entropy `reg` or a lower mass `reg_m`." +
                      "Or a larger absorption threshold `tau`.")
    if n_hists:
        logu = alpha[:, None] / reg + np.log(u)
        logv = beta[:, None] / reg + np.log(v)
    else:
        logu = alpha / reg + np.log(u)
        logv = beta / reg + np.log(v)
    if log:
        log['logu'] = logu
        log['logv'] = logv
    if n_hists:  # return only loss
        res = logsumexp(np.log(M + 1e-100)[:, :, None] + logu[:, None, :] +
                        logv[None, :, :] - M[:, :, None] / reg, axis=(0, 1))
        res = np.exp(res)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        ot_matrix = np.exp(logu[:, None] + logv[None, :] - M / reg)
        if log:
            return ot_matrix, log
        else:
            return ot_matrix


def test_unbalanced_solvers(
    reg_a=10.,
    reg_b=0.1,
    reg=0.1,
    stable_version=False,
    seed=42,
    img_path=None,
):
    np.random.seed(seed)

    n = 10
    mu1, mu2, mu3 = -3, 0, 3
    std = 0.3
    x1 = -np.ones(n) * 0.3
    x2 = -x1 * 0.3
    y1 = np.concatenate(
        [np.random.randn(n // 2) * std + mu1,
         np.random.randn(n // 2) * std + mu2],
    )
    y2 = np.concatenate(
        [np.random.randn(n // 2) * std + mu2,
         np.random.randn(n // 2) * std + mu3],
    )

    source = np.stack([x1, y1], axis=1)
    target = np.stack([x2, y2], axis=1)

    M = np.sqrt(np.sum((source[..., None] - target.T[None, ...]) ** 2, axis=1))
    ot_solver = {
        True: sinkhorn_stabilized_unbalanced,
        False: sinkhorn_knopp_unbalanced,
    }[stable_version]
    gamma = ot_solver(M, reg=reg, reg_a=reg_a, reg_b=reg_b, tau=100)

    source_marg = gamma.sum(axis=1)
    target_marg = gamma.sum(axis=0)

    def tv_to_uniform(probs):
        """Total variation distance to uniform distribution."""
        assert len(probs.shape) == 1
        uniform_probs = np.ones_like(probs) / len(probs)
        return .5 * np.abs(probs - uniform_probs).sum()

    print(f'reg_a: {reg_a}, reg_b: {reg_b}')

    print('---')
    print(f'source_marg: {source_marg}')
    print(f'tv_to_uniform: {tv_to_uniform(source_marg):.4f}, sum: {np.sum(source_marg):.4f}')

    print('---')
    print(f'target_marg: {target_marg}')
    print(f'tv_to_uniform: {tv_to_uniform(target_marg):.4f}, sum: {np.sum(target_marg):.4f}')

    if img_path is not None:
        scatters = [
            dict(x=x1, y=y1, color=utils.get_sns_colors()[0], label='source'),
            dict(x=x2, y=y2, color=utils.get_sns_colors()[1], label='target')
        ]

        alpha_scale = 10
        plots = []
        for s in range(len(gamma)):
            for t in range(len(gamma[0])):
                plots.append(
                    dict(
                        x=[x1[s], x2[t]],
                        y=[y1[s], y2[t]],
                        alpha=max(min(gamma[s, t] * alpha_scale, 1), 0),
                        color=utils.get_sns_colors()[2],
                    )
                )

        utils.plot_wrapper(
            img_path=img_path,
            suffixes=('.png', '.pdf'),
            plots=plots,
            scatters=scatters,
            options=dict(
                title=f'$\epsilon_P$: {reg_a}, $\epsilon_Q$: {reg_b}; darker means higher joint mass'
            )
        )


def plot_4_cases(
    hi=10, lo=0.1, **kwargs
):
    # Check if the different regularization parameters work

    # source marginal uniform, target marginal not uniform and first set matched
    test_unbalanced_solvers(
        reg_a=hi, reg_b=lo, img_path=utils.join('.', 'interpreting_shifts', 'plots', 'high_a_low_b'),
    )
    # source and target marginal both uniform
    test_unbalanced_solvers(
        reg_a=hi, reg_b=hi, img_path=utils.join('.', 'interpreting_shifts', 'plots', 'high_a_high_b')
    )
    # source marginal not uniform and second set matched, target marginal uniform
    test_unbalanced_solvers(
        reg_a=lo, reg_b=hi, img_path=utils.join('.', 'interpreting_shifts', 'plots', 'low_a_high_b')
    )
    # source and target marginal both not uniform
    test_unbalanced_solvers(
        reg_a=lo, reg_b=lo, img_path=utils.join('.', 'interpreting_shifts', 'plots', 'low_a_low_b')
    )


def main(task="plot_4_cases", **kwargs):
    if task == "plot_4_cases":
        plot_4_cases(**kwargs)
    elif task == "test_unbalanced_solvers":
        test_unbalanced_solvers(**kwargs)


if __name__ == '__main__':
    # python -m interpreting_shifts.solvers
    fire.Fire(main)
