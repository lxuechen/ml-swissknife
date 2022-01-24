import warnings

import fire
import numpy as np

from swissknife import utils


def sinkhorn_knopp_unbalanced(M, reg, reg_a, reg_b, numItermax=1000,
                              stopThr=1e-6, verbose=False, log=False,
                              a=np.array([]), b=np.array([]), **unused_kwargs):
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
        u = (a / Kv) ** f_a
        Ktu = K.T.dot(u)
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


# TODO: Also hack the other solver!

def main(
    reg_a=10,
    reg_b=0.1,
    reg=0.01,
):
    # Check if the different regularization parameters work.
    # python -m interpreting_shifts.solvers --reg_a 10 --reg_b 0.1
    #   source marginal uniform, target marginal not uniform and first set matched

    # python -m interpreting_shifts.solvers --reg_a 10 --reg_b 10
    #   source and target marginal both uniform

    # python -m interpreting_shifts.solvers --reg_a 0.1 --reg_b 10
    #   source marginal not uniform and second set matched, target marginal uniform

    # python -m interpreting_shifts.solvers --reg_a 0.1 --reg_b 0.1
    #   source and target marginal both not uniform

    np.random.seed(10)

    n = 10
    mu1, mu2, mu3 = -1, 0, 1
    std = 0.1
    x1 = x2 = np.ones(n)
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
    gamma = sinkhorn_knopp_unbalanced(M, reg=reg, reg_a=reg_a, reg_b=reg_b, )

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


if __name__ == '__main__':
    fire.Fire(main)
