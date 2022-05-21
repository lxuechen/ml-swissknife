"""Common numerical algorithms.

Eigenvalue computation for `lanczos_tridiag` and `orthogonal_iteration_mem_saving` are accurate.

TODO:
    Eigenvector computation for `lanczos_tridiag` and `orthogonal_iteration_mem_saving` not working well.
    Memory scaling for `lanczos_tridiag` is bad.
    Memory scaling for `orthogonal_iteration_mem_saving` is bad in G-S.
"""
import math
from typing import Callable, Optional, Union, Sequence

import numpy as np
import torch
import tqdm


def power_iteration(
    matmul_closure: Callable,
    max_iter,
    dtype: torch.dtype,
    device: torch.device,
    matrix_shape: Union[torch.Size, Sequence[int]],
    v0: Optional[torch.Tensor] = None,
):
    """Run power iteration to find the top eigenvector with maximum absolute eigenvalue.

    Args:
        matmul_closure: Matrix vector product.
        max_iter: The number of iterations to run.
        dtype: For matmul.
        device: For matmul.
        matrix_shape: Shape of matrix. Should not be batched.
        v0: Tensor of the (batch) of vectors to initialize the power iteration.

    Returns:
       The eigenvalues and the eigenvectors.
    """
    if v0 is None:
        v0 = torch.randn(matrix_shape[-1], device=device, dtype=dtype)
        v0 /= v0.norm(2, dim=-1, keepdim=True)

    for _ in range(max_iter):
        mvp = matmul_closure(v0)
        v0 = mvp / mvp.norm(dim=-1, keepdim=True)

    vector = v0
    value = (matmul_closure(vector) * vector).sum(dim=-1) / (vector ** 2).sum(dim=-1)
    return value, vector


def lanczos_tridiag(
    matmul_closure: Callable,
    max_iter: int,
    dtype: torch.dtype,
    device: torch.device,
    matrix_shape: Union[torch.Size, Sequence[int]],
    init_vecs=None,
    num_init_vecs=1,
    tol=1e-5,
):
    """Run Lanczos tridiagonalization with reorthogonalization.

    The returned eigenvalues are mostly accurate.
    """
    if not callable(matmul_closure):
        raise RuntimeError(
            "matmul_closure should be a function callable object that multiples a (Lazy)Tensor "
            "by a vector. Got a {} instead.".format(matmul_closure.__class__.__name__)  # noqa
        )

    if init_vecs is None:
        init_vecs = torch.randn(matrix_shape[-1], num_init_vecs, dtype=dtype, device=device)  # (p, k).
    else:
        num_init_vecs = init_vecs.size(-1)

    # Define some constants
    num_iter = min(max_iter, matrix_shape[-1])
    dim_dimension = -2

    # Create storage for q_mat, alpha, and beta
    # q_mat - orthogonal matrix of decomposition
    # alpha - main diagonal of T
    # beta - off diagonal of T
    q_mat = torch.zeros(num_iter, matrix_shape[-1], num_init_vecs, dtype=dtype, device=device)
    t_mat = torch.zeros(num_iter, num_iter, num_init_vecs, dtype=dtype, device=device)

    # Begin algorithm
    # Initial Q vector: q_0_vec
    q_0_vec = init_vecs / torch.norm(init_vecs, 2, dim=dim_dimension, keepdim=True)
    q_mat[0].copy_(q_0_vec)

    # Initial alpha value: alpha_0
    r_vec = matmul_closure(q_0_vec)
    alpha_0 = q_0_vec.mul(r_vec).sum(dim_dimension)

    # Initial beta value: beta_0
    r_vec.sub_(alpha_0.unsqueeze(dim_dimension).mul(q_0_vec))
    beta_0 = torch.norm(r_vec, 2, dim=dim_dimension)

    # Copy over alpha_0 and beta_0 to t_mat
    t_mat[0, 0].copy_(alpha_0)
    t_mat[0, 1].copy_(beta_0)
    t_mat[1, 0].copy_(beta_0)

    # Compute the first new vector
    q_mat[1].copy_(r_vec.div_(beta_0.unsqueeze(dim_dimension)))

    # Now we start the iteration
    for k in range(1, num_iter):
        torch.cuda.empty_cache()

        # Get previous values
        q_prev_vec = q_mat[k - 1]
        q_curr_vec = q_mat[k]
        beta_prev = t_mat[k, k - 1].unsqueeze(dim_dimension)

        # Compute next alpha value
        r_vec = matmul_closure(q_curr_vec) - q_prev_vec.mul(beta_prev)
        alpha_curr = q_curr_vec.mul(r_vec).sum(dim_dimension, keepdim=True)
        # Copy over to t_mat
        t_mat[k, k].copy_(alpha_curr.squeeze(dim_dimension))

        # Copy over alpha_curr, beta_curr to t_mat
        if (k + 1) < num_iter:
            # Compute next residual value
            r_vec.sub_(alpha_curr.mul(q_curr_vec))
            # Full reorthogonalization: r <- r - Q (Q^T r)
            correction = r_vec.unsqueeze(0).mul(q_mat[: k + 1]).sum(dim_dimension, keepdim=True)
            correction = q_mat[: k + 1].mul(correction).sum(0)
            r_vec.sub_(correction)
            r_vec_norm = torch.norm(r_vec, 2, dim=dim_dimension, keepdim=True)
            r_vec.div_(r_vec_norm)

            # Get next beta value
            beta_curr = r_vec_norm.squeeze_(dim_dimension)
            # Update t_mat with new beta value
            t_mat[k, k + 1].copy_(beta_curr)
            t_mat[k + 1, k].copy_(beta_curr)

            # Run more reorthoganilzation if necessary
            inner_products = q_mat[: k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)
            could_reorthogonalize = False
            for _ in range(10):
                if not torch.sum(inner_products > tol):
                    could_reorthogonalize = True
                    break
                correction = r_vec.unsqueeze(0).mul(q_mat[: k + 1]).sum(dim_dimension, keepdim=True)
                correction = q_mat[: k + 1].mul(correction).sum(0)
                r_vec.sub_(correction)
                r_vec_norm = torch.norm(r_vec, 2, dim=dim_dimension, keepdim=True)
                r_vec.div_(r_vec_norm)
                inner_products = q_mat[: k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)

            # Update q_mat with new q value
            q_mat[k + 1].copy_(r_vec)

            if torch.sum(beta_curr.abs() > 1e-6) == 0 or not could_reorthogonalize:
                break

    # Now let's transpose q_mat, t_mat intot the correct shape
    num_iter = k + 1

    # num_init_vecs x matrix_shape[-1] x num_iter
    q_mat = q_mat[:num_iter].permute(-1, -2, 0).contiguous()
    # num_init_vecs x num_iter x num_iter
    t_mat = t_mat[:num_iter, :num_iter].permute(-1, 0, 1).contiguous()

    q_mat.squeeze_(0)
    t_mat.squeeze_(0)

    return q_mat, t_mat


def lanczos_tridiag_to_diag(t_mat):
    return torch.linalg.eigh(t_mat)


def orthogonal_iteration_mem_saving(
    matmul_closure: Callable,  # CPU -> CPU function.
    k: int,
    dtype: torch.dtype,
    device: torch.device,  # Some GPU device where bulk of computation is performed.
    matrix_shape: torch.Size,
    num_power_iteration: Optional[int] = 1,
    # A function that takes in the current Q matrix on CPU.
    callback: Optional[Callable] = None,
    enable_tqdm=False,
):
    """Orthogonal iteration.

    Different from torch.linalg.eigh, outputs are sorted in descending order.
    Can be used to obtain top eigenvectors/eigenvalues, or to perform PCA.

    Returns a CPU tensor.
    """
    Q = torch.randn(size=(matrix_shape[-1], k), dtype=dtype)
    Q = _gram_schmidt_mem_saving(Q, device)

    for _ in tqdm.tqdm(range(num_power_iteration), disable=not enable_tqdm):
        matmul = matmul_closure(Q)
        Q = _gram_schmidt_mem_saving(matmul, device)

        if callback is not None:
            callback(Q)

    return Q


def _gram_schmidt_mem_saving(Q, device, batch_size=100):
    Q = Q.to(device)
    n, m = Q.size()
    for i in range(m):
        # Normalize the ith column.
        col = Q[:, i: i + 1]  # (n, 1).
        col /= col.norm(2)
        # Remove contribution of this component for remaining columns.
        if i + 1 < m:
            rest = Q[:, i + 1:]  # (p, r).
            r = rest.size(1)

            start_idx = 0
            while start_idx < r:
                batch = rest[:, start_idx:start_idx + batch_size]
                # Broadcast, point-wise multiply, and then reduce seems to
                # suffer from less imprecision than direct matmul or mm.
                batch -= torch.sum(col * batch, dim=0) * col
                start_idx += batch_size
        torch.cuda.empty_cache()

    Q = Q.cpu()
    torch.cuda.empty_cache()
    return Q


def eigenvectors_to_eigenvalues(
    mat: torch.Tensor,  # (p, p).
    eigenvectors: torch.Tensor,  # (p, k).
):
    """Rayleigh quotient computation."""
    return ((mat @ eigenvectors) * eigenvectors).sum(dim=0) / (eigenvectors * eigenvectors).sum(dim=0)


def eigv_to_density(eig_vals, all_weights=None, grids=None,
                    grid_len=10000, sigma_squared=None, grid_expand=1e-2):
    """Compute the smoothed spectral density from a set of eigenvalues.
    Convolves the given eigenvalues with a Gaussian kernel, weighting the values
    by all_weights (or uniform weighting if all_weights is None). Example output
    can be seen in Figure 1 of https://arxiv.org/pdf/1901.10159.pdf. Visualizing
    the estimated density can be done by calling plt.plot(grids, density). There
    is likely not a best value of sigma_squared that works for all use cases,
    so it is recommended to try multiple values in the range [1e-5,1e-1].
    Args:
      eig_vals: Array of shape [num_draws, order]
      all_weights: Array of shape [num_draws, order], if None then weights will be
        taken to be uniform.
      grids: Array of shape [grid_len], the smoothed spectrum will be plotted
        in the interval [grids[0], grids[-1]]. If None then grids will be
        computed based on max and min eigenvalues and grid length.
      grid_len: Integer specifying number of grid cells to use, only used if
        grids is None
      sigma_squared: Scalar. Controls the smoothing of the spectrum estimate.
        If None, an appropriate value is inferred.
      grid_expand: Controls the window of values that grids spans.
        grids[0] = smallest eigenvalue - grid_expand.
        grids[-1] = largest_eigenvalue + grid_expand.
    Returns:
      density: Array of shape [grid_len], the estimated density, averaged over
        all draws.
      grids: Array of shape [grid_len]. The values the density is estimated on.
    """
    if all_weights is None:
        all_weights = np.ones(eig_vals.shape) * 1.0 / float(eig_vals.shape[1])
    num_draws = eig_vals.shape[0]

    lambda_max = np.nanmean(np.max(eig_vals, axis=1), axis=0) + grid_expand
    lambda_min = np.nanmean(np.min(eig_vals, axis=1), axis=0) - grid_expand

    if grids is None:
        assert grid_len is not None, 'grid_len is required if grids is None.'
        grids = np.linspace(lambda_min, lambda_max, num=grid_len)

    grid_len = grids.shape[0]
    if sigma_squared is None:
        sigma = 10 ** -5 * max(1, (lambda_max - lambda_min))
    else:
        sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    density_each_draw = np.zeros((num_draws, grid_len))
    for i in range(num_draws):

        if np.isnan(eig_vals[i, 0]):
            raise ValueError('tridaig has nan values.')
        else:
            for j in range(grid_len):
                x = grids[j]
                vals = _kernel(eig_vals[i, :], x, sigma)
                density_each_draw[i, j] = np.sum(vals * all_weights[i, :])
    density = np.nanmean(density_each_draw, axis=0)
    norm_fact = np.sum(density) * (grids[1] - grids[0])
    density = density / norm_fact
    return density, grids


def _kernel(x, x0, variance):
    """Point estimate of the Gaussian kernel.
    This function computes the Gaussian kernel for
    C exp(-(x - x0) ^2 /(2 * variance)) where C is the appropriate normalization.
    variance should be a list of length 1. Either x0 or x should be a scalar. Only
    one of the x or x0 can be a numpy array.
    Args:
      x: Can be either scalar or array of shape [order]. Points to estimate
        the kernel on.
      x0: Scalar. Mean of the kernel.
      variance: Scalar. Variance of the kernel.
    Returns:
      point_estimate: A scalar corresponding to
        C exp(-(x - x0) ^2 /(2 * variance)).
    """
    coeff = 1.0 / np.sqrt(2 * math.pi * variance)
    val = -(x0 - x) ** 2
    val = val / (2.0 * variance)
    val = np.exp(val)
    point_estimate = coeff * val
    return point_estimate
