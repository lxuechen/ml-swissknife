"""Common numerical algorithms."""
from typing import Callable, Optional, Union, Sequence

import torch


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

    return q_mat, t_mat


def orthogonal_iteration():
    pass
