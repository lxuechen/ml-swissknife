"""Common numerical algorithms."""
from typing import Callable, Optional

import torch


def power_iter(
    mat: Optional[torch.Tensor] = None,
    func: Optional[Callable] = None,
    v0: Optional[torch.Tensor] = None,
    eigenvectors=False,
    num_iterations=100
):
    """Run power iteration to find the top eigenvector with maximum absolute eigenvalue.

    Args:
        mat: Tensor of the (batch) of matrices.
        func: Matrix vector product.
        v0: Tensor of the (batch) of vectors to initialize the power iteration.
        eigenvectors: Returns the eigenvectors if True.
        num_iterations: The number of iterations to run.

    Returns:
       The eigenvalues and the eigenvectors.
    """
    if mat is None:
        if v0 is None:
            raise ValueError(
                f'`v0` should not be None when the input matrix is implicitly defined via a matrix-vector product.')
        if func is None:
            raise ValueError(
                f'`func` should not be None when the input matrix is implicitly defined via a matrix-vector product.')
        eigenvec = v0
    else:
        if v0 is None:
            if mat.dim() == 3:
                eigenvec = torch.randn(size=(*mat.shape[:2], 1), dtype=mat.dtype, device=mat.device)
            elif mat.dim() == 2:
                eigenvec = torch.randn(size=(*mat.shape[:1], 1), dtype=mat.dtype, device=mat.device)
            else:
                raise ValueError(
                    f"`mat` should be of size (batch_size, d, d) or (batch_size, d), but found: {mat.shape}")
        else:
            eigenvec = v0
        func = lambda v: torch.matmul(mat, v)

    for _ in range(num_iterations):
        mvp = func(eigenvec)
        eigenvec = mvp / mvp.norm(dim=-2, keepdim=True)

    eigenval = ((func(eigenvec) * eigenvec).sum(dim=-2) / ((eigenvec ** 2).sum(dim=-2))).squeeze(dim=-1)
    if eigenvectors:
        return eigenval, eigenvec
    return eigenval, None


def _topr_singular(mat, num_iterations):
    matTmat = mat.T.matmul(mat)
    eigenval, rsv = power_iter(matTmat, eigenvectors=True, num_iterations=num_iterations)
    return eigenval.sqrt(), rsv


def _topl_singular(mat, num_iterations):
    matmatT = mat.matmul(mat.T)
    eigenval, lsv = power_iter(matmatT, eigenvectors=True, num_iterations=num_iterations)
    return eigenval.sqrt(), lsv


def top_singular(mat, left_singularvectors=False, right_singularvectors=False, num_iterations=100):
    """Computes the approximate top singular value and vectors of a given matrix.

    Relies on power iteration.
    Currently only works with (n x m)-sized matrices without batching.

    Returns:
        A tuple of top singular value, top left singular vector, and top right singular vector.
    """
    if left_singularvectors:
        singular_value, lsv = _topl_singular(mat, num_iterations=num_iterations)
    else:
        lsv = None

    singular_value = None
    if right_singularvectors:
        singular_value, rsv = _topr_singular(mat, num_iterations=num_iterations)
    else:
        rsv = None

    if singular_value is None:
        dim1, dim2 = mat.shape
        if dim1 < dim2:
            singular_value, _ = _topl_singular(mat, num_iterations=num_iterations)
        else:
            singular_value, _ = _topr_singular(mat, num_iterations=num_iterations)
    return singular_value, lsv, rsv
