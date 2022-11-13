import torch

from ml_swissknife import numerical

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_sympd(d=100):
    A = torch.randn(d, d, device=device)
    A = A @ A.T
    lamb, Q = torch.linalg.eigh(A)
    A = Q @ torch.diag(lamb.abs() + 1e-5) @ Q.T  # PD.
    return A


def test_lanczos(d=100):
    A = _make_sympd(d=d)

    def matmul_closure(v):
        return A @ v

    q_mat, t_mat = numerical.lanczos_tridiag(
        matmul_closure=matmul_closure,
        dtype=torch.get_default_dtype(),
        device=device,
        matrix_shape=A.size(),
        max_iter=100,
        tol=1e-9,
        num_init_vecs=1
    )
    leval, levec = numerical.lanczos_tridiag_to_diag(t_mat=t_mat)
    eval, evec = torch.linalg.eigh(A)
    torch.testing.assert_allclose(leval, eval)


def test_orthogonal_iteration_mem_saving(d=100):
    A = _make_sympd(d=d)

    def matmul_closure(v):
        return (A @ v.to(A.device)).cpu()

    eval, evec = torch.linalg.eigh(A)
    Q = numerical.orthogonal_iteration_mem_saving(
        matmul_closure=matmul_closure,
        k=d,
        dtype=torch.get_default_dtype(),
        device=device,
        matrix_shape=A.size(),
        num_power_iteration=4000,
        enable_tqdm=True
    )
    Q = Q.to(device)
    Q = Q.flip(dims=(1,))
    oeval = numerical.eigenvectors_to_eigenvalues(mat=A, eigenvectors=Q)
    torch.testing.assert_allclose(oeval, eval)
