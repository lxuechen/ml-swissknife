import torch

from swissknife import numerical

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_lanczos(d=100):
    A = torch.randn(d, d, device=device)
    A = A @ A.T
    lamb, Q = torch.linalg.eigh(A)
    A = Q @ torch.diag(lamb.abs() + 1e-7) @ Q.T  # PD.

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
    # TODO: Why aren't the eigenvectors not lining up.


test_lanczos()
