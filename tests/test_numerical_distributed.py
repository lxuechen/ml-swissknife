import torch
from torch.utils.data import DataLoader, TensorDataset

from ml_swissknife import numerical_distributed


def test_orthogonal_iteration(n=100, d=30, k=10, bsz=10):
    torch.set_default_dtype(torch.float64)

    features = torch.randn(n, d)
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=bsz)

    eigenvalues, eigenvectors = numerical_distributed.orthogonal_iteration(
        input_mat=loader,
        k=k,
        num_power_iteration=400,
        dtype=torch.get_default_dtype(),
    )
    eigenvalues_expected, eigenvectors_expected = torch.linalg.eigh(features.T @ features)
    print(eigenvalues)
    print(eigenvalues_expected.flip(dims=(0,)))
    print('---')
    print(eigenvectors)
    print(eigenvectors_expected.flip(dims=(1,)))
    torch.testing.assert_allclose(eigenvalues, eigenvalues_expected.flip(dims=(0,))[:k], atol=1e-4, rtol=1e-4)
