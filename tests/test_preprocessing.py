import numpy as np
import pytest
from scipy import sparse

from multitme.data.preprocessing import preprocess


def _make_counts(n_cells=50, n_genes=20):
    rng = np.random.default_rng(42)
    return rng.poisson(10, size=(n_cells, n_genes)).astype(np.float64)


def test_log1p_shape():
    data = _make_counts()
    result = preprocess(data, method="log1p")
    assert result.shape == data.shape


def test_log1p_nonnegative():
    data = _make_counts()
    result = preprocess(data, method="log1p")
    assert np.all(result >= 0)


def test_clr_shape():
    data = _make_counts()
    result = preprocess(data, method="clr")
    assert result.shape == data.shape


def test_clr_centered():
    data = _make_counts()
    result = preprocess(data, method="clr")
    # CLR should be approximately centered per cell (mean ~0)
    row_means = result.mean(axis=1)
    assert np.allclose(row_means, 0, atol=0.5)


def test_unknown_method_raises():
    data = _make_counts()
    with pytest.raises(ValueError, match="Unknown method"):
        preprocess(data, method="invalid")


def test_log1p_sparse_input_stays_sparse():
    dense = _make_counts()
    dense[dense < 8] = 0  # induce sparsity
    csr = sparse.csr_matrix(dense)
    result = preprocess(csr, method="log1p")
    assert sparse.issparse(result)
    # Same nonzero structure as input (log1p preserves zeros)
    assert result.nnz == csr.nnz
