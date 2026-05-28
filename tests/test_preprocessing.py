import anndata
import numpy as np
import pytest
from scipy import sparse

from multitme.data.preprocessing import get_raw_counts, preprocess


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


def test_get_raw_counts_uses_integer_X():
    counts = _make_counts().astype(np.int32)
    adata = anndata.AnnData(X=counts)
    X = get_raw_counts(adata)
    assert np.array_equal(X, counts)


def test_get_raw_counts_rejects_log1p_uns():
    counts = _make_counts().astype(np.int32)
    adata = anndata.AnnData(X=counts.astype(np.float32))
    adata.uns["log1p"] = {"base": None}
    with pytest.raises(ValueError, match="log1p"):
        get_raw_counts(adata)


def test_get_raw_counts_rejects_continuous_X():
    counts = _make_counts().astype(np.float32)
    counts += 0.5  # make non-integer
    adata = anndata.AnnData(X=counts)
    with pytest.raises(ValueError, match="not integer-valued"):
        get_raw_counts(adata)


def test_get_raw_counts_prefers_raw():
    raw_counts = _make_counts().astype(np.int32)
    raw_adata = anndata.AnnData(X=raw_counts)
    transformed = np.log1p(raw_counts.astype(np.float32))
    adata = anndata.AnnData(X=transformed)
    adata.raw = raw_adata
    adata.uns["log1p"] = {"base": None}
    X = get_raw_counts(adata)
    assert np.array_equal(np.asarray(X), raw_counts)


def test_log1p_sparse_input_stays_sparse():
    dense = _make_counts()
    dense[dense < 8] = 0  # induce sparsity
    csr = sparse.csr_matrix(dense)
    result = preprocess(csr, method="log1p")
    assert sparse.issparse(result)
    # Same nonzero structure as input (log1p preserves zeros)
    assert result.nnz == csr.nnz
