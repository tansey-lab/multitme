import logging

import anndata
import numpy as np
import scanpy as sc
from scipy import sparse

logger = logging.getLogger(__name__)


def downsample_scrna(adata, cell_type_col, max_cells=100_000, seed=0):
    """Downsample scRNA AnnData to at most ``max_cells`` with balanced cell-type representation.

    Each cell type receives an equal quota of ``max_cells // n_cell_types`` cells.
    If a cell type has fewer cells than the quota, all of its cells are kept.

    Parameters
    ----------
    adata : AnnData
        scRNA data with cell-type annotations in ``adata.obs[cell_type_col]``.
    cell_type_col : str
        Column in ``adata.obs`` containing cell-type labels.
    max_cells : int
        Target total number of cells after downsampling.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        Downsampled (or unchanged) AnnData object.
    """
    if adata.n_obs <= max_cells:
        return adata

    rng = np.random.default_rng(seed)
    valid_mask = adata.obs[cell_type_col].notna()
    n_nan = (~valid_mask).sum()
    if n_nan > 0:
        logger.warning(
            "Removing %d cells with NaN values in cell type column '%s'",
            n_nan,
            cell_type_col,
        )
        adata = adata[valid_mask]
    cell_types = adata.obs[cell_type_col].astype(str)
    unique_types = np.unique(cell_types)
    per_type_quota = max_cells // len(unique_types)

    selected = []
    for ct in unique_types:
        idx = np.where(cell_types == ct)[0]
        n = min(per_type_quota, len(idx))
        selected.append(rng.choice(idx, size=n, replace=False))

    keep = np.sort(np.concatenate(selected))
    return adata[keep]


def preprocess(
    data, method="log1p", target_sum=1e4, pseudocount=1e-3, clip_percentile=99.5, chunk_size=2000
):
    """Normalize expression data using log1p or CLR transform.

    Parameters
    ----------
    data : np.ndarray or scipy.sparse matrix
        Raw count matrix (cells x genes).
    method : str
        ``"log1p"`` for library-size normalized log1p, or ``"clr"`` for
        centered log-ratio after outlier clipping and max-scaling.
    target_sum : float
        Target library size for log1p normalization.
    pseudocount : float
        Pseudocount added before log in CLR transform.
    clip_percentile : float
        Per-gene percentile cap for CLR outlier clipping.
    chunk_size : int
        Number of rows to densify at a time for CLR (ignored for log1p).

    Returns
    -------
    np.ndarray
        Transformed expression matrix (float32).
    """
    if method == "log1p":
        adata = anndata.AnnData(X=data)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        return adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()

    elif method == "clr":
        is_sparse = sparse.issparse(data)
        n_cells, n_genes = data.shape

        if is_sparse:
            data_csc = data.tocsc().astype(np.float32)
            caps = _sparse_column_percentile(data_csc, clip_percentile)  # (n_genes,)
            col_max = (
                np.asarray(data_csc.max(axis=0).todense()).ravel().clip(min=1e-8)
            )  # (n_genes,)

            data_csr = data_csc.tocsr()
            output = np.empty((n_cells, n_genes), dtype=np.float32)
            for i in range(0, n_cells, chunk_size):
                _clr_chunk(
                    data_csr[i : i + chunk_size].toarray(),
                    caps,
                    col_max,
                    pseudocount,
                    out=output[i : i + chunk_size],
                )
            return output
        else:
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            caps = np.percentile(data, clip_percentile, axis=0).clip(1)
            col_max = data.max(axis=0, keepdims=True).clip(1e-8)
            np.clip(data, 0, caps, out=data)
            data /= col_max
            data += pseudocount
            np.log(data, out=data)
            data -= data.mean(axis=1, keepdims=True)
            return data

    else:
        raise ValueError(f"Unknown method '{method}', use 'log1p' or 'clr'")


def _clr_chunk(chunk, caps, col_max, pseudocount, out):
    """Apply CLR transform in-place to a dense chunk and write to out."""
    if chunk.dtype != np.float32:
        chunk = chunk.astype(np.float32)
    np.clip(chunk, 0, caps, out=chunk)
    chunk /= col_max
    chunk += pseudocount
    np.log(chunk, out=chunk)
    chunk -= chunk.mean(axis=1, keepdims=True)
    out[:] = chunk


def _sparse_column_percentile(data_csc, clip_percentile):
    """Compute per-column percentile from a CSC matrix without densifying.

    Correctly accounts for implicit zeros in the sparse representation.
    """
    n_cells, n_genes = data_csc.shape
    quantile = clip_percentile / 100.0
    target_pos = quantile * (n_cells - 1)
    caps = np.zeros(n_genes, dtype=np.float32)

    for j in range(n_genes):
        start, end = data_csc.indptr[j], data_csc.indptr[j + 1]
        col_data = data_csc.data[start:end]
        if len(col_data) == 0:
            continue  # all zeros; caps[j] stays 0, clipped to 1 below

        n_zeros = n_cells - len(col_data)
        if target_pos < n_zeros:
            caps[j] = 0.0  # quantile falls within the implicit zeros
        else:
            sorted_col = np.sort(col_data)
            pos_in_nnz = target_pos - n_zeros
            lo = int(pos_in_nnz)
            hi = min(lo + 1, len(sorted_col) - 1)
            frac = pos_in_nnz - lo
            caps[j] = sorted_col[lo] * (1 - frac) + sorted_col[hi] * frac

    return caps.clip(1)
