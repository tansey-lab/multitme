import numpy as np
from scipy import sparse


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
    is_sparse = sparse.issparse(data)

    if method == "log1p":
        if is_sparse:
            data = data.tocsr().astype(np.float32)
            lib_size = np.asarray(data.sum(axis=1)).ravel()  # (n_cells,)
            np.clip(lib_size, 1e-8, None, out=lib_size)
            # Scale each row in-place on stored values only; log1p(0)=0 so zeros stay zero
            scale = target_sum / lib_size
            data = data.multiply(scale[:, np.newaxis]).tocsr()
            np.log1p(data.data, out=data.data)
            return data.toarray()
        else:
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            lib_size = data.sum(axis=1, keepdims=True)
            np.clip(lib_size, 1e-8, None, out=lib_size)
            data /= lib_size
            data *= target_sum
            np.log1p(data, out=data)
            return data

    elif method == "clr":
        n_cells, n_genes = data.shape

        if is_sparse:
            data_csc = data.tocsc().astype(np.float32)
            caps = _sparse_column_percentile(data_csc, clip_percentile)  # (n_genes,)
            col_max = np.asarray(data_csc.max(axis=0)).ravel().clip(1e-8)  # (n_genes,)

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
