import numpy as np


def preprocess(data, method="log1p", target_sum=1e4, pseudocount=1e-3, clip_percentile=99.5):
    """Normalize expression data using log1p or CLR transform.

    Parameters
    ----------
    data : np.ndarray
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

    Returns
    -------
    np.ndarray
        Transformed expression matrix.
    """
    data = data.astype(np.float64)

    if method == "log1p":
        lib_size = data.sum(axis=1, keepdims=True)
        lib_size = np.clip(lib_size, 1e-8, None)
        normalized = data / lib_size * target_sum
        return np.log1p(normalized)

    elif method == "clr":
        caps = np.percentile(data, clip_percentile, axis=0).clip(1)
        data = np.clip(data, 0, caps)
        data = data / data.max(axis=0, keepdims=True).clip(1e-8)
        log_data = np.log(data + pseudocount)
        return log_data - log_data.mean(axis=1, keepdims=True)

    else:
        raise ValueError(f"Unknown method '{method}', use 'log1p' or 'clr'")
