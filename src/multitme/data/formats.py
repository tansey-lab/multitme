"""
Input format detection and loading utilities for Xenium spatial transcriptomics data.

Supports three input formats:
1. H5AD file (AnnData on disk)
2. SpatialData zarr directory
3. Xenium Ranger output directory
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path

import anndata as ad

logger = logging.getLogger(__name__)


class InputFormat(Enum):
    H5AD = "h5ad"
    ZARR = "zarr"
    XENIUM_RANGER = "xenium"
    UNKNOWN = "unknown"


def detect_input_format(path: str | Path) -> InputFormat:
    """Detect the input format from a file or directory path."""
    path = Path(path)

    if path.is_file() and path.suffix == ".h5ad":
        return InputFormat.H5AD

    if path.is_dir():
        # Check for SpatialData zarr format
        spatialdata_markers = ["points/transcripts", "shapes/nucleus_boundaries"]
        if all(os.path.exists(path / p) for p in spatialdata_markers):
            return InputFormat.ZARR

        # Check for Xenium Ranger format
        xenium_markers = ["transcripts.parquet", "nucleus_boundaries.parquet"]
        if all(os.path.exists(path / f) for f in xenium_markers):
            return InputFormat.XENIUM_RANGER

    return InputFormat.UNKNOWN


def load_xenium_adata(path: str | Path) -> ad.AnnData:
    """
    Load a cell x gene AnnData from any supported Xenium input format.

    Args:
        path: Path to an h5ad file, SpatialData zarr directory,
              or Xenium Ranger output directory.

    Returns:
        AnnData with the cell x gene count matrix.
    """
    path = Path(path)
    fmt = detect_input_format(path)
    logger.info(f"Detected input format: {fmt.value} for {path}")

    if fmt == InputFormat.H5AD:
        return _load_h5ad(path)
    elif fmt == InputFormat.ZARR:
        return _load_zarr(path)
    elif fmt == InputFormat.XENIUM_RANGER:
        return _load_xenium_ranger(path)
    else:
        raise ValueError(
            f"Cannot detect input format for '{path}'. "
            "Expected an .h5ad file, SpatialData .zarr directory, "
            "or Xenium Ranger output directory."
        )


def _load_h5ad(path: Path) -> ad.AnnData:
    """Load AnnData from an h5ad file."""
    import scanpy as sc

    return sc.read_h5ad(path)


def _load_zarr(zarr_dir: Path) -> ad.AnnData:
    """Load AnnData from a SpatialData zarr directory."""
    import spatialdata

    sd = spatialdata.read_zarr(str(zarr_dir))
    table_name = next(iter(sd.tables))
    logger.info(f"Using table '{table_name}' from SpatialData zarr")
    return sd.tables[table_name]


def _load_xenium_ranger(xenium_dir: Path) -> ad.AnnData:
    """Load AnnData from a Xenium Ranger output directory via spatialdata_io."""
    import spatialdata_io

    sd = spatialdata_io.xenium(str(xenium_dir), cells_as_circles=True)
    table_name = next(iter(sd.tables))
    logger.info(f"Using table '{table_name}' from Xenium Ranger directory")
    return sd.tables[table_name]
