"""CLI: data loading and CLR/log1p preprocessing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import scanpy as sc

from multitme.config import load_config
from multitme.data import load_xenium_adata, preprocess
from multitme.data.preprocessing import downsample_scrna
from multitme.utils import configure_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Preprocess scRNA/Xenium data")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)

    outdir = Path(cfg.output.dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.data.scrna_path:
        logger.info(f"Loading scRNA data from {cfg.data.scrna_path}")
        adata = sc.read_h5ad(cfg.data.scrna_path)
        adata = adata[adata.X.sum(axis=1) > 0]
        adata = downsample_scrna(
            adata,
            cell_type_col=cfg.data.annotation_column,
            max_cells=cfg.data.scrna_max_cells,
        )
        logger.info(f"scRNA after downsampling: {adata.n_obs} cells")
        data = preprocess(
            adata.X,
            method=cfg.data.preprocess_method,
            pseudocount=cfg.data.pseudocount,
            clip_percentile=cfg.data.clip_percentile,
        )
        np.save(outdir / "scrna_preprocessed.npy", data)
        adata.write_h5ad(outdir / "scrna_filtered.h5ad")
        logger.info(f"Saved preprocessed scRNA: {data.shape}")

    if cfg.data.xenium_path:
        logger.info(f"Loading Xenium data from {cfg.data.xenium_path}")
        adata = load_xenium_adata(cfg.data.xenium_path)
        adata = adata[adata.X.sum(axis=1) > 0]
        data = preprocess(
            adata.X,
            method=cfg.data.preprocess_method,
            pseudocount=cfg.data.pseudocount,
            clip_percentile=cfg.data.clip_percentile,
        )
        np.save(outdir / "xenium_preprocessed.npy", data)
        adata.write_h5ad(outdir / "xenium_filtered.h5ad")
        logger.info(f"Saved preprocessed Xenium: {data.shape}")

    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()
