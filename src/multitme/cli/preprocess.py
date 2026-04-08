"""CLI: data loading and CLR/log1p preprocessing."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--scrna", type=str, required=True, help="Path to scRNA h5ad file")
    parser.add_argument("--xenium", type=str, required=True, help="Path to Xenium data")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)

    outdir = Path(cfg.output.dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading scRNA data from {args.scrna}")
    adata = sc.read_h5ad(args.scrna)
    adata = adata[adata.X.sum(axis=1) > 0]
    ann_col = cfg.data.annotation_column
    labels_pre = adata.obs[ann_col].astype(str)
    counts_before = labels_pre.value_counts().to_dict()
    total_before = int(adata.n_obs)

    adata = downsample_scrna(
        adata,
        cell_type_col=ann_col,
        max_cells=cfg.data.scrna_max_cells,
    )
    labels_post = adata.obs[ann_col].astype(str)
    counts_after = labels_post.value_counts().to_dict()
    total_after = int(adata.n_obs)
    downsampled = total_after < total_before

    all_types = sorted(set(counts_before) | set(counts_after))
    per_cell_type = [
        {
            "cell_type": ct,
            "n_before": int(counts_before.get(ct, 0)),
            "n_after": int(counts_after.get(ct, 0)),
        }
        for ct in all_types
    ]
    celltype_payload = {
        "annotation_column": ann_col,
        "scrna_max_cells": int(cfg.data.scrna_max_cells),
        "downsampled": downsampled,
        "total_before": total_before,
        "total_after": total_after,
        "per_cell_type": per_cell_type,
    }
    (outdir / "scrna_celltype_counts.json").write_text(
        json.dumps(celltype_payload, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"scRNA after downsampling: {adata.n_obs} cells (downsampled={downsampled})")
    data = preprocess(
        adata.X,
        method=cfg.data.preprocess_method,
        pseudocount=cfg.data.pseudocount,
        clip_percentile=cfg.data.clip_percentile,
    )
    np.save(outdir / "scrna_preprocessed.npy", data)
    adata.write_h5ad(outdir / "scrna_filtered.h5ad")
    logger.info(f"Saved preprocessed scRNA: {data.shape}")

    logger.info(f"Loading Xenium data from {args.xenium}")
    adata = load_xenium_adata(args.xenium)
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
