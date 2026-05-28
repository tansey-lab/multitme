"""CLI: model training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from multitme.config import load_config
from multitme.data import pseudo_label_from_markers
from multitme.model import (
    CycleVAETrainer,
    CyclingDataset,
    MultiModalCycleVAE,
    SpatialCycleVAETrainer,
    SpatialMultiModalCycleVAE,
    SpatialTiledDataset,
    spatial_tile_collate,
)
from multitme.utils import configure_logging, get_device, set_seed

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train MultiModal CycleVAE")
    parser.add_argument(
        "--scrna",
        type=str,
        required=True,
        help="Path to scRNA h5ad (with adata.layers['preprocessed'])",
    )
    parser.add_argument(
        "--xenium",
        type=str,
        required=True,
        help="Path to Xenium h5ad (with adata.layers['preprocessed'])",
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Path to checkpoint.pt to resume from"
    )
    parser.add_argument(
        "--scrna-gene-col",
        type=str,
        default=None,
        help="Column in scrna.var holding gene names for overlap (defaults to var index)",
    )
    parser.add_argument(
        "--xenium-gene-col",
        type=str,
        default=None,
        help="Column in xenium.var holding gene names for overlap (defaults to var index)",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.training.seed)
    device = get_device()
    outdir = Path(cfg.output.dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save config to output dir
    OmegaConf.save(cfg, outdir / "config.yaml")

    # Load preprocessed data (output of cli/preprocess.py)
    scrna = sc.read_h5ad(args.scrna)
    xenium = sc.read_h5ad(args.xenium)
    if "preprocessed" not in scrna.layers:
        raise KeyError(f"scRNA h5ad {args.scrna!r} missing layer 'preprocessed'")
    if "preprocessed" not in xenium.layers:
        raise KeyError(f"Xenium h5ad {args.xenium!r} missing layer 'preprocessed'")
    scrna_data = scrna.layers["preprocessed"]
    xenium_data = xenium.layers["preprocessed"]

    # Gene-name vectors (configurable column, default to var index)
    def _gene_names(adata, col, name):
        if col is None:
            return adata.var_names.astype(str).to_numpy()
        if col not in adata.var.columns:
            raise ValueError(
                f"{name} gene column {col!r} not found in .var. "
                f"Available columns: {list(adata.var.columns)}"
            )
        return adata.var[col].astype(str).to_numpy()

    scrna_genes = _gene_names(scrna, args.scrna_gene_col, "scRNA")
    xenium_genes = _gene_names(xenium, args.xenium_gene_col, "Xenium")

    # Common genes
    common_genes = np.intersect1d(scrna_genes, xenium_genes)
    if len(common_genes) == 0:
        raise ValueError(
            "Zero gene overlap between scRNA and Xenium datasets. "
            f"scRNA has {scrna.n_vars} genes (using "
            f"{args.scrna_gene_col or 'var index'}), Xenium has {xenium.n_vars} genes "
            f"(using {args.xenium_gene_col or 'var index'}), but none match. "
            "Check that gene identifiers use the same convention "
            "(e.g., both gene symbols or both Ensembl IDs), or specify "
            "--scrna-gene-col / --xenium-gene-col."
        )
    # Map common gene names back to positional indices in each modality
    scrna_pos = {g: i for i, g in enumerate(scrna_genes)}
    xenium_pos = {g: i for i, g in enumerate(xenium_genes)}
    indices_scrna = np.array([scrna_pos[g] for g in common_genes])
    indices_xenium = np.array([xenium_pos[g] for g in common_genes])

    # Cell types
    unique_types = sorted(set(scrna.obs[cfg.data.annotation_column]))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    logger.info(f"Cell types ({len(unique_types)}): {unique_types}")

    # Labels
    scrna_labels = torch.tensor(
        [type_to_idx[t] for t in scrna.obs[cfg.data.annotation_column]],
        dtype=torch.long,
    )

    # Pseudo labels from marker config (expects marker_dict in config)
    if "marker_dict" in cfg:
        marker_dict = dict(cfg.marker_dict)
        xenium_labels = pseudo_label_from_markers(
            data=xenium_data,
            gene_names=list(xenium_genes),
            marker_dict=marker_dict,
            type_to_idx=type_to_idx,
            top_k=cfg.pseudo_labels.top_k,
            normalize=cfg.pseudo_labels.normalize,
            min_score=cfg.pseudo_labels.min_score,
        )
    else:
        xenium_labels = torch.full((xenium_data.shape[0],), -1, dtype=torch.long)

    logger.info(f"scRNA:  {scrna_data.shape}  (labeled: {(scrna_labels >= 0).sum().item()})")
    logger.info(f"Xenium: {xenium_data.shape}  (labeled: {(xenium_labels >= 0).sum().item()})")

    spatial_cfg = cfg.get("spatial", None)
    use_spatial = bool(spatial_cfg) and bool(spatial_cfg.get("enabled", False))

    model_kwargs = dict(
        modality_dims={"scrna": scrna_data.shape[1], "xenium": xenium_data.shape[1]},
        n_latent=cfg.model.n_latent,
        hidden_dims=list(cfg.model.hidden_dims),
        common_masks={"scrna": indices_scrna, "xenium": indices_xenium},
        cycle_pairs=[("scrna", "xenium"), ("xenium", "scrna")],
        n_cell_types=len(unique_types),
        aux_loss_multiplier=cfg.model.aux_loss_multiplier,
        type_alignment_weight=cfg.model.type_alignment_weight,
        alignment_method=cfg.model.alignment_method,
        cycle_cls_weight=cfg.model.cycle_cls_weight,
        labeled_modality=cfg.model.labeled_modality,
        common_feature_weight=cfg.model.get("common_feature_weight", 1.0),
    )

    if use_spatial:
        obsm_key = spatial_cfg.get("obsm_key", "spatial")
        if obsm_key not in xenium.obsm:
            available = list(xenium.obsm.keys())
            raise KeyError(
                f"Xenium adata.obsm missing key {obsm_key!r}; "
                f"available: {available}. Set spatial.obsm_key to a valid entry."
            )
        coords = np.asarray(xenium.obsm[obsm_key])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"xenium.obsm[{obsm_key!r}] must have shape (n_cells, 2+); got {coords.shape}"
            )
        coords = coords[:, :2].astype(np.float32)
        logger.info(
            f"Spatial mode enabled (xenium.obsm[{obsm_key!r}]: {coords.shape}, "
            f"tile_size={spatial_cfg.get('tile_size', 500.0)}, "
            f"halo={spatial_cfg.get('halo', 150.0)})"
        )

        model = SpatialMultiModalCycleVAE(
            **model_kwargs,
            spatial_k=spatial_cfg.get("k", 10),
            spatial_tau=spatial_cfg.get("tau", 100.0),
            spatial_weight=spatial_cfg.get("weight", 1.0),
        ).to(device)

        dataset = SpatialTiledDataset(
            modality_dict={"scrna": scrna_data, "xenium": xenium_data},
            coord_dict={"xenium": coords},
            label_dict={"scrna": scrna_labels, "xenium": xenium_labels},
            tile_size=spatial_cfg.get("tile_size", 500.0),
            halo=spatial_cfg.get("halo", 150.0),
            min_core_cells=spatial_cfg.get("min_core_cells", 1),
            nonspatial_batch_size=spatial_cfg.get("nonspatial_batch_size", cfg.training.batch_size),
        )
        loader = DataLoader(dataset, batch_size=1, collate_fn=spatial_tile_collate)
        TrainerCls = SpatialCycleVAETrainer
        logger.info(f"SpatialTiledDataset: {len(dataset)} tiles")
    else:
        model = MultiModalCycleVAE(**model_kwargs).to(device)
        dataset = CyclingDataset(
            modality_dict={"scrna": scrna_data, "xenium": xenium_data},
            label_dict={"scrna": scrna_labels, "xenium": xenium_labels},
            target_batch_size=cfg.training.batch_size,
        )
        loader = DataLoader(dataset, batch_size=None, shuffle=False)
        TrainerCls = CycleVAETrainer

    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # wandb config
    wandb_config = None
    if cfg.wandb.enabled:
        wandb_config = {
            "project": cfg.wandb.project,
            "entity": cfg.wandb.entity,
            "name": cfg.wandb.name,
            "tags": list(cfg.wandb.tags) if cfg.wandb.tags else [],
            "full_config": OmegaConf.to_container(cfg, resolve=True),
        }

    metadata = {
        "type_to_idx": type_to_idx,
        "unique_types": unique_types,
        "config": cfg,
    }

    trainer = TrainerCls(
        model,
        learning_rate=cfg.training.learning_rate,
        cycle_weight=cfg.training.cycle_weight,
        beta=cfg.training.beta,
        beta_warmup_epochs=cfg.training.beta_warmup_epochs,
        output_dir=str(outdir),
        save_freq=cfg.training.get("save_freq", 1),
        metadata=metadata,
        wandb_enabled=cfg.wandb.enabled,
        wandb_config=wandb_config,
    )

    # Resume from checkpoint
    start_epoch = 0
    resume_path = args.resume_from
    if resume_path is None:
        # Auto-detect checkpoint in output dir
        auto_ckpt = outdir / "checkpoint.pt"
        if auto_ckpt.exists():
            resume_path = str(auto_ckpt)
            logger.info(f"Auto-detected checkpoint: {resume_path}")

    if resume_path:
        start_epoch = trainer.load_checkpoint(resume_path, device=device)

    trainer.fit(
        loader,
        n_epochs=cfg.training.n_epochs,
        max_epochs=cfg.training.max_epochs,
        slope_window=cfg.training.slope_window,
        slope_threshold=cfg.training.slope_threshold,
        print_every=cfg.training.print_every,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
