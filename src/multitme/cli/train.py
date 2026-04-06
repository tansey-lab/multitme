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
from multitme.model import CycleVAETrainer, CyclingDataset, MultiModalCycleVAE
from multitme.utils import configure_logging, get_device, set_seed

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train MultiModal CycleVAE")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Path to checkpoint.pt to resume from"
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
    scrna = sc.read_h5ad(outdir / "scrna_filtered.h5ad")
    xenium = sc.read_h5ad(outdir / "xenium_filtered.h5ad")
    scrna_data = np.load(outdir / "scrna_preprocessed.npy")
    xenium_data = np.load(outdir / "xenium_preprocessed.npy")

    # Common genes
    common_genes = np.intersect1d(scrna.var_names, xenium.var_names)
    indices_scrna = scrna.var.index.get_indexer(common_genes)
    indices_xenium = xenium.var.index.get_indexer(common_genes)

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
            gene_names=list(xenium.var_names),
            marker_dict=marker_dict,
            type_to_idx=type_to_idx,
            top_k=cfg.pseudo_labels.top_k,
            normalize=cfg.pseudo_labels.normalize,
            min_score=cfg.pseudo_labels.min_score,
        )
    else:
        xenium_labels = torch.full((xenium_data.shape[0],), -1, dtype=torch.long)

    scrna_tensor = torch.tensor(scrna_data, dtype=torch.float32)
    xenium_tensor = torch.tensor(xenium_data, dtype=torch.float32)

    logger.info(f"scRNA:  {scrna_tensor.shape}  (labeled: {(scrna_labels >= 0).sum().item()})")
    logger.info(f"Xenium: {xenium_tensor.shape}  (labeled: {(xenium_labels >= 0).sum().item()})")

    # Model
    model = MultiModalCycleVAE(
        modality_dims={"scrna": scrna_tensor.shape[1], "xenium": xenium_tensor.shape[1]},
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
    ).to(device)

    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset & training
    dataset = CyclingDataset(
        modality_dict={"scrna": scrna_tensor, "xenium": xenium_tensor},
        label_dict={"scrna": scrna_labels, "xenium": xenium_labels},
        target_batch_size=cfg.training.batch_size,
    )
    loader = DataLoader(dataset, batch_size=None, shuffle=False)

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

    trainer = CycleVAETrainer(
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
        print_every=cfg.training.print_every,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
