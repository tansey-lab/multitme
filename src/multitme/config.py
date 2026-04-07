"""Configuration loading via OmegaConf."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = {
    "data": {
        "preprocess_method": "clr",
        "pseudocount": 1e-3,
        "clip_percentile": 99.5,
        "annotation_column": "major_annotation",
        "scrna_max_cells": 100_000,
    },
    "model": {
        "n_latent": 20,
        "hidden_dims": [512, 256],
        "alignment_method": "swd",
        "aux_loss_multiplier": 1000.0,
        "type_alignment_weight": 100.0,
        "cycle_cls_weight": 1000.0,
        "labeled_modality": "scrna",
    },
    "training": {
        "n_epochs": 50,
        "learning_rate": 1e-3,
        "batch_size": 4096,
        "cycle_weight": 1.0,
        "beta": 1.0,
        "beta_warmup_epochs": 10,
        "print_every": 5,
        "save_freq": 1,
        "seed": 1,
    },
    "pseudo_labels": {
        "top_k": 50,
        "normalize": False,
        "min_score": None,
    },
    "wandb": {
        "enabled": False,
        "project": "multitme",
        "entity": None,
        "name": None,
        "tags": [],
    },
    "output": {
        "dir": "results",
        "save_latent": True,
        "save_predictions": True,
    },
}


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> DictConfig:
    """Load configuration, merging defaults with a YAML file and CLI overrides.

    Parameters
    ----------
    path : str or Path, optional
        Path to a YAML config file.
    overrides : list of str, optional
        Dot-notation overrides, e.g. ``["model.n_latent=32", "training.n_epochs=100"]``.
    """
    cfg = OmegaConf.structured(DEFAULT_CONFIG)

    if path is not None:
        user_cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(cfg, user_cfg)

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg
