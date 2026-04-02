"""CLI: cell type prediction on new data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from multitme.data import load_xenium_adata, preprocess
from multitme.model import MultiModalCycleVAE
from multitme.utils import configure_logging, get_device

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Predict cell types with trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input (h5ad file, SpatialData zarr dir, or Xenium Ranger dir)",
    )
    parser.add_argument("--modality", type=str, default="xenium", help="Modality name")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--preprocess-method", type=str, default="clr", help="Preprocessing method")
    args = parser.parse_args(argv)

    device = get_device()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = ckpt["metadata"]
    unique_types = metadata["unique_types"]
    cfg = metadata["config"]
    state_dict = ckpt["model"]

    # Load data
    adata = load_xenium_adata(args.input)
    adata = adata[adata.X.sum(axis=1) > 0]
    data = preprocess(np.array(adata.X.todense()), method=args.preprocess_method)
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Reconstruct model from checkpoint — infer dummy modality dim from last decoder bias
    modality_dims = {args.modality: data_tensor.shape[1]}
    dummy_mod = [m for m in ["scrna", "xenium"] if m != args.modality][0]
    # Find the last bias in the dummy decoder to get its output dim
    decoder_keys = sorted(
        k for k in state_dict if k.startswith(f"decoders.{dummy_mod}.") and k.endswith(".bias")
    )
    dummy_dim = state_dict[decoder_keys[-1]].shape[0]
    modality_dims[dummy_mod] = dummy_dim

    model = MultiModalCycleVAE(
        modality_dims=modality_dims,
        n_latent=cfg.model.n_latent,
        hidden_dims=list(cfg.model.hidden_dims),
        n_cell_types=len(unique_types),
        alignment_method=cfg.model.alignment_method,
        aux_loss_multiplier=cfg.model.aux_loss_multiplier,
        type_alignment_weight=cfg.model.type_alignment_weight,
        cycle_cls_weight=cfg.model.cycle_cls_weight,
        labeled_modality=cfg.model.labeled_modality,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        z = model.get_latent(data_tensor.to(device), args.modality).cpu().numpy()
        probs = model.predict_cell_types(data_tensor.to(device), args.modality).cpu().numpy()

    pred_idx = probs.argmax(axis=1)
    pred_types = np.array([unique_types[i] for i in pred_idx])

    # Save results
    np.save(outdir / "latent.npy", z)
    np.save(outdir / "pred_idx.npy", pred_idx)
    np.save(outdir / "pred_probs.npy", probs)
    adata.obs["predicted_type"] = pred_types
    adata.write_h5ad(outdir / "predictions.h5ad")

    logger.info(f"Predictions saved to {outdir}")
    for ct in unique_types:
        count = (pred_types == ct).sum()
        logger.info(f"  {ct}: {count}")


if __name__ == "__main__":
    main()
