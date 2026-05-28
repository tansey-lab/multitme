"""CLI: cell type prediction on new data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from multitme.data import get_raw_counts, load_xenium_adata, preprocess
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
    parser.add_argument(
        "--scrna",
        type=str,
        default=None,
        help="Optional scRNA h5ad to also project into latent space (saved alongside xenium)",
    )
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
    data = preprocess(get_raw_counts(adata), method=args.preprocess_method)
    if hasattr(data, "toarray"):
        data = data.toarray()
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
        common_feature_weight=cfg.model.get("common_feature_weight", 1.0),
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        z = model.get_latent(data_tensor.to(device), args.modality).cpu().numpy()
        probs = model.predict_cell_types(data_tensor.to(device), args.modality).cpu().numpy()

    pred_idx = probs.argmax(axis=1)
    pred_types = np.array([unique_types[i] for i in pred_idx])

    # Latent projection — include scRNA cells too if provided so the saved
    # latent.npz covers every cell the model has seen, indexed by cell id + modality.
    latent_blocks = [z]
    cell_id_blocks = [adata.obs_names.to_numpy().astype(str)]
    modality_blocks = [np.full(z.shape[0], args.modality, dtype=object)]

    if args.scrna is not None:
        scrna_adata = sc.read_h5ad(args.scrna)
        scrna_adata = scrna_adata[scrna_adata.X.sum(axis=1) > 0]
        scrna_data = preprocess(get_raw_counts(scrna_adata), method=args.preprocess_method)
        if hasattr(scrna_data, "toarray"):
            scrna_data = scrna_data.toarray()
        scrna_tensor = torch.tensor(scrna_data, dtype=torch.float32)
        with torch.no_grad():
            z_scrna = model.get_latent(scrna_tensor.to(device), "scrna").cpu().numpy()
        latent_blocks.append(z_scrna)
        cell_id_blocks.append(scrna_adata.obs_names.to_numpy().astype(str))
        modality_blocks.append(np.full(z_scrna.shape[0], "scrna", dtype=object))

    latent_all = np.concatenate(latent_blocks, axis=0)
    cell_ids = np.concatenate(cell_id_blocks, axis=0).astype(str)
    modalities = np.concatenate(modality_blocks, axis=0).astype(str)

    np.savez(
        outdir / "latent.npz",
        latent=latent_all,
        cell_id=cell_ids,
        modality=modalities,
    )
    np.save(outdir / "pred_idx.npy", pred_idx)
    np.save(outdir / "pred_probs.npy", probs)
    adata.obs["predicted_type"] = pred_types
    adata.uns["cell_type_names"] = list(unique_types)
    adata.write_h5ad(outdir / "predictions.h5ad")

    logger.info(f"Predictions saved to {outdir}")
    for ct in unique_types:
        count = (pred_types == ct).sum()
        logger.info(f"  {ct}: {count}")


if __name__ == "__main__":
    main()
