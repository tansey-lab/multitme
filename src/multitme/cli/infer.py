"""CLI: cell type prediction on new data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from multitme.data import preprocess
from multitme.model import MultiModalCycleVAE
from multitme.utils import configure_logging, get_device

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Predict cell types with trained model")
    parser.add_argument(
        "--model-dir", type=str, required=True, help="Directory with model.pt and metadata.pt"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input h5ad file")
    parser.add_argument("--modality", type=str, default="xenium", help="Modality name")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--preprocess-method", type=str, default="clr", help="Preprocessing method")
    args = parser.parse_args(argv)

    device = get_device()
    model_dir = Path(args.model_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = torch.load(model_dir / "metadata.pt", map_location="cpu", weights_only=False)
    unique_types = metadata["unique_types"]
    cfg = metadata["config"]

    # Load data
    adata = sc.read_h5ad(args.input)
    adata = adata[adata.X.sum(axis=1) > 0]
    data = preprocess(np.array(adata.X.todense()), method=args.preprocess_method)
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Reconstruct model
    # We need modality dims from the saved config; for inference on a single
    # modality we only need that modality's encoder/decoder to exist.
    modality_dims = {args.modality: data_tensor.shape[1]}
    # Add a dummy second modality so the model architecture matches the checkpoint
    dummy_mod = [m for m in ["scrna", "xenium"] if m != args.modality][0]
    state_dict = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    # Infer the other modality dim from the state dict
    dummy_dim = state_dict[f"decoders.{dummy_mod}.6.weight"].shape[0]
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
