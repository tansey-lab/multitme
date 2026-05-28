import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def check_loss_slope_convergence(losses: list, window: int, threshold: float) -> bool:
    """Return True if the recent per-epoch total loss has flattened.

    Fits a line over the last *window* values of *losses* and checks whether
    the slope, normalised by the mean loss in that window, falls below
    *threshold*.

    Parameters
    ----------
    losses : list of float
        Per-epoch loss history (e.g. ``trainer.history["total"]``).
    window : int
        Number of recent epochs to fit the line over.
    threshold : float
        Stop when ``|slope / mean_loss| < threshold``.
    """
    if len(losses) < window:
        return False
    recent = np.array(losses[-window:], dtype=float)
    x = np.arange(window, dtype=float)
    slope = np.polyfit(x, recent, 1)[0]
    mean_loss = recent.mean()
    if mean_loss == 0.0:
        return False
    return abs(slope / mean_loss) < threshold


def save_loss_history(history: dict, output_dir: str | os.PathLike) -> tuple[Path, Path]:
    """Write a per-epoch loss plot (PNG) and the raw history (JSON).

    Returns the (png_path, json_path). Matplotlib is imported lazily and uses
    the non-interactive ``Agg`` backend so this works on headless workers.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "loss_history.json"
    json_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    png_path = out / "loss_curve.png"
    total = history.get("total", [])
    if not total:
        return png_path, json_path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(total) + 1)
    components = [k for k in history if k != "total" and len(history[k]) == len(total)]

    fig, (ax_total, ax_comp) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_total.plot(epochs, total, color="black", linewidth=1.8)
    ax_total.set_title("Total loss")
    ax_total.set_xlabel("epoch")
    ax_total.set_ylabel("loss")
    ax_total.grid(alpha=0.3)

    for k in components:
        ax_comp.plot(epochs, history[k], label=k, linewidth=1.2)
    ax_comp.set_title("Loss components")
    ax_comp.set_xlabel("epoch")
    ax_comp.set_ylabel("loss")
    ax_comp.grid(alpha=0.3)
    if components:
        ax_comp.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path, json_path


class MultiModalCycleVAE(nn.Module):
    def __init__(
        self,
        modality_dims,
        n_latent=20,
        hidden_dims=None,
        common_masks=None,
        cycle_pairs=None,
        n_cell_types=None,
        aux_loss_multiplier=1.0,
        type_alignment_weight=10.0,
        alignment_method="swd",
        cycle_cls_weight=1000.0,
        labeled_modality="scrna",
        common_feature_weight=1.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.modality_names = list(modality_dims.keys())
        self.modality_dims = modality_dims
        self.n_modalities = len(self.modality_names)
        self.n_latent = n_latent
        self.common_masks = common_masks or {}
        self.n_cell_types = n_cell_types
        self.aux_loss_multiplier = aux_loss_multiplier
        self.type_alignment_weight = type_alignment_weight
        self.alignment_method = alignment_method
        self.cycle_cls_weight = cycle_cls_weight
        self.labeled_modality = labeled_modality
        self.common_feature_weight = common_feature_weight

        if cycle_pairs is None:
            self.cycle_pairs = [
                (m1, m2)
                for i, m1 in enumerate(self.modality_names)
                for m2 in self.modality_names[i + 1 :]
            ]
        else:
            self.cycle_pairs = cycle_pairs

        common_dim = hidden_dims[0]

        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, common_dim),
                    nn.LayerNorm(common_dim),
                    nn.ReLU(),
                )
                for name, dim in modality_dims.items()
            }
        )

        self.shared_encoder = nn.Sequential(
            nn.Linear(common_dim, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.encoder_mu = nn.Linear(hidden_dims[1], n_latent)
        self.encoder_logvar = nn.Linear(hidden_dims[1], n_latent)

        if self.n_cell_types is not None:
            self.classifier = nn.Sequential(
                nn.Linear(n_latent, hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], n_cell_types),
            )

        self.decoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(n_latent, hidden_dims[1]),
                    nn.LayerNorm(hidden_dims[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[1], common_dim),
                    nn.ReLU(),
                    nn.Linear(common_dim, dim),
                )
                for name, dim in modality_dims.items()
            }
        )

        self.log_sigmas = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in self.modality_names}
        )

    def encode(self, x, modality_name):
        h = self.projections[modality_name](x)
        h = self.shared_encoder(h)
        mu = self.encoder_mu(h)
        logvar = torch.clamp(self.encoder_logvar(h), -4.0, 4.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, modality_name):
        return self.decoders[modality_name](z)

    def classify(self, z):
        if self.n_cell_types is None:
            raise ValueError("No classifier")
        return self.classifier(z)

    @staticmethod
    def _kl_divergence(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    def _reconstruction_nll(self, recon, target, modality_name):
        sigma = torch.exp(self.log_sigmas[modality_name])
        nll = 0.5 * ((target - recon) / sigma).pow(2) + torch.log(sigma)
        return nll.sum(dim=-1)

    def _cycle_nll(self, src_cycle, src_data, modality_name, cycle_weight):
        sigma = torch.exp(self.log_sigmas[modality_name])
        effective_sigma = sigma * cycle_weight
        nll = 0.5 * ((src_data - src_cycle) / effective_sigma).pow(2) + torch.log(effective_sigma)
        return nll.sum(dim=-1)

    def _common_feature_nll(self, tgt_recon_common, src_data_common):
        nll = 0.5 * (src_data_common - tgt_recon_common).pow(2)
        return nll.sum(dim=-1)

    def _classification_loss(self, logits, labels):
        labeled_mask = labels >= 0
        if labeled_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        return self.aux_loss_multiplier * F.cross_entropy(
            logits[labeled_mask], labels[labeled_mask], label_smoothing=0.1
        )

    def _compute_swd(self, z1, z2, num_projections=50):
        n1, n2 = z1.size(0), z2.size(0)
        if n1 != n2:
            min_n = min(n1, n2)
            z1 = z1[torch.randperm(n1, device=z1.device)[:min_n]]
            z2 = z2[torch.randperm(n2, device=z2.device)[:min_n]]

        dim = z1.size(1)
        theta = torch.randn(dim, num_projections, device=z1.device)
        theta = theta / torch.sqrt((theta**2).sum(dim=0, keepdim=True))

        proj1, _ = torch.sort(z1 @ theta, dim=0)
        proj2, _ = torch.sort(z2 @ theta, dim=0)
        return torch.mean((proj1 - proj2).pow(2))

    def _type_alignment_loss(self, z_samples, cell_type_logits):
        predicted_types = {mod: logits.argmax(dim=1) for mod, logits in cell_type_logits.items()}

        total_loss = 0.0
        pairs_count = 0

        for i, mod1 in enumerate(self.modality_names):
            for mod2 in self.modality_names[i + 1 :]:
                if mod1 not in z_samples or mod2 not in z_samples:
                    continue

                z1, z2 = z_samples[mod1], z_samples[mod2]
                types1, types2 = predicted_types[mod1], predicted_types[mod2]

                for ct in range(self.n_cell_types):
                    mask1 = types1 == ct
                    mask2 = types2 == ct

                    if mask1.sum() < 5 or mask2.sum() < 5:
                        continue

                    if self.alignment_method == "swd":
                        loss = self._compute_swd(z1[mask1], z2[mask2])
                    elif self.alignment_method == "moment_matching":
                        loss = (z1[mask1].mean(0) - z2[mask2].mean(0)).pow(2).sum()
                    else:
                        loss = torch.tensor(0.0, device=z1.device)

                    total_loss = total_loss + loss
                    pairs_count += 1

        if pairs_count > 0:
            return (total_loss / pairs_count) * self.type_alignment_weight
        return torch.tensor(0.0, device=next(iter(z_samples.values())).device)

    def forward(self, data_dict, labels_dict=None, cycle_weight=1.0, beta=1.0):
        device = next(self.parameters()).device
        losses = {
            k: torch.tensor(0.0, device=device)
            for k in [
                "reconstruction",
                "kl",
                "classification",
                "cycle",
                "common_feature",
                "alignment",
                "cycle_cls",
                "total",
            ]
        }

        z_samples = {}
        mu_dict = {}
        cell_type_logits = {}

        for mod_name, data in data_dict.items():
            mu, logvar = self.encode(data, mod_name)
            z = self.reparameterize(mu, logvar)
            z_samples[mod_name] = z
            mu_dict[mod_name] = mu

            recon = self.decode(z, mod_name)
            losses["reconstruction"] += self._reconstruction_nll(recon, data, mod_name).mean()
            losses["kl"] += self._kl_divergence(mu, logvar).mean()

            if self.n_cell_types is not None:
                logits = self.classify(mu)
                cell_type_logits[mod_name] = logits

                if labels_dict is not None and mod_name in labels_dict:
                    losses["classification"] += self._classification_loss(
                        logits, labels_dict[mod_name]
                    )

        if self.n_cell_types is not None and len(z_samples) > 1:
            losses["alignment"] = self._type_alignment_loss(z_samples, cell_type_logits)

        for mod_src, mod_tgt in self.cycle_pairs:
            if mod_src not in data_dict or mod_tgt not in data_dict:
                continue

            z_src = z_samples[mod_src]
            tgt_recon = self.decode(z_src, mod_tgt)
            z_cycle_mu, _ = self.encode(tgt_recon, mod_tgt)
            src_cycle = self.decode(z_cycle_mu, mod_src)

            losses["cycle"] += self._cycle_nll(
                src_cycle, data_dict[mod_src], mod_src, cycle_weight
            ).mean()

            if self.n_cell_types is not None and mod_src == self.labeled_modality:
                logits_direct = self.classify(mu_dict[mod_src])
                logits_cycle = self.classify(z_cycle_mu)
                q = F.softmax(logits_direct.detach(), dim=-1)
                p = F.log_softmax(logits_cycle, dim=-1)
                losses["cycle_cls"] += self.cycle_cls_weight * F.kl_div(p, q, reduction="batchmean")

            if mod_src in self.common_masks and mod_tgt in self.common_masks:
                mask_src = self.common_masks[mod_src]
                mask_tgt = self.common_masks[mod_tgt]
                losses["common_feature"] += self._common_feature_nll(
                    tgt_recon[:, mask_tgt], data_dict[mod_src][:, mask_src]
                ).mean()

        losses["total"] = (
            losses["reconstruction"]
            + beta * losses["kl"]
            + losses["classification"]
            + losses["cycle"]
            + self.common_feature_weight * losses["common_feature"]
            + losses["alignment"]
            + losses["cycle_cls"]
        )

        return losses

    @torch.no_grad()
    def get_latent(self, data, modality_name):
        self.eval()
        mu, _ = self.encode(data, modality_name)
        return mu

    @torch.no_grad()
    def translate(self, data, source_modality, target_modality):
        self.eval()
        mu, _ = self.encode(data, source_modality)
        return self.decode(mu, target_modality)

    @torch.no_grad()
    def predict_cell_types(self, data, modality_name):
        self.eval()
        mu, _ = self.encode(data, modality_name)
        logits = self.classify(mu)
        return F.softmax(logits, dim=-1)


class CyclingDataset(Dataset):
    """Cycles minibatches across modalities, densifying sparse data per-batch.

    ``modality_dict`` values may be torch.Tensor, np.ndarray, or
    scipy.sparse matrix. Sparse inputs stay sparse in memory and only the
    selected rows are densified when a batch is requested.
    """

    def __init__(self, modality_dict, label_dict=None, target_batch_size=256):
        self.modality_dict = {name: _as_indexable(d) for name, d in modality_dict.items()}
        self.label_dict = label_dict or {}
        self.modality_names = list(self.modality_dict.keys())

        self.modality_sizes = {name: d.shape[0] for name, d in self.modality_dict.items()}
        self.largest_modality = max(self.modality_sizes, key=self.modality_sizes.get)
        self.n_cells_max = self.modality_sizes[self.largest_modality]
        self.n_batches = max(1, self.n_cells_max // target_batch_size)
        self.batch_sizes = {
            name: self.modality_sizes[name] // self.n_batches for name in self.modality_names
        }
        self.reshuffle()

    def reshuffle(self):
        self.indices = {
            name: np.random.permutation(self.modality_sizes[name]) for name in self.modality_names
        }

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        batch = {}
        labels = {}
        for name in self.modality_names:
            bs = self.batch_sizes[name]
            start = idx * bs
            end = start + bs
            inds = self.indices[name][start:end]
            batch[name] = _select_rows(self.modality_dict[name], inds)
            if name in self.label_dict:
                labels[name] = self.label_dict[name][inds]
        return batch, labels


def _as_indexable(data):
    """Coerce input to a row-indexable form (CSR if sparse, else as-is)."""
    if sparse.issparse(data):
        return data.tocsr()
    return data


def _select_rows(data, inds):
    """Return a float32 torch tensor of the rows ``inds`` from ``data``."""
    if sparse.issparse(data):
        return torch.from_numpy(data[inds].toarray().astype(np.float32, copy=False))
    if isinstance(data, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(data[inds], dtype=np.float32))
    # torch.Tensor
    return data[inds]


class CycleVAETrainer:
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        cycle_weight=1.0,
        beta=1.0,
        beta_warmup_epochs=0,
        output_dir=None,
        save_freq=1,
        metadata=None,
        wandb_enabled=False,
        wandb_config=None,
    ):
        self.model = model
        self.cycle_weight = cycle_weight
        self.beta = beta
        self.beta_warmup_epochs = beta_warmup_epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.output_dir = output_dir
        self.save_freq = save_freq
        self.metadata = metadata
        self.wandb_enabled = wandb_enabled
        self.run = None
        self.history = {
            "total": [],
            "reconstruction": [],
            "kl": [],
            "cycle": [],
            "classification": [],
            "alignment": [],
            "common_feature": [],
            "cycle_cls": [],
        }

        if self.wandb_enabled:
            try:
                import wandb

                wandb_mode = "online"
                if not os.environ.get("WANDB_API_KEY"):
                    wandb_mode = "offline"
                else:
                    wandb.login()

                wb_cfg = wandb_config or {}
                self.run = wandb.init(
                    project=wb_cfg.get("project", "multitme"),
                    entity=wb_cfg.get("entity"),
                    name=wb_cfg.get("name"),
                    tags=wb_cfg.get("tags", []),
                    config=wb_cfg.get("full_config"),
                    mode=wandb_mode,
                )
                logger.info(f"wandb initialized (mode={wandb_mode})")
            except ImportError:
                logger.warning("wandb not installed, disabling experiment tracking")
                self.wandb_enabled = False

    def _check_slope_convergence(self, window, threshold):
        return check_loss_slope_convergence(self.history["total"], window, threshold)

    def _get_beta(self, epoch):
        if self.beta_warmup_epochs <= 0:
            return self.beta
        return self.beta * min(1.0, epoch / self.beta_warmup_epochs)

    def save_checkpoint(self, epoch):
        if self.output_dir is None:
            return
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "metadata": self.metadata,
        }
        path = os.path.join(self.output_dir, "checkpoint.pt")
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: epoch {epoch} -> {path}")

    def load_checkpoint(self, path, device=None):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        logger.info(f"Resumed from checkpoint: {path} (epoch {start_epoch})")
        return start_epoch

    def train_epoch(self, dataloader, epoch=0):
        self.model.train()
        device = next(self.model.parameters()).device
        epoch_losses = {k: 0.0 for k in self.history}
        n_batches = 0
        current_beta = self._get_beta(epoch)

        for batch, labels in dataloader:
            data_dict = {m: batch[m].float().to(device) for m in batch}
            label_dict = {m: labels[m].to(device) for m in labels} if labels else None

            losses = self.model(
                data_dict,
                labels_dict=label_dict,
                cycle_weight=self.cycle_weight,
                beta=current_beta,
            )

            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()

            step_loss = losses["total"].item()
            if self.run is not None:
                self.run.log({"train/loss_step": step_loss})

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += losses[k].item()
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
            self.history[k].append(epoch_losses[k])

        if self.run is not None:
            self.run.log({"epoch": epoch, **{f"train/{k}": v for k, v in epoch_losses.items()}})

        return epoch_losses

    def fit(
        self,
        dataloader,
        n_epochs=None,
        max_epochs=500,
        slope_window=10,
        slope_threshold=1e-4,
        print_every=5,
        start_epoch=0,
    ):
        """Train the model.

        Parameters
        ----------
        n_epochs : int or None
            If an integer, train for exactly this many epochs (no early stopping).
            If None (default), use slope-based early stopping: training halts once
            the normalised slope of the total-loss curve over the last
            *slope_window* epochs drops below *slope_threshold*.
        max_epochs : int
            Hard upper limit on epochs when slope-based stopping is active.
        slope_window : int
            Number of recent epochs used for the linear-slope estimate.
        slope_threshold : float
            Stop when |slope / mean_loss| < threshold.
        """
        use_slope_stopping = n_epochs is None
        epoch_limit = max_epochs if use_slope_stopping else n_epochs

        device = next(self.model.parameters()).device
        stop_desc = (
            f"slope stopping (window={slope_window}, threshold={slope_threshold}, max={max_epochs})"
            if use_slope_stopping
            else f"fixed {n_epochs} epochs"
        )
        logger.info(
            f"Training MultiModal CycleVAE on {device} | modalities={self.model.modality_names} "
            f"| latent={self.model.n_latent} | batches/epoch={len(dataloader)} "
            f"| beta={self.beta:.2f} (warmup={self.beta_warmup_epochs}) "
            f"| stopping={stop_desc}"
        )

        start_time = time.time()
        stopped_epoch = epoch_limit - 1

        for epoch in range(start_epoch, epoch_limit):
            epoch_start = time.time()

            if hasattr(dataloader.dataset, "reshuffle"):
                dataloader.dataset.reshuffle()

            epoch_losses = self.train_epoch(dataloader, epoch=epoch)
            et = time.time() - epoch_start

            if epoch % print_every == 0:
                beta_now = self._get_beta(epoch)
                logger.info(
                    f"Epoch {epoch:3d} | total={epoch_losses['total']:.2f} "
                    f"recon={epoch_losses['reconstruction']:.2f} kl={epoch_losses['kl']:.2f} "
                    f"cycle={epoch_losses['cycle']:.2f} cls={epoch_losses['classification']:.3f} "
                    f"align={epoch_losses['alignment']:.3f} cycle_cls={epoch_losses['cycle_cls']:.3f} "
                    f"common={epoch_losses['common_feature']:.2f} beta={beta_now:.2f} {et:.1f}s"
                )

            if self.output_dir and epoch % self.save_freq == 0:
                self.save_checkpoint(epoch)

            if device.type == "mps" and epoch % 5 == 0 and epoch > 0:
                torch.mps.empty_cache()

            if use_slope_stopping and self._check_slope_convergence(slope_window, slope_threshold):
                logger.info(
                    f"Epoch {epoch}: loss slope converged "
                    f"(|slope/mean| < {slope_threshold} over last {slope_window} epochs) — stopping early."
                )
                stopped_epoch = epoch
                break

        # Final checkpoint + loss plot/history
        if self.output_dir:
            self.save_checkpoint(stopped_epoch)
            try:
                png_path, _ = save_loss_history(self.history, self.output_dir)
                logger.info(f"Wrote loss curve: {png_path}")
            except Exception as e:
                logger.warning(f"Failed to write loss plot: {e}")

        total_time = time.time() - start_time
        n_trained = stopped_epoch - start_epoch + 1
        logger.info(
            f"Training complete: {total_time / 60:.1f} min, {total_time / max(n_trained, 1):.1f}s/epoch"
        )
        return self.history
