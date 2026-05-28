import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch.utils.data import Dataset

from multitme.model.cycle_vae import save_loss_history

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


def _as_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def _as_indexable(data):
    """Coerce a modality matrix to a row-indexable form (CSR if sparse)."""
    if sparse.issparse(data):
        return data.tocsr()
    return data


def _n_rows(data):
    if sparse.issparse(data):
        return data.shape[0]
    if torch.is_tensor(data):
        return data.size(0)
    return data.shape[0]


def _select_rows(data, inds):
    """Return a float32 torch tensor of rows ``inds`` from ``data``.

    ``inds`` may be a numpy array or a torch LongTensor. Sparse inputs are
    densified only for the selected rows.
    """
    if sparse.issparse(data):
        idx_np = inds.cpu().numpy() if torch.is_tensor(inds) else np.asarray(inds)
        return torch.from_numpy(data[idx_np].toarray().astype(np.float32, copy=False))
    if isinstance(data, np.ndarray):
        idx_np = inds.cpu().numpy() if torch.is_tensor(inds) else inds
        return torch.from_numpy(np.ascontiguousarray(data[idx_np], dtype=np.float32))
    return data[inds]


def spatial_tile_collate(items):
    """Collate spatial tile samples.

    Use as ``DataLoader(dataset, batch_size=1, collate_fn=spatial_tile_collate)``.
    """
    if len(items) == 1:
        return items[0]
    raise ValueError("SpatialTiledDataset requires DataLoader(..., batch_size=1)")


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
        spatial_pairs=None,
        spatial_k=10,
        spatial_tau=100.0,
        spatial_weight=1.0,
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
        self.spatial_k = spatial_k
        self.spatial_tau = spatial_tau
        self.spatial_weight = spatial_weight
        self.common_feature_weight = common_feature_weight

        if cycle_pairs is None:
            self.cycle_pairs = [
                (m1, m2)
                for i, m1 in enumerate(self.modality_names)
                for m2 in self.modality_names[i + 1 :]
            ]
        else:
            self.cycle_pairs = cycle_pairs

        self.spatial_pairs = self.cycle_pairs if spatial_pairs is None else spatial_pairs

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
    def _masked_mean(values, mask=None):
        if mask is None:
            return values.mean()
        mask = mask.to(device=values.device, dtype=torch.bool)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=values.device)
        return values[mask].mean()

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

    def _classification_loss(self, logits, labels, mask=None):
        labeled_mask = labels >= 0
        if mask is not None:
            labeled_mask = labeled_mask & mask.to(device=labels.device, dtype=torch.bool)
        if labeled_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        return self.aux_loss_multiplier * F.cross_entropy(
            logits[labeled_mask], labels[labeled_mask], label_smoothing=0.1
        )

    def _compute_swd(self, z1, z2, num_projections=50):
        n1, n2 = z1.size(0), z2.size(0)
        if n1 == 0 or n2 == 0:
            return torch.tensor(0.0, device=z1.device)
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

    def _type_alignment_loss(self, z_samples, cell_type_logits, core_masks=None):
        predicted_types = {mod: logits.argmax(dim=1) for mod, logits in cell_type_logits.items()}

        total_loss = 0.0
        pairs_count = 0

        for i, mod1 in enumerate(self.modality_names):
            for mod2 in self.modality_names[i + 1 :]:
                if mod1 not in z_samples or mod2 not in z_samples:
                    continue

                z1, z2 = z_samples[mod1], z_samples[mod2]
                types1, types2 = predicted_types[mod1], predicted_types[mod2]
                core1 = self._get_core_mask(core_masks, mod1, z1.device, z1.size(0))
                core2 = self._get_core_mask(core_masks, mod2, z2.device, z2.size(0))

                for ct in range(self.n_cell_types):
                    mask1 = (types1 == ct) & core1
                    mask2 = (types2 == ct) & core2

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

    @staticmethod
    def _get_core_mask(core_masks, modality_name, device, size):
        if core_masks is None or modality_name not in core_masks:
            return torch.ones(size, device=device, dtype=torch.bool)
        return core_masks[modality_name].to(device=device, dtype=torch.bool)

    def _cycle_classification_loss(self, logits_direct, logits_cycle, core_mask=None):
        q = F.softmax(logits_direct.detach(), dim=-1)
        p = F.log_softmax(logits_cycle, dim=-1)
        per_sample = F.kl_div(p, q, reduction="none").sum(dim=-1)
        return self.cycle_cls_weight * self._masked_mean(per_sample, core_mask)

    def _spatial_regularizer_loss(self, z_dict, cell_type_logits, coords_dict, core_masks=None):
        if coords_dict is None or self.n_cell_types is None:
            return torch.tensor(0.0, device=next(iter(z_dict.values())).device)

        probs = {
            mod: F.softmax(logits, dim=-1)
            for mod, logits in cell_type_logits.items()
            if mod in z_dict and mod in coords_dict
        }
        if not probs:
            return torch.tensor(0.0, device=next(iter(z_dict.values())).device)

        total_loss = 0.0
        pairs_count = 0

        for mod_a, mod_b in self.spatial_pairs:
            if mod_a not in z_dict or mod_b not in z_dict:
                continue
            if mod_a not in coords_dict or mod_b not in coords_dict:
                continue
            if mod_a not in probs or mod_b not in probs:
                continue

            z_a = z_dict[mod_a]
            z_b = z_dict[mod_b]
            p_a = probs[mod_a]
            p_b = probs[mod_b]
            s_a = coords_dict[mod_a].to(device=z_a.device, dtype=z_a.dtype)
            s_b = coords_dict[mod_b].to(device=z_b.device, dtype=z_b.dtype)

            query_mask = self._get_core_mask(core_masks, mod_a, z_a.device, z_a.size(0))
            if query_mask.sum() == 0 or z_b.size(0) == 0:
                continue

            z_a_core = z_a[query_mask]
            p_a_core = p_a[query_mask]
            s_a_core = s_a[query_mask]

            dist2 = torch.cdist(s_a_core, s_b).pow(2)
            k = min(self.spatial_k, s_b.size(0))
            nn_dist2, nn_idx = torch.topk(dist2, k=k, largest=False, dim=1)

            neighbor_z = z_b[nn_idx]
            neighbor_p = p_b[nn_idx]
            spatial_w = torch.exp(-nn_dist2 / (2.0 * self.spatial_tau**2))

            weighted_type = spatial_w.unsqueeze(-1) * neighbor_p
            denom = weighted_type.sum(dim=1).clamp_min(1e-8)
            target = torch.einsum("ikt,ikd->itd", weighted_type, neighbor_z)
            target = target / denom.unsqueeze(-1)

            diff2 = (z_a_core.unsqueeze(1) - target).pow(2).sum(dim=-1)
            total_loss = total_loss + (p_a_core * diff2).sum(dim=1).mean()
            pairs_count += 1

        if pairs_count == 0:
            return torch.tensor(0.0, device=next(iter(z_dict.values())).device)
        return self.spatial_weight * total_loss / pairs_count

    def forward(
        self,
        data_dict,
        labels_dict=None,
        coords_dict=None,
        core_masks=None,
        cycle_weight=1.0,
        beta=1.0,
    ):
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
                "spatial",
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

            core_mask = self._get_core_mask(core_masks, mod_name, device, data.size(0))
            recon = self.decode(z, mod_name)
            recon_nll = self._reconstruction_nll(recon, data, mod_name)
            kl = self._kl_divergence(mu, logvar)
            losses["reconstruction"] += self._masked_mean(recon_nll, core_mask)
            losses["kl"] += self._masked_mean(kl, core_mask)

            if self.n_cell_types is not None:
                logits = self.classify(mu)
                cell_type_logits[mod_name] = logits

                if labels_dict is not None and mod_name in labels_dict:
                    losses["classification"] += self._classification_loss(
                        logits, labels_dict[mod_name], core_mask
                    )

        if self.n_cell_types is not None and len(z_samples) > 1:
            losses["alignment"] = self._type_alignment_loss(
                z_samples, cell_type_logits, core_masks=core_masks
            )
            losses["spatial"] = self._spatial_regularizer_loss(
                mu_dict, cell_type_logits, coords_dict, core_masks=core_masks
            )

        for mod_src, mod_tgt in self.cycle_pairs:
            if mod_src not in data_dict or mod_tgt not in data_dict:
                continue

            src_core_mask = self._get_core_mask(
                core_masks, mod_src, device, data_dict[mod_src].size(0)
            )
            z_src = z_samples[mod_src]
            tgt_recon = self.decode(z_src, mod_tgt)
            z_cycle_mu, _ = self.encode(tgt_recon, mod_tgt)
            src_cycle = self.decode(z_cycle_mu, mod_src)

            cycle_nll = self._cycle_nll(src_cycle, data_dict[mod_src], mod_src, cycle_weight)
            losses["cycle"] += self._masked_mean(cycle_nll, src_core_mask)

            if self.n_cell_types is not None and mod_src == self.labeled_modality:
                logits_direct = self.classify(mu_dict[mod_src])
                logits_cycle = self.classify(z_cycle_mu)
                losses["cycle_cls"] += self._cycle_classification_loss(
                    logits_direct, logits_cycle, src_core_mask
                )

            if mod_src in self.common_masks and mod_tgt in self.common_masks:
                mask_src = self.common_masks[mod_src]
                mask_tgt = self.common_masks[mod_tgt]
                common_nll = self._common_feature_nll(
                    tgt_recon[:, mask_tgt], data_dict[mod_src][:, mask_src]
                )
                losses["common_feature"] += self._masked_mean(common_nll, src_core_mask)

        losses["total"] = (
            losses["reconstruction"]
            + beta * losses["kl"]
            + losses["classification"]
            + losses["cycle"]
            + self.common_feature_weight * losses["common_feature"]
            + losses["alignment"]
            + losses["cycle_cls"]
            + losses["spatial"]
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
    """Cycles minibatches across modalities, densifying sparse data per-batch."""

    def __init__(self, modality_dict, label_dict=None, target_batch_size=256):
        self.modality_dict = {name: _as_indexable(d) for name, d in modality_dict.items()}
        self.label_dict = label_dict or {}
        self.modality_names = list(self.modality_dict.keys())

        self.modality_sizes = {name: _n_rows(d) for name, d in self.modality_dict.items()}
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


class SpatialTiledDataset(Dataset):
    """Tile co-registered spatial modalities with halo cells.

    Core cells contribute to reconstruction, KL, classification, cycle, and
    common-feature losses. Core plus halo cells are available as neighbor
    candidates for the spatial regularizer.
    """

    def __init__(
        self,
        modality_dict,
        coord_dict,
        label_dict=None,
        tile_size=500.0,
        halo=150.0,
        min_core_cells=1,
        drop_empty=True,
        nonspatial_batch_size=256,
    ):
        self.modality_dict = {name: _as_indexable(data) for name, data in modality_dict.items()}
        self.coord_dict = {name: _as_tensor(coords) for name, coords in coord_dict.items()}
        self.label_dict = label_dict or {}
        self.tile_size = float(tile_size)
        self.halo = float(halo)
        self.min_core_cells = int(min_core_cells)
        self.drop_empty = drop_empty
        self.nonspatial_batch_size = int(nonspatial_batch_size)
        self.modality_names = list(self.modality_dict.keys())
        self.spatial_modalities = [name for name in self.modality_names if name in self.coord_dict]
        self.nonspatial_modalities = [
            name for name in self.modality_names if name not in self.coord_dict
        ]

        if not self.spatial_modalities:
            raise ValueError("SpatialTiledDataset requires coordinates for at least one modality")

        for name in self.spatial_modalities:
            if self.coord_dict[name].ndim != 2 or self.coord_dict[name].size(1) != 2:
                raise ValueError(f"Coordinates for {name!r} must have shape (n_cells, 2)")
            if self.coord_dict[name].size(0) != _n_rows(self.modality_dict[name]):
                raise ValueError(f"Data and coordinates for {name!r} have different lengths")

        all_coords = torch.cat([self.coord_dict[name] for name in self.spatial_modalities], dim=0)
        min_xy = all_coords.min(dim=0).values
        max_xy = all_coords.max(dim=0).values

        x_edges = torch.arange(min_xy[0], max_xy[0] + self.tile_size, self.tile_size)
        y_edges = torch.arange(min_xy[1], max_xy[1] + self.tile_size, self.tile_size)
        if x_edges.numel() < 2:
            x_edges = torch.tensor([min_xy[0], min_xy[0] + self.tile_size])
        if y_edges.numel() < 2:
            y_edges = torch.tensor([min_xy[1], min_xy[1] + self.tile_size])

        self.tiles = []
        for x0 in x_edges[:-1]:
            for y0 in y_edges[:-1]:
                tile = (
                    float(x0),
                    float(y0),
                    float(x0 + self.tile_size),
                    float(y0 + self.tile_size),
                )
                if not drop_empty or self._tile_core_count(tile) >= self.min_core_cells:
                    self.tiles.append(tile)

        if not self.tiles:
            raise ValueError("No spatial tiles contain enough core cells")

    def _tile_core_count(self, tile):
        x0, y0, x1, y1 = tile
        count = 0
        for name in self.spatial_modalities:
            coords = self.coord_dict[name]
            core = (
                (coords[:, 0] >= x0)
                & (coords[:, 0] < x1)
                & (coords[:, 1] >= y0)
                & (coords[:, 1] < y1)
            )
            count += int(core.sum().item())
        return count

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        x0, y0, x1, y1 = self.tiles[idx]
        batch = {}
        labels = {}
        coords_batch = {}
        core_masks = {}

        for name in self.spatial_modalities:
            coords = self.coord_dict[name]
            in_halo = (
                (coords[:, 0] >= x0 - self.halo)
                & (coords[:, 0] < x1 + self.halo)
                & (coords[:, 1] >= y0 - self.halo)
                & (coords[:, 1] < y1 + self.halo)
            )
            core = (
                (coords[:, 0] >= x0)
                & (coords[:, 0] < x1)
                & (coords[:, 1] >= y0)
                & (coords[:, 1] < y1)
            )

            inds = torch.nonzero(in_halo, as_tuple=False).squeeze(1)
            batch[name] = _select_rows(self.modality_dict[name], inds)
            coords_batch[name] = coords[inds]
            core_masks[name] = core[inds]
            if name in self.label_dict:
                labels[name] = self.label_dict[name][inds]

        # Non-spatial modalities: random sample per tile so they still contribute
        # to cycle / alignment / classification losses. No coords or core mask
        # entry — treated as all-core by the model.
        for name in self.nonspatial_modalities:
            n_total = _n_rows(self.modality_dict[name])
            k = min(self.nonspatial_batch_size, n_total)
            inds = torch.from_numpy(np.random.choice(n_total, size=k, replace=False))
            batch[name] = _select_rows(self.modality_dict[name], inds)
            if name in self.label_dict:
                labels[name] = self.label_dict[name][inds]

        return batch, labels, coords_batch, core_masks


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
            "spatial": [],
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

    @staticmethod
    def _move_dict(values, device, dtype=None):
        if not values:
            return None
        out = {}
        for name, value in values.items():
            if dtype is None:
                out[name] = value.to(device)
            else:
                out[name] = value.to(device=device, dtype=dtype)
        return out

    def train_epoch(self, dataloader, epoch=0):
        self.model.train()
        device = next(self.model.parameters()).device
        epoch_losses = {k: 0.0 for k in self.history}
        n_batches = 0
        current_beta = self._get_beta(epoch)

        for batch_item in dataloader:
            if len(batch_item) == 4:
                batch, labels, coords, core_masks = batch_item
            else:
                batch, labels = batch_item
                coords = None
                core_masks = None

            data_dict = self._move_dict(batch, device, dtype=torch.float32)
            label_dict = self._move_dict(labels, device) if labels else None
            coords_dict = self._move_dict(coords, device, dtype=torch.float32) if coords else None
            core_masks = (
                self._move_dict(core_masks, device, dtype=torch.bool) if core_masks else None
            )

            losses = self.model(
                data_dict,
                labels_dict=label_dict,
                coords_dict=coords_dict,
                core_masks=core_masks,
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
            f"Training spatial MultiModal CycleVAE on {device} "
            f"| modalities={self.model.modality_names} | latent={self.model.n_latent} "
            f"| batches/epoch={len(dataloader)} | beta={self.beta:.2f} "
            f"(warmup={self.beta_warmup_epochs}) | stopping={stop_desc}"
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
                    f"align={epoch_losses['alignment']:.3f} spatial={epoch_losses['spatial']:.3f} "
                    f"cycle_cls={epoch_losses['cycle_cls']:.3f} "
                    f"common={epoch_losses['common_feature']:.2f} beta={beta_now:.2f} {et:.1f}s"
                )

            if self.output_dir and epoch % self.save_freq == 0:
                self.save_checkpoint(epoch)

            if device.type == "mps" and epoch % 5 == 0 and epoch > 0:
                torch.mps.empty_cache()

            if use_slope_stopping and self._check_slope_convergence(slope_window, slope_threshold):
                logger.info(
                    f"Epoch {epoch}: loss slope converged "
                    f"(|slope/mean| < {slope_threshold} over last {slope_window} epochs) - "
                    "stopping early."
                )
                stopped_epoch = epoch
                break

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
            f"Training complete: {total_time / 60:.1f} min, "
            f"{total_time / max(n_trained, 1):.1f}s/epoch"
        )
        return self.history


MultiModalCycleVAESpatial = MultiModalCycleVAE
CycleVAESpatialTrainer = CycleVAETrainer

__all__ = [
    "MultiModalCycleVAE",
    "CycleVAETrainer",
    "MultiModalCycleVAESpatial",
    "CycleVAESpatialTrainer",
    "CyclingDataset",
    "SpatialTiledDataset",
    "spatial_tile_collate",
    "check_loss_slope_convergence",
]
