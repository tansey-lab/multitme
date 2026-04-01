import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time

def preprocess(data, method='log1p', target_sum=1e4, pseudocount=1e-3, clip_percentile=99.5):
    """
    log1p or clr transform
    """
    data = data.astype(np.float64)
    
    if method == 'log1p':
        lib_size = data.sum(axis=1, keepdims=True)
        lib_size = np.clip(lib_size, 1e-8, None)
        normalized = data / lib_size * target_sum
        return np.log1p(normalized)
    
    elif method == 'clr':
        # Clip outliers per gene
        caps = np.percentile(data, clip_percentile, axis=0).clip(1)
        data = np.clip(data, 0, caps)
        # Max-scale per gene to [0, 1]
        data = data / data.max(axis=0, keepdims=True).clip(1e-8)
        # Centered log-ratio
        log_data = np.log(data + pseudocount)
        return log_data - log_data.mean(axis=1, keepdims=True)
    
    else:
        raise ValueError(f"Unknown method '{method}', use 'log1p' or 'clr'")

class MultiModalCycleVAE(nn.Module):

    def __init__(
        self,
        modality_dims,
        n_latent=20,
        hidden_dims=[128, 64],
        common_masks=None,
        cycle_pairs=None,
        n_cell_types=None,
        aux_loss_multiplier=1.0,
        type_alignment_weight=10.0,
        alignment_method='swd',
        cycle_cls_weight=1000.0,
        labeled_modality='scrna',
    ):
        super().__init__()

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

        if cycle_pairs is None:
            self.cycle_pairs = [
                (m1, m2)
                for i, m1 in enumerate(self.modality_names)
                for m2 in self.modality_names[i + 1:]
            ]
        else:
            self.cycle_pairs = cycle_pairs
            
        common_dim = hidden_dims[0]

        # Modality-specific projections
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, common_dim),
                nn.LayerNorm(common_dim),
                nn.ReLU(),
            )
            for name, dim in modality_dims.items()
        })

        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(common_dim, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.encoder_mu = nn.Linear(hidden_dims[1], n_latent)
        self.encoder_logvar = nn.Linear(hidden_dims[1], n_latent)

        # Cell type classifier
        if self.n_cell_types is not None:
            self.classifier = nn.Sequential(
                nn.Linear(n_latent, hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], n_cell_types),
            )

        # Modality-specific decoders
        self.decoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(n_latent, hidden_dims[1]),
                nn.LayerNorm(hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], common_dim),
                nn.ReLU(),
                nn.Linear(common_dim, dim),
            )
            for name, dim in modality_dims.items()
        })

        # Learnable observation noise
        self.log_sigmas = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in self.modality_names
        })

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
        #KL(N(mu, exp(logvar)) || N(0, I)), mean over latent dims, per sample
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    def _reconstruction_nll(self, recon, target, modality_name):
        # -log N(target | recon, sigma) per sample, summed over features
        sigma = torch.exp(self.log_sigmas[modality_name])
        nll = 0.5 * ((target - recon) / sigma).pow(2) + torch.log(sigma)
        return nll.sum(dim=-1)  # (batch,)

    def _cycle_nll(self, src_cycle, src_data, modality_name, cycle_weight):
        # -log N(src_data | src_cycle, sigma * cycle_weight) per sample.
        sigma = torch.exp(self.log_sigmas[modality_name])
        effective_sigma = sigma * cycle_weight
        nll = 0.5 * ((src_data - src_cycle) / effective_sigma).pow(2) + torch.log(effective_sigma)
        return nll.sum(dim=-1)

    def _common_feature_nll(self, tgt_recon_common, src_data_common):
        # -log N(src_data_common | tgt_recon_common, 1.0) per sample
        nll = 0.5 * (src_data_common - tgt_recon_common).pow(2)  # sigma=1, log(1)=0
        return nll.sum(dim=-1)

    def _classification_loss(self, logits, labels):
        labeled_mask = labels >= 0
        if labeled_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        # cross_entropy returns mean over samples
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
        theta = theta / torch.sqrt((theta ** 2).sum(dim=0, keepdim=True))

        proj1, _ = torch.sort(z1 @ theta, dim=0)
        proj2, _ = torch.sort(z2 @ theta, dim=0)
        return torch.mean((proj1 - proj2).pow(2))

    def _type_alignment_loss(self, z_samples, cell_type_logits):
        # swd cell type alignment
        predicted_types = {
            mod: logits.argmax(dim=1)
            for mod, logits in cell_type_logits.items()
        }

        total_loss = 0.0
        pairs_count = 0

        for i, mod1 in enumerate(self.modality_names):
            for mod2 in self.modality_names[i + 1:]:
                if mod1 not in z_samples or mod2 not in z_samples:
                    continue

                z1, z2 = z_samples[mod1], z_samples[mod2]
                types1, types2 = predicted_types[mod1], predicted_types[mod2]

                for ct in range(self.n_cell_types):
                    mask1 = types1 == ct
                    mask2 = types2 == ct

                    if mask1.sum() < 5 or mask2.sum() < 5:
                        continue

                    if self.alignment_method == 'swd':
                        loss = self._compute_swd(z1[mask1], z2[mask2])
                    elif self.alignment_method == 'moment_matching':
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
        losses = {k: torch.tensor(0.0, device=device) for k in [
            'reconstruction', 'kl', 'classification', 'cycle',
            'common_feature', 'alignment', 'cycle_cls', 'total',
        ]}

        z_samples = {}
        mu_dict = {}
        cell_type_logits = {}

        # encode reconstruct classify
        for mod_name, data in data_dict.items():
            mu, logvar = self.encode(data, mod_name)
            z = self.reparameterize(mu, logvar)
            z_samples[mod_name] = z
            mu_dict[mod_name] = mu

            recon = self.decode(z, mod_name)
            losses['reconstruction'] += self._reconstruction_nll(recon, data, mod_name).mean()

            losses['kl'] += self._kl_divergence(mu, logvar).mean()

            # classification from mu
            if self.n_cell_types is not None:
                logits = self.classify(mu)
                cell_type_logits[mod_name] = logits

                if labels_dict is not None and mod_name in labels_dict:
                    losses['classification'] += self._classification_loss(
                        logits, labels_dict[mod_name]
                    )

        # cell type alignment
        if self.n_cell_types is not None and len(z_samples) > 1:
            losses['alignment'] = self._type_alignment_loss(z_samples, cell_type_logits)

        # cycle consistency
        for mod_src, mod_tgt in self.cycle_pairs:
            if mod_src not in data_dict or mod_tgt not in data_dict:
                continue

            z_src = z_samples[mod_src]

            # src -> tgt -> src (data space)
            tgt_recon = self.decode(z_src, mod_tgt)
            z_cycle_mu, _ = self.encode(tgt_recon, mod_tgt)
            src_cycle = self.decode(z_cycle_mu, mod_src)

            losses['cycle'] += self._cycle_nll(
                src_cycle, data_dict[mod_src], mod_src, cycle_weight
            ).mean()

            # cycle classification consistency
            if self.n_cell_types is not None and mod_src == self.labeled_modality:
                logits_direct = self.classify(mu_dict[mod_src])
                logits_cycle = self.classify(z_cycle_mu)
                q = F.softmax(logits_direct.detach(), dim=-1)
                p = F.log_softmax(logits_cycle, dim=-1)
                losses['cycle_cls'] += self.cycle_cls_weight * F.kl_div(
                    p, q, reduction='batchmean'
                )

            # common feature constraint
            if mod_src in self.common_masks and mod_tgt in self.common_masks:
                mask_src = self.common_masks[mod_src]
                mask_tgt = self.common_masks[mod_tgt]
                losses['common_feature'] += self._common_feature_nll(
                    tgt_recon[:, mask_tgt], data_dict[mod_src][:, mask_src]
                ).mean()

        losses['total'] = (
            losses['reconstruction']
            + beta * losses['kl']
            + losses['classification']
            + losses['cycle']
            + losses['common_feature']
            + losses['alignment']
            + losses['cycle_cls']
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
    def __init__(self, modality_dict, label_dict=None, target_batch_size=256):
        self.modality_dict = modality_dict
        self.label_dict = label_dict or {}
        self.modality_names = list(modality_dict.keys())

        self.modality_sizes = {
            name: data.shape[0] for name, data in modality_dict.items()
        }
        self.largest_modality = max(self.modality_sizes, key=self.modality_sizes.get)
        self.n_cells_max = self.modality_sizes[self.largest_modality]
        self.n_batches = max(1, self.n_cells_max // target_batch_size)
        self.batch_sizes = {
            name: self.modality_sizes[name] // self.n_batches
            for name in self.modality_names
        }
        self.reshuffle()

    def reshuffle(self):
        self.indices = {
            name: np.random.permutation(self.modality_sizes[name])
            for name in self.modality_names
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
            batch[name] = self.modality_dict[name][inds]
            if name in self.label_dict:
                labels[name] = self.label_dict[name][inds]
        return batch, labels

class CycleVAETrainer:
    def __init__(self, model, learning_rate=1e-3, cycle_weight=1.0, beta=1.0, beta_warmup_epochs=0):
        self.model = model
        self.cycle_weight = cycle_weight
        self.beta = beta
        self.beta_warmup_epochs = beta_warmup_epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.history = {
            'total': [], 'reconstruction': [], 'kl': [], 'cycle': [],
            'classification': [], 'alignment': [], 'common_feature': [],
            'cycle_cls': [],
        }

    def _get_beta(self, epoch):
        if self.beta_warmup_epochs <= 0:
            return self.beta
        return self.beta * min(1.0, epoch / self.beta_warmup_epochs)

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
                data_dict, labels_dict=label_dict,
                cycle_weight=self.cycle_weight, beta=current_beta,
            )

            self.optimizer.zero_grad()
            losses['total'].backward()
            self.optimizer.step()

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += losses[k].item()
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
            self.history[k].append(epoch_losses[k])
        return epoch_losses

    def fit(self, dataloader, n_epochs=50, print_every=5, scrna_test_data=None, scrna_labels=None, test_data=None, ref=None, marker_labels=None):
        device = next(self.model.parameters()).device
        print(f"Training MultiModal CycleVAE on {device}")
        print(f"  Modalities: {self.model.modality_names}")
        print(f"  Latent dim: {self.model.n_latent}")
        print(f"  Batches/epoch: {len(dataloader)}")
        print(f"  Beta: {self.beta} (warmup: {self.beta_warmup_epochs} epochs)")
        print("-" * 70)

        start_time = time.time()
        epoch_times = []

        for epoch in range(n_epochs):
            epoch_start = time.time()

            if hasattr(dataloader.dataset, 'reshuffle'):
                dataloader.dataset.reshuffle()

            epoch_losses = self.train_epoch(dataloader, epoch=epoch)

            et = time.time() - epoch_start
            epoch_times.append(et)

            if epoch % print_every == 0:
                elapsed = time.time() - start_time
                beta_now = self._get_beta(epoch)

                print(
                    f"Epoch {epoch:3d} | "
                    f"Total: {epoch_losses['total']:10.2f} | "
                    f"Recon: {epoch_losses['reconstruction']:8.2f} | "
                    f"KL: {epoch_losses['kl']:7.2f} | "
                    f"Cycle: {epoch_losses['cycle']:8.2f} | "
                    f"Class: {epoch_losses['classification']:7.3f} | "
                    f"Align: {epoch_losses['alignment']:7.3f} | "
                    f"CycleCls: {epoch_losses['cycle_cls']:7.3f} | "
                    f"Common: {epoch_losses['common_feature']:7.2f} | "
                    f"beta={beta_now:.2f} | "
                    f"{et:.1f}s"
                )

            if device.type == 'mps' and epoch % 5 == 0 and epoch > 0:
                torch.mps.empty_cache()

        total_time = time.time() - start_time
        print("-" * 70)
        print(f"Done! {total_time / 60:.1f} min, {total_time / n_epochs:.1f}s/epoch")
        return self.history