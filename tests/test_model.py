import numpy as np
import torch
from scipy import sparse

from multitme.model import CycleVAETrainer, CyclingDataset, MultiModalCycleVAE


def _make_model(n_genes_a=50, n_genes_b=30, n_latent=10, n_cell_types=5):
    return MultiModalCycleVAE(
        modality_dims={"a": n_genes_a, "b": n_genes_b},
        n_latent=n_latent,
        hidden_dims=[32, 16],
        n_cell_types=n_cell_types,
        cycle_pairs=[("a", "b"), ("b", "a")],
    )


def test_encode_decode_shapes():
    model = _make_model()
    x = torch.randn(8, 50)
    mu, logvar = model.encode(x, "a")
    assert mu.shape == (8, 10)
    assert logvar.shape == (8, 10)

    z = model.reparameterize(mu, logvar)
    recon = model.decode(z, "a")
    assert recon.shape == (8, 50)


def test_forward_produces_all_loss_keys():
    model = _make_model()
    data = {"a": torch.randn(16, 50), "b": torch.randn(16, 30)}
    labels = {"a": torch.randint(0, 5, (16,))}
    losses = model(data, labels_dict=labels)

    expected_keys = {
        "total",
        "reconstruction",
        "kl",
        "classification",
        "cycle",
        "common_feature",
        "alignment",
        "cycle_cls",
    }
    assert set(losses.keys()) == expected_keys
    assert losses["total"].requires_grad


def test_forward_backward():
    model = _make_model()
    data = {"a": torch.randn(16, 50), "b": torch.randn(16, 30)}
    labels = {"a": torch.randint(0, 5, (16,))}
    losses = model(data, labels_dict=labels)
    losses["total"].backward()

    # Check that encoder/decoder params got gradients
    for name, p in model.named_parameters():
        if p.requires_grad and "classifier" not in name:
            assert p.grad is not None, f"{name} has no gradient"


def test_predict_cell_types():
    model = _make_model()
    x = torch.randn(8, 50)
    probs = model.predict_cell_types(x, "a")
    assert probs.shape == (8, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)


def test_translate():
    model = _make_model()
    x = torch.randn(8, 50)
    translated = model.translate(x, "a", "b")
    assert translated.shape == (8, 30)


def test_get_latent():
    model = _make_model()
    x = torch.randn(8, 50)
    z = model.get_latent(x, "a")
    assert z.shape == (8, 10)


def test_cycling_dataset():
    data_a = torch.randn(100, 50)
    data_b = torch.randn(80, 30)
    labels_a = torch.randint(0, 5, (100,))

    ds = CyclingDataset(
        modality_dict={"a": data_a, "b": data_b},
        label_dict={"a": labels_a},
        target_batch_size=32,
    )
    assert len(ds) > 0
    batch, labels = ds[0]
    assert "a" in batch and "b" in batch
    assert "a" in labels


def test_cycling_dataset_sparse_input():
    rng = np.random.default_rng(0)
    dense_a = rng.poisson(0.3, size=(100, 50)).astype(np.float32)
    dense_b = rng.poisson(0.3, size=(80, 30)).astype(np.float32)
    sp_a = sparse.csr_matrix(dense_a)
    sp_b = sparse.csr_matrix(dense_b)

    ds = CyclingDataset(
        modality_dict={"a": sp_a, "b": sp_b},
        target_batch_size=32,
    )
    batch, _ = ds[0]
    assert isinstance(batch["a"], torch.Tensor) and batch["a"].dtype == torch.float32
    assert isinstance(batch["b"], torch.Tensor) and batch["b"].dtype == torch.float32
    assert batch["a"].shape[1] == 50 and batch["b"].shape[1] == 30


def test_trainer_writes_loss_plot(tmp_path):
    model = _make_model()
    data_a = torch.randn(64, 50)
    data_b = torch.randn(64, 30)
    labels_a = torch.randint(0, 5, (64,))

    ds = CyclingDataset(
        modality_dict={"a": data_a, "b": data_b},
        label_dict={"a": labels_a},
        target_batch_size=32,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=None)
    trainer = CycleVAETrainer(model, learning_rate=1e-3, output_dir=str(tmp_path))
    trainer.fit(loader, n_epochs=2, print_every=1)

    assert (tmp_path / "loss_curve.png").exists()
    assert (tmp_path / "loss_history.json").exists()


def test_spatial_trainer_one_epoch(tmp_path):
    from multitme.model import (
        SpatialCycleVAETrainer,
        SpatialMultiModalCycleVAE,
        SpatialTiledDataset,
        spatial_tile_collate,
    )

    rng = np.random.default_rng(0)
    n_a, n_b = 64, 80
    data_a = torch.randn(n_a, 50)  # scrna-like, no coords
    data_b = torch.randn(n_b, 30)  # xenium-like, has coords
    coords_b = torch.from_numpy(rng.uniform(0, 1000, size=(n_b, 2)).astype("float32"))
    labels_a = torch.randint(0, 5, (n_a,))

    ds = SpatialTiledDataset(
        modality_dict={"a": data_a, "b": data_b},
        coord_dict={"b": coords_b},
        label_dict={"a": labels_a},
        tile_size=400.0,
        halo=100.0,
        min_core_cells=1,
        nonspatial_batch_size=16,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=spatial_tile_collate)

    model = SpatialMultiModalCycleVAE(
        modality_dims={"a": 50, "b": 30},
        n_latent=10,
        hidden_dims=[32, 16],
        n_cell_types=5,
        cycle_pairs=[("a", "b"), ("b", "a")],
        spatial_k=5,
        spatial_tau=50.0,
        spatial_weight=0.5,
    )
    trainer = SpatialCycleVAETrainer(model, learning_rate=1e-3, output_dir=str(tmp_path))
    losses = trainer.train_epoch(loader, epoch=0)
    assert "total" in losses
    assert "spatial" in losses


def test_trainer_one_epoch():
    model = _make_model()
    data_a = torch.randn(64, 50)
    data_b = torch.randn(64, 30)
    labels_a = torch.randint(0, 5, (64,))

    ds = CyclingDataset(
        modality_dict={"a": data_a, "b": data_b},
        label_dict={"a": labels_a},
        target_batch_size=32,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=None)
    trainer = CycleVAETrainer(model, learning_rate=1e-3)
    losses = trainer.train_epoch(loader, epoch=0)
    assert "total" in losses
    assert losses["total"] > 0
