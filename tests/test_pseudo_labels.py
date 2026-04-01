import numpy as np
import torch

from multitme.data.pseudo_labels import (
    compute_marker_scores,
    pseudo_label_discriminative,
    pseudo_label_from_markers,
)


def _make_data(n_cells=100, n_genes=10):
    rng = np.random.default_rng(42)
    data = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float64)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    return data, gene_names


def test_compute_marker_scores():
    data, gene_names = _make_data()
    marker_dict = {"typeA": ["gene_0", "gene_1"], "typeB": ["gene_2", "gene_3"]}
    scores = compute_marker_scores(data, gene_names, marker_dict, normalize=True)
    assert "typeA" in scores
    assert "typeB" in scores
    assert scores["typeA"].shape == (100,)


def test_compute_marker_scores_missing_genes():
    data, gene_names = _make_data()
    marker_dict = {"typeA": ["gene_0", "MISSING_GENE"]}
    scores = compute_marker_scores(data, gene_names, marker_dict, normalize=False)
    assert "typeA" in scores


def test_pseudo_label_from_markers():
    data, gene_names = _make_data()
    marker_dict = {"typeA": ["gene_0", "gene_1"], "typeB": ["gene_2", "gene_3"]}
    type_to_idx = {"typeA": 0, "typeB": 1}
    labels = pseudo_label_from_markers(
        data, gene_names, marker_dict, type_to_idx, top_k=10, normalize=False
    )
    assert labels.shape == (100,)
    assert labels.dtype == torch.long
    assert (labels >= 0).sum() <= 20  # at most top_k per type
    assert (labels == -1).sum() > 0  # some unlabeled


def test_pseudo_label_discriminative():
    data, gene_names = _make_data()
    marker_dict_pos = {"typeA": ["gene_0"], "typeB": ["gene_2"]}
    marker_dict_neg = {"typeA": ["gene_2"], "typeB": ["gene_0"]}
    type_to_idx = {"typeA": 0, "typeB": 1}
    labels = pseudo_label_discriminative(
        data,
        gene_names,
        marker_dict_pos,
        marker_dict_neg,
        type_to_idx,
        top_k=10,
        normalize=False,
    )
    assert labels.shape == (100,)
    assert (labels >= 0).sum() <= 20
