from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_marker_scores(
    data: np.ndarray,
    gene_names: list[str],
    marker_dict: dict[str, list[str]],
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Compute per-cell mean expression scores for each cell type's marker genes."""
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    if normalize:
        lib_size = data.sum(axis=1, keepdims=True) + 1e-8
        data = np.log1p(data / lib_size * 1e4)

    scores = {}
    for cell_type, markers in marker_dict.items():
        valid_idx = [gene_to_idx[g] for g in markers if g in gene_to_idx]
        if len(valid_idx) == 0:
            logger.warning(f"No markers found for {cell_type}, skipping")
            continue

        found = [g for g in markers if g in gene_to_idx]
        missing = [g for g in markers if g not in gene_to_idx]
        if missing:
            logger.debug(f"{cell_type}: using {found}, missing {missing}")

        scores[cell_type] = data[:, valid_idx].mean(axis=1)

    return scores


def pseudo_label_discriminative(
    data,
    gene_names,
    marker_dict_pos,
    marker_dict_neg,
    type_to_idx,
    top_k=200,
    normalize=False,
):
    """Assign pseudo labels using positive minus negative marker scores.

    Parameters
    ----------
    marker_dict_pos : dict
        Positive markers per cell type.
    marker_dict_neg : dict
        Negative markers per cell type.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_cells = data.shape[0]
    labels = torch.full((n_cells,), -1, dtype=torch.long)

    scores = {}
    for ct in marker_dict_pos:
        pos_idx = [gene_to_idx[g] for g in marker_dict_pos[ct] if g in gene_to_idx]
        neg_idx = [gene_to_idx[g] for g in marker_dict_neg.get(ct, []) if g in gene_to_idx]

        pos_score = data[:, pos_idx].mean(axis=1) if pos_idx else np.zeros(n_cells)
        neg_score = data[:, neg_idx].mean(axis=1) if neg_idx else np.zeros(n_cells)
        scores[ct] = pos_score - neg_score

    candidates = []
    for ct, cell_scores in scores.items():
        if ct not in type_to_idx:
            continue
        type_idx = type_to_idx[ct]
        for i in range(n_cells):
            candidates.append((cell_scores[i], i, type_idx, ct))

    candidates.sort(key=lambda x: -x[0])
    claimed = np.zeros(n_cells, dtype=bool)
    type_counts = {ct: 0 for ct in scores}

    for _score, cell_idx, type_idx, ct in candidates:
        if claimed[cell_idx] or type_counts[ct] >= top_k:
            continue
        labels[cell_idx] = type_idx
        claimed[cell_idx] = True
        type_counts[ct] += 1

    return labels


def pseudo_label_from_markers(
    data: np.ndarray,
    gene_names: list[str],
    marker_dict: dict[str, list[str]],
    type_to_idx: dict[str, int],
    top_k: int = 100,
    normalize: bool = True,
    min_score: float | None = None,
) -> torch.Tensor:
    """Assign pseudo labels to cells based on marker gene expression.

    For each cell type, the ``top_k`` highest-scoring cells are labeled.
    Cells are claimed greedily in descending score order so each cell
    receives at most one label.
    """
    n_cells = data.shape[0]
    labels = torch.full((n_cells,), -1, dtype=torch.long)

    scores = compute_marker_scores(data, gene_names, marker_dict, normalize)

    claimed = np.zeros(n_cells, dtype=bool)

    candidates = []
    for cell_type, cell_scores in scores.items():
        if cell_type not in type_to_idx:
            logger.warning(f"{cell_type} not in type_to_idx, skipping")
            continue
        type_idx = type_to_idx[cell_type]
        for i in range(n_cells):
            if min_score is not None and cell_scores[i] < min_score:
                continue
            candidates.append((cell_scores[i], i, type_idx, cell_type))

    candidates.sort(key=lambda x: -x[0])

    type_counts = {ct: 0 for ct in scores}
    labeled_count = 0

    for _score, cell_idx, type_idx, cell_type in candidates:
        if claimed[cell_idx]:
            continue
        if type_counts[cell_type] >= top_k:
            continue

        labels[cell_idx] = type_idx
        claimed[cell_idx] = True
        type_counts[cell_type] += 1
        labeled_count += 1

    logger.info(f"Pseudo-labeling: {labeled_count} cells labeled")
    for cell_type in sorted(type_counts.keys()):
        count = type_counts[cell_type]
        if cell_type in scores:
            top_scores = np.sort(scores[cell_type])[-min(count, 5) :][::-1]
            score_str = ", ".join(f"{s:.2f}" for s in top_scores)
            logger.debug(f"{cell_type}: {count} cells (top scores: {score_str})")

    return labels
