import torch
import numpy as np
from typing import Dict, List, Optional


def compute_marker_scores(
    data: np.ndarray,
    gene_names: List[str],
    marker_dict: Dict[str, List[str]],
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    if normalize:
        lib_size = data.sum(axis=1, keepdims=True) + 1e-8
        data = np.log1p(data / lib_size * 1e4)

    scores = {}
    for cell_type, markers in marker_dict.items():
        # Find which markers are actually in the panel
        valid_idx = [gene_to_idx[g] for g in markers if g in gene_to_idx]
        if len(valid_idx) == 0:
            print(f"  WARNING: no markers found for {cell_type}, skipping")
            continue

        found = [g for g in markers if g in gene_to_idx]
        missing = [g for g in markers if g not in gene_to_idx]
        if missing:
            print(f"  {cell_type}: using {found}, missing {missing}")

        # Mean expression of marker genes per cell
        scores[cell_type] = data[:, valid_idx].mean(axis=1)

    return scores

def pseudo_label_discriminative(
    data, gene_names, marker_dict_pos, marker_dict_neg,
    type_to_idx, top_k=200, normalize=False
):
    """
    marker_dict_pos: {'B-CD22-CD40': ['CD22', 'CD40', 'CD19'], ...}
    marker_dict_neg: {'B-CD22-CD40': ['CD3', 'CD8'], 'CD4 T': ['CD19', 'CD8'], ...}
    score = mean(positive markers) - mean(negative markers)
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
    
    # Same top-k assignment
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
    
    for score, cell_idx, type_idx, ct in candidates:
        if claimed[cell_idx] or type_counts[ct] >= top_k:
            continue
        labels[cell_idx] = type_idx
        claimed[cell_idx] = True
        type_counts[ct] += 1
    
    return labels

def pseudo_label_from_markers(
    data: np.ndarray,
    gene_names: List[str],
    marker_dict: Dict[str, List[str]],
    type_to_idx: Dict[str, int],
    top_k: int = 100,
    normalize: bool = True,
    min_score: Optional[float] = None,
) -> torch.Tensor:
    n_cells = data.shape[0]
    labels = torch.full((n_cells,), -1, dtype=torch.long)

    scores = compute_marker_scores(data, gene_names, marker_dict, normalize)

    # Track which cells have been claimed (a cell can only get one label)
    claimed = np.zeros(n_cells, dtype=bool)

    # Assign labels, highest confidence first across all types
    # Build list of (score, cell_idx, type_idx) for all candidates
    candidates = []
    for cell_type, cell_scores in scores.items():
        if cell_type not in type_to_idx:
            print(f"  WARNING: {cell_type} not in type_to_idx, skipping")
            continue
        type_idx = type_to_idx[cell_type]
        for i in range(n_cells):
            if min_score is not None and cell_scores[i] < min_score:
                continue
            candidates.append((cell_scores[i], i, type_idx, cell_type))

    # Sort descending by score
    candidates.sort(key=lambda x: -x[0])

    # Assign top_k per type
    type_counts = {ct: 0 for ct in scores}
    labeled_count = 0

    for score, cell_idx, type_idx, cell_type in candidates:
        if claimed[cell_idx]:
            continue
        if type_counts[cell_type] >= top_k:
            continue

        labels[cell_idx] = type_idx
        claimed[cell_idx] = True
        type_counts[cell_type] += 1
        labeled_count += 1

    # Report
    print(f"\nPseudo-labeling summary ({labeled_count} cells labeled):")
    for cell_type in sorted(type_counts.keys()):
        count = type_counts[cell_type]
        if cell_type in scores:
            top_scores = np.sort(scores[cell_type])[-min(count, 5):][::-1]
            score_str = ", ".join(f"{s:.2f}" for s in top_scores)
            print(f"  {cell_type}: {count} cells (top scores: {score_str})")

    return labels
