# multitme

Multimodal CycleVAE for tumor microenvironment cell typing across scRNA-seq and spatial transcriptomics (Xenium).

## Overview

**multitme** trains a cycle-consistent variational autoencoder that jointly embeds scRNA-seq and Xenium spatial transcriptomics data into a shared latent space. This enables:

- **Cell type transfer** from annotated scRNA-seq to unannotated spatial data
- **Cross-modality translation** (e.g., impute full transcriptome from spatial panels)
- **Joint latent embeddings** for downstream analysis

## Installation

```bash
# Clone and install with uv
git clone https://github.com/your-org/multitme.git
cd multitme
uv sync --group dev
```

## CLI Usage

### Preprocess

```bash
multitme-preprocess --config configs/example.yaml
```

### Train

```bash
multitme-train --config configs/example.yaml
```

Override any parameter via dot notation:

```bash
multitme-train --config configs/example.yaml training.n_epochs=100 model.n_latent=32
```

### Infer

```bash
multitme-infer \
    --model-dir results/ \
    --input new_xenium.h5ad \
    --modality xenium \
    --output-dir predictions/
```

## Nextflow Pipeline

For production-scale runs with GPU support and checkpointing:

```bash
cd nextflow
nextflow run main.nf \
    -profile docker,gpu \
    --input samplesheet.csv \
    --outdir results
```

## Development

```bash
make dev          # Install deps + pre-commit hooks
make test         # Run pytest
make lint         # Run ruff
make format       # Auto-format
```

## Project Structure

```
src/multitme/
├── __init__.py
├── config.py              # OmegaConf config loading
├── utils.py               # Device selection, seeding
├── model/
│   ├── __init__.py
│   └── cycle_vae.py       # MultiModalCycleVAE, CyclingDataset, CycleVAETrainer
├── data/
│   ├── __init__.py
│   ├── preprocessing.py   # log1p / CLR transforms
│   └── pseudo_labels.py   # Marker-based pseudo labeling
└── cli/
    ├── __init__.py
    ├── preprocess.py       # multitme-preprocess
    ├── train.py            # multitme-train
    └── infer.py            # multitme-infer
```
