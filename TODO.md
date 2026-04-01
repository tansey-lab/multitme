# multitme Project Setup TODOs

Organizing this project to match the structure of `~/Code/ebbf`.

---

## Python Package Structure

- [ ] Create `src/multitme/` package directory
- [ ] Move `multitme_simple_sc_xenium.py` → `src/multitme/model/cycle_vae.py`
- [ ] Move `pseudo_label_markers.py` → `src/multitme/data/pseudo_labels.py`
- [ ] Extract preprocessing logic → `src/multitme/data/preprocessing.py`
- [ ] Create `src/multitme/cli/` with entry points:
  - [ ] `preprocess.py` - data loading and CLR transform
  - [ ] `train.py` - model training
  - [ ] `infer.py` - cell type prediction on new data
- [ ] Create `src/multitme/__init__.py` with `__version__`
- [ ] Create `src/multitme/config.py` for OmegaConf config loading
- [ ] Create `src/multitme/utils.py` for shared utilities
- [ ] Delete `celltyping.py` after refactoring into CLI

---

## pyproject.toml

- [ ] Create `pyproject.toml` with:
  - [ ] `[build-system]` using setuptools
  - [ ] `[project]` metadata (name, version, description, authors, keywords)
  - [ ] `requires-python = ">=3.12"`
  - [ ] `dependencies` list (torch, numpy, scanpy, etc.)
  - [ ] `[project.optional-dependencies]` for `dev` group
  - [ ] `[project.scripts]` entry points:
    - `multitme-preprocess`
    - `multitme-train`
    - `multitme-infer`
  - [ ] `[tool.setuptools.packages.find]` with `where = ["src"]`
  - [ ] `[tool.ruff]` config (line-length, target-version, lint rules)
  - [ ] `[tool.pytest.ini_options]`
  - [ ] `[dependency-groups]` for uv

---

## uv Setup

- [ ] Run `uv init` or create `pyproject.toml` manually
- [ ] Run `uv sync --group dev` to generate `uv.lock`
- [ ] Create `.venv/` via uv
- [ ] Add `.python-version` file (3.12)

---

## Pre-commit

- [ ] Create `.pre-commit-config.yaml` with:
  - [ ] `pytest` hook (local, `uv run pytest`)
  - [ ] `ruff` hook (lint + fix)
  - [ ] `ruff-format` hook
  - [ ] `prettier` hook for YAML/JSON/MD
  - [ ] `trailing-whitespace` hook
  - [ ] `end-of-file-fixer` hook
- [ ] Run `uv run pre-commit install`

---

## Tests

- [ ] Create `tests/` directory
- [ ] Create `tests/__init__.py`
- [ ] Create `tests/test_model.py` - unit tests for CycleVAE
- [ ] Create `tests/test_pseudo_labels.py` - test marker scoring
- [ ] Create `tests/test_preprocessing.py` - test CLR transform

---

## GitHub Actions

- [ ] Create `.github/workflows/ci.yml`:
  - [ ] Trigger on push to main, PRs, tags
  - [ ] `lint-and-test` job:
    - [ ] `actions/checkout@v4`
    - [ ] `astral-sh/setup-uv@v5` with cache
    - [ ] `uv sync --group dev`
    - [ ] `uv run ruff check .`
    - [ ] `uv run ruff format --check .`
    - [ ] `uv run pytest` (optional)
- [ ] Create `.github/workflows/docker.yml` for container builds

---

## Nextflow Pipeline

- [ ] Create `nextflow/` directory with nf-core structure
- [ ] Create `nextflow/main.nf` - pipeline entry point
- [ ] Create `nextflow/nextflow.config` - default config
- [ ] Create `nextflow/workflows/multitme.nf` - main workflow
- [ ] Create `nextflow/modules/local/`:
  - [ ] `preprocess.nf` - load h5ad, CLR transform
  - [ ] `train.nf` - train CycleVAE with checkpointing
  - [ ] `infer.nf` - predict cell types
- [ ] Create `nextflow/conf/`:
  - [ ] `base.config` - default resources
  - [ ] `test.config` - test profile
  - [ ] `modules.config` - module-specific args
- [ ] Create `nextflow/assets/`:
  - [ ] `schema_input.json` - samplesheet schema
  - [ ] `multiqc_config.yml`
- [ ] Create `nextflow/nextflow_schema.json` - parameter schema
- [ ] Add GPU support (`label 'process_gpu'`)
- [ ] Add checkpoint resume logic (beforeScript/afterScript)
- [ ] Add wandb integration for experiment tracking
- [ ] Create `nextflow/.pre-commit-config.yaml` for nf-core linting

---

## Docker

- [ ] Create `Dockerfile`:
  - [ ] Base image with CUDA support
  - [ ] Install uv
  - [ ] Copy and install package
  - [ ] Set entry point
- [ ] Create `.dockerignore`
- [ ] Add Docker build to GitHub Actions

---

## Config Management

- [ ] Create `configs/` directory for example configs
- [ ] Create `configs/example.yaml` with all training params
- [ ] Use OmegaConf for config loading in CLI

---

## Makefile

- [ ] Create `Makefile` with targets:
  - [ ] `dev` - full dev setup
  - [ ] `install-uv`
  - [ ] `install-deps` (`uv sync --group dev`)
  - [ ] `install-hooks` (`uv run pre-commit install`)
  - [ ] `install-nextflow`
  - [ ] `install-nf-test`
  - [ ] `test-nf` - run nextflow integration tests

---

## Documentation

- [ ] Create `README.md` with:
  - [ ] Project description
  - [ ] Installation instructions
  - [ ] CLI usage examples
  - [ ] Nextflow usage
- [ ] Create `CHANGELOG.md`

---

## Git/Misc

- [ ] Create comprehensive `.gitignore`:
  - [ ] Python: `__pycache__/`, `*.egg-info/`, `.venv/`
  - [ ] Data: `*.h5ad`, `*.npy`, `results/`
  - [ ] Nextflow: `work/`, `.nextflow/`, `.nextflow.log*`
  - [ ] IDE: `.vscode/`, `.idea/`
- [ ] Create `version.config` for centralized versioning
- [ ] Create `environment.yml` for conda fallback
- [ ] Create `samplesheets/` with example input CSVs
