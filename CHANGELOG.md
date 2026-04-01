# Changelog

## [0.1.0] - 2026-04-01

### Added

- Initial project structure with `src/multitme/` package layout
- MultiModalCycleVAE model with cycle consistency, cell type alignment, and classification
- CLR and log1p preprocessing transforms
- Marker-based pseudo labeling for spatial data
- CLI entry points: `multitme-preprocess`, `multitme-train`, `multitme-infer`
- OmegaConf-based configuration system
- Nextflow pipeline with preprocess, train, and infer modules
- Dockerfile with CUDA support
- GitHub Actions CI and Docker workflows
- Unit tests for model, preprocessing, and pseudo labeling
