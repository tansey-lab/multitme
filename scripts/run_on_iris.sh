#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <scrna_h5ad> <xenium_h5ad> <output_dir> [conda_env_path]"
    echo ""
    echo "Arguments:"
    echo "  scrna_h5ad       Path to scRNA-seq h5ad file"
    echo "  xenium_h5ad      Path to Xenium h5ad file"
    echo "  output_dir       Path to output directory"
    echo "  conda_env_path   Path to conda environment (default: /home/quinnj2/.conda/multitme)"
    exit 1
}

if [[ $# -lt 3 ]]; then
    usage
fi

SCRNA="$1"
XENIUM="$2"
OUTPUT_DIR="$3"
CONDA_ENV="${4:-/home/quinnj2/.conda/multitme}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")/nextflow"

# Create temporary samplesheet
SAMPLE_NAME=$(basename "$SCRNA" | sed 's/\.h5ad$//')
SAMPLESHEET=$(mktemp --suffix=.csv)
echo "sample,scrna,xenium" > "$SAMPLESHEET"
echo "${SAMPLE_NAME},${SCRNA},${XENIUM}" >> "$SAMPLESHEET"

WANDB_ARGS=""
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ARGS="--wandb_enabled true --wandb_api_key $WANDB_API_KEY"
fi

echo "Running MultiTME pipeline on iris"
echo "  scRNA:  $SCRNA"
echo "  Xenium: $XENIUM"
echo "  Output: $OUTPUT_DIR"
echo "  Conda:  $CONDA_ENV"
echo "  Samplesheet: $SAMPLESHEET"
[[ -n "$WANDB_ARGS" ]] && echo "  W&B:    enabled"

nextflow run "$PIPELINE_DIR" \
    -w "$OUTPUT_DIR"/nf \
    -profile mskcc_iris,conda \
    --input "$SAMPLESHEET" \
    --outdir "$OUTPUT_DIR" \
    --conda_env_path "$CONDA_ENV" \
    $WANDB_ARGS \
    -resume

rm -f "$SAMPLESHEET"
