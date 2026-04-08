#!/usr/bin/env bash
# Integration test: run the full nextflow pipeline on test_data with 2 epochs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NF_DIR="$PROJECT_DIR/nextflow"

# Local venv so multitme-* CLIs are on PATH (non-container runs)
_VENV="${MULTITME_VENV:-$PROJECT_DIR/.venv}"
if [[ -f "$_VENV/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$_VENV/bin/activate"
  echo "Using virtualenv: $_VENV"
elif [[ -f "$PROJECT_DIR/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$PROJECT_DIR/venv/bin/activate"
  echo "Using virtualenv: $PROJECT_DIR/venv"
fi

# Create a test samplesheet pointing at test_data
SAMPLESHEET=$(mktemp /tmp/multitme_test_samplesheet_XXXXXX).csv
cat > "$SAMPLESHEET" <<EOF
sample,scrna,xenium
test_sample,${PROJECT_DIR}/test_data/scRNA_sample.h5ad,${PROJECT_DIR}/test_data/xenium_sample.h5ad
EOF

echo "=== MultiTME integration test ==="
echo "Samplesheet: $SAMPLESHEET"
cat "$SAMPLESHEET"
echo ""

nextflow run "$NF_DIR" \
    --input "$SAMPLESHEET" \
    --outdir "$PROJECT_DIR/test_output" \
    --n_epochs 2 \
    --skip_report false \
    -profile test \
    -work-dir "$PROJECT_DIR/test_work" \
    "$@"

echo ""
echo "=== Test complete. Results in $PROJECT_DIR/test_output ==="
