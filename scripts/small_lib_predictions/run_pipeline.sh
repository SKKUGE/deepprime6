#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect REPO_ROOT dynamically
curr="$SCRIPT_DIR"
while [ "$curr" != "/" ]; do
    if [ -f "$curr/.project-root" ] || ([ -d "$curr/PRIDICT2" ] && [ -d "$curr/src" ]); then
        REPO_ROOT="$curr"
        break
    fi
    curr="$(dirname "$curr")"
done

if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi

OUT_DIR="${REPO_ROOT}/data/frt-library-prediction"
PLOTS_DIR="${REPO_ROOT}/notebooks/plots/frt_small_library"
cd "${REPO_ROOT}"

PYTHON_EXEC="${PYTHON_EXEC:-python}"

echo "========================================================================"
echo "=== Reproducing FRT Small Library Benchmark Pipeline ==="
echo "  Repo root: ${REPO_ROOT}"
echo "  Python:    ${PYTHON_EXEC}"
echo "========================================================================"

# Step 1: Model Inference across 6 models
echo -e "\n>>> [Step 1/2] Running multi-model inference..."
$PYTHON_EXEC "${SCRIPT_DIR}/01_predict_frt_small_library.py"

# Step 2: Visualization and distribution analysis
echo -e "\n>>> [Step 2/2] Generating distribution plots and correlation heatmaps..."
$PYTHON_EXEC "${SCRIPT_DIR}/02_plot_frt_predictions.py" \
    --results_file "${OUT_DIR}/frt_small_lib_predictions.csv" \
    --plots_dir "${PLOTS_DIR}"

echo -e "\n========================================================================"
echo " [Complete] Pipeline finished successfully!"
echo " Results: ${OUT_DIR}/"
echo " Plots:   ${PLOTS_DIR}/"
echo "========================================================================"
