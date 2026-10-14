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

EXP_DIR="${REPO_ROOT}/experiments/20261009_frt_small_library"
cd "${REPO_ROOT}"

PYTHON_EXEC="${PYTHON_EXEC:-python3}"
if [ -f "$REPO_ROOT/../DeepPrime6/.conda/bin/python" ]; then
    PYTHON_EXEC="$REPO_ROOT/../DeepPrime6/.conda/bin/python"
elif [ -f "$HOME/.conda/bin/python" ]; then
    PYTHON_EXEC="$HOME/.conda/bin/python"
fi

echo "========================================================================"
echo "=== Reproducing FRT Small Library Benchmark Pipeline ==="
echo "  Repo root: ${REPO_ROOT}"
echo "  Exp dir:   ${EXP_DIR}"
echo "  Python:    ${PYTHON_EXEC}"
echo "========================================================================"

# Step 1: Model Inference across 6 models
echo -e "\n>>> [Step 1/2] Running multi-model inference..."
$PYTHON_EXEC "${EXP_DIR}/scripts/01_predict_frt_small_library.py"

# Step 2: Visualization and distribution analysis
echo -e "\n>>> [Step 2/2] Generating distribution plots and correlation heatmaps..."
$PYTHON_EXEC "${EXP_DIR}/scripts/02_plot_frt_predictions.py" \
    --results_file "${EXP_DIR}/results/frt_small_lib_predictions.csv" \
    --plots_dir "${EXP_DIR}/plots"

echo -e "\n========================================================================"
echo " [Complete] Pipeline finished successfully!"
echo " Results: ${EXP_DIR}/results/"
echo " Plots:   ${EXP_DIR}/plots/"
echo "========================================================================"
