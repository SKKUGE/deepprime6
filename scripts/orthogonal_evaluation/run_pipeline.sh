#!/usr/bin/env bash
# ==============================================================================
# Script: run_pipeline.sh
# Purpose: Execute Full Orthogonal Benchmark Cross-Evaluation Pipeline
# Models: DeepPrime-base, PRIDICT2.0, DeepPrime6 (pe6a, pe6b, pe6c, pemaxdrnaseh)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect PROJECT_ROOT dynamically
curr="$SCRIPT_DIR"
while [ "$curr" != "/" ]; do
    if [ -f "$curr/.project-root" ] || ([ -d "$curr/PRIDICT2" ] && [ -d "$curr/src" ]); then
        PROJECT_ROOT="$curr"
        break
    fi
    curr="$(dirname "$curr")"
done

if [ -z "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi

cd "${PROJECT_ROOT}"

PYTHON_EXEC="${PYTHON_EXEC:-python}"

echo "========================================================================"
echo " Starting Orthogonal Benchmark Cross-Evaluation Pipeline"
echo " Project Root: ${PROJECT_ROOT}"
echo " Python: ${PYTHON_EXEC}"
echo " Timestamp: $(date)"
echo "========================================================================"

# Step 1: Data Filtering & Target Locus Stratification
echo ""
echo ">>> [Step 1/3] Filtering Benchmark Datasets (Strict Target Locus Stratification)..."
$PYTHON_EXEC "${SCRIPT_DIR}/01_filter_testsets.py"

# Step 2: Model Inference & Correlation Evaluation
echo ""
echo ">>> [Step 2/3] Running GPU Inference & Computing Spearman / Pearson Metrics..."
$PYTHON_EXEC "${SCRIPT_DIR}/02_run_inference.py"

# Step 3: Boxplots & Visualizations Generation
echo ""
echo ">>> [Step 3/3] Generating Boxplots and Correlation Figures..."
$PYTHON_EXEC "${SCRIPT_DIR}/03_plot_benchmark_results.py"

echo ""
echo "========================================================================"
echo " Pipeline Finished Successfully!"
echo " Outputs saved to:"
echo "   1) data/orthogonal_testset/"
echo "   2) notebooks/plots/orthogonal_benchmark/"
echo " Timestamp: $(date)"
echo "========================================================================"
