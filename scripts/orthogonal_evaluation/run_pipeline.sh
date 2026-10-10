#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/orthogonal_evaluation/run_pipeline.sh
# Purpose: Execute Full Orthogonal Benchmark Cross-Evaluation Pipeline
# Models: DeepPrime-base, PRIDICT2.0, DeepPrime6 (pe6a, pe6b, pe6c, pemaxdrnaseh)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../.." && pwd))"

cd "${PROJECT_ROOT}"

echo "========================================================================"
echo " Starting Orthogonal Benchmark Cross-Evaluation Pipeline"
echo " Project Root: ${PROJECT_ROOT}"
echo " Timestamp: $(date)"
echo "========================================================================"

# Step 1: Data Filtering & Target Locus Stratification
echo ""
echo ">>> [Step 1/3] Filtering Benchmark Datasets (Strict Target Locus Stratification)..."
python "${SCRIPT_DIR}/01_filter_testsets.py"

# Step 2: Model Inference & Correlation Evaluation
echo ""
echo ">>> [Step 2/3] Running GPU Inference & Computing Spearman / Pearson Metrics..."
python "${SCRIPT_DIR}/02_run_inference.py"

# Step 3: Boxplots & Visualizations Generation
echo ""
echo ">>> [Step 3/3] Generating Boxplots and Correlation Figures..."
python "${SCRIPT_DIR}/03_plot_benchmark_results.py"

echo ""
echo "========================================================================"
echo " Pipeline Finished Successfully!"
echo " Outputs saved to:"
echo "   1) data/orthogonal_testset/"
echo "   2) experiments/20261009_orthogonal_benchmark_leakage_stratification/"
echo "   3) notebooks/plots/orthogonal_benchmark/"
echo " Timestamp: $(date)"
echo "========================================================================"
