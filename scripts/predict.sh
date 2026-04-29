#!/bin/bash
# Generalized evaluation/inference script for DeepPrime6
# Usage: bash scripts/predict.sh <experiment_name> <ckpt_path> [hydra_overrides...]
#
# Examples:
#   bash scripts/predict.sh pe6a-DP-baseline logs/pe6a-DP-baseline/runs/2026-04-28_15-00-00/checkpoints/epoch_004.ckpt
#   bash scripts/predict.sh pe6a-DP-baseline logs/pe6a-DP-baseline/runs/2026-04-28_15-00-00/checkpoints/epoch_004.ckpt trainer=cpu

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <experiment_name> <ckpt_path> [additional_overrides...]"
    echo "Example: $0 pe6a-DP-baseline logs/pe6a-DP-baseline/runs/2026-04-28_15-00-00/checkpoints/epoch_004.ckpt trainer=gpu"
    exit 1
fi

EXPERIMENT=$1
CKPT_PATH=$2
shift 2

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CKPT_PATH"
    exit 1
fi

echo "=================================================================="
echo "Starting evaluation for experiment: $EXPERIMENT"
echo "Using checkpoint: $CKPT_PATH"
if [ "$#" -gt 0 ]; then
    echo "Additional overrides: $@"
fi
echo "=================================================================="

python src/eval.py experiment="$EXPERIMENT" ckpt_path="$CKPT_PATH" "$@"

echo "Evaluation completed."
