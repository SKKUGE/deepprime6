#!/bin/bash
# Generalized training script for DeepPrime6
# Usage: bash scripts/train.sh <experiment_name> [hydra_overrides...]
#
# Examples:
#   bash scripts/train.sh pe6a-DP-baseline
#   bash scripts/train.sh from-scratch/pe6a-DP-from-scratch
#   bash scripts/train.sh pe6a-DP-baseline trainer=cpu data.batch_size=128

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [additional_overrides...]"
    echo "Example: $0 pe6a-DP-baseline trainer=gpu"
    exit 1
fi

EXPERIMENT=$1
shift

LOG_NAME="${EXPERIMENT//\//_}"

echo "=================================================================="
echo "Starting training for experiment: $EXPERIMENT"
if [ "$#" -gt 0 ]; then
    echo "Additional overrides: $@"
fi
echo "=================================================================="

PYTHON_EXEC="${PYTHON_EXEC:-python}"
nohup "$PYTHON_EXEC" src/train.py experiment="$EXPERIMENT" "$@" > "${LOG_NAME}_train.out" 2>&1 &

echo "Training started. Check ${LOG_NAME}_train.out for progress."
