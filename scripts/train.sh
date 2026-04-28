#!/bin/bash
# Generalized training script for DeepPrime6
# Usage: bash scripts/train.sh <experiment_name> [hydra_overrides...]
#
# Examples:
#   bash scripts/train.sh GEI-13/pe6a-DP-baseline
#   bash scripts/train.sh GEI-13/pe6a-DP-baseline trainer=cpu data.batch_size=128

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [additional_overrides...]"
    echo "Example: $0 GEI-13/pe6a-DP-baseline trainer=gpu"
    exit 1
fi

EXPERIMENT=$1
shift

echo "=================================================================="
echo "Starting training for experiment: $EXPERIMENT"
if [ "$#" -gt 0 ]; then
    echo "Additional overrides: $@"
fi
echo "=================================================================="

python src/train.py experiment="$EXPERIMENT" "$@"

echo "Training completed."
