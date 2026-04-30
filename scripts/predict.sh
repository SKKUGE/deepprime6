#!/bin/bash

# Default values
EXPERIMENT="pe6a-DP-baseline"
CKPT_PATH="src/models/weights/DeepPrime6-weights/pe6a_mainft.ckpt"

# Function to display help
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --experiment <name>    Hydra experiment name (default: $EXPERIMENT)"
    echo "  --ckpt <path>         Path to checkpoint (default: $CKPT_PATH)"
    echo "  --help                Show this help message"
    echo ""
    echo "Any additional arguments will be passed to hydra."
}

# Parse arguments
POS_COUNT=0
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment) EXPERIMENT="$2"; shift ;;
        --ckpt) CKPT_PATH="$2"; shift ;;
        --help) show_help; exit 0 ;;
        -*) echo "Unknown option: $1"; show_help; exit 1 ;;
        *) 
            # If it doesn't start with --, and is not a hydra override (no =),
            # treat first positional as experiment and second as checkpoint.
            if [[ "$1" != *=* ]]; then
                if [ $POS_COUNT -eq 0 ]; then
                    EXPERIMENT="$1"
                    POS_COUNT=1
                elif [ $POS_COUNT -eq 1 ]; then
                    CKPT_PATH="$1"
                    POS_COUNT=2
                else
                    break # Pass remaining to hydra
                fi
            else
                break # Pass remaining to hydra (includes key=value pairs)
            fi
            ;;
    esac
    shift
done

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CKPT_PATH"
    exit 1
fi

echo "=================================================================="
echo "Starting inference for experiment: $EXPERIMENT"
echo "Using checkpoint: $CKPT_PATH"
if [ "$#" -gt 0 ]; then
    echo "Additional overrides: $@"
fi
echo "=================================================================="

python src/predict.py experiment="$EXPERIMENT" ckpt_path="$CKPT_PATH" "$@"

echo "Inference completed."
