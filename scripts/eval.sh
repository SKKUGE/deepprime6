#!/bin/bash

# Default values
EXPERIMENT="pe6a-DP-baseline"
CKPT_PATH="src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/*.pt"

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
            if [[ "$1" != *=* ]]; then
                if [ $POS_COUNT -eq 0 ]; then
                    EXPERIMENT="$1"
                    POS_COUNT=1
                elif [ $POS_COUNT -eq 1 ]; then
                    CKPT_PATH="$1"
                    POS_COUNT=2
                else
                    break
                fi
            else
                break
            fi
            ;;
    esac
    shift
done

echo "=================================================================="
echo "Starting evaluation for experiment: $EXPERIMENT"
echo "Using checkpoint: $CKPT_PATH"
if [ "$#" -gt 0 ]; then
    echo "Additional overrides: $@"
fi
echo "=================================================================="

python src/eval.py experiment="$EXPERIMENT" ckpt_path="$CKPT_PATH" "$@"

echo "Evaluation completed."
