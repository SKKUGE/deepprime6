#!/bin/bash
# Exit on error
set -e

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

if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi

EXP_DIR="$PROJECT_ROOT/experiments/20261008_panelB_base_architecture_revalidation"
cd "$EXP_DIR"

PYTHON_EXEC="${PYTHON_EXEC:-python3}"
if [ -f "$PROJECT_ROOT/../DeepPrime6/.conda/bin/python" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/../DeepPrime6/.conda/bin/python"
elif [ -f "$HOME/.conda/bin/python" ]; then
    PYTHON_EXEC="$HOME/.conda/bin/python"
fi

# Parse arguments
FRESH_START=false
for arg in "$@"; do
    if [ "$arg" == "--fresh" ]; then
        FRESH_START=true
    fi
done

echo "=== Reproducing Track A Benchmark Figures and Analysis ==="
echo "Project root: $PROJECT_ROOT"
echo "Working directory: $EXP_DIR"
echo "Python interpreter: $PYTHON_EXEC"

if [ "$FRESH_START" = true ]; then
    echo "--------------------------------------------------------"
    echo "[--fresh detected] Re-running all model inferences from scratch..."
    echo "--------------------------------------------------------"
    
    if [ -f "$PROJECT_ROOT/scripts/evaluation/run_all_track_a_predictions.py" ]; then
        echo "Running fresh evaluation on deduplicated test library..."
        $PYTHON_EXEC "$PROJECT_ROOT/scripts/evaluation/run_all_track_a_predictions.py"
        $PYTHON_EXEC "$PROJECT_ROOT/scripts/generate_scratch_predictions.py"
        echo "All fresh inferences complete and synced to predictions/!"
    else
        echo "Error: Repo root with full model weights not found for --fresh run."
        exit 1
    fi
    echo "--------------------------------------------------------"
fi

echo "Running the visualization script to generate predictive modelling capacity figures:"
echo "- Panel B (Model comparison: DeepPrime-Base vs PRIDICT2.0 vs From Scratch vs DeepPrime6)"
echo "- Panel C (Edit type comparison for DeepPrime6)"
echo "- Boxplots, pairplots, and correlation heatmaps"
echo "--------------------------------------------------------"

$PYTHON_EXEC "$EXP_DIR/scripts/run_track_a_analysis.py"

echo "--------------------------------------------------------"
echo "Analysis and plotting complete!"
echo "Outputs are saved in panels/, figures/, and results/ (and mirrored to notebooks/plots/)."
