#!/bin/bash
# Exit on error
set -e

# Automatically move to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
CONDA_ENV="$PROJECT_ROOT/../DeepPrime6/.conda"  # Change if necessary
PYTHON_EXEC="$CONDA_ENV/bin/python"

# Fallback if conda python doesn't exist
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python"
fi

# Parse arguments
FRESH_START=false
for arg in "$@"; do
    if [ "$arg" == "--fresh" ]; then
        FRESH_START=true
    fi
done

echo "=== Reproducing Track A Benchmark Figures and Analysis ==="

if [ "$FRESH_START" = true ]; then
    echo "--------------------------------------------------------"
    echo "[--fresh detected] Re-running all model inferences from scratch..."
    echo "--------------------------------------------------------"
    
    echo "1/4: Converting Parquet to CSV"
    $PYTHON_EXEC -c "
import pandas as pd
df = pd.read_parquet('data/rq3_test_library.parquet')
df.to_csv('data/rq3_test_library.csv', index=False)
"

    echo "2/4: PRIDICT2.0 Score Extraction"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    $PYTHON_EXEC src/utils/predict_pridict2.py
    
    echo "3/4: DeepPrime-Base Predictions"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    $PYTHON_EXEC src/predict.py \
      experiment=pe6a-DP-baseline \
      trainer.accelerator=cpu \
      data.dataloader.num_workers=0 \
      data.skip_preprocessing=true \
      data.csv_path=data/rq3_test_library.csv \
      model.prediction_save_path=data/rq3_predictions_dp_base.csv \
      ckpt_path=null
      
    echo "4/4: DeepPrime6 Predictions (Iterating over Domains)"
    checkpoints=(
      "pe6a_mainft.ckpt:data/rq3_predictions_dp6.csv"
      "pe6b_mainft.ckpt:data/rq3_predictions_dp6_pe6b.csv"
      "pe6c_mainft.ckpt:data/rq3_predictions_dp6_pe6c.csv"
      "pemaxdrnaseh_mainft.ckpt:data/rq3_predictions_dp6_drnaseh.csv"
    )
    for pair in "${checkpoints[@]}"; do
      ckpt="${pair%%:*}"
      output="${pair##*:}"
      echo "   Evaluating model checkpoint $ckpt -> $output"
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      $PYTHON_EXEC src/predict.py \
        experiment=pe6a-DP-baseline \
        trainer.accelerator=gpu \
        data.dataloader.num_workers=0 \
        data.skip_preprocessing=true \
        data.csv_path=data/rq3_test_library.csv \
        model.prediction_save_path=$output \
        ckpt_path=src/models/weights/DeepPrime6-weights/$ckpt
    done
    echo "All fresh inferences complete!"
    echo "--------------------------------------------------------"
fi

echo "=== Reproducing Track A Benchmark Figures and Analysis ==="
echo "Running the visualization script to generate predictive modelling capacity figures:"
echo "- Panel B (Model comparison)"
echo "- Panel C (Edit type comparison for DeepPrime6)"
echo "- Boxplots, pairplots, and correlation heatmaps"
echo "--------------------------------------------------------"

$PYTHON_EXEC src/utils/run_track_a_analysis.py

echo "--------------------------------------------------------"
echo "Analysis and plotting complete!"
echo "Outputs (PNG and raw CSVs) are saved in notebooks/plots/"
