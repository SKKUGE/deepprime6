#!/bin/bash
# Exit on error
set -e

PROJECT_ROOT=$(pwd)
CONDA_ENV="$PROJECT_ROOT/../DeepPrime6/.conda"  # Change if necessary
PYTHON_EXEC="$CONDA_ENV/bin/python"

echo "=== Step 1: Converting Parquet to CSV ==="
$PYTHON_EXEC -c "
import pandas as pd
df = pd.read_parquet('data/rq3_test_library.parquet')
df.to_csv('data/rq3_test_library.csv', index=False)
"
echo "CSV conversion complete: data/rq3_test_library.csv"

echo "=== Step 2: PRIDICT2.0 Score Extraction ==="
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
$PYTHON_EXEC src/utils/predict_pridict2.py
echo "PRIDICT2.0 score extraction complete: data/rq3_predictions_pridict2.csv"

echo "=== Step 3: DeepPrime-Base predictions ==="
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
$PYTHON_EXEC src/predict.py \
  experiment=pe6a-DP-baseline \
  trainer.accelerator=cpu \
  data.dataloader.num_workers=0 \
  data.skip_preprocessing=true \
  data.csv_path=data/rq3_test_library.csv \
  model.prediction_save_path=data/rq3_predictions_dp_base.csv \
  ckpt_path=null
echo "DeepPrime-base predictions complete: data/rq3_predictions_dp_base.csv"

echo "=== Step 4: DeepPrime6 predictions ==="
# We run predictions sequentially using domain-specific checkpoints on GPU
checkpoints=(
  "pe6a_mainft.ckpt:data/rq3_predictions_dp6.csv"
  "pe6b_mainft.ckpt:data/rq3_predictions_dp6_pe6b.csv"
  "pe6c_mainft.ckpt:data/rq3_predictions_dp6_pe6c.csv"
  "pemaxdrnaseh_mainft.ckpt:data/rq3_predictions_dp6_drnaseh.csv"
)

for pair in "${checkpoints[@]}"; do
  ckpt="${pair%%:*}"
  output="${pair##*:}"
  
  echo "Evaluating model checkpoint $ckpt -> $output"
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
echo "DeepPrime6 predictions complete."

echo "=== Step 5: Merging correlations and Generating Plots ==="
$PYTHON_EXEC src/utils/run_track_a_analysis.py
echo "Plotting and correlation extraction complete."
echo "Workflow completed successfully!"
