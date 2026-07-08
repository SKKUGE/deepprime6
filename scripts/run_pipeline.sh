#!/bin/bash

# Exit on error
set -e

# Data directories (can be overridden by environment variables)
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/predictions}"

mkdir -p "$OUTPUT_DIR"

echo "=== Starting end-to-end pegRNA Prediction and Candidate Selection Pipeline ==="
echo "Input directory (DATA_DIR): $DATA_DIR"
echo "Output directory (OUTPUT_DIR): $OUTPUT_DIR"

PYTHON_BIN="/home/work/workdir/DeepPrime6/.conda/bin/python"

# Step 1: Preprocess raw ClinVar pegRNA library using resolved genomic details
echo "Step 1: Running parallel target sequence reconstruction and feature preprocessing..."
$PYTHON_BIN scripts/supp3_pipeline/generate_prediction_parquet.py --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"

# Step 2: Run DeepPrime (Base) model prediction on GPU
echo "Step 2: Running DeepPrime (Base) GPU prediction..."
$PYTHON_BIN src/predict.py \
    trainer=gpu \
    data.csv_path="$OUTPUT_DIR/preprocessed_pegrna_prediction.parquet" \
    ckpt_path=null \
    model.prediction_save_path="$OUTPUT_DIR/pegrna_predictions.csv" \
    data.skip_preprocessing=True \
    data.batch_size=4096

# Step 3: Run DeepPrime6 fine-tuned checkpoints (PE6a, PE6b, PE6c, PEmax-dRNaseH)
echo "Step 3a: Running DeepPrime6 PE6a GPU prediction..."
$PYTHON_BIN src/predict.py \
    trainer=gpu \
    data.csv_path="$OUTPUT_DIR/preprocessed_pegrna_prediction.parquet" \
    ckpt_path=src/models/weights/DeepPrime6-weights/pe6a_mainft.ckpt \
    data.datafilter.PE_types='["PE6a(+PEmaxCas9)"]' \
    model.prediction_save_path="$OUTPUT_DIR/predictions_pe6a.csv" \
    data.skip_preprocessing=True \
    data.batch_size=4096

echo "Step 3b: Running DeepPrime6 PE6b GPU prediction..."
$PYTHON_BIN src/predict.py \
    trainer=gpu \
    data.csv_path="$OUTPUT_DIR/preprocessed_pegrna_prediction.parquet" \
    ckpt_path=src/models/weights/DeepPrime6-weights/pe6b_mainft.ckpt \
    data.datafilter.PE_types='["PE6b(+PEmaxCas9)"]' \
    model.prediction_save_path="$OUTPUT_DIR/predictions_pe6b.csv" \
    data.skip_preprocessing=True \
    data.batch_size=4096

echo "Step 3c: Running DeepPrime6 PE6c GPU prediction..."
$PYTHON_BIN src/predict.py \
    trainer=gpu \
    data.csv_path="$OUTPUT_DIR/preprocessed_pegrna_prediction.parquet" \
    ckpt_path=src/models/weights/DeepPrime6-weights/pe6c_mainft.ckpt \
    data.datafilter.PE_types='["PE6c(+PEmaxCas9)"]' \
    model.prediction_save_path="$OUTPUT_DIR/predictions_pe6c.csv" \
    data.skip_preprocessing=True \
    data.batch_size=4096

echo "Step 3d: Running DeepPrime6 PEmaxdRNaseH GPU prediction..."
$PYTHON_BIN src/predict.py \
    trainer=gpu \
    data.csv_path="$OUTPUT_DIR/preprocessed_pegrna_prediction.parquet" \
    ckpt_path=src/models/weights/DeepPrime6-weights/pemaxdrnaseh_mainft.ckpt \
    data.datafilter.PE_types='["PEmaxdRNaseH"]' \
    model.prediction_save_path="$OUTPUT_DIR/predictions_pemaxdrnaseh.csv" \
    data.skip_preprocessing=True \
    data.batch_size=4096

# Step 4: Run high-throughput PRIDICT2.0 model prediction on GPU
echo "Step 4: Running parallelized PRIDICT2.0 feature extraction & GPU prediction..."
$PYTHON_BIN scripts/supp3_pipeline/predict_pridict2_library.py --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"

# Step 5: Perform percentile normalization, consensus/disagreement candidate selection, and verification
echo "Step 5: Running validation candidate selection..."
$PYTHON_BIN scripts/supp3_pipeline/select_candidates.py --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"

echo "=== Pipeline Completed Successfully! ==="
