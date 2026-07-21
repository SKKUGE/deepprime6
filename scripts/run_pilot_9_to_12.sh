#!/bin/bash
set -e

echo "=========================================="
echo " Starting Pilot Run Phases 9 to 12"
echo "=========================================="

export PYTHONPATH=$PYTHONPATH:$(pwd)
CONDA_ENV="/home/work/workdir/DeepPrime6/.conda/bin/python"

echo "------------------------------------------"
echo " Phase 9: Merge and Normalize Scores"
echo "------------------------------------------"
$CONDA_ENV src/pegrna_pipeline/09_merge_and_normalize_scores.py \
    --input data/pilot_output/pegrna_designs_pridict2_scored.parquet \
    --output-dir data/pilot_output/

echo "------------------------------------------"
echo " Phase 10: Select Final Candidates"
echo "------------------------------------------"
$CONDA_ENV src/pegrna_pipeline/10_select_candidates.py \
    --input data/pilot_output/pegrna_designs_scored_normalized.parquet \
    --config config/pipeline_config.yaml \
    --output-dir data/pilot_output/

echo "------------------------------------------"
echo " Phase 11: Independent Final Validation"
echo "------------------------------------------"
$CONDA_ENV src/pegrna_pipeline/11_independent_final_validation.py \
    --input data/pilot_output/final_candidates.parquet \
    --output-dir data/pilot_output/

echo "------------------------------------------"
echo " Phase 12: Export Cloning Ready Table"
echo "------------------------------------------"
$CONDA_ENV src/pegrna_pipeline/12_export_cloning_ready.py \
    --input data/pilot_output/final_candidates_validated.parquet \
    --output-dir data/pilot_output/

echo "=========================================="
echo " Pilot Run 9-12 Complete!"
echo "=========================================="
