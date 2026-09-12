#!/bin/bash
set -e

echo "Starting Pilot Run (Phase 6-8) on Phase 5 outputs..."

ENV_PYTHON="/home/work/workdir/DeepPrime6/.conda/bin/python"
export LD_LIBRARY_PATH="/home/work/workdir/DeepPrime6/.conda/lib:$LD_LIBRARY_PATH"
export PATH="/home/work/workdir/DeepPrime6/.conda/bin:$PATH"
OUTDIR="data/pilot_output"

echo "Phase 6: Score DeepPrime(base)"
$ENV_PYTHON src/pegrna_pipeline/06_score_deepprime.py \
    --input $OUTDIR/pegrna_designs_qc_pass.parquet \
    --config config/pipeline_config.yaml \
    --model-versions config/model_versions.yaml \
    --output-dir $OUTDIR

echo "Phase 7: Score DeepPrime6"
$ENV_PYTHON src/pegrna_pipeline/07_score_deepprime6.py \
    --input $OUTDIR/pegrna_designs_dp_scored.parquet \
    --features $OUTDIR/pegrna_designs_biofeatures.parquet \
    --config config/pipeline_config.yaml \
    --model-versions config/model_versions.yaml \
    --output-dir $OUTDIR

echo "Phase 8: Score PRIDICT2"
$ENV_PYTHON src/pegrna_pipeline/08_score_pridict2.py \
    --input $OUTDIR/pegrna_designs_dp6_scored.parquet \
    --features $OUTDIR/pegrna_designs_biofeatures.parquet \
    --config config/pipeline_config.yaml \
    --model-versions config/model_versions.yaml \
    --output-dir $OUTDIR

echo "Phase 6-8 completed successfully!"
ls -lh $OUTDIR
