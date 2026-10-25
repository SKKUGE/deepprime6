#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting Pilot Run (Phase 6-8) on Phase 5 outputs..."

ENV_PYTHON="${ENV_PYTHON:-${PYTHON_EXEC:-python}}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
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
