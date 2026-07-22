#!/bin/bash
set -e

echo "Starting Full Run (Phase 1-13) on entire clinvar_fix.csv..."

OUTDIR="data/full_output"
mkdir -p $OUTDIR
ENV_PYTHON="/home/work/workdir/DeepPrime6/.conda/bin/python"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LD_LIBRARY_PATH="/home/work/workdir/DeepPrime6/.conda/lib:$LD_LIBRARY_PATH"
export PATH="/home/work/workdir/DeepPrime6/.conda/bin:$PATH"

echo "Phase 1: Normalize Variants"
$ENV_PYTHON src/pegrna_pipeline/01_normalize_variants.py \
    --input data/clinvar_fix.csv \
    --fasta data/hg38.fa \
    --run-id "FULL_RUN_01" \
    --output-dir $OUTDIR

echo "Phase 2: Build Contexts & Designability"
$ENV_PYTHON src/pegrna_pipeline/02_build_contexts_and_designability.py \
    --input $OUTDIR/normalized_variants.parquet \
    --fasta data/hg38.fa \
    --config config/pipeline_config.yaml \
    --output-dir $OUTDIR

echo "Phase 3: Select Mutation Panel"
$ENV_PYTHON src/pegrna_pipeline/03_select_mutation_panel.py \
    --input $OUTDIR/variants_designability.parquet \
    --config config/pipeline_config.yaml \
    --output-dir $OUTDIR

echo "Phase 4: Design pegRNAs"
$ENV_PYTHON src/pegrna_pipeline/04_design_pegrnas.py \
    --input $OUTDIR/mutation_panel_primary.parquet \
    --config config/pipeline_config.yaml \
    --output-dir $OUTDIR

echo "Phase 5: Validate pegRNA Designs"
$ENV_PYTHON src/pegrna_pipeline/05_validate_pegrna_designs.py \
    --input $OUTDIR/pegrna_designs_raw.parquet \
    --output-dir $OUTDIR

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

echo "Phase 9: Merge and Normalize Scores"
$ENV_PYTHON src/pegrna_pipeline/09_merge_and_normalize_scores.py \
    --input $OUTDIR/pegrna_designs_pridict2_scored.parquet \
    --output-dir $OUTDIR/

echo "Phase 10: Select Final Candidates"
$ENV_PYTHON src/pegrna_pipeline/10_select_candidates.py \
    --input $OUTDIR/pegrna_designs_scored_normalized.parquet \
    --config config/pipeline_config.yaml \
    --output-dir $OUTDIR/

echo "Phase 11: Independent Final Validation"
$ENV_PYTHON src/pegrna_pipeline/11_independent_final_validation.py \
    --input $OUTDIR/final_candidates.parquet \
    --output-dir $OUTDIR/

echo "Phase 12: Export Cloning Ready Table"
$ENV_PYTHON src/pegrna_pipeline/12_export_cloning_ready.py \
    --input $OUTDIR/final_candidates_validated.parquet \
    --output-dir $OUTDIR/

echo "Phase 13: Plot Model Score Scatter"
$ENV_PYTHON src/pegrna_pipeline/13_plot_score_scatter.py \
    --input $OUTDIR/pegrna_designs_scored_normalized.parquet \
    --output-dir $OUTDIR/

echo "Full Run Completed Successfully!"
