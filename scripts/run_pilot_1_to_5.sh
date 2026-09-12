#!/bin/bash
set -e

echo "Starting Pilot Run (Phase 1-5) on first 500 variants..."

ENV_PYTHON="${ENV_PYTHON:-python}"
OUTDIR="data/pilot_output"
mkdir -p $OUTDIR

# Extract the header and first 500 lines for the pilot run
head -n 501 data/clinvar_fix.csv > data/clinvar_fix_pilot.csv

echo "Phase 1: Normalize Variants"
$ENV_PYTHON src/pegrna_pipeline/01_normalize_variants.py \
    --input data/clinvar_fix_pilot.csv \
    --fasta data/hg38.fa \
    --run-id "PILOT_01" \
    --output-dir $OUTDIR

echo "Phase 2: Build Contexts & Designability"
$ENV_PYTHON src/pegrna_pipeline/02_build_contexts_and_designability.py \
    --input $OUTDIR/normalized_variants.parquet \
    --fasta data/hg38.fa \
    --config config/pipeline_config.yaml \
    --output-dir $OUTDIR

echo "Phase 3: Select Mutation Panel"
# For the pilot, the panel might be smaller than 1000, so we update the target count via config override or just let it select everything available.
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

echo "Pilot run completed successfully!"
ls -lh $OUTDIR
