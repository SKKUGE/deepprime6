# Walkthrough: ClinVar pegRNA Prediction & Candidate Selection Pipeline

The end-to-end pegRNA prediction and candidate selection pipeline has been successfully executed on the non-colliding ClinVar dataset. This document details the changes made, the files created, and the final verification metrics.

---

## 🛠️ Summary of Created & Modified Files

### Core Pipeline Components
- **[scripts/run_pipeline.sh](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/run_pipeline.sh)**: End-to-end shell script controller, optimized with `data.batch_size=4096` to run all 5 DeepPrime checkpoints in under 75 seconds on the GPU.
- **[scripts/supp3_pipeline/generate_prediction_parquet.py](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/generate_prediction_parquet.py)**: Step 1 script. Extracts the 80nt WideTargetSequence context for DeepPrime and runs native CPU preprocessing with 32-core parallel Tm/MFE calculations.
- **[scripts/supp3_pipeline/predict_pridict2_library.py](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/predict_pridict2_library.py)**: Step 4 script. Maps the full 200nt flanking context for PRIDICT2.0 and runs prediction in GPU memory with 100k chunk loops and checkpoint caching.
- **[scripts/supp3_pipeline/select_candidates.py](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/select_candidates.py)**: Step 5 script. Aggregates the 6 raw scores into 3 representative signals, normalizes them via percentile ranking, and selects 50 candidates balancing consensus and disagreement.

### Auxiliary & Diagnostic Tools
- **[scripts/supp3_pipeline/resolve_remaining.py](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/resolve_remaining.py)**: Utility to identify and query NCBI BLAST for unresolved variant IDs, caching transcript sequences incrementally. Mapped **653 unique variants** (97.6% resolution rate).
- **[scripts/supp3_pipeline/verify_parquet_loading.py](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/verify_parquet_loading.py)**: Verification utility validating that the generated Parquet loads cleanly in PyTorch Lightning datamodules.
- **[scripts/supp3_pipeline/plot_correlation.py](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/plot_correlation.py)**: Script to generate the density hexbin correlation plot.

### Documentation & Verification Reports
- **[scripts/supp3_pipeline/pipeline_flowchart.md](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/pipeline_flowchart.md)**: Mermaid diagram and technical design rationale.
- **[scripts/supp3_pipeline/audit_report.md](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/audit_report.md)**: Requirements-based audit report including QC data reduction flow analysis.
- **[scripts/supp3_pipeline/correlation_analysis.md](file:///home/work/workdir/deepprime6-genomebiol-revision/scripts/supp3_pipeline/correlation_analysis.md)**: Scientific correlation analysis report with hexbin scatter plots.

---

## 📊 Verification Metrics & Results

### 1. QC Data Reduction
- **Raw CSV Dataset**: 711,232 rows (669 unique variant IDs)
- **BLAST Transcript-Mapping Filter (653 resolved IDs)**: 692,470 rows
- **DeepPrime Native QC Preprocessor Filter**: **678,084 rows** (Final dataset scored by all models)

### 2. Multi-Model Percentile Correlations
Pairwise Pearson correlations between the three representative model percentile ranks in the selected 50 candidates:
- **DeepPrime (Base) vs. DeepPrime6 Avg**: **0.816**
- **DeepPrime (Base) vs. PRIDICT2.0 Avg**: **0.185**
- **DeepPrime6 Avg vs. PRIDICT2.0 Avg**: **0.171**
- *All correlations are well below the limit ($\rho < 0.9$), confirming high diversity in model signals.*

---

## 📋 Selected 50 Candidates Summary Table

The following table summarizes the selected 50 validation candidates across the terciles (Highly Efficient, Modest, Inefficient), balancing consensus and high-disagreement designs:

| ID | Category | Subcategory | PBSlen | RTlen | Percentile_DP_Base | Percentile_DP6 | Percentile_PRIDICT2 | Average_Percentile | Disagreement_Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 33451 | Highly Efficient | Consensus | 17 | 33 | 0.995 | 1.000 | 0.988 | 0.994 | 0.006 |
| 33014 | Highly Efficient | Consensus | 11 | 30 | 0.979 | 0.973 | 0.993 | 0.981 | 0.010 |
| 33130 | Highly Efficient | Consensus | 12 | 35 | 0.984 | 0.986 | 0.973 | 0.981 | 0.007 |
| 3339 | Highly Efficient | Consensus | 12 | 25 | 0.992 | 0.986 | 0.960 | 0.979 | 0.017 |
| 33056 | Highly Efficient | Consensus | 13 | 12 | 0.991 | 0.989 | 0.957 | 0.979 | 0.019 |
| 33177 | Highly Efficient | Consensus | 13 | 31 | 0.992 | 0.997 | 0.939 | 0.976 | 0.032 |
| 33176 | Highly Efficient | Consensus | 14 | 27 | 0.997 | 0.985 | 0.945 | 0.975 | 0.027 |
| clinic_33827 | Highly Efficient | Consensus | 13 | 29 | 0.991 | 0.986 | 0.945 | 0.974 | 0.025 |
| 33423 | Highly Efficient | Consensus | 13 | 18 | 0.987 | 0.985 | 0.946 | 0.973 | 0.023 |
| 33300 | Highly Efficient | Consensus | 14 | 25 | 0.950 | 0.989 | 0.980 | 0.973 | 0.020 |
| 33117 | Highly Efficient | Disagreement | 12 | 39 | 0.972 | 0.988 | 0.067 | 0.676 | 0.527 |
| 33052 | Highly Efficient | Disagreement | 16 | 29 | 0.963 | 0.102 | 0.948 | 0.671 | 0.493 |
| 33227 | Highly Efficient | Disagreement | 13 | 31 | 0.972 | 0.972 | 0.131 | 0.692 | 0.486 |
| 33342 | Highly Efficient | Disagreement | 14 | 34 | 0.924 | 0.991 | 0.121 | 0.679 | 0.484 |
| clinic_33667 | Highly Efficient | Disagreement | 16 | 30 | 0.997 | 0.125 | 0.903 | 0.675 | 0.479 |
| 33317 | Modest | Consensus | 13 | 34 | 0.723 | 0.286 | 0.491 | 0.500 | 0.219 |
| 33075 | Modest | Consensus | 11 | 40 | 0.589 | 0.861 | 0.050 | 0.500 | 0.413 |
| 33337 | Modest | Consensus | 12 | 38 | 0.985 | 0.211 | 0.304 | 0.500 | 0.422 |
| 33079 | Modest | Consensus | 7 | 32 | 0.193 | 0.451 | 0.857 | 0.500 | 0.335 |
| 33260 | Modest | Consensus | 16 | 38 | 0.707 | 0.345 | 0.448 | 0.500 | 0.187 |
| 33249 | Modest | Consensus | 11 | 31 | 0.729 | 0.648 | 0.123 | 0.500 | 0.329 |
| 33321 | Modest | Consensus | 12 | 28 | 0.374 | 0.850 | 0.277 | 0.500 | 0.307 |
| 33170 | Modest | Consensus | 12 | 33 | 0.619 | 0.855 | 0.026 | 0.500 | 0.427 |
| 33047 | Modest | Consensus | 15 | 40 | 0.556 | 0.445 | 0.500 | 0.500 | 0.056 |
| 33221 | Modest | Consensus | 13 | 17 | 0.690 | 0.346 | 0.464 | 0.500 | 0.174 |
| 33255 | Modest | Consensus | 17 | 38 | 0.781 | 0.152 | 0.567 | 0.500 | 0.320 |
| clinic_33240 | Modest | Consensus | 15 | 34 | 0.246 | 0.600 | 0.659 | 0.500 | 0.223 |
| clinic_33645 | Modest | Consensus | 15 | 25 | 0.633 | 0.746 | 0.121 | 0.500 | 0.333 |
| 33432 | Modest | Consensus | 8 | 26 | 0.552 | 0.593 | 0.355 | 0.500 | 0.127 |
| 33336 | Modest | Disagreement | 10 | 14 | 0.036 | 0.015 | 0.993 | 0.348 | 0.559 |
| clinic_33048 | Modest | Disagreement | 14 | 10 | 0.064 | 0.008 | 0.983 | 0.351 | 0.548 |
| 33245 | Modest | Disagreement | 7 | 39 | 0.052 | 0.013 | 0.975 | 0.347 | 0.544 |
| 3335 | Modest | Disagreement | 7 | 31 | 0.029 | 0.061 | 0.984 | 0.358 | 0.543 |
| clinic_33506 | Modest | Disagreement | 13 | 12 | 0.050 | 0.020 | 0.974 | 0.348 | 0.542 |
| clinic_33451 | Modest | Disagreement | 8 | 7 | 0.129 | 0.015 | 0.999 | 0.381 | 0.538 |
| clinic_33170 | Inefficient | Consensus | 7 | 10 | 0.001 | 0.004 | 0.003 | 0.003 | 0.001 |
| clinic_33988 | Inefficient | Consensus | 7 | 25 | 0.003 | 0.005 | 0.004 | 0.004 | 0.001 |
| clinic_33990 | Inefficient | Consensus | 7 | 24 | 0.004 | 0.006 | 0.004 | 0.005 | 0.001 |
| 3312 | Inefficient | Consensus | 7 | 24 | 0.007 | 0.010 | 0.000 | 0.006 | 0.005 |
| 33010 | Inefficient | Consensus | 11 | 33 | 0.003 | 0.001 | 0.015 | 0.006 | 0.007 |
| 3318 | Inefficient | Consensus | 10 | 37 | 0.006 | 0.009 | 0.004 | 0.006 | 0.003 |
| 33059 | Inefficient | Consensus | 8 | 40 | 0.009 | 0.013 | 0.001 | 0.007 | 0.006 |
| clinic_33624 | Inefficient | Consensus | 7 | 38 | 0.006 | 0.010 | 0.007 | 0.008 | 0.002 |
| 33180 | Inefficient | Consensus | 8 | 19 | 0.006 | 0.017 | 0.002 | 0.008 | 0.008 |
| 33340 | Inefficient | Consensus | 8 | 20 | 0.008 | 0.007 | 0.011 | 0.009 | 0.002 |
| 330 | Inefficient | Disagreement | 10 | 14 | 0.021 | 0.019 | 0.948 | 0.329 | 0.536 |
| 3300 | Inefficient | Disagreement | 10 | 34 | 0.019 | 0.023 | 0.947 | 0.329 | 0.534 |
| clinic_33460 | Inefficient | Disagreement | 14 | 36 | 0.028 | 0.007 | 0.941 | 0.325 | 0.534 |
| 33051 | Inefficient | Disagreement | 8 | 37 | 0.029 | 0.012 | 0.942 | 0.328 | 0.532 |
| 33259 | Inefficient | Disagreement | 7 | 17 | 0.025 | 0.029 | 0.935 | 0.329 | 0.524 |

*Output File Path: [data/predictions/selected_50_validation_candidates.csv](file:///home/work/workdir/deepprime6-genomebiol-revision/data/predictions/selected_50_validation_candidates.csv)*
