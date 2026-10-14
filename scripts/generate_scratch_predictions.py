#!/usr/bin/env python3
"""
Generate predictions for From Scratch models on the Main Library Test Set (rq3_test_library_dedup.parquet)
using the canonical codebase inference engine (src.utils.inference).
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def find_project_root():
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, ".project-root")) or (
            os.path.exists(os.path.join(curr, "src")) and os.path.exists(os.path.join(curr, "configs"))
        ):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.getcwd())

PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.inference import run_comprehensive_inference

PE_TARGET_COLS = {
    "PEmaxdRNaseH": ["PEmaxdRNaseH", "comb_vis+HEK-M-2-7D%", "Normalized+3rep_HEK-M-2-7D+pe_ratio_%"],
    "PE6a": ["PE6a(+PEmaxCas9)", "comb_vis+HEK-M-3-7D%", "Normalized+3rep_HEK-M-3-7D+pe_ratio_%"],
    "PE6b": ["PE6b(+PEmaxCas9)", "comb_vis+HEK-M-4-7D%", "Normalized+3rep_HEK-M-4-7D+pe_ratio_%"],
    "PE6c": ["PE6c(+PEmaxCas9)", "comb_vis+HEK-M-5-7D%", "Normalized+3rep_HEK-M-5-7D+pe_ratio_%"],
}

def get_target_values(df, candidate_cols):
    for col in candidate_cols:
        if col in df.columns:
            return df[col].values
    return np.full(len(df), np.nan)

def main():
    test_path = "data/rq3_test_library_dedup.parquet"
    print(f"Loading deduplicated test set from {test_path}...", flush=True)
    df_test = pd.read_parquet(test_path)
    print(f"Loaded {len(df_test):,} test sequences.")

    # Run inference using canonical codebase inference engine
    scored_df = run_comprehensive_inference(
        df_or_path=df_test,
        include_base=False,
        include_dp6_ft=False,
        include_dp6_scratch=True,
        include_pridict2=False,
        batch_size=2048
    )

    id_col = scored_df["ID"].values if "ID" in scored_df.columns else scored_df.index.values

    scratch_exports = {
        "PE6a": ("pred_scratch_pe6a", "data/rq3_predictions_scratch_pe6a.csv"),
        "PE6b": ("pred_scratch_pe6b", "data/rq3_predictions_scratch_pe6b.csv"),
        "PE6c": ("pred_scratch_pe6c", "data/rq3_predictions_scratch_pe6c.csv"),
        "PEmaxdRNaseH": ("pred_scratch_drnaseh", "data/rq3_predictions_scratch_drnaseh.csv"),
    }

    exp_pred_dir = "experiments/20261008_panelB_base_architecture_revalidation/predictions"
    os.makedirs(exp_pred_dir, exist_ok=True)

    print("\n--- Generating From Scratch Predictions on Test Set ---")
    for domain, (pred_col, out_csv) in scratch_exports.items():
        preds = scored_df[pred_col].values
        pred_df = pd.DataFrame({"ID": id_col, "orig_index": id_col, "Prediction": preds})
        pred_df.to_csv(out_csv, index=False)
        shutil.copy(out_csv, os.path.join(exp_pred_dir, os.path.basename(out_csv)))
        print(f"Saved {domain} scratch predictions to {out_csv} and {exp_pred_dir}")

        y_true = get_target_values(df_test, PE_TARGET_COLS[domain])
        mask = ~(np.isnan(y_true) | np.isnan(preds))
        r, _ = pearsonr(y_true[mask], preds[mask])
        rho, _ = spearmanr(y_true[mask], preds[mask])
        print(f"  {domain} (From Scratch): Pearson r = {r:.4f}, Spearman rho = {rho:.4f} (N={mask.sum()})")

    print("\nAll From Scratch predictions generated successfully via codebase engine!")

if __name__ == "__main__":
    main()
