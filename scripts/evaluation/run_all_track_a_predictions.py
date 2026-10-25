#!/usr/bin/env python3
"""
Compute all model predictions on the deduplicated test set (data/rq3_test_library_dedup.parquet)
using the canonical codebase inference engine (src.utils.inference).

Models:
1. DeepPrime-FT (Vanilla 20-model ensemble)
2. DeepPrime6-pe6a (pe6a_mainft.ckpt)
3. DeepPrime6-pe6b (pe6b_mainft.ckpt)
4. DeepPrime6-pe6c (pe6c_mainft.ckpt)
5. DeepPrime6-drnaseh (pemaxdrnaseh_mainft.ckpt)
6. PRIDICT2.0 (Official HEK & K562 scores)
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np

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

def main():
    test_path = "data/rq3_test_library_dedup.parquet"
    print(f"Loading deduplicated test set from {test_path}...", flush=True)
    df = pd.read_parquet(test_path)
    print(f"Loaded {len(df):,} test sequences.", flush=True)

    has_cached_pridict = False
    if os.path.exists("data/rq3_predictions_pridict2.csv"):
        pr_cached = pd.read_csv("data/rq3_predictions_pridict2.csv")
        if len(pr_cached) == len(df) and pr_cached["PRIDICT2_0_editing_Score_deep_HEK"].notna().sum() == len(df):
            has_cached_pridict = True
            print("Found complete cached PRIDICT2.0 predictions (2,325/2,325 valid) in data/rq3_predictions_pridict2.csv.", flush=True)

    # Execute inference via canonical codebase engine
    scored_df = run_comprehensive_inference(
        df_or_path=df,
        include_base=True,
        include_dp6_ft=True,
        include_dp6_scratch=False,
        include_pridict2=not has_cached_pridict,
        batch_size=2048
    )
        cache_id = "ID" if "ID" in pr_cached.columns else "REF_ID"
        cache = pr_cached.set_index(cache_id)
        scored_ids = scored_df["ID"] if "ID" in scored_df.columns else pd.Series(scored_df.index, index=scored_df.index)
        scored_df["pred_pridict2_hek"] = cache["PRIDICT2_0_editing_Score_deep_HEK"].reindex(scored_ids).to_numpy()
        scored_df["pred_pridict2_k562"] = cache["PRIDICT2_0_editing_Score_deep_K562"].reindex(scored_ids).to_numpy()

    # Save individual prediction CSVs matching Track A format
    id_col = scored_df["ID"].values if "ID" in scored_df.columns else scored_df.index.values

    pred_exports = {
        "pred_dp_base": "data/rq3_predictions_dp_base.csv",
        "pred_dp6_pe6a": "data/rq3_predictions_dp6.csv",
        "pred_dp6_pe6b": "data/rq3_predictions_dp6_pe6b.csv",
        "pred_dp6_pe6c": "data/rq3_predictions_dp6_pe6c.csv",
        "pred_dp6_drnaseh": "data/rq3_predictions_dp6_drnaseh.csv",
    }

    for col, filepath in pred_exports.items():
        out_df = pd.DataFrame({"ID": id_col, "orig_index": id_col, "Prediction": scored_df[col].values})
        out_df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}", flush=True)

    # PRIDICT2 export
    if "pred_pridict2_hek" in scored_df.columns:
        pr_export_df = pd.DataFrame({
            "ID": id_col,
            "orig_index": id_col,
            "Prediction": scored_df["pred_pridict2_hek"].values,
            "Prediction_K562": scored_df.get("pred_pridict2_k562", scored_df["pred_pridict2_hek"]).values,
            "PRIDICT2_0_editing_Score_deep_HEK": scored_df["pred_pridict2_hek"].values,
            "PRIDICT2_0_editing_Score_deep_K562": scored_df.get("pred_pridict2_k562", scored_df["pred_pridict2_hek"]).values,
        })
        pr_export_df.to_csv("data/rq3_predictions_pridict2.csv", index=False)
        print("Saved: data/rq3_predictions_pridict2.csv", flush=True)

    # Save combined parquet
    scored_df.to_parquet("data/rq3_test_library_with_pridict2.parquet", index=False)
    print("Saved: data/rq3_test_library_with_pridict2.parquet", flush=True)

    print("\n--- Track A Predictions Generation Complete ---", flush=True)

if __name__ == "__main__":
    main()
