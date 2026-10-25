#!/usr/bin/env python3
"""
Paired Bootstrap Significance Testing for DeepPrime6 vs Baselines (DeepPrime-base, PRIDICT2.0, From-Scratch)
Evaluated on the deduplicated test set (data/rq3_test_library_dedup.parquet, N=2,325).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(REPO_ROOT)

def main():
    print("=== Running Paired Bootstrap Significance Testing ===", flush=True)
    gt_path = "data/rq3_test_library_dedup.parquet"
    if not os.path.exists(gt_path):
        gt_path = "data/rq3_test_library.parquet"
    gt = pd.read_parquet(gt_path).reset_index(drop=True)

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from src.utils.dataprep import IDX_TO_PE_TYPE, get_pe6_qc_mask

    # Ensure clean 2-rep average columns exist and apply canonical QC filter (matching PE6DeepPrimeDataset)
    for i in range(1, 10):
        col_clean = f"Normalized+2rep_HEK-M-{i}-7D+pe_ratio_%"
        rep_cols = [
            f"Normalized+HEK-M-{i}-7D-B1+pe_ratio_%",
            f"Normalized+HEK-M2-{i}-7D-B1+pe_ratio_%",
        ]
        valid_reps = [c for c in rep_cols if c in gt.columns]
        if len(valid_reps) > 0:
            gt[col_clean] = np.maximum(0.0, gt[valid_reps].mean(axis=1))
            pe_type = IDX_TO_PE_TYPE[i]
            qc_mask = get_pe6_qc_mask(gt, pe_type=pe_type, read_count_filter=200, max_ctrl_bg=5.0)
            gt.loc[~qc_mask, col_clean] = np.nan
            gt[f"Normalized+3rep_HEK-M-{i}-7D+pe_ratio_%"] = gt[col_clean]
            if pe_type in gt.columns:
                gt.loc[~qc_mask, pe_type] = np.nan

    def load_pred(csv_path, name):
        df = pd.read_csv(csv_path)
        id_col = "ID" if "ID" in df.columns else "orig_index"
        return df.set_index(id_col)["Prediction"].rename(name)

    dp_base = load_pred("data/rq3_predictions_dp_base.csv", "pred_dp_base")
    dp6_a = load_pred("data/rq3_predictions_dp6.csv", "pred_dp6_pe6a")
    dp6_b = load_pred("data/rq3_predictions_dp6_pe6b.csv", "pred_dp6_pe6b")
    dp6_c = load_pred("data/rq3_predictions_dp6_pe6c.csv", "pred_dp6_pe6c")
    dp6_drn = load_pred("data/rq3_predictions_dp6_drnaseh.csv", "pred_dp6_drnaseh")

    sc_a = load_pred("data/rq3_predictions_scratch_pe6a.csv", "pred_scratch_pe6a")
    sc_b = load_pred("data/rq3_predictions_scratch_pe6b.csv", "pred_scratch_pe6b")
    sc_c = load_pred("data/rq3_predictions_scratch_pe6c.csv", "pred_scratch_pe6c")
    sc_drn = load_pred("data/rq3_predictions_scratch_drnaseh.csv", "pred_scratch_drnaseh")

    pridict = pd.read_csv("data/rq3_predictions_pridict2.csv")
    pr_id = "ID" if "ID" in pridict.columns else "REF_ID"
    pr_hek = pridict.set_index(pr_id)["PRIDICT2_0_editing_Score_deep_HEK"].rename("pred_pridict2_hek")

    gt_id = "ID" if "ID" in gt.columns else "REF_ID"
    gt["dp_base"] = gt[gt_id].map(dp_base)
    gt["dp6_pe6a"] = gt[gt_id].map(dp6_a)
    gt["dp6_pe6b"] = gt[gt_id].map(dp6_b)
    gt["dp6_pe6c"] = gt[gt_id].map(dp6_c)
    gt["dp6_drnaseh"] = gt[gt_id].map(dp6_drn)
    gt["sc_pe6a"] = gt[gt_id].map(sc_a)
    gt["sc_pe6b"] = gt[gt_id].map(sc_b)
    gt["sc_pe6c"] = gt[gt_id].map(sc_c)
    gt["sc_drnaseh"] = gt[gt_id].map(sc_drn)
    gt["pridict2_hek"] = gt[gt_id].map(pr_hek)

    eval_domains = [
        ("PE2max-dRNaseH", "Normalized+3rep_HEK-M-2-7D+pe_ratio_%", "dp6_drnaseh", "sc_drnaseh"),
        ("PE6a", "Normalized+3rep_HEK-M-3-7D+pe_ratio_%", "dp6_pe6a", "sc_pe6a"),
        ("PE6b", "Normalized+3rep_HEK-M-4-7D+pe_ratio_%", "dp6_pe6b", "sc_pe6b"),
        ("PE6c", "Normalized+3rep_HEK-M-5-7D+pe_ratio_%", "dp6_pe6c", "sc_pe6c"),
    ]

    rng = np.random.default_rng(42)
    results = []

    for dom_name, target_col, dp6_col, sc_col in eval_domains:
        sub = gt[[target_col, "dp_base", dp6_col, sc_col, "pridict2_hek"]].dropna()
        n = len(sub)
        y = sub[target_col].values
        base = sub["dp_base"].values
        dp6 = sub[dp6_col].values
        sc = sub[sc_col].values
        pr = sub["pridict2_hek"].values

        rho_base = spearmanr(y, base)[0]
        rho_pr = spearmanr(y, pr)[0]
        rho_sc = spearmanr(y, sc)[0]
        rho_dp6 = spearmanr(y, dp6)[0]
        delta_over_base = rho_dp6 - rho_base
        delta_over_pr = rho_dp6 - rho_pr
        delta_over_sc = rho_dp6 - rho_sc

        diffs_base = []
        diffs_pr = []
        diffs_sc = []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            sp_dp6_i = spearmanr(y[idx], dp6[idx])[0]
            diffs_base.append(sp_dp6_i - spearmanr(y[idx], base[idx])[0])
            diffs_pr.append(sp_dp6_i - spearmanr(y[idx], pr[idx])[0])
            diffs_sc.append(sp_dp6_i - spearmanr(y[idx], sc[idx])[0])

        diffs_base = np.array(diffs_base)
        diffs_pr = np.array(diffs_pr)
        diffs_sc = np.array(diffs_sc)

        results.append({
            "Domain": dom_name,
            "N": n,
            "DeepPrime_FT_rho": round(rho_base, 4),
            "PRIDICT2_rho": round(rho_pr, 4),
            "From_Scratch_rho": round(rho_sc, 4),
            "DeepPrime6_rho": round(rho_dp6, 4),
            "Delta_over_FT": round(delta_over_base, 4),
            "95%_CI_Delta_FT": f"[{np.percentile(diffs_base, 2.5):.4f}, {np.percentile(diffs_base, 97.5):.4f}]",
            "Bootstrap_Probability_Delta_FT_gt0": float((diffs_base > 0).mean()),
            "One_Sided_P_val_Delta_FT": float((diffs_base <= 0).mean()),
            "Delta_over_PRIDICT2": round(delta_over_pr, 4),
            "95%_CI_Delta_PRIDICT2": f"[{np.percentile(diffs_pr, 2.5):.4f}, {np.percentile(diffs_pr, 97.5):.4f}]",
            "One_Sided_P_val_Delta_PRIDICT2": float((diffs_pr <= 0).mean()),
            "Delta_over_Scratch": round(delta_over_sc, 4),
            "95%_CI_Delta_Scratch": f"[{np.percentile(diffs_sc, 2.5):.4f}, {np.percentile(diffs_sc, 97.5):.4f}]",
            "One_Sided_P_val_Delta_Scratch": float((diffs_sc <= 0).mean()),
        })

    res_df = pd.DataFrame(results)
    print("\n--- Paired Bootstrap Results (All Edit Types, N=2,325) ---")
    print(res_df.to_markdown(index=False))

    os.makedirs("notebooks/plots", exist_ok=True)
    out_plots = "notebooks/plots/paired_bootstrap_results.csv"
    res_df.to_csv(out_plots, index=False)
    print(f"\nSaved bootstrap results to {out_plots}")

if __name__ == "__main__":
    main()
