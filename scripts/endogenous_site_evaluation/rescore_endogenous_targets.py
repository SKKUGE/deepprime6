#!/usr/bin/env python3
"""
Endogenous Targets Rescoring Pipeline (DeepPrime (PE2) & DeepPrime6).

Updates `experiments/20261011_endogenous_rescore/` across all three N=50 cohorts:
  1. final_50_validated_candidates
  2. selected_50_candidates
  3. FINAL_v6_candidates

Evaluates DeepPrime (PE2, HEK293T 5-model ensemble, exp(pred) - 1) alongside
DeepPrime6 (pe6a, pe6b, pe6c, pemaxdrnaseh) and PRIDICT2.0 (HEK293T),
regenerating all comparison tables and figures.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def find_project_root() -> str:
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, ".project-root")) or (
            os.path.exists(os.path.join(curr, "PRIDICT2")) and os.path.exists(os.path.join(curr, "src"))
        ):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.getcwd())


PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.components.pe6_preprocess_data import preprocess_data
from src.utils.inference import (
    char_to_onehot_tensor,
    load_deepprime6_models,
    load_deepprime_pe2_models,
    predict_deepprime6,
    predict_deepprime_pe2,
    prepare_biofeatures,
)


def rescore_cohort(
    df: pd.DataFrame,
    dataset_name: str,
    pe2_bundle,
    dp6_models,
    device: torch.device,
) -> pd.DataFrame:
    """Computes DeepPrime (PE2) and DeepPrime6 scores using canonical feature preprocessing."""
    pe2_models, mean_pe2, std_pe2 = pe2_bundle
    work_df = df.copy()

    if "Target" in work_df.columns and "Masked_EditSeq" in work_df.columns and "Tm1_PBS" in work_df.columns:
        feat_df = work_df
    else:
        prep_df = work_df.rename(
            columns={
                "PBSlen": "PBS_length",
                "RTlen": "RT_length",
                "PBS_len": "PBS_length",
                "RTT_len": "RT_length",
                "Edit_len": "Edit length",
                "Edit_pos": "Edit position",
                "ID": "REF_ID",
            }
        ).copy()
        if "PBS_pegRNA_DNA" in prep_df.columns and "PBS" not in prep_df.columns:
            prep_df["PBS"] = prep_df["PBS_pegRNA_DNA"]
        if "RTT_template_DNA" in prep_df.columns and "RTT" not in prep_df.columns:
            prep_df["RTT"] = prep_df["RTT_template_DNA"]
        prep_df["ContextSeqUsed"] = "WideTargetSequence"
        prep_df["leading G"] = ""
        prep_df["Nicking"] = 21
        feat_df = preprocess_data(data=prep_df, aggregate_duplicates=False)
        if "RHA_len" in work_df.columns and work_df["RHA_len"].notna().all():
            feat_df["RHA_len"] = work_df["RHA_len"].values

    g_tensor = char_to_onehot_tensor(
        feat_df["Target"].astype(str).tolist(),
        feat_df["Masked_EditSeq"].astype(str).tolist(),
        device,
    )
    b_pe2 = prepare_biofeatures(feat_df, mean_pe2, std_pe2, device=device)
    b_pe6 = prepare_biofeatures(feat_df, "pe6", device=device)

    pred_pe2 = predict_deepprime_pe2(pe2_models, g_tensor, b_pe2, exp_transform=True)
    preds_dp6 = predict_deepprime6(dp6_models, g_tensor, b_pe6)

    out_df = work_df.copy()
    out_df["New_Score_DP_Base"] = pred_pe2
    out_df["New_Score_PE6a"] = preds_dp6["pe6a"]
    out_df["New_Score_PE6b"] = preds_dp6["pe6b"]
    out_df["New_Score_PE6c"] = preds_dp6["pe6c"]
    out_df["New_Score_PEmax_dRNaseH"] = preds_dp6["drnaseh"]
    out_df["New_DP6_Avg"] = (
        preds_dp6["pe6a"] + preds_dp6["pe6b"] + preds_dp6["pe6c"] + preds_dp6["drnaseh"]
    ) / 4.0

    print(
        f"[{dataset_name}] DeepPrime (PE2) mean={pred_pe2.mean():.3f}%, "
        f"DeepPrime6-Avg mean={out_df['New_DP6_Avg'].mean():.3f}%",
        flush=True,
    )
    return out_df


def summarize_and_plot(res_df: pd.DataFrame, dataset_name: str, out_dir: str) -> pd.DataFrame:
    """Saves updated CSV summaries and regenerates scatter & correlation heatmap figures."""
    results_dir = os.path.join(out_dir, "results")
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    old_base_col = next(c for c in ["pred_dp_base", "Score_DP_Base", "DP_Base"] if c in res_df.columns)
    old_dp6_col = next(c for c in ["pred_dp6", "DP6_Avg"] if c in res_df.columns)
    old_pri_col = next(c for c in ["pred_pridict2", "Score_PRIDICT_HEK", "PRIDICT2_Avg"] if c in res_df.columns)

    if dataset_name == "FINAL_v6_candidates":
        model_pairs = [
            ("DeepPrime (PE2)", old_base_col, "New_Score_DP_Base"),
            ("DeepPrime6-Average", old_dp6_col, "New_DP6_Avg"),
            ("PRIDICT2.0", old_pri_col, "New_Score_PRIDICT_HEK"),
        ]
    else:
        model_pairs = [
            ("DeepPrime (PE2)", old_base_col, "New_Score_DP_Base"),
            (
                "DeepPrime6-PE6a",
                next((c for c in ["pred_dp6_pe6a", "Score_PE6a"] if c in res_df.columns), None),
                "New_Score_PE6a",
            ),
            (
                "DeepPrime6-PE6b",
                next((c for c in ["pred_dp6_pe6b", "Score_PE6b"] if c in res_df.columns), None),
                "New_Score_PE6b",
            ),
            (
                "DeepPrime6-PE6c",
                next((c for c in ["pred_dp6_pe6c", "Score_PE6c"] if c in res_df.columns), None),
                "New_Score_PE6c",
            ),
            (
                "DeepPrime6-PEmaxdRNaseH",
                next((c for c in ["pred_dp6_pemaxdrnaseh", "Score_PEmax_dRNaseH"] if c in res_df.columns), None),
                "New_Score_PEmax_dRNaseH",
            ),
            ("DeepPrime6-Average", old_dp6_col, "New_DP6_Avg"),
            ("PRIDICT2.0", old_pri_col, "New_Score_PRIDICT_HEK"),
        ]

    records = []
    for model_name, old_col, new_col in model_pairs:
        if old_col and old_col in res_df.columns and new_col in res_df.columns:
            mask = ~(res_df[old_col].isna() | res_df[new_col].isna())
            y_old = res_df.loc[mask, old_col].to_numpy(dtype=float)
            y_new = res_df.loc[mask, new_col].to_numpy(dtype=float)
            r_val, _ = pearsonr(y_old, y_new)
            rho_val, _ = spearmanr(y_old, y_new)
            records.append(
                {
                    "Model": model_name,
                    "Old_Mean": round(float(np.mean(y_old)), 3),
                    "New_Mean": round(float(np.mean(y_new)), 3),
                    "Delta_Mean": round(float(np.mean(y_new) - np.mean(y_old)), 3),
                    "Pearson_r_Old_vs_New": round(float(r_val), 4),
                    "Spearman_rho_Old_vs_New": round(float(rho_val), 4),
                    "N": int(mask.sum()),
                }
            )

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(os.path.join(results_dir, f"{dataset_name}_comparison_summary.csv"), index=False)
    summary_df.to_csv(os.path.join(results_dir, f"{dataset_name}_score_comparison_summary.csv"), index=False)
    res_df.to_csv(os.path.join(results_dir, f"{dataset_name}_rescored.csv"), index=False)

    # Scatter plot: Old vs New
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    axes[0].scatter(res_df[old_base_col], res_df["New_Score_DP_Base"], color="#2b5c8f", alpha=0.7, edgecolors="k", s=60)
    axes[0].set_title("DeepPrime (PE2): Old vs New Rescored", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(f"Old Score ({old_base_col})", fontsize=11)
    axes[0].set_ylabel("New DeepPrime (PE2) Score (%)", fontsize=11)
    row_pe2 = summary_df.loc[summary_df["Model"] == "DeepPrime (PE2)"].iloc[0]
    axes[0].text(
        0.05,
        0.88,
        f"Pearson r = {row_pe2['Pearson_r_Old_vs_New']:.4f}\nSpearman ρ = {row_pe2['Spearman_rho_Old_vs_New']:.4f}",
        transform=axes[0].transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        fontsize=10,
    )
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].scatter(res_df[old_dp6_col], res_df["New_DP6_Avg"], color="#c23b22", alpha=0.7, edgecolors="k", s=60)
    axes[1].set_title("DeepPrime6-Avg: Old vs New Retrained", fontsize=13, fontweight="bold")
    axes[1].set_xlabel(f"Old DP6 Score ({old_dp6_col})", fontsize=11)
    axes[1].set_ylabel("New DP6 Score (%)", fontsize=11)
    row_dp6 = summary_df.loc[summary_df["Model"] == "DeepPrime6-Average"].iloc[0]
    axes[1].text(
        0.05,
        0.88,
        f"Pearson r = {row_dp6['Pearson_r_Old_vs_New']:.4f}\nSpearman ρ = {row_dp6['Spearman_rho_Old_vs_New']:.4f}",
        transform=axes[1].transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        fontsize=10,
    )
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{dataset_name}_old_vs_new_scatters.png"), dpi=300)
    plt.close()

    # Heatmap
    heat_cols = [
        ("New_Score_DP_Base", "DeepPrime (PE2)"),
        ("New_Score_PE6a", "PE6a"),
        ("New_Score_PE6b", "PE6b"),
        ("New_Score_PE6c", "PE6c"),
        ("New_Score_PEmax_dRNaseH", "PEmax_dRNaseH"),
        ("New_Score_PRIDICT_HEK", "PRIDICT_HEK"),
    ]
    valid = [(c, lbl) for c, lbl in heat_cols if c in res_df.columns and not res_df[c].isna().all()]
    corr_mat = res_df[[c for c, _ in valid]].corr(method="spearman")
    labels = [lbl for _, lbl in valid]
    plt.figure(figsize=(7.2, 6.2), dpi=300)
    sns.heatmap(
        corr_mat,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Spearman's rank correlation"},
    )
    plt.title(f"{dataset_name}: Inter-Model Correlation (Rescored)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{dataset_name}_rescored_correlation_heatmap.png"), dpi=300)
    plt.close()

    return summary_df


def main() -> None:
    out_dir = "experiments/20261011_endogenous_rescore"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pe2_bundle = load_deepprime_pe2_models(device)
    dp6_models = load_deepprime6_models(device=device, is_scratch=False)

    cohorts = [
        "final_50_validated_candidates",
        "selected_50_candidates",
        "FINAL_v6_candidates",
    ]
    for name in cohorts:
        csv_path = os.path.join(out_dir, "results", f"{name}_rescored.csv")
        df = pd.read_csv(csv_path)
        res_df = rescore_cohort(df, name, pe2_bundle, dp6_models, device)
        summary_df = summarize_and_plot(res_df, name, out_dir)
        print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
