#!/usr/bin/env python3
"""
Endogenous Targets Rescoring Experiment:
Evaluates endogenous validation candidate sets using the newly retrained DeepPrime6 models
and canonical normalization, and compares the new predictions with the existing (July) predictions.

Evaluated datasets:
1. data/predictions/selected_50_validation_candidates.csv (Selected 50 validation candidates)
2. data/FINAL_v6_ALL_INFO.csv (50 ClinVar annotated candidates)

Outputs saved to:
experiments/20261011_endogenous_rescore/
  - results/
  - figures/
  - README.md
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

def find_project_root():
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

pridict2_dir = os.path.join(PROJECT_ROOT, "PRIDICT2")
if pridict2_dir not in sys.path:
    sys.path.insert(0, pridict2_dir)

from src.models.pe6_module_dp_only import PE6DeepPrimeModule
from src.models.components.dprime import GeneInteractionModelVanilla
from src.data.components.pe6_preprocess_data import preprocess_data
from src.utils.predict_pridict2 import (
    load_pridict_model,
    deeppridict,
    compute_average_predictions,
    extract_single_pegRNA_features,
)

FEATURE_NAMES = [
    "PBSlen", "RTlen", "RT-PBSlen", "Edit_pos", "Edit_len", "RHA_len",
    "type_sub", "type_ins", "type_del", "Tm1", "Tm2", "Tm2new", "Tm3", "Tm4", "TmD",
    "nGCcnt1", "nGCcnt2", "nGCcnt3", "fGCcont1", "fGCcont2", "fGCcont3",
    "MFE3", "MFE4", "DeepSpCas9_score"
]

FEATURE_RENAME_MAP = {
    "PBS_len": "PBSlen", "RTT_len": "RTlen", "RT-PBS_len": "RT-PBSlen",
    "Tm1_PBS": "Tm1", "Tm2_RTT_cTarget_sameLength": "Tm2",
    "Tm3_RTT_cTarget_replaced": "Tm2new", "Tm4_cDNA_PAM-oppositeTarget": "Tm3",
    "Tm5_RTT_cDNA": "Tm4", "deltaTm_Tm4-Tm2": "TmD",
    "GC_count_PBS": "nGCcnt1", "GC_count_RTT": "nGCcnt2", "GC_count_RT-PBS": "nGCcnt3",
    "GC_contents_PBS": "fGCcont1", "GC_contents_RTT": "fGCcont2", "GC_contents_RT-PBS": "fGCcont3",
    "MFE_RT-PBS-polyT": "MFE3", "MFE_Spacer": "MFE4",
}

def load_deepprime_base_models(weights_dir, device):
    files = sorted(glob.glob(os.path.join(weights_dir, "*.pt")))
    models = []
    for f in files:
        m = GeneInteractionModelVanilla()
from src.utils.inference import (
    FEATURE_NAMES,
    FEATURE_RENAME_MAP,
    char_to_onehot_tensor,
    prepare_biofeatures,
    load_deepprime_base_models,
    load_deepprime6_models,
    predict_deepprime_base,
    predict_deepprime6,
)

def score_dataset(df_input, dataset_name, device):
    print(f"\n>>> Scoring dataset: {dataset_name} ({len(df_input)} rows)...", flush=True)

    df = df_input.copy()
    # Normalize column names for preprocessing
    rename_for_prep = {}
    if "PBS_pegRNA_DNA" in df.columns:
        rename_for_prep["PBS_pegRNA_DNA"] = "PBS"
    if "RTT_template_DNA" in df.columns:
        rename_for_prep["RTT_template_DNA"] = "RTT"
    if "PBSlen" in df.columns:
        rename_for_prep["PBSlen"] = "PBS_length"
    if "RTlen" in df.columns:
        rename_for_prep["RTlen"] = "RT_length"
    if "ID" in df.columns:
        rename_for_prep["ID"] = "REF_ID"
    
    prep_df = df.rename(columns=rename_for_prep).copy()
    prep_df["ContextSeqUsed"] = "WideTargetSequence"
    prep_df["leading G"] = ""
    prep_df["Nicking"] = 21
    
    print("  Extracting DeepPrime biofeatures via pe6_preprocess_data...", flush=True)
    processed = preprocess_data(data=prep_df)

    # Normalization statistics
    base_norm_dir = "src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428"
    norm_mean_base = pd.read_csv(os.path.join(base_norm_dir, "mean.csv"), header=None, index_col=0).squeeze()
    norm_std_base = pd.read_csv(os.path.join(base_norm_dir, "std.csv"), header=None, index_col=0).squeeze()

    pe6_norm_dir = "src/models/weights/DeepPrime6-weights"
    norm_mean_pe6 = pd.read_csv(os.path.join(pe6_norm_dir, "pe6_mean.csv"), header=None, index_col=0).squeeze()
    norm_std_pe6 = pd.read_csv(os.path.join(pe6_norm_dir, "pe6_std.csv"), header=None, index_col=0).squeeze()

    b_tensor_base = prepare_biofeatures(processed, norm_mean_base, norm_std_base, device)
    b_tensor_pe6 = prepare_biofeatures(processed, norm_mean_pe6, norm_std_pe6, device)

    wt_seqs = processed["Target"].tolist() if "Target" in processed.columns else processed["WideTargetSequence"].tolist()
    ed_seqs = processed["Masked_EditSeq"].tolist() if "Masked_EditSeq" in processed.columns else wt_seqs
    g_batch = char_to_onehot_tensor(wt_seqs, ed_seqs, device)

    # 1. DeepPrime-base inference
    print("  Running DeepPrime-base 20-ensemble inference via canonical engine...", flush=True)
    dp_base_models = load_deepprime_base_models(base_norm_dir, device)
    pred_base = predict_deepprime_base(dp_base_models, g_batch, b_tensor_base)

    # 2. DeepPrime6 retrained checkpoints inference
    print("  Running DeepPrime6 4 retrained models inference via canonical engine...", flush=True)
    dp6_models = load_deepprime6_models(pe6_norm_dir, device, is_scratch=False)
    preds_dp6 = predict_deepprime6(dp6_models, g_batch, b_tensor_pe6)

    # 3. PRIDICT2.0 inference
    print("  Running PRIDICT2.0 inference...", flush=True)
    preds_pridict2 = run_pridict2_inference(df)

    res_df = df.copy()
    res_df["New_Score_DP_Base"] = pred_base
    res_df["New_Score_PE6a"] = preds_dp6["pe6a"]
    res_df["New_Score_PE6b"] = preds_dp6["pe6b"]
    res_df["New_Score_PE6c"] = preds_dp6["pe6c"]
    res_df["New_Score_PEmax_dRNaseH"] = preds_dp6["drnaseh"]
    res_df["New_DP6_Avg"] = (preds_dp6["pe6a"] + preds_dp6["pe6b"] + preds_dp6["pe6c"] + preds_dp6["drnaseh"]) / 4.0
    res_df["New_Score_PRIDICT_HEK"] = preds_pridict2

    return res_df

def run_pridict2_inference(df):
    pr_rows = []
    for _, r in df.iterrows():
        # Resolve columns
        wt = r.get("WildTypeSequence", r.get("WideTargetSequence", ""))
        ed = r.get("PrimeEditedSequence", wt)
        ref_id = r.get("ID", r.get("clean_id", "seq"))
        guide = r.get("Guide", r.get("Spacer", ""))
        pbs = r.get("PBS_pegRNA_DNA", r.get("PBS", ""))
        rtt = r.get("RTT_template_DNA", r.get("RTT", ""))
        etype = str(r.get("Edit_type", "Sub")).capitalize()

        pr_rows.append(pd.Series({
            "REF_ID": ref_id,
            "WildTypeSequence": wt,
            "PrimeEditedSequence": ed,
            "Edit_type": etype,
            "Guide": guide,
            "PBS": pbs,
            "RTT": rtt
        }))

    features_list_pr = []
    valid_indices = []
    for i, pr_row in enumerate(pr_rows):
        try:
            feat = extract_single_pegRNA_features(pr_row)
            if feat is not None:
                features_list_pr.append(feat)
                valid_indices.append(i)
        except Exception:
            pass

    scores = np.full(len(df), np.nan)
    if len(features_list_pr) > 0:
        pegdf = pd.DataFrame(features_list_pr)
        deepdfcols = [
            "wide_initial_target", "wide_mutated_target", "deepeditposition", 
            "deepeditposition_lst", "Correction_Type", "Correction_Length", 
            "protospacerlocation_only_initial", "PBSlocation",
            "RT_initial_location", "RT_mutated_location",
            "RToverhangmatches", "RToverhanglength", 
            "RTlength", "PBSlength", "RTmt", "RToverhangmt", "PBSmt", "protospacermt",
            "extensionmt", "original_base_mt", "edited_base_mt", "original_base_mt_nan",
            "edited_base_mt_nan"
        ]
        for c in deepdfcols:
            if c not in pegdf.columns:
                pegdf[c] = 0

        pr_models = load_pridict_model(run_ids=[0])
        all_avg_preds = deeppridict(pegdf, pr_models)
        tmp = [all_avg_preds[m] for m in all_avg_preds]
        tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
        agg_df = compute_average_predictions(tmp_df, grp_cols=["seq_id", "dataset_name"])
        hek_df = agg_df[agg_df["dataset_name"] == "HEK"].sort_values("seq_id")
        hek_scores = (hek_df["pred_averageedited"].values) * 100.0

        for i, val_idx in enumerate(valid_indices):
            scores[val_idx] = hek_scores[i]

    return scores

def analyze_and_plot(res_df, dataset_name, out_dir):
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)

    print(f"\n=======================================================", flush=True)
    print(f"=== Comparative Analysis for {dataset_name} ===", flush=True)
    print(f"=======================================================", flush=True)

    # Detect old score columns
    old_base_col = "Score_DP_Base" if "Score_DP_Base" in res_df.columns else "DP_Base"
    old_dp6_col = "DP6_Avg"
    old_pri_col = "Score_PRIDICT_HEK" if "Score_PRIDICT_HEK" in res_df.columns else "PRIDICT2_Avg"

    analysis_records = []
    model_pairs = [
        ("DeepPrime-Base", old_base_col, "New_Score_DP_Base"),
        ("DeepPrime6-PE6a", "Score_PE6a" if "Score_PE6a" in res_df.columns else None, "New_Score_PE6a"),
        ("DeepPrime6-PE6b", "Score_PE6b" if "Score_PE6b" in res_df.columns else None, "New_Score_PE6b"),
        ("DeepPrime6-PE6c", "Score_PE6c" if "Score_PE6c" in res_df.columns else None, "New_Score_PE6c"),
        ("DeepPrime6-PEmaxdRNaseH", "Score_PEmax_dRNaseH" if "Score_PEmax_dRNaseH" in res_df.columns else None, "New_Score_PEmax_dRNaseH"),
        ("DeepPrime6-Average", old_dp6_col, "New_DP6_Avg"),
        ("PRIDICT2.0", old_pri_col, "New_Score_PRIDICT_HEK"),
    ]

    for model_name, old_col, new_col in model_pairs:
        if old_col and old_col in res_df.columns and new_col in res_df.columns:
            valid_mask = ~(res_df[old_col].isna() | res_df[new_col].isna())
            y_old = res_df.loc[valid_mask, old_col].values
            y_new = res_df.loc[valid_mask, new_col].values
            
            r, _ = pearsonr(y_old, y_new)
            rho, _ = spearmanr(y_old, y_new)
            mean_old = np.mean(y_old)
            mean_new = np.mean(y_new)
            delta_mean = mean_new - mean_old

            analysis_records.append({
                "Model": model_name,
                "Old_Mean": round(mean_old, 3),
                "New_Mean": round(mean_new, 3),
                "Delta_Mean": round(delta_mean, 3),
                "Pearson_r_Old_vs_New": round(r, 4),
                "Spearman_rho_Old_vs_New": round(rho, 4),
                "N": len(y_old)
            })

    summary_df = pd.DataFrame(analysis_records)
    print("\n[Old vs New Score Comparison]")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(out_dir, "results", f"{dataset_name}_score_comparison_summary.csv"), index=False)

    # Save full rescored CSV
    res_df.to_csv(os.path.join(out_dir, "results", f"{dataset_name}_rescored.csv"), index=False)

    # Plot 1: Scatter plot Old vs New for DeepPrime6-Avg and DeepPrime-Base
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Base
    ax = axes[0]
    ax.scatter(res_df[old_base_col], res_df["New_Score_DP_Base"], color="#2b5c8f", alpha=0.7, edgecolors="k", s=60)
    ax.set_title("DeepPrime-Base: Old vs New Rescored", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Old Base Score ({old_base_col})", fontsize=11)
    ax.set_ylabel("New Base Score (Canonical Normalization)", fontsize=11)
    r_val = summary_df.loc[summary_df["Model"] == "DeepPrime-Base", "Pearson_r_Old_vs_New"].values[0]
    rho_val = summary_df.loc[summary_df["Model"] == "DeepPrime-Base", "Spearman_rho_Old_vs_New"].values[0]
    ax.text(0.05, 0.90, f"Pearson r = {r_val:.4f}\nSpearman ρ = {rho_val:.4f}", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9), fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # DeepPrime6 Avg
    ax = axes[1]
    ax.scatter(res_df[old_dp6_col], res_df["New_DP6_Avg"], color="#c23b22", alpha=0.7, edgecolors="k", s=60)
    ax.set_title("DeepPrime6-Avg: Old vs New Retrained", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Old DP6 Score ({old_dp6_col})", fontsize=11)
    ax.set_ylabel("New DP6 Score (Retrained Checkpoints)", fontsize=11)
    r_val6 = summary_df.loc[summary_df["Model"] == "DeepPrime6-Average", "Pearson_r_Old_vs_New"].values[0]
    rho_val6 = summary_df.loc[summary_df["Model"] == "DeepPrime6-Average", "Spearman_rho_Old_vs_New"].values[0]
    ax.text(0.05, 0.90, f"Pearson r = {r_val6:.4f}\nSpearman ρ = {rho_val6:.4f}", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9), fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "figures", f"{dataset_name}_old_vs_new_scatters.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved scatter plot to: {fig_path}")

    # Plot 2: Correlation Heatmap (New Models)
    new_cols = ["New_Score_DP_Base", "New_Score_PE6a", "New_Score_PE6b", "New_Score_PE6c", "New_Score_PEmax_dRNaseH", "New_Score_PRIDICT_HEK"]
    valid_cols = [c for c in new_cols if c in res_df.columns and not res_df[c].isna().all()]
    if len(valid_cols) >= 3:
        corr_matrix = res_df[valid_cols].corr(method="spearman")
        clean_labels = [c.replace("New_Score_", "").replace("New_", "") for c in valid_cols]

        plt.figure(figsize=(7, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="Blues", xticklabels=clean_labels, yticklabels=clean_labels,
                    cbar_kws={'label': "Spearman's rank correlation"})
        plt.title(f"{dataset_name}: Inter-Model Correlation (Rescored)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        hm_path = os.path.join(out_dir, "figures", f"{dataset_name}_rescored_correlation_heatmap.png")
        plt.savefig(hm_path, dpi=300)
        plt.close()
        print(f"  Saved heatmap to: {hm_path}")

    # Rank shift analysis
    res_df["Old_Rank_DP6"] = res_df[old_dp6_col].rank(ascending=False)
    res_df["New_Rank_DP6"] = res_df["New_DP6_Avg"].rank(ascending=False)
    res_df["Rank_Shift_DP6"] = res_df["Old_Rank_DP6"] - res_df["New_Rank_DP6"]
    
    print("\n[Top 5 Positive Rank Gainers (DeepPrime6)]")
    top_gainers = res_df.sort_values(by="Rank_Shift_DP6", ascending=False).head(5)
    id_col = "ID" if "ID" in res_df.columns else "REF_ID"
    print(top_gainers[[id_col, old_dp6_col, "New_DP6_Avg", "Old_Rank_DP6", "New_Rank_DP6", "Rank_Shift_DP6"]].to_string(index=False))

    print("\n[Top 5 Rank Decliners (DeepPrime6)]")
    top_decliners = res_df.sort_values(by="Rank_Shift_DP6", ascending=True).head(5)
    print(top_decliners[[id_col, old_dp6_col, "New_DP6_Avg", "Old_Rank_DP6", "New_Rank_DP6", "Rank_Shift_DP6"]].to_string(index=False))

    return summary_df, res_df

def main():
    print("==========================================================================", flush=True)
    print("=== Master Endogenous Targets Rescoring Experiment ===", flush=True)
    print("==========================================================================", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}", flush=True)

    out_base = "experiments/20261011_endogenous_rescore"
    os.makedirs(out_base, exist_ok=True)

    # 1. Rescore Selected 50 Validation Candidates
    df_50 = pd.read_csv("data/predictions/selected_50_validation_candidates.csv")
    res_50 = score_dataset(df_50, "selected_50_validation_candidates", device)
    summary_50, rescored_50 = analyze_and_plot(res_50, "selected_50_candidates", out_base)

    # 2. Rescore FINAL_v6_ALL_INFO (ClinVar annotated 50 candidates)
    df_v6 = pd.read_csv("data/FINAL_v6_ALL_INFO.csv")
    res_v6 = score_dataset(df_v6, "FINAL_v6_ALL_INFO", device)
    summary_v6, rescored_v6 = analyze_and_plot(res_v6, "FINAL_v6_candidates", out_base)

    print(f"\n[DONE] Endogenous rescoring completed successfully! Results archived in {out_base}/")

if __name__ == "__main__":
    main()
