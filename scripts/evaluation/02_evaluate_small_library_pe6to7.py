#!/usr/bin/env python3
"""
Small Library Orthogonal Benchmark: PE6 to PE7 Generalization
Comprehensive evaluation script for Small Library PE6 and PE7 generalization.

Evaluates:
- Small Library dataset (pe6to7_with_avgs.parquet, N=1,960)
- DeepPrime-FT (20-model ensemble)
- PRIDICT2.0 (HEK293T) predictions
- DeepPrime6 (PEmaxdRNaseH, PE6a, PE6b, PE6c) with canonical fine-tuned weights
- Ground truth across PE6 systems (PE6-2, PE6-3, PE6-4) and PE7 systems (PE7-2, PE7-3, PE7-4)
- Replicate-level metrics (rep1, rep2, rep3) and 3-rep averages (avg)

Produces:
- Predictions CSV: predictions/small_library_pe6to7_predictions.csv (with DeepPrime-FT, PRIDICT2.0, DeepPrime6)
- Summary metrics CSV: results/small_library_pe6to7_summary.csv
- Replicate metrics CSV: results/small_library_pe6to7_replicates.csv
- 6-panel orthogonal correlation plot (PNG & SVG)
- Bar plots comparing DeepPrime-FT, PRIDICT2.0, and DeepPrime6 (Pearson & Spearman) (PNG & SVG)
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

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

from src.models.pe6_module_dp_only import PE6DeepPrimeModule
from src.models.components.dprime import GeneInteractionModelVanilla
from src.data.components.pe6_preprocess_data import preprocess_data
from src.utils.inference import (
    char_to_onehot_tensor,
    prepare_biofeatures,
    load_deepprime_base_models,
    load_deepprime6_models,
    predict_deepprime_base,
    predict_deepprime6,
    FEATURE_NAMES,
    FEATURE_RENAME_MAP,
)


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    if n < 2:
        return np.nan, np.nan, np.nan, n
    r, _ = pearsonr(yp, yt)
    rho, _ = spearmanr(yp, yt)
    mse = mean_squared_error(yt, yp)
    return r, rho, mse, n


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstraps=1000):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    if n < 2:
        return np.nan, np.nan
    boot_vals = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        val, _ = metric_fn(yt[idx], yp[idx])
        if not np.isnan(val):
            boot_vals.append(val)
    if len(boot_vals) == 0:
        return np.nan, np.nan
    return np.percentile(boot_vals, 2.5), np.percentile(boot_vals, 97.5)


def load_or_run_pridict2(df, predictions_parquet_path="data/small_library/pe6to7_predictions.parquet"):
    """Loads pre-computed PRIDICT2.0 scores and fills any missing NaN rows via on-the-fly inference."""
    mapped_scores = np.full(len(df), np.nan)
    pred_df = None
    if os.path.exists(predictions_parquet_path):
        pred_df = pd.read_parquet(predictions_parquet_path)
        if "pred_pridict2" in pred_df.columns:
            pred_map = dict(zip(pred_df["ID"], pred_df["pred_pridict2"]))
            mapped_scores = df["ID"].map(pred_map).values.astype(float)
            valid_cnt = int(np.sum(~np.isnan(mapped_scores)))
            if valid_cnt == len(df):
                print(f"Loaded complete PRIDICT2.0 predictions ({valid_cnt:,}/{len(df):,} valid) from {predictions_parquet_path}", flush=True)
                return mapped_scores
            print(f"Loaded {valid_cnt:,}/{len(df):,} PRIDICT2.0 predictions from {predictions_parquet_path}; computing {len(df) - valid_cnt} missing rows...", flush=True)

    missing_mask = np.isnan(mapped_scores)
    try:
        sys.path.append(os.path.join(PROJECT_ROOT, "PRIDICT2"))
        from src.utils.predict_pridict2 import load_pridict_model, deeppridict, compute_average_predictions, extract_single_pegRNA_features

        pr_input_df = df.loc[missing_mask].copy()
        def get_pr_edit_type(row):
            t = str(row.get("Edit_type", "")).lower()
            if t == "sub": return "Sub"
            elif t == "del": return "Del"
            elif t == "ins": return "Ins"
            return "Sub"
        pr_input_df["Edit_type"] = pr_input_df.apply(get_pr_edit_type, axis=1)
        pr_input_df["REF"] = pr_input_df.get("OriginalBase", "")
        pr_input_df["ALT"] = pr_input_df.get("EditedBase", "")
        pr_input_df["nick_position"] = pr_input_df.get("Nicking", 0)
        pr_input_df["WT_context"] = pr_input_df.get("WildTypeSequence", pr_input_df.get("Target", ""))
        pr_input_df["REF_ID"] = pr_input_df["ID"]

        pridict_models = load_pridict_model([0])
        features_list_pr = []

        for idx, row in pr_input_df.iterrows():
            try:
                feat = extract_single_pegRNA_features(row)
                if feat is not None:
                    features_list_pr.append(feat)
            except Exception:
                pass

        if len(features_list_pr) > 0:
            pegdataframe = pd.DataFrame(features_list_pr)
            for col in ['wide_initial_target', 'wide_mutated_target', 'deepeditposition',
                        'deepeditposition_lst', 'Correction_Type', 'Correction_Length',
                        'protospacerlocation_only_initial', 'PBSlocation',
                        'RT_initial_location', 'RT_mutated_location',
                        'RToverhangmatches', 'RToverhanglength',
                        'RTlength', 'PBSlength', 'RTmt', 'RToverhangmt','PBSmt','protospacermt',
                        'extensionmt','original_base_mt','edited_base_mt','original_base_mt_nan',
                        'edited_base_mt_nan']:
                if col not in pegdataframe.columns:
                    pegdataframe[col] = 0

            all_avg_preds = deeppridict(pegdataframe, pridict_models)
            tmp = [all_avg_preds[m_id] for m_id in all_avg_preds]
            tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
            agg_df = compute_average_predictions(tmp_df, grp_cols=['seq_id', 'dataset_name'])
            hek_cond = agg_df['dataset_name'] == 'HEK'
            agg_hek = agg_df[hek_cond].sort_values('seq_id')

            for _, r in agg_hek.iterrows():
                v_idx = int(r['seq_id'])
                ref_id = pegdataframe.iloc[v_idx]['sequence_name']
                row_match = df.index[df['ID'] == ref_id]
                if len(row_match) > 0:
                    mapped_scores[row_match[0]] = r['pred_averageedited'] * 100

        if pred_df is not None and os.path.exists(predictions_parquet_path):
            updated_map = dict(zip(df["ID"], mapped_scores))
            pred_df["pred_pridict2"] = pred_df["ID"].map(updated_map)
            if "pridict2_293t_score" in pred_df.columns:
                pred_df["pridict2_293t_score"] = pred_df["pred_pridict2"]
            pred_df.to_parquet(predictions_parquet_path, index=False)

        print(f"Total valid PRIDICT2.0 predictions: {np.sum(~np.isnan(mapped_scores)):,}/{len(df):,}.", flush=True)
        return mapped_scores
    except Exception as e:
        print(f"Warning: Failed to compute PRIDICT2.0 on-the-fly ({e}). Returning partial scores.", flush=True)
        return mapped_scores


def main():
    figures_dir = "figures/evaluation"
    results_dir = "notebooks/plots/small_library_pe6to7"
    preds_dir = "data/small_library"

    for d in [preds_dir, results_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    print("==========================================================================", flush=True)
    print("=== Small Library Orthogonal Benchmark: PE6 & PE7 Generalization ===", flush=True)
    print("==========================================================================", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    data_path = "data/small_library/pe6to7_with_avgs.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Small library file not found: {data_path}")

    raw_df = pd.read_parquet(data_path)
    print(f"Loaded small library: {len(raw_df):,} rows", flush=True)

    # Feature preprocessing
    processed_cache = "data/small_library/pe6to7_processed.parquet"
    if os.path.exists(processed_cache):
        print(f"Loading pre-computed biofeatures from {processed_cache}...", flush=True)
        df = pd.read_parquet(processed_cache)
    else:
        print("Computing biofeatures for small library...", flush=True)
        df = preprocess_data(data=raw_df)

    # Restore replicate-level QC filtering (NaNs in raw_df represent replicates failing QC,
    # which were overwritten by .fillna(0) in preprocess_data) and clip non-null values at 0.0.
    # Require all 3 replicates to pass QC (skipna=False) for the 3-replicate average, matching PE6DeepPrimeDataset.
    small_lib_rep_map = [
        ("PE6-2", "PEmaxdRNaseH"),
        ("PE6-3", "PE6b"),
        ("PE6-4", "PE6c"),
        ("PE7-2", "PE7maxdRNaseH"),
        ("PE7-3", "PE7b"),
        ("PE7-4", "PE7c"),
    ]
    for raw_pfx, proc_pfx in small_lib_rep_map:
        raw_reps = [f"{raw_pfx}rep1", f"{raw_pfx}rep2", f"{raw_pfx}rep3"]
        proc_reps = [f"{proc_pfx}-#1", f"{proc_pfx}-#2", f"{proc_pfx}-#3"]
        if all(c in raw_df.columns for c in raw_reps):
            for r_col, p_col in zip(raw_reps, proc_reps):
                df[p_col] = raw_df[r_col].clip(lower=0.0).values
                df[r_col] = df[p_col]
            df[f"{proc_pfx}-avg"] = df[proc_reps].mean(axis=1, skipna=False)
            df[f"{raw_pfx}avg"] = df[f"{proc_pfx}-avg"]
    df.to_parquet(processed_cache, index=False)

    # 1. Base normalization for DeepPrime-base
    norm_mean_base = pd.read_csv("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/mean.csv", index_col=0, header=None).squeeze()
    norm_std_base = pd.read_csv("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/std.csv", index_col=0, header=None).squeeze()

    # 2. Canonical PE6 normalization for DeepPrime6 checkpoints
    pe6_ckpt_dir = "src/models/weights/DeepPrime6-weights"
    norm_mean_pe6 = pd.read_csv(os.path.join(pe6_ckpt_dir, "pe6_mean.csv"), index_col=0, header=None).squeeze()
    norm_std_pe6 = pd.read_csv(os.path.join(pe6_ckpt_dir, "pe6_std.csv"), index_col=0, header=None).squeeze()

    b_tensor_base = prepare_biofeatures(df, norm_mean_base, norm_std_base, device)
    b_tensor_pe6 = prepare_biofeatures(df, norm_mean_pe6, norm_std_pe6, device)

    wt_seqs = df["Target"].tolist() if "Target" in df.columns else df["WideTargetSequence"].tolist()
    ed_seqs = df["Masked_EditSeq"].tolist() if "Masked_EditSeq" in df.columns else wt_seqs
    g_batch = char_to_onehot_tensor(wt_seqs, ed_seqs, device)

    # Load models via canonical engine
    dp_base_models = load_deepprime_base_models("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428", device)
    ckpt_dir = "src/models/weights/DeepPrime6-weights"
    dp6_models = load_deepprime6_models(ckpt_dir, device, is_scratch=False)

    print("Running forward passes for DeepPrime-base and DeepPrime6 via canonical engine...", flush=True)
    preds_dp_base = predict_deepprime_base(dp_base_models, g_batch, b_tensor_base, batch_size=2048)
    preds_dp6 = predict_deepprime6(dp6_models, g_batch, b_tensor_pe6, batch_size=2048)

    preds_dp6["PEmaxdRNaseH"] = preds_dp6["drnaseh"]
    preds_dp6["PE6a"] = preds_dp6["pe6a"]
    preds_dp6["PE6b"] = preds_dp6["pe6b"]
    preds_dp6["PE6c"] = preds_dp6["pe6c"]

    df["Pred_DeepPrime_Base"] = preds_dp_base
    df["Pred_DeepPrime6_PEmaxdRNaseH"] = preds_dp6["drnaseh"]
    df["Pred_DeepPrime6_PE6a"] = preds_dp6["pe6a"]
    df["Pred_DeepPrime6_PE6b"] = preds_dp6["pe6b"]
    df["Pred_DeepPrime6_PE6c"] = preds_dp6["pe6c"]

    # Predict / Load PRIDICT2.0
    preds_pridict2 = load_or_run_pridict2(df)
    df["Pred_PRIDICT2"] = preds_pridict2

    # Save predictions
    preds_out_file = os.path.join(preds_dir, "small_library_pe6to7_predictions.csv")
    df.to_csv(preds_out_file, index=False)
    print(f"Saved complete predictions table to {preds_out_file}", flush=True)

    def resolve_col(candidates):
        if isinstance(candidates, str):
            candidates = [candidates]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Benchmark pairs: Average targets
    eval_pairs = [
        ("PEmaxdRNaseH (PE6-2)", ["PEmaxdRNaseH-avg", "PE6-2avg"], "PEmaxdRNaseH"),
        ("PE6b (PE6-3)", ["PE6b-avg", "PE6-3avg"], "PE6b"),
        ("PE6c (PE6-4)", ["PE6c-avg", "PE6-4avg"], "PE6c"),
        ("PE7maxdRNaseH (PE7-2)", ["PE7maxdRNaseH-avg", "PE7-2avg"], "PEmaxdRNaseH"),
        ("PE7b (PE7-3)", ["PE7b-avg", "PE7-3avg"], "PE6b"),
        ("PE7c (PE7-4)", ["PE7c-avg", "PE7-4avg"], "PE6c"),
    ]

    results = []
    valid_eval_pairs = []
    for test_label, meas_candidates, model_key in eval_pairs:
        col = resolve_col(meas_candidates)
        if col is not None:
            valid_eval_pairs.append((test_label, col, model_key))
            y_meas = df[col].values
            r_base, rho_base, mse_base, n_base = compute_metrics(y_meas, preds_dp_base)
            r_pri, rho_pri, mse_pri, n_pri = compute_metrics(y_meas, preds_pridict2)
            r_dp6, rho_dp6, mse_dp6, n_dp6 = compute_metrics(y_meas, preds_dp6[model_key])
            
            ci_r_dp6 = bootstrap_ci(y_meas, preds_dp6[model_key], pearsonr)
            ci_rho_dp6 = bootstrap_ci(y_meas, preds_dp6[model_key], spearmanr)

            results.append({
                "Evaluation Target": test_label,
                "Model Evaluated": f"DeepPrime6-{model_key}",
                "DeepPrime-FT (Pearson r)": round(r_base, 4),
                "DeepPrime-FT (Spearman rho)": round(rho_base, 4),
                "DeepPrime-FT (MSE)": round(mse_base, 4),
                "PRIDICT2.0 (Pearson r)": round(r_pri, 4),
                "PRIDICT2.0 (Spearman rho)": round(rho_pri, 4),
                "PRIDICT2.0 (MSE)": round(mse_pri, 4),
                "DeepPrime6 (Pearson r)": round(r_dp6, 4),
                "DeepPrime6 (Spearman rho)": round(rho_dp6, 4),
                "DeepPrime6 (MSE)": round(mse_dp6, 4),
                "Delta Pearson r (vs FT)": round(r_dp6 - r_base, 4),
                "Delta Spearman rho (vs FT)": round(rho_dp6 - rho_base, 4),
                "Delta Pearson r (vs PRIDICT2)": round(r_dp6 - r_pri, 4),
                "Delta Spearman rho (vs PRIDICT2)": round(rho_dp6 - rho_pri, 4),
                "95% CI Pearson r": f"[{ci_r_dp6[0]:.4f}, {ci_r_dp6[1]:.4f}]",
                "95% CI Spearman rho": f"[{ci_rho_dp6[0]:.4f}, {ci_rho_dp6[1]:.4f}]",
                "N (Valid FT/DP6)": n_base,
                "N (Valid PRIDICT2)": n_pri,
            })

    df_res = pd.DataFrame(results)
    print("\n--- Small Library (PE6 & PE7) Benchmark Results ---", flush=True)
    print(df_res.to_string(index=False), flush=True)

    summary_csv = os.path.join(results_dir, "small_library_pe6to7_summary.csv")
    df_res.to_csv(summary_csv, index=False)
    # Mirror to results, figures/evaluation, and notebooks/plots for backward compatibility
    os.makedirs("figures/evaluation", exist_ok=True)
    os.makedirs("notebooks/plots", exist_ok=True)
    df_res.to_csv(os.path.join(results_dir, "Small_library_orthogonal_benchmark.csv"), index=False)
    df_res.to_csv("figures/evaluation/Small_library_orthogonal_benchmark.csv", index=False)
    df_res.to_csv("notebooks/plots/Small_library_orthogonal_benchmark.csv", index=False)
    print(f"\nSaved summary to {summary_csv}", flush=True)

    # Replicate breakdown
    reps_results = []
    for editor_prefix, model_key in [
        ("PEmaxdRNaseH", "PEmaxdRNaseH"),
        ("PE6b", "PE6b"),
        ("PE6c", "PE6c"),
        ("PE7maxdRNaseH", "PEmaxdRNaseH"),
        ("PE7b", "PE6b"),
        ("PE7c", "PE6c")
    ]:
        for rep, rep_sfx in [("rep1", "#1"), ("rep2", "#2"), ("rep3", "#3"), ("avg", "avg")]:
            col_candidates = [f"{editor_prefix}-{rep_sfx}", f"{editor_prefix}-{rep}", f"{editor_prefix}_{rep}"]
            col = resolve_col(col_candidates)
            if col:
                y_meas = df[col].values
                r_base, rho_base, _, n_b = compute_metrics(y_meas, preds_dp_base)
                r_pri, rho_pri, _, n_p = compute_metrics(y_meas, preds_pridict2)
                r_dp6, rho_dp6, _, n_d = compute_metrics(y_meas, preds_dp6[model_key])
                reps_results.append({
                    "System": editor_prefix,
                    "Replicate": rep,
                    "Target Column": col,
                    "Model": f"DeepPrime6-{model_key}",
                    "DeepPrime-FT (Pearson r)": round(r_base, 4),
                    "DeepPrime-FT (Spearman rho)": round(rho_base, 4),
                    "PRIDICT2.0 (Pearson r)": round(r_pri, 4),
                    "PRIDICT2.0 (Spearman rho)": round(rho_pri, 4),
                    "DeepPrime6 (Pearson r)": round(r_dp6, 4),
                    "DeepPrime6 (Spearman rho)": round(rho_dp6, 4),
                    "Delta Pearson r (DP6 vs FT)": round(r_dp6 - r_base, 4),
                    "Delta Spearman rho (DP6 vs FT)": round(rho_dp6 - rho_base, 4),
                    "N (FT/DP6)": n_b,
                    "N (PRIDICT2)": n_p,
                })
    df_reps = pd.DataFrame(reps_results)
    reps_csv = os.path.join(results_dir, "small_library_pe6to7_replicates.csv")
    df_reps.to_csv(reps_csv, index=False)
    print(f"Saved replicate breakdown to {reps_csv}", flush=True)

    # 1. 6-panel Scatter Plot
    sns.set_theme(style="ticks", font_scale=1.1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    axes = axes.flatten()

    for idx, (test_label, col, model_key) in enumerate(valid_eval_pairs):
        ax = axes[idx]
        yt = df[col].values
        yp = preds_dp6[model_key]
        r, rho, _, n = compute_metrics(yt, yp)

        sns.regplot(
            x=yt, y=yp, ax=ax,
            scatter_kws={"alpha": 0.35, "s": 22, "color": "#2b5c8f" if "PE6" in test_label else "#c0392b"},
            line_kws={"color": "#111111", "linewidth": 2.0}
        )
        ax.set_title(f"{test_label}\nDeepPrime6 {model_key} ($r={r:.3f}, \\rho={rho:.3f}, N={n:,}$)", fontsize=13, weight="bold")
        ax.set_xlabel("Measured Average Efficiency (%)", fontsize=11)
        ax.set_ylabel("DeepPrime6 Predicted (%)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Small Library Benchmark: DeepPrime6 Accuracy Across PE6 & PE7 Systems", fontsize=16, weight="bold", y=0.99)
    plt.tight_layout()

    out_png = os.path.join(figures_dir, "Small_library_orthogonal_benchmark.png")
    out_svg = os.path.join(figures_dir, "Small_library_orthogonal_benchmark.svg")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.savefig("figures/evaluation/Small_library_orthogonal_benchmark.png", dpi=300, bbox_inches="tight")
    plt.savefig("figures/evaluation/Small_library_orthogonal_benchmark.svg", format="svg", bbox_inches="tight")
    plt.savefig("notebooks/plots/Small_library_orthogonal_benchmark.png", dpi=300, bbox_inches="tight")
    plt.savefig("notebooks/plots/Small_library_orthogonal_benchmark.svg", format="svg", bbox_inches="tight")
    plt.close()

    # 2. Spearman Comparison Bar Plot (3-Model: DeepPrime-base vs PRIDICT2.0 vs DeepPrime6)
    x = np.arange(len(df_res))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    bars1 = ax.bar(x - width, df_res["DeepPrime-FT (Spearman rho)"], width, label="DeepPrime-FT", color="#7f8c8d", alpha=0.85, edgecolor="black")
    bars2 = ax.bar(x, df_res["PRIDICT2.0 (Spearman rho)"], width, label="PRIDICT2.0", color="#8DD3C7", alpha=0.9, edgecolor="black")
    bars3 = ax.bar(x + width, df_res["DeepPrime6 (Spearman rho)"], width, label="DeepPrime6 (Fine-Tuned)", color="#2980b9", alpha=0.9, edgecolor="black")

    for b in bars1:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for b in bars2:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for b in bars3:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, weight="bold")

    ax.set_ylabel("Spearman Correlation ($\\rho$)", fontsize=13, weight="bold")
    ax.set_title("Rank Accuracy: DeepPrime-FT vs PRIDICT2.0 vs DeepPrime6 on Small Library", fontsize=14, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" (", "\n(") for t in df_res["Evaluation Target"]], fontsize=11)
    ax.set_ylim(0, 0.88)
    ax.legend(frameon=True, fontsize=10.5, loc="upper right")
    sns.despine()
    plt.tight_layout()

    bar_png = os.path.join(figures_dir, "Small_library_spearman_gain_bars.png")
    bar_svg = os.path.join(figures_dir, "Small_library_spearman_gain_bars.svg")
    plt.savefig(bar_png, dpi=300, bbox_inches="tight")
    plt.savefig(bar_svg, format="svg", bbox_inches="tight")
    plt.savefig("figures/evaluation/Small_library_spearman_gain_bars.png", dpi=300, bbox_inches="tight")
    plt.savefig("figures/evaluation/Small_library_spearman_gain_bars.svg", format="svg", bbox_inches="tight")
    plt.close()

    # 3. Pearson Quantitative Comparison Bar Plot (3-Model: DeepPrime-FT vs PRIDICT2.0 vs DeepPrime6)
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    bars_p1 = ax.bar(x - width, df_res["DeepPrime-FT (Pearson r)"], width, label="DeepPrime-FT", color="#7f8c8d", alpha=0.85, edgecolor="black")
    bars_p2 = ax.bar(x, df_res["PRIDICT2.0 (Pearson r)"], width, label="PRIDICT2.0", color="#8DD3C7", alpha=0.9, edgecolor="black")
    bars_p3 = ax.bar(x + width, df_res["DeepPrime6 (Pearson r)"], width, label="DeepPrime6 (Fine-Tuned)", color="#d35400", alpha=0.9, edgecolor="black")

    for idx, (b1, b2, b3) in enumerate(zip(bars_p1, bars_p2, bars_p3)):
        h1 = b1.get_height()
        h2 = b2.get_height()
        h3 = b3.get_height()
        delta = h3 - h1
        ax.annotate(f"{h1:.3f}", xy=(b1.get_x() + b1.get_width()/2, h1), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
        ax.annotate(f"{h2:.3f}", xy=(b2.get_x() + b2.get_width()/2, h2), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
        ax.annotate(f"{h3:.3f}", xy=(b3.get_x() + b3.get_width()/2, h3), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, weight="bold")
        pct_gain = (delta / h1) * 100
        ax.annotate(f"$\\Delta r = +{delta:.3f}$\n(+{pct_gain:.0f}%)", xy=(b3.get_x() + b3.get_width()/2, h3 + 0.05),
                    ha="center", va="bottom", fontsize=8, color="#b03a2e", weight="bold")

    ax.set_ylabel("Pearson Correlation ($r$)", fontsize=13, weight="bold")
    ax.set_title("Quantitative Linear Accuracy: DeepPrime-FT vs PRIDICT2.0 vs DeepPrime6 on Small Library", fontsize=14, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" (", "\n(") for t in df_res["Evaluation Target"]], fontsize=11)
    ax.set_ylim(0, 0.98)
    ax.legend(frameon=True, fontsize=10.5, loc="upper left")
    sns.despine()
    plt.tight_layout()

    pearson_png = os.path.join(figures_dir, "Small_library_pearson_gain_bars.png")
    pearson_svg = os.path.join(figures_dir, "Small_library_pearson_gain_bars.svg")
    plt.savefig(pearson_png, dpi=300, bbox_inches="tight")
    plt.savefig(pearson_svg, format="svg", bbox_inches="tight")
    plt.savefig("figures/evaluation/Small_library_pearson_gain_bars.png", dpi=300, bbox_inches="tight")
    plt.savefig("figures/evaluation/Small_library_pearson_gain_bars.svg", format="svg", bbox_inches="tight")
    plt.close()

    # 4. Dual-Metric Side-by-Side Panel Plot
    fig, (ax_r, ax_rho) = plt.subplots(1, 2, figsize=(20, 6.0), dpi=300)

    # Panel A: Pearson
    ax_r.bar(x - width, df_res["DeepPrime-FT (Pearson r)"], width, label="DeepPrime-FT", color="#7f8c8d", alpha=0.85, edgecolor="black")
    ax_r.bar(x, df_res["PRIDICT2.0 (Pearson r)"], width, label="PRIDICT2.0", color="#8DD3C7", alpha=0.9, edgecolor="black")
    ax_r.bar(x + width, df_res["DeepPrime6 (Pearson r)"], width, label="DeepPrime6", color="#d35400", alpha=0.9, edgecolor="black")

    for idx, row in df_res.iterrows():
        delta = row["Delta Pearson r (vs FT)"]
        ax_r.annotate(f"{row['DeepPrime-FT (Pearson r)']:.3f}", xy=(x[idx] - width, row["DeepPrime-FT (Pearson r)"] + 0.01),
                      ha="center", va="bottom", fontsize=8)
        ax_r.annotate(f"{row['PRIDICT2.0 (Pearson r)']:.3f}", xy=(x[idx], row["PRIDICT2.0 (Pearson r)"] + 0.01),
                      ha="center", va="bottom", fontsize=8)
        ax_r.annotate(f"{row['DeepPrime6 (Pearson r)']:.3f}", xy=(x[idx] + width, row["DeepPrime6 (Pearson r)"] + 0.01),
                      ha="center", va="bottom", fontsize=8, weight="bold")
        ax_r.annotate(f"+{delta:.3f}", xy=(x[idx] + width, row["DeepPrime6 (Pearson r)"] + 0.06),
                      ha="center", va="bottom", fontsize=8.5, color="#900c3f", weight="bold")

    ax_r.set_ylabel("Pearson Correlation ($r$)", fontsize=13, weight="bold")
    ax_r.set_title("A. Linear Calibration & Quantitative Accuracy ($r$)", fontsize=14, weight="bold")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([t.replace(" (", "\n(") for t in df_res["Evaluation Target"]], fontsize=10)
    ax_r.set_ylim(0, 0.98)
    ax_r.legend(frameon=True, fontsize=10.5, loc="upper left")
    sns.despine(ax=ax_r)

    # Panel B: Spearman
    ax_rho.bar(x - width, df_res["DeepPrime-FT (Spearman rho)"], width, label="DeepPrime-FT", color="#7f8c8d", alpha=0.85, edgecolor="black")
    ax_rho.bar(x, df_res["PRIDICT2.0 (Spearman rho)"], width, label="PRIDICT2.0", color="#8DD3C7", alpha=0.9, edgecolor="black")
    ax_rho.bar(x + width, df_res["DeepPrime6 (Spearman rho)"], width, label="DeepPrime6", color="#2980b9", alpha=0.9, edgecolor="black")

    for idx, row in df_res.iterrows():
        delta = row["Delta Spearman rho (vs FT)"]
        sign = "+" if delta >= 0 else ""
        color = "#1f618d" if delta >= 0 else "#555555"
        ax_rho.annotate(f"{row['DeepPrime-FT (Spearman rho)']:.3f}", xy=(x[idx] - width, row["DeepPrime-FT (Spearman rho)"] + 0.01),
                        ha="center", va="bottom", fontsize=8)
        ax_rho.annotate(f"{row['PRIDICT2.0 (Spearman rho)']:.3f}", xy=(x[idx], row["PRIDICT2.0 (Spearman rho)"] + 0.01),
                        ha="center", va="bottom", fontsize=8)
        ax_rho.annotate(f"{row['DeepPrime6 (Spearman rho)']:.3f}", xy=(x[idx] + width, row["DeepPrime6 (Spearman rho)"] + 0.01),
                        ha="center", va="bottom", fontsize=8, weight="bold")
        ax_rho.annotate(f"{sign}{delta:.3f}", xy=(x[idx] + width, row["DeepPrime6 (Spearman rho)"] + 0.06),
                        ha="center", va="bottom", fontsize=8.5, color=color, weight="bold")

    ax_rho.set_ylabel("Spearman Correlation ($\\rho$)", fontsize=13, weight="bold")
    ax_rho.set_title("B. Rank Ordering Consistency ($\\rho$)", fontsize=14, weight="bold")
    ax_rho.set_xticks(x)
    ax_rho.set_xticklabels([t.replace(" (", "\n(") for t in df_res["Evaluation Target"]], fontsize=10)
    ax_rho.set_ylim(0, 0.92)
    ax_rho.legend(frameon=True, fontsize=10.5, loc="upper right")
    sns.despine(ax=ax_rho)

    plt.suptitle("Small Library Generalization: DeepPrime-FT vs PRIDICT2.0 vs DeepPrime6 Across PE6 & PE7 Systems", fontsize=16, weight="bold", y=1.02)
    plt.tight_layout()

    dual_png = os.path.join(figures_dir, "Small_library_dual_metric_bars.png")
    dual_svg = os.path.join(figures_dir, "Small_library_dual_metric_bars.svg")
    plt.savefig(dual_png, dpi=300, bbox_inches="tight")
    plt.savefig(dual_svg, format="svg", bbox_inches="tight")
    plt.savefig("figures/evaluation/Small_library_dual_metric_bars.png", dpi=300, bbox_inches="tight")
    plt.savefig("figures/evaluation/Small_library_dual_metric_bars.svg", format="svg", bbox_inches="tight")
    plt.close()

    print(f"Saved all figures (Scatter, Pearson, Spearman, Dual-Metric) to {figures_dir}")
    print("Small Library Orthogonal Benchmark Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
