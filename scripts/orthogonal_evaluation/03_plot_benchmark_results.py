#!/usr/bin/env python3
"""
Orthogonal Benchmark: Step 3 - Optimal Visualization & Correlation Analysis Module
Author: DeepPrime6 Project

Purpose:
1. Loads raw prediction CSVs directly from results directory:
   - `deepprime_test_raw_predictions.csv` (or `deepprime_test_predictions.csv`)
   - `pridict2_test_raw_predictions.csv` (or `pridict2_test_predictions.csv`)
   - `performance_summary.csv`
2. Generates all 8 publication-ready benchmark figures into panels directory:
   - `ground_truth_correlation_heatmap.png`: Model vs Benchmark Test Sets correlation matrix
   - `spearman_heatmap_pairwise_matrices.png`: Full 7x7 pairwise Spearman correlation heatmaps
   - `pearson_heatmap_pairwise_matrices.png`: Full 7x7 pairwise Pearson correlation heatmaps
   - `decile_calibration_curves.png`: Model predicted decile vs measured editing efficiency
   - `barplot_correlation_comparison.png`: Correlation comparison bar chart
   - `boxplot_deepprime_test_predictions.png`: DeepPrime test set prediction distribution boxplot
   - `boxplot_pridict2_test_predictions.png`: PRIDICT2.0 test set prediction distribution boxplot
   - `boxplot_combined_benchmarks.png`: 2-panel combined prediction distribution boxplot
3. Exports correlation matrix CSVs:
   - `correlation_matrix_deepprime_spearman.csv`
   - `correlation_matrix_deepprime_pearson.csv`
   - `correlation_matrix_pridict2_spearman.csv`
   - `correlation_matrix_pridict2_pearson.csv`
   - `ground_truth_correlations_summary.csv`
"""

import os
import shutil
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.matrix as sm
from scipy.stats import spearmanr, pearsonr

# Monkeypatch Seaborn 0.12.2 bug with Matplotlib >= 3.8 where QuadMesh.get_array() returns a 2D array,
# causing zip() in _HeatMapper._annotate_heatmap to terminate prematurely after only nrows elements.
_orig_annotate_heatmap = sm._HeatMapper._annotate_heatmap

def _patched_annotate_heatmap(self, ax, mesh):
    mesh.update_scalarmappable()
    height, width = self.annot_data.shape
    xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
    array = mesh.get_array()
    array_iter = array.flat if hasattr(array, 'flat') else np.ravel(array)
    for x, y, m, color, val in zip(xpos.flat, ypos.flat,
                                   array_iter, mesh.get_facecolors(),
                                   self.annot_data.flat):
        if m is not np.ma.masked:
            lum = sm.relative_luminance(color)
            text_color = ".15" if lum > .408 else "w"
            annotation = ("{:" + self.fmt + "}").format(val)
            text_kwargs = dict(color=text_color, ha="center", va="center")
            if self.annot_kws:
                text_kwargs.update(self.annot_kws)
            ax.text(x, y, annotation, **text_kwargs)

sm._HeatMapper._annotate_heatmap = _patched_annotate_heatmap


def set_plot_style():
    """Sets standard aesthetic styling matching notebooks/plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 16,
        'legend.fontsize': 11,
    })


def find_pred_file(dir_path, candidates):
    for c in candidates:
        full = os.path.join(dir_path, c)
        if os.path.exists(full):
            return full
    raise FileNotFoundError(f"Could not find any of {candidates} in {dir_path}")


def run_supp_fig7_cross_inference(
    results_dir: str,
    cache_dir: str = "data/orthogonal_testset/supp_fig7_testsets",
    excel_path: str = "data/Library design/pairwise_library_design/OnTarget/fetching_from_DPlt=3nt;6K/src/Table S2.xlsx",
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Runs actual inference across all 5 Table S2 test datasets using:
    1. Pretrained DeepPrime and DeepPrime-FT models from genet (HEK293T variants).
    2. DeepPrime6 models (PE2max dRNaseH, PE6a, PE6b, PE6c).
    Computes Pearson correlation (r) for all 5 datasets x 9 models from real forward passes.
    """
    csv_path = os.path.join(results_dir, "Supplementary_Figure_7_generalizability_heatmap.csv")
    preds_path = os.path.join(results_dir, "Supplementary_Figure_7_all_predictions.parquet")
    if os.path.exists(csv_path) and not force_recompute:
        print(f"  Found existing Supplementary Figure 7 inference results: {csv_path}", flush=True)
        return pd.read_csv(csv_path)

    import glob
    import torch
    from scipy.stats import pearsonr
    from genet.predict.PrimeEditor import GeneInteractionModel
    from genet.models import LoadModel
    from src.utils.inference import (
        char_to_onehot_tensor,
        prepare_biofeatures,
        load_deepprime6_models,
        predict_deepprime6,
        FEATURE_RENAME_MAP
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Executing real model inference for Supplementary Figure 7 on {device}...", flush=True)

    os.makedirs(cache_dir, exist_ok=True)
    test_files = [
        ("PE2 (n = 28,883)", "pe2_test_28883.parquet"),
        ("PE2max (n = 639)", "pe2max_test_639.parquet"),
        ("PE2max-e (n = 589)", "pe2max_e_test_589.parquet"),
        ("PE4max (n = 491)", "pe4max_test_491.parquet"),
        ("PE4max-e (n = 644)", "pe4max_e_test_644.parquet")
    ]
    datasets = {}
    for label, fname in test_files:
        fpath = os.path.join(cache_dir, fname)
        if os.path.exists(fpath):
            datasets[label] = pd.read_parquet(fpath)
        else:
            if "PE2 " in label:
                csv_src = "data/Library design/pairwise_library_design/OnTarget/fetching_from_DPlt=3nt;6K/dst/Generalization_Test/Table_S2_1_Test_Set.csv"
                df_src = pd.read_csv(csv_src)
            else:
                sheet_idx = {
                    "PE2max (n = 639)": "3",
                    "PE2max-e (n = 589)": "7",
                    "PE4max (n = 491)": "4",
                    "PE4max-e (n = 644)": "8"
                }[label]
                df_all = pd.read_excel(excel_path, sheet_name=sheet_idx, header=3)
                df_src = df_all[df_all["Data set name"] == "Test"].reset_index(drop=True)
            df_src.to_parquet(fpath, index=False)
            datasets[label] = df_src

    genet_specs = {
        'DeepPrime': ('PE2', 'HEK293T'),
        'DeepPrime-FT, PE2max': ('PE2max', 'HEK293T'),
        'DeepPrime-FT, PE2max + epegRNA': ('PE2max-e', 'HEK293T'),
        'DeepPrime-FT, PE4max': ('PE4max', 'HEK293T'),
        'DeepPrime-FT, PE4max + epegRNA': ('PE4max-e', 'HEK293T'),
    }
    loaded_genet = {}
    for name, (pe_sys, cell) in genet_specs.items():
        mdir = LoadModel('DeepPrime', pe_sys, cell).model_dir
        mean = pd.read_csv(f"{mdir}/mean_231124.csv", header=None, index_col=0).squeeze().rename(index=FEATURE_RENAME_MAP)
        std = pd.read_csv(f"{mdir}/std_231124.csv", header=None, index_col=0).squeeze().rename(index=FEATURE_RENAME_MAP)
        pts = sorted(glob.glob(f"{mdir}/*.pt"))
        models = []
        for p in pts:
            m = GeneInteractionModel(128, 1).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            m.eval()
            models.append(m)
        loaded_genet[name] = {'models': models, 'mean': mean, 'std': std}

    dp6_models = load_deepprime6_models(device=device)

    corrs_rows = []
    all_pred_dfs = []

    for dname, df_data in datasets.items():
        y_true = df_data['Measured_PE_efficiency'].values
        wt_col = [c for c in df_data.columns if 'Wide target' in c][0]
        ed_col = [c for c in df_data.columns if 'Edited target' in c][0]
        g_tensor = char_to_onehot_tensor(df_data[wt_col], df_data[ed_col], device)

        pred_record = pd.DataFrame({
            'Dataset': [dname] * len(y_true),
            'Measured_PE_efficiency': y_true
        })
        row_corr = {'Test dataset': dname}

        for mname, minfo in loaded_genet.items():
            b_tensor = prepare_biofeatures(df_data, norm_mean_or_type=minfo['mean'], norm_std=minfo['std'], device=device)
            preds = []
            with torch.no_grad():
                for m in minfo['models']:
                    p = m(g_tensor, b_tensor).cpu().numpy().flatten()
                    preds.append(p)
            pred_raw = np.mean(preds, axis=0)
            pred_exp = np.exp(pred_raw) - 1.0
            r_val, _ = pearsonr(pred_exp, y_true)
            row_corr[mname] = round(float(r_val), 2)
            pred_record[f"pred_{mname}"] = pred_exp

        b_tensor_dp6 = prepare_biofeatures(df_data, norm_mean_or_type='pe6', device=device)
        preds_dp6 = predict_deepprime6(dp6_models, g_tensor, b_tensor_dp6)

        dp6_col_map = {
            'DeepPrime6, PE2max ΔRNaseH': 'drnaseh',
            'DeepPrime6, PE6a': 'pe6a',
            'DeepPrime6, PE6b': 'pe6b',
            'DeepPrime6, PE6c': 'pe6c'
        }
        for col_name, key in dp6_col_map.items():
            pred_dp6_vals = preds_dp6[key]
            r_val, _ = pearsonr(pred_dp6_vals, y_true)
            row_corr[col_name] = round(float(r_val), 2)
            pred_record[f"pred_{col_name}"] = pred_dp6_vals

        corrs_rows.append(row_corr)
        all_pred_dfs.append(pred_record)

    final_corr_df = pd.DataFrame(corrs_rows)
    final_corr_df.to_csv(csv_path, index=False)
    print(f"  Saved inference correlation matrix to: {csv_path}", flush=True)

    all_preds_df = pd.concat(all_pred_dfs, ignore_index=True)
    all_preds_df.to_parquet(preds_path, index=False)
    print(f"  Saved all raw predictions to: {preds_path}", flush=True)

    return final_corr_df


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations from results CSV files.")
    parser.add_argument("--results_dir", type=str,
                        default="data/orthogonal_testset",
                        help="Directory containing predictions and summary CSV files.")
    parser.add_argument("--panels_dir", type=str,
                        default="notebooks/plots/orthogonal_benchmark",
                        help="Directory to save generated figure panels.")
    parser.add_argument("--plots_dir", type=str,
                        default="notebooks/plots/orthogonal_benchmark",
                        help="Secondary directory to mirror figures and CSVs.")
    parser.add_argument("--recompute_supp_fig7", action="store_true", default=False,
                        help="Force real GPU/CPU inference across all 5 test sets and 9 models for Supplementary Figure 7.")
    args = parser.parse_args()

    set_plot_style()
    print("==========================================================================", flush=True)
    print("=== Step 3: Generating Benchmark Visualizations from Results CSVs ===", flush=True)
    print(f"  Results source: {args.results_dir}", flush=True)
    print(f"  Panels target:  {args.panels_dir}", flush=True)
    if args.plots_dir:
        print(f"  Plots mirror:   {args.plots_dir}", flush=True)
    print("==========================================================================", flush=True)

    os.makedirs(args.panels_dir, exist_ok=True)
    if args.plots_dir:
        os.makedirs(args.plots_dir, exist_ok=True)

    dp_pred_path = find_pred_file(args.results_dir, ["deepprime_test_predictions.csv", "deepprime_test_raw_predictions.csv"])
    pr_pred_path = find_pred_file(args.results_dir, ["pridict2_test_predictions.csv", "pridict2_test_raw_predictions.csv"])
    print(f"  Loaded DeepPrime predictions from: {dp_pred_path}", flush=True)
    print(f"  Loaded PRIDICT2 predictions from:  {pr_pred_path}", flush=True)
    perf_path = os.path.join(args.results_dir, "performance_summary.csv")

    dp_df = pd.read_csv(dp_pred_path)
    pr_df = pd.read_csv(pr_pred_path)

    # Standard model display names & tailored colors
    model_cols = [
        "Measured_PE_efficiency",
        "pred_dp_base",
        "pred_pridict2_hek",
        "pred_dp6_pe6a",
        "pred_dp6_pe6b",
        "pred_dp6_pe6c",
        "pred_dp6_drnaseh"
    ]
    model_labels = [
        "Ground Truth",
        "DeepPrime-FT",
        "PRIDICT2.0",
        "DP6 (PE6a)",
        "DP6 (PE6b)",
        "DP6 (PE6c)",
        "DP6 (dRNaseH)"
    ]
    palette = ["#4A5568", "#3182CE", "#319795", "#DD6B20", "#E53E3E", "#805AD5", "#2B6CB0"]

    dp_sub = dp_df[model_cols].copy()
    dp_sub.columns = model_labels

    pr_sub = pr_df[model_cols].copy()
    pr_sub.columns = model_labels

    # Compute correlation matrices
    dp_spearman = dp_sub.corr(method="spearman")
    dp_pearson = dp_sub.corr(method="pearson")
    pr_spearman = pr_sub.corr(method="spearman")
    pr_pearson = pr_sub.corr(method="pearson")

    # Export correlation matrices to CSV
    dp_spearman.to_csv(os.path.join(args.results_dir, "correlation_matrix_deepprime_spearman.csv"))
    dp_pearson.to_csv(os.path.join(args.results_dir, "correlation_matrix_deepprime_pearson.csv"))
    pr_spearman.to_csv(os.path.join(args.results_dir, "correlation_matrix_pridict2_spearman.csv"))
    pr_pearson.to_csv(os.path.join(args.results_dir, "correlation_matrix_pridict2_pearson.csv"))

    # Compute performance summary dataframe if not present or verify
    if os.path.exists(perf_path):
        perf_df = pd.read_csv(perf_path)
    else:
        perf_records = []
        eval_models_map = [
            ("DeepPrime-FT", "pred_dp_base"),
            ("PRIDICT2.0", "pred_pridict2_hek"),
            ("DeepPrime6-pe6a", "pred_dp6_pe6a"),
            ("DeepPrime6-pe6b", "pred_dp6_pe6b"),
            ("DeepPrime6-pe6c", "pred_dp6_pe6c"),
            ("DeepPrime6-pemaxdrnaseh", "pred_dp6_drnaseh"),
        ]
        for m_name, col in eval_models_map:
            sp, _ = spearmanr(dp_df["Measured_PE_efficiency"], dp_df[col])
            pe, _ = pearsonr(dp_df["Measured_PE_efficiency"], dp_df[col])
            perf_records.append({
                "Evaluation Benchmark": "DeepPrime Official Generalization Test Set",
                "Condition": "Strict Target Locus Stratification",
                "Model": m_name,
                "Spearman": round(sp, 4),
                "Pearson": round(pe, 4),
                "N": len(dp_df)
            })
        for m_name, col in eval_models_map:
            sp, _ = spearmanr(pr_df["Measured_PE_efficiency"], pr_df[col])
            pe, _ = pearsonr(pr_df["Measured_PE_efficiency"], pr_df[col])
            perf_records.append({
                "Evaluation Benchmark": "PRIDICT2.0 Official Fold 0 Test Set",
                "Condition": "Strict Target Locus Stratification",
                "Model": m_name,
                "Spearman": round(sp, 4),
                "Pearson": round(pe, 4),
                "N": len(pr_df)
            })
        perf_df = pd.DataFrame(perf_records)
        perf_df.to_csv(perf_path, index=False)

    generated_panels = []

    # -------------------------------------------------------------------------
    # PLOT 1: Ground Truth Correlation Heatmap
    # -------------------------------------------------------------------------
    print("Generating Plot 1: Ground Truth Focused Correlation Heatmaps...", flush=True)
    models_only = [m for m in model_labels if m != "Ground Truth"]
    gt_corr_data = {
        "Spearman ($\\rho$)\nDeepPrime Test": [dp_spearman.loc["Ground Truth", m] for m in models_only],
        "Spearman ($\\rho$)\nPRIDICT2.0 Test": [pr_spearman.loc["Ground Truth", m] for m in models_only],
        "Pearson ($r$)\nDeepPrime Test": [dp_pearson.loc["Ground Truth", m] for m in models_only],
        "Pearson ($r$)\nPRIDICT2.0 Test": [pr_pearson.loc["Ground Truth", m] for m in models_only],
    }
    gt_corr_df = pd.DataFrame(gt_corr_data, index=models_only)
    gt_corr_df.to_csv(os.path.join(args.results_dir, "ground_truth_correlations_summary.csv"))

    fig, ax = plt.subplots(figsize=(9.5, 6))
    sns.heatmap(
        gt_corr_df,
        annot=True,
        cmap="Blues",
        fmt=".3f",
        linewidths=1.5,
        vmin=-0.25,
        vmax=1.0,
        annot_kws={"size": 12, "weight": "bold"},
        cbar_kws={'label': "Correlation with Ground Truth"},
        ax=ax
    )
    ax.set_title("Cross-Evaluation Correlation with Ground Truth (Strict Target Stratification)", fontweight='bold', pad=15)
    ax.set_ylabel("Evaluated Model", fontweight='bold')
    plt.tight_layout()

    out_gt_heat = os.path.join(args.panels_dir, "ground_truth_correlation_heatmap.png")
    plt.savefig(out_gt_heat, dpi=300, bbox_inches='tight')
    plt.close()
    generated_panels.append(out_gt_heat)
    print(f"  Saved: {out_gt_heat}", flush=True)

    # -------------------------------------------------------------------------
    # PLOT 2: Full Pairwise Spearman Correlation Heatmap (2 Panels)
    # -------------------------------------------------------------------------
    print("Generating Plot 2: Full Pairwise Spearman Correlation Heatmaps...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    sns.heatmap(
        dp_spearman,
        annot=True,
        cmap="vlag",
        center=0,
        vmin=-0.3,
        vmax=1.0,
        fmt=".3f",
        linewidths=1.0,
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={'label': "Spearman Correlation ($\\rho$)"},
        ax=axes[0]
    )
    axes[0].set_title(f"DeepPrime Official Test Set (N = {len(dp_df):,})\nStrictly Unseen Targets", fontweight='bold', pad=12)

    sns.heatmap(
        pr_spearman,
        annot=True,
        cmap="vlag",
        center=0,
        vmin=-0.3,
        vmax=1.0,
        fmt=".3f",
        linewidths=1.0,
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={'label': "Spearman Correlation ($\\rho$)"},
        ax=axes[1]
    )
    axes[1].set_title(f"PRIDICT2.0 Official Fold 0 Test Set (N = {len(pr_df):,})\nStrictly Unseen Targets", fontweight='bold', pad=12)

    plt.suptitle("Pairwise Spearman Correlation Matrices Across Models & Ground Truth", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_sp_heat = os.path.join(args.panels_dir, "spearman_heatmap_pairwise_matrices.png")
    plt.savefig(out_sp_heat, dpi=300, bbox_inches='tight')
    plt.close()
    generated_panels.append(out_sp_heat)
    print(f"  Saved: {out_sp_heat}", flush=True)

    # -------------------------------------------------------------------------
    # PLOT 3: Full Pairwise Pearson Correlation Heatmap (2 Panels)
    # -------------------------------------------------------------------------
    print("Generating Plot 3: Full Pairwise Pearson Correlation Heatmaps...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    sns.heatmap(
        dp_pearson,
        annot=True,
        cmap="vlag",
        center=0,
        vmin=-0.3,
        vmax=1.0,
        fmt=".3f",
        linewidths=1.0,
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={'label': "Pearson Correlation ($r$)"},
        ax=axes[0]
    )
    axes[0].set_title(f"DeepPrime Official Test Set (N = {len(dp_df):,})\nStrictly Unseen Targets", fontweight='bold', pad=12)

    sns.heatmap(
        pr_pearson,
        annot=True,
        cmap="vlag",
        center=0,
        vmin=-0.3,
        vmax=1.0,
        fmt=".3f",
        linewidths=1.0,
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={'label': "Pearson Correlation ($r$)"},
        ax=axes[1]
    )
    axes[1].set_title(f"PRIDICT2.0 Official Fold 0 Test Set (N = {len(pr_df):,})\nStrictly Unseen Targets", fontweight='bold', pad=12)

    plt.suptitle("Pairwise Pearson Correlation Matrices Across Models & Ground Truth", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_pe_heat = os.path.join(args.panels_dir, "pearson_heatmap_pairwise_matrices.png")
    plt.savefig(out_pe_heat, dpi=300, bbox_inches='tight')
    plt.close()
    generated_panels.append(out_pe_heat)
    print(f"  Saved: {out_pe_heat}", flush=True)

    # -------------------------------------------------------------------------
    # PLOT 4: Decile Calibration Curves
    # -------------------------------------------------------------------------
    print("Generating Plot 4: Decile Calibration & Predictive Ranking Curves...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    eval_models = [
        ("DeepPrime-FT", "pred_dp_base", "#3182CE", "s-"),
        ("PRIDICT2.0", "pred_pridict2_hek", "#319795", "^-"),
        ("DP6 (PE6a)", "pred_dp6_pe6a", "#DD6B20", "o-"),
        ("DP6 (PE6b)", "pred_dp6_pe6b", "#E53E3E", "D-"),
        ("DP6 (PE6c)", "pred_dp6_pe6c", "#805AD5", "v-"),
        ("DP6 (dRNaseH)", "pred_dp6_drnaseh", "#2B6CB0", "P-"),
    ]

    # Panel A: DeepPrime Test Set
    for name, col, color, m_style in eval_models:
        ranks = pd.qcut(dp_df[col].rank(method='first'), q=10, labels=False) + 1
        decile_means = dp_df.groupby(ranks)["Measured_PE_efficiency"].mean()
        axes[0].plot(decile_means.index, decile_means.values, m_style, label=name, color=color, linewidth=2, markersize=6)
    axes[0].set_title(f"DeepPrime Official Test Set (N = {len(dp_df):,})", fontweight='bold')
    axes[0].set_xlabel("Predicted Efficiency Decile (1 = Lowest, 10 = Highest)", fontweight='bold')
    axes[0].set_ylabel("Mean Measured PE Efficiency (%)", fontweight='bold')
    axes[0].set_xticks(range(1, 11))
    axes[0].legend(loc="upper left", frameon=True)

    # Panel B: PRIDICT2.0 Test Set
    for name, col, color, m_style in eval_models:
        ranks = pd.qcut(pr_df[col].rank(method='first'), q=10, labels=False) + 1
        decile_means = pr_df.groupby(ranks)["Measured_PE_efficiency"].mean()
        axes[1].plot(decile_means.index, decile_means.values, m_style, label=name, color=color, linewidth=2, markersize=6)
    axes[1].set_title(f"PRIDICT2.0 Official Fold 0 Test Set (N = {len(pr_df):,})", fontweight='bold')
    axes[1].set_xlabel("Predicted Efficiency Decile (1 = Lowest, 10 = Highest)", fontweight='bold')
    axes[1].set_ylabel("Mean Measured PE Efficiency (%)", fontweight='bold')
    axes[1].set_xticks(range(1, 11))
    axes[1].legend(loc="upper left", frameon=True)

    plt.suptitle("Model Decile Calibration: Predictive Ranking Power Across Dynamic Ranges", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_decile = os.path.join(args.panels_dir, "decile_calibration_curves.png")
    plt.savefig(out_decile, dpi=300, bbox_inches='tight')
    plt.close()
    generated_panels.append(out_decile)
    print(f"  Saved: {out_decile}", flush=True)

    # -------------------------------------------------------------------------
    # PLOT 5: Barplot Correlation Comparison
    # -------------------------------------------------------------------------
    print("Generating Plot 5: Correlation Comparison Bar Chart...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    dp_perf = perf_df[perf_df["Evaluation Benchmark"].str.contains("DeepPrime")].copy()
    pr_perf = perf_df[perf_df["Evaluation Benchmark"].str.contains("PRIDICT2")].copy()

    # Panel A: DeepPrime Test Set
    x = np.arange(len(dp_perf))
    width = 0.35
    axes[0].bar(x - width/2, dp_perf["Spearman"], width, label="Spearman ($\\rho$)", color="#3182CE", alpha=0.9)
    axes[0].bar(x + width/2, dp_perf["Pearson"], width, label="Pearson ($r$)", color="#ED8936", alpha=0.9)
    axes[0].set_title(f"DeepPrime Official Test Set (N = {dp_perf['N'].iloc[0]:,})", fontweight='bold')
    axes[0].set_ylabel("Correlation Coefficient", fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dp_perf["Model"], rotation=30, ha='right')
    axes[0].set_ylim(-0.1, 0.85)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].legend(loc="upper right", frameon=True)

    for i in range(len(dp_perf)):
        axes[0].text(x[i] - width/2, dp_perf["Spearman"].iloc[i] + 0.02, f"{dp_perf['Spearman'].iloc[i]:.3f}", ha='center', va='bottom', fontsize=9)
        axes[0].text(x[i] + width/2, dp_perf["Pearson"].iloc[i] + 0.02, f"{dp_perf['Pearson'].iloc[i]:.3f}", ha='center', va='bottom', fontsize=9)

    # Panel B: PRIDICT2 Test Set
    x2 = np.arange(len(pr_perf))
    axes[1].bar(x2 - width/2, pr_perf["Spearman"], width, label="Spearman ($\\rho$)", color="#3182CE", alpha=0.9)
    axes[1].bar(x2 + width/2, pr_perf["Pearson"], width, label="Pearson ($r$)", color="#ED8936", alpha=0.9)
    axes[1].set_title(f"PRIDICT2.0 Fold 0 Test Set (N = {pr_perf['N'].iloc[0]:,})", fontweight='bold')
    axes[1].set_ylabel("")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(pr_perf["Model"], rotation=30, ha='right')
    axes[1].set_ylim(-0.25, 1.0)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].legend(loc="upper left", frameon=True)

    for i in range(len(pr_perf)):
        val_sp = pr_perf["Spearman"].iloc[i]
        val_pe = pr_perf["Pearson"].iloc[i]
        axes[1].text(x2[i] - width/2, val_sp + 0.02 if val_sp >= 0 else val_sp - 0.05, f"{val_sp:.3f}", ha='center', va='bottom' if val_sp >= 0 else 'top', fontsize=9)
        axes[1].text(x2[i] + width/2, val_pe + 0.02 if val_pe >= 0 else val_pe - 0.05, f"{val_pe:.3f}", ha='center', va='bottom' if val_pe >= 0 else 'top', fontsize=9)

    plt.suptitle("Orthogonal Benchmark Cross-Evaluation: Model Correlation Performance", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_bar = os.path.join(args.panels_dir, "barplot_correlation_comparison.png")
    plt.savefig(out_bar, dpi=300, bbox_inches='tight')
    plt.close()
    generated_panels.append(out_bar)
    print(f"  Saved: {out_bar}", flush=True)

    # -------------------------------------------------------------------------
    # PLOT 6: Boxplot DeepPrime Generalization Test Set
    # -------------------------------------------------------------------------
    print("Generating Plot 6: Boxplot DeepPrime Generalization Test Set...", flush=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dp_sub, palette=palette, width=0.6, fliersize=2, linewidth=1.2)
    plt.title(f"Prediction Distributions on DeepPrime Generalization Test Set (N = {len(dp_df):,})\nStrict Target Locus Stratification", fontweight='bold', pad=12)
    plt.ylabel("Prime Editing Efficiency (%)", fontweight='bold')
    plt.xlabel("Model / Ground Truth", fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    out_dp_box = os.path.join(args.panels_dir, "boxplot_deepprime_test_predictions.png")
    plt.savefig(out_dp_box, dpi=300, bbox_inches='tight')
    plt.close()
    generated_panels.append(out_dp_box)
    print(f"  Saved: {out_dp_box}", flush=True)

    # -------------------------------------------------------------------------
    # PLOT 7: Boxplot PRIDICT2.0 Fold 0 Test Set
    # -------------------------------------------------------------------------
    print("Generating Plot 7: Boxplot PRIDICT2.0 Fold 0 Test Set...", flush=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pr_sub, palette=palette, width=0.6, fliersize=2, linewidth=1.2)
    plt.title(f"Prediction Distributions on PRIDICT2.0 Fold 0 Test Set (N = {len(pr_df):,})\nStrict Target Locus Stratification", fontweight='bold', pad=12)
    plt.ylabel("Prime Editing Efficiency (%)", fontweight='bold')
    plt.xlabel("Model / Ground Truth", fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    out_pr_box = os.path.join(args.panels_dir, "boxplot_pridict2_test_predictions.png")
    plt.savefig(out_pr_box, dpi=300, bbox_inches='tight')
    plt.close()
    generated_panels.append(out_pr_box)
    print(f"  Saved: {out_pr_box}", flush=True)

    # -------------------------------------------------------------------------
    # PLOT 8: Combined 2-Panel Boxplot Figure
    # -------------------------------------------------------------------------
    print("Generating Plot 8: Combined 2-Panel Boxplot Figure...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    sns.boxplot(data=dp_sub, palette=palette, width=0.55, fliersize=1.5, linewidth=1.1, ax=axes[0])
    axes[0].set_title(f"DeepPrime Official Test Set (N = {len(dp_df):,})", fontweight='bold')
    axes[0].set_ylabel("Editing Efficiency (%)", fontweight='bold')
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=30)

    sns.boxplot(data=pr_sub, palette=palette, width=0.55, fliersize=1.5, linewidth=1.1, ax=axes[1])
    axes[1].set_title(f"PRIDICT2.0 Official Fold 0 Test Set (N = {len(pr_df):,})", fontweight='bold')
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=30)

    plt.suptitle("Orthogonal Benchmark Cross-Evaluation: Model Prediction Distributions", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_comb_box = os.path.join(args.panels_dir, "boxplot_combined_benchmarks.png")
    plt.savefig(out_comb_box, dpi=300, bbox_inches='tight')
    plt.close()
    generated_panels.append(out_comb_box)
    print(f"  Saved: {out_comb_box}", flush=True)

    # -------------------------------------------------------------------------
    # PLOT 9: Supplementary Figure 7 Generalizability Heatmap (5 Test Sets x 9 Models)
    # -------------------------------------------------------------------------
    print("Generating Plot 9: Supplementary Figure 7 Generalizability Heatmap...", flush=True)
    supp_fig7_csv_df = run_supp_fig7_cross_inference(
        results_dir=args.results_dir,
        force_recompute=args.recompute_supp_fig7
    )

    plot_df = supp_fig7_csv_df.copy()
    plot_df["Test dataset"] = (
        plot_df["Test dataset"]
        .str.replace(" (n = ", "\n(n = ", regex=False)
    )
    supp_fig7_plot_df = plot_df.set_index("Test dataset")
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)
    hm = sns.heatmap(
        supp_fig7_plot_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.4,
        vmax=0.70,
        cbar_kws={"label": "Pearson's correlation (r)", "ticks": [0.0, 0.7]},
        linewidths=0.0,
        ax=ax,
        annot_kws={"size": 11},
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.set_xlabel("Prediction model", fontsize=13, labelpad=10)
    ax.set_ylabel("Test dataset", fontsize=13, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    plt.tight_layout()

    out_supp7_png = os.path.join(args.panels_dir, "Supplementary_Figure_7_generalizability_heatmap.png")
    out_supp7_svg = os.path.join(args.panels_dir, "Supplementary_Figure_7_generalizability_heatmap.svg")
    plt.savefig(out_supp7_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_supp7_svg, format="svg", bbox_inches="tight")
    plt.close()
    generated_panels.extend([out_supp7_png, out_supp7_svg])
    print(f"  Saved: {out_supp7_png} & {out_supp7_svg}", flush=True)

    # -------------------------------------------------------------
    # Mirror all figures and tables to plots_dir if specified
    # -------------------------------------------------------------
    if args.plots_dir and os.path.abspath(args.plots_dir) != os.path.abspath(args.panels_dir):
        for f in generated_panels:
            shutil.copy(f, args.plots_dir)
        csv_files = [
            "correlation_matrix_deepprime_spearman.csv",
            "correlation_matrix_deepprime_pearson.csv",
            "correlation_matrix_pridict2_spearman.csv",
            "correlation_matrix_pridict2_pearson.csv",
            "ground_truth_correlations_summary.csv",
            "Supplementary_Figure_7_generalizability_heatmap.csv",
            "Supplementary_Figure_7_all_predictions.parquet",
        ]
        for cf in csv_files:
            src_cf = os.path.join(args.results_dir, cf)
            if os.path.exists(src_cf):
                shutil.copy(src_cf, args.plots_dir)
        print(f"Mirrored all {len(generated_panels)} figures to: {args.plots_dir}", flush=True)

    print(f"\n[Complete] Successfully generated all {len(generated_panels)} figure panels directly from results CSVs!", flush=True)


if __name__ == "__main__":
    main()
