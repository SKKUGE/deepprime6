#!/usr/bin/env python3
"""
Generate comprehensive analysis plots for FRT small library predictions:
1. Model-to-model correlation heatmap (Pearson r and Spearman rho)
2. Overall model score distributions (boxplots & stripplots by model)
3. Breakdown by design Group (1bp, 2bp, 3bp, 5bp, 10bp, attB, structured, co-edit, off-target)
4. Prediction status breakdown pie chart

Reproducible directly from:
- `data/frt-library-prediction/frt_small_lib_predictions.csv`
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Generate analysis plots for FRT small library predictions.")
    parser.add_argument(
        "--results_file",
        type=str,
        default="data/frt-library-prediction/frt_small_lib_predictions.csv",
        help="Path to frt_small_lib_predictions.csv"
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="notebooks/plots/frt_small_library",
        help="Directory to save generated figure panels."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_file = args.results_file
    plots_dir = args.plots_dir

    if not os.path.exists(results_file):
        fallback = "data/frt-library-prediction/frt_small_lib_predictions.csv"
        if os.path.exists(fallback):
            results_file = fallback
        else:
            raise FileNotFoundError(f"Predictions CSV not found at: {results_file} or {fallback}")

    os.makedirs(plots_dir, exist_ok=True)
    print("==========================================================================", flush=True)
    print("=== Generating Analysis Figures for FRT Small Library Predictions ===", flush=True)
    print(f"  Source CSV: {results_file}", flush=True)
    print(f"  Plots Dir:  {plots_dir}", flush=True)
    print("==========================================================================", flush=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    df = pd.read_csv(results_file)

    model_cols = [
        "pred_dp_base",
        "pred_dp6_drnaseh",
        "pred_dp6_pe6a",
        "pred_dp6_pe6b",
        "pred_dp6_pe6c",
        "pred_pridict2"
    ]
    model_labels = [
        "DeepPrime-FT",
        "PE2max-ΔRNaseH",
        "PE6a",
        "PE6b",
        "PE6c",
        "PRIDICT2.0"
    ]
    model_name_map = dict(zip(model_cols, model_labels))

    # -------------------------------------------------------------
    # 1. Model-to-Model Correlation Heatmap (Commonly Scored N=109)
    # -------------------------------------------------------------
    common_df = df.dropna(subset=model_cols)
    n_common = len(common_df)

    corr_pearson = common_df[model_cols].corr(method="pearson").rename(index=model_name_map, columns=model_name_map)
    corr_spearman = common_df[model_cols].corr(method="spearman").rename(index=model_name_map, columns=model_name_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(corr_pearson, annot=False, cmap="Blues", vmin=0, vmax=1, ax=axes[0], cbar_kws={"label": "Pearson r"})
    for i in range(corr_pearson.shape[0]):
        for j in range(corr_pearson.shape[1]):
            val = corr_pearson.iloc[i, j]
            axes[0].text(j + 0.5, i + 0.5, f"{val:.3f}", ha="center", va="center",
                         color="white" if val > 0.65 else "#111111", fontsize=10, fontweight="bold")
    axes[0].set_title(f"Inter-Model Pearson Correlation (N={n_common})", fontsize=13, fontweight="bold", pad=12)

    sns.heatmap(corr_spearman, annot=False, cmap="YlGnBu", vmin=0, vmax=1, ax=axes[1], cbar_kws={"label": "Spearman rho"})
    for i in range(corr_spearman.shape[0]):
        for j in range(corr_spearman.shape[1]):
            val = corr_spearman.iloc[i, j]
            axes[1].text(j + 0.5, i + 0.5, f"{val:.3f}", ha="center", va="center",
                         color="white" if val > 0.65 else "#111111", fontsize=10, fontweight="bold")
    axes[1].set_title(f"Inter-Model Spearman Correlation (N={n_common})", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    heatmap_path = os.path.join(plots_dir, "inter_model_correlation_heatmaps.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"  [1/4] Saved: {heatmap_path}", flush=True)

    # -------------------------------------------------------------
    # 2. Overall Model Score Distributions (Box + Strip plot)
    # -------------------------------------------------------------
    melted = df.melt(
        id_vars=["INDEX", "Group", "Edit type"],
        value_vars=model_cols,
        var_name="Model",
        value_name="Predicted Efficiency (%)"
    ).dropna()
    melted["Model Name"] = melted["Model"].map(model_name_map)

    palette = ["#4A90E2", "#50E3C2", "#F5A623", "#E94E77", "#9013FE", "#7ED321"]

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=melted,
        x="Model Name",
        y="Predicted Efficiency (%)",
        palette=palette,
        boxprops=dict(alpha=0.7),
        showmeans=True,
        meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white")
    )
    sns.stripplot(
        data=melted,
        x="Model Name",
        y="Predicted Efficiency (%)",
        color="black",
        alpha=0.35,
        jitter=0.2,
        size=5
    )
    plt.title("Predicted Editing Efficiency Distribution Across 6 Models (FRT Small Library)", fontsize=13, fontweight="bold", pad=14)
    plt.xlabel("Model", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Efficiency (%)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    box_path = os.path.join(plots_dir, "model_efficiency_distributions.png")
    plt.savefig(box_path, dpi=300)
    plt.close()
    print(f"  [2/4] Saved: {box_path}", flush=True)

    # -------------------------------------------------------------
    # 3. Efficiency by Library Group
    # -------------------------------------------------------------
    group_order = ["1bp", "2bp", "3bp", "5bp", "10bp", "structured", "co-edit", "off-target", "attB"]
    plt.figure(figsize=(15, 7))
    sns.barplot(
        data=melted,
        x="Group",
        y="Predicted Efficiency (%)",
        hue="Model Name",
        order=group_order,
        palette=palette,
        errorbar="se"
    )
    plt.title("Predicted Efficiency by Design Group (Mean ± SE)", fontsize=14, fontweight="bold", pad=14)
    plt.xlabel("Library Group", fontsize=12, fontweight="bold")
    plt.ylabel("Predicted Efficiency (%)", fontsize=12, fontweight="bold")
    plt.legend(title="Model", title_fontsize="10", loc="upper right")
    plt.tight_layout()
    group_path = os.path.join(plots_dir, "predicted_efficiency_by_group.png")
    plt.savefig(group_path, dpi=300)
    plt.close()
    print(f"  [3/4] Saved: {group_path}", flush=True)

    # -------------------------------------------------------------
    # 4. Status Breakdown Pie Chart
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 7))
    status_counts = df["prediction_status"].value_counts()
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    explode = [0.05 if i == 0 else 0.1 for i in range(len(status_counts))]

    plt.pie(
        status_counts.values,
        labels=[f"{k}\n(N={v})" for k, v in status_counts.items()],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors[:len(status_counts)],
        explode=explode,
        textprops={"fontsize": 11, "fontweight": "bold"}
    )
    plt.title(f"FRT Small Library Prediction Completeness (Total N={len(df)})", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    pie_path = os.path.join(plots_dir, "prediction_status_breakdown.png")
    plt.savefig(pie_path, dpi=300)
    plt.close()
    print(f"  [4/4] Saved: {pie_path}", flush=True)

    print("\n[Complete] Successfully generated all 4 analysis figures from predictions CSV!", flush=True)


if __name__ == "__main__":
    main()
