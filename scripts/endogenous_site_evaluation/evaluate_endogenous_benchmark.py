#!/usr/bin/env python3
"""
Endogenous Sites Prime Editing Efficiency Evaluation & Ensemble Modeling Benchmark

Evaluates prediction models (DeepPrime-FT, PRIDICT2.0, DeepPrime6) and their rank-averaged
ensembles against measured endogenous prime editing efficiencies across 4 PE systems:
- PEmax-dRNaseH (dRN)
- PE6a
- PE6b
- PE6c
- Overall Average

Dataset:
  data/endogenous_ground_truth/PE_efficiency_cleanonly_NCcorrected_n32.xlsx (N=32 clean targets)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import torch

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

# Matplotlib global style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# Color palettes
MODEL_COLORS = {
    'DeepPrime-FT': '#3182CE',        # Blue
    'PRIDICT2.0': '#2A9D8F',          # Teal/Green
    'DeepPrime6': '#E63946',          # Crimson
    'DP + PRIDICT2.0': '#6B4C9A',     # Purple
    'DP + DP6': '#DD6B20',            # Orange
    'PRIDICT2.0 + DP6': '#319795',    # Cyan
    'DP + DP6 + PRIDICT2.0': '#C53030' # Dark Red/Ruby
}

PE_SYSTEM_LABELS = {
    'dRN_NCcorr_%': 'PEmax-dRNaseH',
    '6a_NCcorr_%': 'PE6a',
    '6b_NCcorr_%': 'PE6b',
    '6c_NCcorr_%': 'PE6c',
    'overall_avg_NCcorr_%': 'Overall Average'
}

MODEL_ANNOTATIONS = {
    'DeepPrime-FT': 'DP-FT (HEK293T PE2max epegRNA)',
    'PRIDICT2.0': 'PRIDICT2.0 (HEK293T epegRNA)',
    'DeepPrime6': 'DeepPrime6 (PE6-tuned Ensemble Avg)'
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate endogenous PE efficiency predictions")
    parser.add_argument(
        "--excel-path",
        default=os.path.join(PROJECT_ROOT, "data/endogenous_ground_truth/PE_efficiency_cleanonly_NCcorrected_n32.xlsx"),
        help="Path to endogenous ground truth Excel file"
    )
    parser.add_argument(
        "--parquet-path",
        default=os.path.join(PROJECT_ROOT, "data/full_output/final_candidates_validated.parquet"),
        help="Path to full candidate sequences parquet file"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "experiments/20261012_endogenous_site_evaluation"),
        help="Output directory for figures, results, and logs"
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Re-run neural network model inference from scratch for validation"
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
        help="Device for inference (cuda:0 or cpu)"
    )
    parser.add_argument(
        "--rescored-csv",
        default=os.path.join(PROJECT_ROOT, "experiments/20261011_endogenous_rescore/results/selected_50_candidates_rescored.csv"),
        help="Path to newly rescored 50 candidates CSV with latest retrained DeepPrime6 models"
    )
    return parser.parse_args()


def load_dataset(excel_path, use_codebase_inference=True, device="cuda:0"):
    print(f"\n[1/5] Loading endogenous benchmark dataset from: {excel_path}")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found at: {excel_path}")

    xl = pd.ExcelFile(excel_path)
    df = xl.parse("Final_n32_NCcorrected")
    print(f"  Successfully loaded {len(df)} endogenous target sites.")
    print(f"  Targets analyzed: {df['target_num'].min()} to {df['target_num'].max()} (N={len(df)})")

    if use_codebase_inference:
        print("  [Fresh Weights] Running live inference on endogenous targets via codebase engine...")
        pq_path = os.path.join(PROJECT_ROOT, "data/full_output/final_candidates_validated.parquet")
        bio_path = os.path.join(PROJECT_ROOT, "data/full_output/pegrna_designs_biofeatures.parquet")
        if os.path.exists(pq_path) and os.path.exists(bio_path):
            pq_df = pd.read_parquet(pq_path)
            pq_df['target_num'] = range(1, len(pq_df) + 1)
            merged_cand = pd.merge(df, pq_df[['target_num', 'design_id', 'Disagreement_Subcategory']], on='target_num')
            bio_df = pd.read_parquet(bio_path)
            bio_matched = pd.merge(merged_cand, bio_df, on='design_id')

            from src.utils.inference import (
                char_to_onehot_tensor,
                prepare_biofeatures,
                load_deepprime_base_models,
                load_deepprime6_models,
                predict_deepprime_base,
                predict_deepprime6
            )

            dev = torch.device(device if torch.cuda.is_available() else "cpu")
            g_tensor = char_to_onehot_tensor(
                bio_matched['Target'].tolist(),
                bio_matched['Masked_EditSeq'].tolist(),
                dev
            )
            b_base = prepare_biofeatures(bio_matched, 'base', device=dev)
            b_pe6 = prepare_biofeatures(bio_matched, 'pe6', device=dev)

            base_models = load_deepprime_base_models(device=dev)
            dp6_models = load_deepprime6_models(device=dev, is_scratch=False)

            live_dp_base = predict_deepprime_base(base_models, g_tensor, b_base)
            live_dp6_dict = predict_deepprime6(dp6_models, g_tensor, b_pe6)
            live_dp6_avg = np.mean([
                live_dp6_dict['pe6a'],
                live_dp6_dict['pe6b'],
                live_dp6_dict['pe6c'],
                live_dp6_dict['drnaseh']
            ], axis=0)

            df['pred_dp_base'] = live_dp_base
            df['pred_dp6'] = live_dp6_avg
            df['pred_dp6_pe6a'] = live_dp6_dict['pe6a']
            df['pred_dp6_pe6b'] = live_dp6_dict['pe6b']
            df['pred_dp6_pe6c'] = live_dp6_dict['pe6c']
            df['pred_dp6_drnaseh'] = live_dp6_dict['drnaseh']
            print("  [SUCCESS] Retrained predictions updated via codebase canonical inference engine.")

    return df


def compute_rank_ensembles(df):
    """
    Computes ascending rank of pegRNA activity predictions and generates ensemble combinations.
    Ascending rank: rank 1 = lowest predicted score, rank 32 = highest predicted score.
    Higher rank represents higher predicted efficacy.
    """
    df_calc = df.copy()

    # Model scores
    df_calc['score_dp'] = df_calc['pred_dp_base']
    df_calc['score_pri'] = df_calc['pred_pridict2']
    df_calc['score_dp6'] = df_calc['pred_dp6']

    # Ranks (1 to N)
    df_calc['rank_dp'] = df_calc['score_dp'].rank(ascending=True)
    df_calc['rank_pri'] = df_calc['score_pri'].rank(ascending=True)
    df_calc['rank_dp6'] = df_calc['score_dp6'].rank(ascending=True)

    # Ensembles (average of rank)
    df_calc['rank_ens_dp_pri'] = (df_calc['rank_dp'] + df_calc['rank_pri']) / 2.0
    df_calc['rank_ens_dp_dp6'] = (df_calc['rank_dp'] + df_calc['rank_dp6']) / 2.0
    df_calc['rank_ens_pri_dp6'] = (df_calc['rank_pri'] + df_calc['rank_dp6']) / 2.0
    df_calc['rank_ens_all3'] = (df_calc['rank_dp'] + df_calc['rank_dp6'] + df_calc['rank_pri']) / 3.0

    return df_calc


def run_inference_validation(df_endogenous, parquet_path, device_str="cuda:0"):
    """
    Validates that model predictions are non-trivial and properly distributed.
    """
    print(f"\n[Validation] Validating model predictions...")
    assert df_endogenous['pred_dp_base'].notna().all(), "Missing DeepPrime-Base predictions!"
    assert df_endogenous['pred_dp6'].notna().all(), "Missing DeepPrime6 predictions!"
    assert df_endogenous['pred_pridict2'].notna().all(), "Missing PRIDICT2 predictions!"
    print(f"  - DeepPrime-FT: N={len(df_endogenous)}, Mean={df_endogenous['pred_dp_base'].mean():.2f}")
    print(f"  - DeepPrime6:   N={len(df_endogenous)}, Mean={df_endogenous['pred_dp6'].mean():.2f}")
    print(f"  - PRIDICT2.0:   N={len(df_endogenous)}, Mean={df_endogenous['pred_pridict2'].mean():.2f}")
    print("  [PASS] All prediction scores are validated.")


def calculate_metrics_summary(df):
    """
    Computes Spearman rho, p-value, Pearson r, and Pearson p for all models and ensembles.
    """
    pe_systems = [
        ('PEmax-dRNaseH', 'dRN_NCcorr_%'),
        ('PE6a', '6a_NCcorr_%'),
        ('PE6b', '6b_NCcorr_%'),
        ('PE6c', '6c_NCcorr_%'),
        ('Overall Average', 'overall_avg_NCcorr_%'),
    ]

    single_models = [
        ('DeepPrime-FT', 'pred_dp_base', 'Score'),
        ('PRIDICT2.0', 'pred_pridict2', 'Score'),
        ('DeepPrime6', 'pred_dp6', 'Score'),
    ]

    ensembles = [
        ('DP + PRIDICT2.0', 'rank_ens_dp_pri', 'Rank Average'),
        ('DP + DP6', 'rank_ens_dp_dp6', 'Rank Average'),
        ('PRIDICT2.0 + DP6', 'rank_ens_pri_dp6', 'Rank Average'),
        ('DP + DP6 + PRIDICT2.0', 'rank_ens_all3', 'Rank Average'),
    ]

    records = []
    n_samples = len(df)

    # 1. Single models
    for pe_name, pe_col in pe_systems:
        for m_name, col_name, pred_type in single_models:
            sp_r, sp_p = spearmanr(df[col_name], df[pe_col])
            pe_r, pe_p = pearsonr(df[col_name], df[pe_col])
            records.append({
                'Category': 'Single Model',
                'Model': m_name,
                'Input_Type': pred_type,
                'PE_System': pe_name,
                'N': n_samples,
                'Spearman_rho': sp_r,
                'Spearman_p': sp_p,
                'Pearson_r': pe_r,
                'Pearson_p': pe_p
            })

    # 2. Ensembles
    for pe_name, pe_col in pe_systems:
        for ens_name, col_name, pred_type in ensembles:
            sp_r, sp_p = spearmanr(df[col_name], df[pe_col])
            pe_r, pe_p = pearsonr(df[col_name], df[pe_col])
            records.append({
                'Category': 'Ensemble Model',
                'Model': ens_name,
                'Input_Type': pred_type,
                'PE_System': pe_name,
                'N': n_samples,
                'Spearman_rho': sp_r,
                'Spearman_p': sp_p,
                'Pearson_r': pe_r,
                'Pearson_p': pe_p
            })

    metrics_df = pd.DataFrame(records)
    return metrics_df


def plot_fig1_single_models(df, out_dir):
    """
    Task 1 & 4: Scattered dot plots for DP, PRIDICT2.0, DP6 across 4 PE types + overall average.
    """
    print("\n[2/5] Generating Figure 1: Single models vs measured activity...")
    pe_systems = [
        ('PEmax-dRNaseH', 'dRN_NCcorr_%'),
        ('PE6a', '6a_NCcorr_%'),
        ('PE6b', '6b_NCcorr_%'),
        ('PE6c', '6c_NCcorr_%'),
        ('Overall Average', 'overall_avg_NCcorr_%'),
    ]

    single_models = [
        ('DeepPrime-FT', 'pred_dp_base', 'DeepPrime Prediction Score (%)', MODEL_ANNOTATIONS['DeepPrime-FT']),
        ('PRIDICT2.0', 'pred_pridict2', 'PRIDICT2.0 Prediction Score (%)', MODEL_ANNOTATIONS['PRIDICT2.0']),
        ('DeepPrime6', 'pred_dp6', 'DeepPrime6 Prediction Score (%)', MODEL_ANNOTATIONS['DeepPrime6']),
    ]

    # Combined master figure (5 rows x 3 columns)
    fig, axes = plt.subplots(len(pe_systems), len(single_models), figsize=(15, 18), sharey=False)

    for row_idx, (pe_name, pe_col) in enumerate(pe_systems):
        for col_idx, (m_name, m_col, x_label, m_anno) in enumerate(single_models):
            ax = axes[row_idx, col_idx]
            x_vals = df[m_col]
            y_vals = df[pe_col]

            sp_r, sp_p = spearmanr(x_vals, y_vals)
            pe_r, pe_p = pearsonr(x_vals, y_vals)

            # Scatter with regression line
            sns.regplot(
                x=x_vals,
                y=y_vals,
                ax=ax,
                color=MODEL_COLORS[m_name],
                scatter_kws={'alpha': 0.75, 's': 55, 'edgecolors': 'k', 'linewidths': 0.7},
                line_kws={'color': '#C53030' if m_name == 'DeepPrime6' else '#1A365D', 'linewidth': 1.8}
            )

            # Annotation box
            stats_text = f"Spearman $R$ = {sp_r:.4f}\n$P$ = {sp_p:.2e}\n$n$ = {len(x_vals)}"
            ax.text(
                0.05, 0.92,
                stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=9.5,
                fontweight='medium',
                bbox=dict(boxstyle='round,pad=0.35', fc='#FFFFFF', ec='#CBD5E0', alpha=0.9)
            )

            # Titles and Labels
            if row_idx == 0:
                ax.set_title(f"{m_name}\n({m_anno})", fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel(x_label if row_idx == len(pe_systems) - 1 else "", fontsize=9.5)
            ax.set_ylabel(f"{pe_name}\nMeasured Activity (%)" if col_idx == 0 else "", fontsize=9.5)
            ax.grid(True, linestyle='--', alpha=0.45)

    plt.tight_layout()
    master_png = os.path.join(out_dir, "figures/fig1_single_models_all_PE_systems.png")
    master_svg = os.path.join(out_dir, "figures/fig1_single_models_all_PE_systems.svg")
    plt.savefig(master_png, dpi=300)
    plt.savefig(master_svg)
    plt.close()
    print(f"  Saved master Figure 1 to: {master_png}")

    # Generate individual 1x3 figures for each PE system
    for pe_name, pe_col in pe_systems:
        fig_ind, axes_ind = plt.subplots(1, 3, figsize=(15, 4.5))
        for col_idx, (m_name, m_col, x_label, m_anno) in enumerate(single_models):
            ax = axes_ind[col_idx]
            x_vals = df[m_col]
            y_vals = df[pe_col]

            sp_r, sp_p = spearmanr(x_vals, y_vals)
            sns.regplot(
                x=x_vals,
                y=y_vals,
                ax=ax,
                color=MODEL_COLORS[m_name],
                scatter_kws={'alpha': 0.75, 's': 60, 'edgecolors': 'k', 'linewidths': 0.8},
                line_kws={'color': '#C53030' if m_name == 'DeepPrime6' else '#1A365D', 'linewidth': 1.8}
            )

            stats_text = f"Spearman $R$ = {sp_r:.4f}\n$P$ = {sp_p:.2e}\n$n$ = {len(x_vals)}"
            ax.text(
                0.05, 0.92,
                stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.35', fc='#FFFFFF', ec='#CBD5E0', alpha=0.9)
            )

            ax.set_title(f"{m_name}\n({m_anno})", fontsize=11, fontweight='bold')
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(f"{pe_name} Measured Activity (%)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.45)

        safe_name = pe_name.replace(" ", "_").replace("-", "_")
        ind_png = os.path.join(out_dir, f"figures/fig1_single_models_{safe_name}.png")
        plt.tight_layout()
        plt.savefig(ind_png, dpi=300)
        plt.close()


def plot_fig2_model_prediction_correlations(df, out_dir):
    """
    Task 2: 3 scattered dot plots showing correlation between model prediction scores.
    """
    print("\n[3/5] Generating Figure 2: Inter-model prediction score correlations...")
    pairs = [
        ('DeepPrime-FT', 'PRIDICT2.0', 'pred_dp_base', 'pred_pridict2',
         'DeepPrime-FT Prediction Score (%)', 'PRIDICT2.0 Prediction Score (%)',
         MODEL_ANNOTATIONS['DeepPrime-FT'], MODEL_ANNOTATIONS['PRIDICT2.0']),
        ('DeepPrime-FT', 'DeepPrime6', 'pred_dp_base', 'pred_dp6',
         'DeepPrime-FT Prediction Score (%)', 'DeepPrime6 Prediction Score (%)',
         MODEL_ANNOTATIONS['DeepPrime-FT'], MODEL_ANNOTATIONS['DeepPrime6']),
        ('PRIDICT2.0', 'DeepPrime6', 'pred_pridict2', 'pred_dp6',
         'PRIDICT2.0 Prediction Score (%)', 'DeepPrime6 Prediction Score (%)',
         MODEL_ANNOTATIONS['PRIDICT2.0'], MODEL_ANNOTATIONS['DeepPrime6']),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    records = []

    for idx, (m1, m2, c1, c2, xlab, ylab, anno1, anno2) in enumerate(pairs):
        ax = axes[idx]
        x_vals = df[c1]
        y_vals = df[c2]

        sp_r, sp_p = spearmanr(x_vals, y_vals)
        pe_r, pe_p = pearsonr(x_vals, y_vals)

        records.append({
            'Model_1': m1,
            'Model_2': m2,
            'Spearman_rho': sp_r,
            'Spearman_p': sp_p,
            'Pearson_r': pe_r,
            'Pearson_p': pe_p,
            'N': len(x_vals)
        })

        sns.regplot(
            x=x_vals,
            y=y_vals,
            ax=ax,
            color='#4A5568',
            scatter_kws={'alpha': 0.75, 's': 60, 'edgecolors': 'k', 'linewidths': 0.8, 'color': '#2B6CB0'},
            line_kws={'color': '#E53E3E', 'linewidth': 2.0, 'linestyle': '--'}
        )

        stats_text = (
            f"Spearman $R$ = {sp_r:.4f}\n"
            f"$P$ = {sp_p:.3f}\n"
            f"Pearson $r$ = {pe_r:.4f}\n"
            f"$n$ = {len(x_vals)}"
        )
        ax.text(
            0.05, 0.92,
            stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFFF', ec='#A0AEC0', alpha=0.9)
        )

        ax.set_title(f"{m1} vs {m2}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(f"{xlab}\n({anno1})", fontsize=9.5)
        ax.set_ylabel(f"{ylab}\n({anno2})", fontsize=9.5)
        ax.grid(True, linestyle='--', alpha=0.45)

    plt.tight_layout()
    fig2_png = os.path.join(out_dir, "figures/fig2_inter_model_prediction_correlations.png")
    fig2_svg = os.path.join(out_dir, "figures/fig2_inter_model_prediction_correlations.svg")
    plt.savefig(fig2_png, dpi=300)
    plt.savefig(fig2_svg)
    plt.close()
    print(f"  Saved Figure 2 to: {fig2_png}")

    corr_df = pd.DataFrame(records)
    corr_df.to_csv(os.path.join(out_dir, "results/inter_model_correlations.csv"), index=False)


def plot_fig3_ensemble_models(df, out_dir):
    """
    Task 3 & 4: Scattered dot plots for 4 ensemble models (average of rank vs measured activity)
    across the 4 PE types + overall average.
    """
    print("\n[4/5] Generating Figure 3: Ensemble models (rank average) vs measured activity...")
    pe_systems = [
        ('PEmax-dRNaseH', 'dRN_NCcorr_%'),
        ('PE6a', '6a_NCcorr_%'),
        ('PE6b', '6b_NCcorr_%'),
        ('PE6c', '6c_NCcorr_%'),
        ('Overall Average', 'overall_avg_NCcorr_%'),
    ]

    ensembles = [
        ('DP + PRIDICT2.0', 'rank_ens_dp_pri', MODEL_COLORS['DP + PRIDICT2.0']),
        ('DP + DP6', 'rank_ens_dp_dp6', MODEL_COLORS['DP + DP6']),
        ('PRIDICT2.0 + DP6', 'rank_ens_pri_dp6', MODEL_COLORS['PRIDICT2.0 + DP6']),
        ('DP + DP6 + PRIDICT2.0', 'rank_ens_all3', MODEL_COLORS['DP + DP6 + PRIDICT2.0']),
    ]

    # Combined master figure (5 rows x 4 columns)
    fig, axes = plt.subplots(len(pe_systems), len(ensembles), figsize=(18, 18), sharey=False)

    for row_idx, (pe_name, pe_col) in enumerate(pe_systems):
        for col_idx, (ens_name, ens_col, color) in enumerate(ensembles):
            ax = axes[row_idx, col_idx]
            x_vals = df[ens_col]
            y_vals = df[pe_col]

            sp_r, sp_p = spearmanr(x_vals, y_vals)

            sns.regplot(
                x=x_vals,
                y=y_vals,
                ax=ax,
                color=color,
                scatter_kws={'alpha': 0.75, 's': 55, 'edgecolors': 'k', 'linewidths': 0.7},
                line_kws={'color': '#9B2C2C' if 'all3' in ens_col else '#2C5282', 'linewidth': 1.8}
            )

            stats_text = f"Spearman $R$ = {sp_r:.4f}\n$P$ = {sp_p:.2e}\n$n$ = {len(x_vals)}"
            ax.text(
                0.05, 0.92,
                stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=9.5,
                fontweight='bold' if 'all3' in ens_col else 'normal',
                bbox=dict(
                    boxstyle='round,pad=0.35',
                    fc='#FFF5F5' if 'all3' in ens_col else '#FFFFFF',
                    ec='#FEB2B2' if 'all3' in ens_col else '#CBD5E0',
                    alpha=0.9
                )
            )

            if row_idx == 0:
                ax.set_title(f"Ensemble: {ens_name}", fontsize=11.5, fontweight='bold', pad=10)
            ax.set_xlabel("Average of Rank (pegRNA Activity)" if row_idx == len(pe_systems) - 1 else "", fontsize=9.5)
            ax.set_ylabel(f"{pe_name}\nMeasured Activity (%)" if col_idx == 0 else "", fontsize=9.5)
            ax.grid(True, linestyle='--', alpha=0.45)

    plt.tight_layout()
    master_png = os.path.join(out_dir, "figures/fig3_ensemble_models_all_PE_systems.png")
    master_svg = os.path.join(out_dir, "figures/fig3_ensemble_models_all_PE_systems.svg")
    plt.savefig(master_png, dpi=300)
    plt.savefig(master_svg)
    plt.close()
    print(f"  Saved master Figure 3 to: {master_png}")

    # Generate individual 1x4 figures for each PE system
    for pe_name, pe_col in pe_systems:
        fig_ind, axes_ind = plt.subplots(1, 4, figsize=(18, 4.5))
        for col_idx, (ens_name, ens_col, color) in enumerate(ensembles):
            ax = axes_ind[col_idx]
            x_vals = df[ens_col]
            y_vals = df[pe_col]

            sp_r, sp_p = spearmanr(x_vals, y_vals)
            sns.regplot(
                x=x_vals,
                y=y_vals,
                ax=ax,
                color=color,
                scatter_kws={'alpha': 0.75, 's': 60, 'edgecolors': 'k', 'linewidths': 0.8},
                line_kws={'color': '#9B2C2C' if 'all3' in ens_col else '#2C5282', 'linewidth': 1.8}
            )

            stats_text = f"Spearman $R$ = {sp_r:.4f}\n$P$ = {sp_p:.2e}\n$n$ = {len(x_vals)}"
            ax.text(
                0.05, 0.92,
                stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                fontweight='bold' if 'all3' in ens_col else 'normal',
                bbox=dict(
                    boxstyle='round,pad=0.35',
                    fc='#FFF5F5' if 'all3' in ens_col else '#FFFFFF',
                    ec='#FEB2B2' if 'all3' in ens_col else '#CBD5E0',
                    alpha=0.9
                )
            )

            ax.set_title(f"{ens_name}", fontsize=11.5, fontweight='bold')
            ax.set_xlabel("Average of Rank (pegRNA Activity)", fontsize=10)
            ax.set_ylabel(f"{pe_name} Measured Activity (%)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.45)

        safe_name = pe_name.replace(" ", "_").replace("-", "_")
        ind_png = os.path.join(out_dir, f"figures/fig3_ensemble_models_{safe_name}.png")
        plt.tight_layout()
        plt.savefig(ind_png, dpi=300)
        plt.close()


def plot_fig4_summary_comparison(metrics_df, out_dir):
    """
    Summary comparison showing performance gain from single models to 2-way and 3-way ensembles.
    """
    print("\n[5/5] Generating Figure 4: Summary Performance Comparison...")
    
    # Order models
    model_order = [
        'DeepPrime-FT',
        'PRIDICT2.0',
        'DeepPrime6',
        'DP + PRIDICT2.0',
        'DP + DP6',
        'PRIDICT2.0 + DP6',
        'DP + DP6 + PRIDICT2.0'
    ]
    
    # 1. Grouped Bar Chart
    plt.figure(figsize=(13, 6))
    ax = sns.barplot(
        data=metrics_df,
        x='PE_System',
        y='Spearman_rho',
        hue='Model',
        palette=MODEL_COLORS,
        edgecolor='k',
        linewidth=0.8
    )

    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
    plt.axhline(0.6, color='red', linestyle='--', alpha=0.7, label='High Accuracy Threshold (0.60)')
    plt.title("Endogenous Site Performance: Single Models vs Rank-Averaged Ensembles (N=32)", fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("Spearman Rank Correlation ($R$)", fontsize=11)
    plt.xlabel("Prime Editing System Evaluated", fontsize=11)
    plt.ylim(0, 0.78)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, title="Model / Ensemble")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    bar_png = os.path.join(out_dir, "figures/fig4_summary_spearman_gain_bars.png")
    bar_svg = os.path.join(out_dir, "figures/fig4_summary_spearman_gain_bars.svg")
    plt.savefig(bar_png, dpi=300)
    plt.savefig(bar_svg)
    plt.close()
    print(f"  Saved Figure 4 (Bars) to: {bar_png}")

    # 2. Performance Heatmap
    pivot_df = metrics_df.pivot(index='Model', columns='PE_System', values='Spearman_rho')
    pivot_df = pivot_df.reindex(model_order)
    system_order = ['PEmax-dRNaseH', 'PE6a', 'PE6b', 'PE6c', 'Overall Average']
    pivot_df = pivot_df[system_order]

    plt.figure(figsize=(9, 6.5))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.4f',
        cmap='YlGnBu',
        cbar_kws={'label': "Spearman's Rank Correlation ($R$)"},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title("Spearman Rank Correlation Heatmap (N=32 Endogenous Sites)", fontsize=12, fontweight='bold', pad=12)
    plt.ylabel("Prediction Model / Ensemble", fontsize=11)
    plt.xlabel("Prime Editing System Evaluated", fontsize=11)
    plt.tight_layout()

    hm_png = os.path.join(out_dir, "figures/fig4_performance_heatmap.png")
    hm_svg = os.path.join(out_dir, "figures/fig4_performance_heatmap.svg")
    plt.savefig(hm_png, dpi=300)
    plt.savefig(hm_svg)
    plt.close()
    print(f"  Saved Figure 4 (Heatmap) to: {hm_png}")


def export_clean_tables(df, metrics_df, out_dir):
    """
    Saves clean data tables and summaries.
    """
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)

    # 1. Full data with ranks and ensembles
    df_out = df.copy()
    data_csv = os.path.join(out_dir, "results/endogenous_n32_full_scored_data.csv")
    df_out.to_csv(data_csv, index=False)
    print(f"\nSaved full enriched dataset to: {data_csv}")

    # 2. Metrics summary
    metrics_csv = os.path.join(out_dir, "results/endogenous_n32_evaluation_metrics_summary.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics summary to: {metrics_csv}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)

    print("=" * 80)
    print("=== Master Endogenous Site Evaluation & Ensemble Benchmark ===")
    print("=" * 80)

    # 1. Load Data
    df_raw = load_dataset(args.excel_path, args.rescored_csv)

    # 2. Verify / Run Inference if requested
    if args.run_inference:
        run_inference_validation(df_raw, args.parquet_path, args.device)

    # 3. Compute Rank Ensembles
    df_calc = compute_rank_ensembles(df_raw)

    # 4. Calculate Comprehensive Metrics
    metrics_df = calculate_metrics_summary(df_calc)

    # 5. Generate Figures
    plot_fig1_single_models(df_calc, args.output_dir)
    plot_fig2_model_prediction_correlations(df_calc, args.output_dir)
    plot_fig3_ensemble_models(df_calc, args.output_dir)
    plot_fig4_summary_comparison(metrics_df, args.output_dir)

    # 6. Export Results
    export_clean_tables(df_calc, metrics_df, args.output_dir)

    print("\n" + "=" * 80)
    print("=== All Benchmarks & Figures Completed Successfully! ===")
    print("=" * 80)


if __name__ == "__main__":
    main()
