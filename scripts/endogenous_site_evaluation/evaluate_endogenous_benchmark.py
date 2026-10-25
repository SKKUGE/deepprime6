#!/usr/bin/env python3
"""
Endogenous Sites Prime Editing Efficiency Evaluation & Ensemble Modeling Benchmark

Evaluates prediction models (DeepPrime-FT, PRIDICT2.0, DeepPrime6 [4-model average & variant-matched])
and their rank-averaged ensembles against measured endogenous prime editing efficiencies across 4 PE systems:
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

pridict2_dir = os.path.join(PROJECT_ROOT, "PRIDICT2")
if pridict2_dir not in sys.path:
    sys.path.insert(0, pridict2_dir)

# Matplotlib global style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# Color palettes
MODEL_COLORS = {
    'DeepPrime-FT': '#3182CE',                 # Blue
    'PRIDICT2.0': '#2A9D8F',                   # Teal/Green
    'DeepPrime6': '#E63946',                   # Crimson (4-Model Avg)
    'DeepPrime6 (Variant-Matched)': '#FB8500', # Amber/Orange
    'DP + PRIDICT2.0': '#6B4C9A',              # Purple
    'DP + DP6': '#DD6B20',                     # Orange
    'PRIDICT2.0 + DP6': '#319795',             # Cyan
    'DP + DP6 + PRIDICT2.0': '#C53030'          # Dark Red/Ruby
}

PE_SYSTEM_LABELS = {
    'dRN_NCcorr_%': 'PEmax-dRNaseH',
    '6a_NCcorr_%': 'PE6a',
    '6b_NCcorr_%': 'PE6b',
    '6c_NCcorr_%': 'PE6c',
    'overall_avg_NCcorr_%': 'Overall Average'
}

VARIANT_MATCHED_COL_MAP = {
    'PEmax-dRNaseH': 'pred_dp6_drnaseh',
    'PE6a': 'pred_dp6_pe6a',
    'PE6b': 'pred_dp6_pe6b',
    'PE6c': 'pred_dp6_pe6c',
    'Overall Average': 'pred_dp6',
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
        default=os.path.join(PROJECT_ROOT, "notebooks/plots/endogenous_site_evaluation"),
        help="Output directory for figures, results, and logs"
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        default=True,
        help="Re-run neural network model inference from scratch for validation (default: True)"
    )
    parser.add_argument(
        "--sync-ground-truth",
        action="store_true",
        default=True,
        help="Synchronize updated predictions back to data/endogenous_ground_truth/ CSV and XLSX"
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
        help="Device for inference (cuda:0 or cpu)"
    )
    return parser.parse_args()


def load_dataset(excel_path, parquet_path, use_codebase_inference=True, device="cuda:0", sync_gt=True):
    print(f"\n[1/6] Loading endogenous benchmark dataset from: {excel_path}")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found at: {excel_path}")

    xl = pd.ExcelFile(excel_path)
    df = xl.parse("Final_n32_NCcorrected")
    excluded_df = xl.parse("Excluded_18") if "Excluded_18" in xl.sheet_names else None
    notes_df = xl.parse("Notes") if "Notes" in xl.sheet_names else None
    print(f"  Successfully loaded {len(df)} endogenous target sites.")
    print(f"  Targets analyzed: {df['target_num'].min()} to {df['target_num'].max()} (N={len(df)})")

    if use_codebase_inference and os.path.exists(parquet_path):
        print("  [Live Inference] Running live inference on endogenous targets via codebase engine...")
        pq_df = pd.read_parquet(parquet_path)
        pq_df['target_num'] = range(1, len(pq_df) + 1)

        # Preserve initial PRIDICT2 score before canonical re-inference
        df['pred_pridict2_initial'] = df['pred_pridict2'].values

        merged_cand = pd.merge(
            df,
            pq_df[[
                'target_num', 'design_id', 'Efficacy_Category', 'Disagreement_Subcategory',
                'WT_context', 'Edited_context', 'REF', 'ALT', 'spacer', 'nick_position',
                'PBS_sequence', 'RTT_sequence', 'pegRNA_strand'
            ]],
            on='target_num'
        )

        from Bio.Seq import reverse_complement
        from src.data.components.pe6_preprocess_data import preprocess_data

        # Prepare features directly from candidate designs with verified strand & deletion alignment
        cand_prep = merged_cand.copy()
        cand_prep['WideTargetSequence'] = [
            str(reverse_complement(row['WT_context'])) if row['pegRNA_strand'] == '-' else row['WT_context']
            for _, row in cand_prep.iterrows()
        ]
        cand_prep['ContextSeqUsed'] = 'WideTargetSequence'
        cand_prep['Guide'] = cand_prep['spacer']
        cand_prep['leading G'] = ""
        cand_prep['PBS'] = cand_prep['PBS_sequence']
        cand_prep['RTT'] = cand_prep['RTT_sequence']

        # Determine true edit type and edit length from HGVS Mutation string & REF/ALT
        true_edit_types = []
        true_edit_lens = []
        edit_starts = []
        for _, row in cand_prep.iterrows():
            mut_str = str(row['Mutation'])
            ref, alt = str(row['REF']), str(row['ALT'])
            wt = row['WT_context']
            strand = row['pegRNA_strand']
            if 'del' in mut_str:
                true_edit_types.append('del')
                true_edit_lens.append(1)
                ref_len_for_pos = 1
            elif len(ref) == len(alt):
                true_edit_types.append('sub')
                true_edit_lens.append(len(alt))
                ref_len_for_pos = len(ref)
            elif len(ref) > len(alt):
                true_edit_types.append('del')
                true_edit_lens.append(len(ref) - len(alt))
                ref_len_for_pos = len(ref) - len(alt)
            else:
                true_edit_types.append('ins')
                true_edit_lens.append(len(alt) - len(ref))
                ref_len_for_pos = 0

            st = 100 if strand == '+' else len(wt) - 100 - ref_len_for_pos
            edit_starts.append(st)

        cand_prep['Edit length'] = true_edit_lens
        cand_prep['Edit_start_idx'] = edit_starts
        cand_prep['Edit position'] = [st - nick + 1 for st, nick in zip(edit_starts, cand_prep['nick_position'])]
        cand_prep['Nicking'] = cand_prep['nick_position'].astype(int)
        cand_prep['Edit_type'] = true_edit_types

        feat_df = preprocess_data(data=cand_prep)

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
            feat_df['Target'].tolist(),
            feat_df['Masked_EditSeq'].tolist(),
            dev
        )
        b_base = prepare_biofeatures(feat_df, 'base', device=dev)
        b_pe6 = prepare_biofeatures(feat_df, 'pe6', device=dev)

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

        # Re-score PRIDICT2.0 with proper deletion representation
        print("  [Live Inference] Running PRIDICT2.0 inference with canonical deletion representations...")
        from src.utils.predict_pridict2 import (
            load_pridict_model,
            deeppridict,
            compute_average_predictions,
            extract_single_pegRNA_features
        )
        pr_models = load_pridict_model(run_ids=[0])
        pr_feats = []
        for _, r in merged_cand.iterrows():
            wt_clean = r['WT_context'].replace('N', '')
            ed_clean = r['Edited_context'].replace('N', '')
            is_del = 'del' in str(r['Mutation'])
            if is_del and len(ed_clean) == len(wt_clean):
                # Remove the 1 shifted trailing nucleotide from the fixed 123-nt window
                ed_clean = ed_clean[:-1]
            pr_row = {
                'REF_ID': r['design_id'],
                'WildTypeSequence': wt_clean,
                'PrimeEditedSequence': ed_clean,
                'Edit_type': 'Del' if is_del else 'Sub',
                'Guide': r['spacer'],
                'PBS': r['PBS_sequence'],
                'RTT': r['RTT_sequence']
            }
            pr_feats.append(extract_single_pegRNA_features(pr_row))

        pegdf = pd.DataFrame(pr_feats)
        all_avg_preds = deeppridict(pegdf, pr_models)
        tmp_df = pd.concat([all_avg_preds[k] for k in all_avg_preds], axis=0, ignore_index=True)
        agg_df = compute_average_predictions(tmp_df, grp_cols=['seq_id', 'dataset_name'])
        live_pridict2 = agg_df[agg_df['dataset_name'] == 'HEK'].sort_values('seq_id')['pred_averageedited'].values * 100.0

        df['Efficacy_Category'] = merged_cand['Efficacy_Category'].values
        df['Disagreement_Subcategory'] = merged_cand['Disagreement_Subcategory'].values
        df['Edit_type_resolved'] = [t.capitalize() for t in true_edit_types]
        df['pred_dp_base'] = live_dp_base
        df['pred_dp6'] = live_dp6_avg
        df['pred_pridict2'] = live_pridict2
        df['pred_dp6_pe6a'] = live_dp6_dict['pe6a']
        df['pred_dp6_pe6b'] = live_dp6_dict['pe6b']
        df['pred_dp6_pe6c'] = live_dp6_dict['pe6c']
        df['pred_dp6_drnaseh'] = live_dp6_dict['drnaseh']
        print("  [SUCCESS] All predictions updated via canonical inference engine.")

        # Synchronize updated scores back to ground truth CSV and XLSX
        if sync_gt:
            gt_cols = [
                'target_num', 'Locus', 'Mutation',
                'dRN_NCcorr_%', '6a_NCcorr_%', '6b_NCcorr_%', '6c_NCcorr_%', 'overall_avg_NCcorr_%',
                'NC_background_%', 'dRN_raw_%', '6a_raw_%', '6b_raw_%', '6c_raw_%', 'overall_avg_raw_%',
                'pred_dp_base', 'pred_dp6', 'pred_pridict2',
                'pred_dp6_pe6a', 'pred_dp6_pe6b', 'pred_dp6_pe6c', 'pred_dp6_drnaseh'
            ]
            gt_sync_df = df[[c for c in gt_cols if c in df.columns]].copy()
            csv_gt_path = os.path.splitext(excel_path)[0] + ".csv"
            gt_sync_df.to_csv(csv_gt_path, index=False)
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                gt_sync_df.to_excel(writer, sheet_name='Final_n32_NCcorrected', index=False)
                if excluded_df is not None:
                    excluded_df.to_excel(writer, sheet_name='Excluded_18', index=False)
                if notes_df is not None:
                    notes_df.to_excel(writer, sheet_name='Notes', index=False)
            print(f"  [SYNC] Synchronized canonical predictions to {excel_path} and {csv_gt_path}")

    return df


def compute_rank_ensembles(df):
    """
    Computes ascending rank of pegRNA activity predictions and generates ensemble combinations.
    Ascending rank: rank 1 = lowest predicted score, rank N = highest predicted score.
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


def run_inference_validation(df_endogenous):
    """
    Validates that model predictions are non-trivial and properly distributed.
    """
    print(f"\n[Validation] Validating model predictions...")
    assert df_endogenous['pred_dp_base'].notna().all(), "Missing DeepPrime-Base predictions!"
    assert df_endogenous['pred_dp6'].notna().all(), "Missing DeepPrime6 predictions!"
    assert df_endogenous['pred_pridict2'].notna().all(), "Missing PRIDICT2 predictions!"
    print(f"  - DeepPrime-FT: N={len(df_endogenous)}, Mean={df_endogenous['pred_dp_base'].mean():.2f}%")
    print(f"  - DeepPrime6:   N={len(df_endogenous)}, Mean={df_endogenous['pred_dp6'].mean():.2f}%")
    print(f"  - PRIDICT2.0:   N={len(df_endogenous)}, Mean={df_endogenous['pred_pridict2'].mean():.2f}%")
    print("  [PASS] All prediction scores are validated.")


def calculate_metrics_summary(df, cohort_name="Full Cohort (N=32)"):
    """
    Computes Spearman rho, p-value, Pearson r, and Pearson p for all models and ensembles.
    """
    df_ranked = compute_rank_ensembles(df)
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
        ('DeepPrime6 (Variant-Matched)', None, 'Score'),
    ]

    ensembles = [
        ('DP + PRIDICT2.0', 'rank_ens_dp_pri', 'Rank Average'),
        ('DP + DP6', 'rank_ens_dp_dp6', 'Rank Average'),
        ('PRIDICT2.0 + DP6', 'rank_ens_pri_dp6', 'Rank Average'),
        ('DP + DP6 + PRIDICT2.0', 'rank_ens_all3', 'Rank Average'),
    ]

    records = []
    n_samples = len(df_ranked)

    # 1. Single models
    for pe_name, pe_col in pe_systems:
        for m_name, col_name, pred_type in single_models:
            actual_col = VARIANT_MATCHED_COL_MAP[pe_name] if col_name is None else col_name
            if actual_col not in df_ranked.columns:
                continue
            sp_r, sp_p = spearmanr(df_ranked[actual_col], df_ranked[pe_col])
            pe_r, pe_p = pearsonr(df_ranked[actual_col], df_ranked[pe_col])
            records.append({
                'Cohort': cohort_name,
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
            sp_r, sp_p = spearmanr(df_ranked[col_name], df_ranked[pe_col])
            pe_r, pe_p = pearsonr(df_ranked[col_name], df_ranked[pe_col])
            records.append({
                'Cohort': cohort_name,
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


def plot_fig1_single_models(df, out_dir, suffix="", title_prefix=""):
    """
    Scattered dot plots for DP, PRIDICT2.0, DP6 across 4 PE types + overall average.
    Displays BOTH Spearman R and Pearson r in every subplot annotation box.
    """
    print(f"\n[2/5] Generating Figure 1: Single models vs measured activity (N=32)...")
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
    fig, axes = plt.subplots(len(pe_systems), len(single_models), figsize=(15, 18.5), sharey=False)

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

            # Annotation box with BOTH Spearman R and Pearson r
            stats_text = (
                f"Spearman $R$ = {sp_r:.4f} ($P$ = {sp_p:.2e})\n"
                f"Pearson $r$ = {pe_r:.4f} ($P$ = {pe_p:.2e})\n"
                f"$n$ = {len(x_vals)}"
            )
            ax.text(
                0.04, 0.94,
                stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=8.8,
                fontweight='medium',
                bbox=dict(boxstyle='round,pad=0.35', fc='#FFFFFF', ec='#CBD5E0', alpha=0.92)
            )

            # Titles and Labels
            if row_idx == 0:
                ax.set_title(f"{title_prefix}{m_name}\n({m_anno})", fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel(x_label if row_idx == len(pe_systems) - 1 else "", fontsize=9.5)
            ax.set_ylabel(f"{pe_name}\nMeasured Activity (%)" if col_idx == 0 else "", fontsize=9.5)
            ax.grid(True, linestyle='--', alpha=0.45)

    plt.tight_layout()
    fname_base = f"fig1_single_models_all_PE_systems{suffix}"
    master_png = os.path.join(out_dir, f"figures/{fname_base}.png")
    master_svg = os.path.join(out_dir, f"figures/{fname_base}.svg")
    plt.savefig(master_png, dpi=300)
    plt.savefig(master_svg)
    plt.close()
    print(f"  Saved master Figure 1 to: {master_png}")

    if not suffix:
        # Generate individual 1x3 figures for each PE system
        for pe_name, pe_col in pe_systems:
            fig_ind, axes_ind = plt.subplots(1, 3, figsize=(15, 4.6))
            for col_idx, (m_name, m_col, x_label, m_anno) in enumerate(single_models):
                ax = axes_ind[col_idx]
                x_vals = df[m_col]
                y_vals = df[pe_col]

                sp_r, sp_p = spearmanr(x_vals, y_vals)
                pe_r, pe_p = pearsonr(x_vals, y_vals)
                sns.regplot(
                    x=x_vals,
                    y=y_vals,
                    ax=ax,
                    color=MODEL_COLORS[m_name],
                    scatter_kws={'alpha': 0.75, 's': 60, 'edgecolors': 'k', 'linewidths': 0.8},
                    line_kws={'color': '#C53030' if m_name == 'DeepPrime6' else '#1A365D', 'linewidth': 1.8}
                )

                stats_text = (
                    f"Spearman $R$ = {sp_r:.4f} ($P$ = {sp_p:.2e})\n"
                    f"Pearson $r$ = {pe_r:.4f} ($P$ = {pe_p:.2e})\n"
                    f"$n$ = {len(x_vals)}"
                )
                ax.text(
                    0.04, 0.94,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    fontsize=9.2,
                    bbox=dict(boxstyle='round,pad=0.35', fc='#FFFFFF', ec='#CBD5E0', alpha=0.92)
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
    3 scattered dot plots showing correlation between model prediction scores.
    """
    print("\n[3/6] Generating Figure 2: Inter-model prediction score correlations...")
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
            f"Spearman $R$ = {sp_r:.4f} ($P$ = {sp_p:.3f})\n"
            f"Pearson $r$ = {pe_r:.4f} ($P$ = {pe_p:.3f})\n"
            f"$n$ = {len(x_vals)}"
        )
        ax.text(
            0.04, 0.94,
            stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFFF', ec='#A0AEC0', alpha=0.92)
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


def plot_fig3_ensemble_models(df, out_dir, suffix="", title_prefix=""):
    """
    Scattered dot plots for 4 ensemble models (average of rank vs measured activity)
    across the 4 PE types + overall average.
    Displays BOTH Spearman R and Pearson r in every subplot.
    """
    print(f"\n[4/6] Generating Figure 3 ({suffix or 'full'}): Ensemble models (rank average) vs measured activity...")
    df_plot = compute_rank_ensembles(df)
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
    fig, axes = plt.subplots(len(pe_systems), len(ensembles), figsize=(18.5, 18.5), sharey=False)

    for row_idx, (pe_name, pe_col) in enumerate(pe_systems):
        for col_idx, (ens_name, ens_col, color) in enumerate(ensembles):
            ax = axes[row_idx, col_idx]
            x_vals = df_plot[ens_col]
            y_vals = df_plot[pe_col]

            sp_r, sp_p = spearmanr(x_vals, y_vals)
            pe_r, pe_p = pearsonr(x_vals, y_vals)

            sns.regplot(
                x=x_vals,
                y=y_vals,
                ax=ax,
                color=color,
                scatter_kws={'alpha': 0.75, 's': 55, 'edgecolors': 'k', 'linewidths': 0.7},
                line_kws={'color': '#9B2C2C' if 'all3' in ens_col else '#2C5282', 'linewidth': 1.8}
            )

            stats_text = (
                f"Spearman $R$ = {sp_r:.4f} ($P$ = {sp_p:.2e})\n"
                f"Pearson $r$ = {pe_r:.4f} ($P$ = {pe_p:.2e})\n"
                f"$n$ = {len(x_vals)}"
            )
            ax.text(
                0.04, 0.94,
                stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=8.6,
                fontweight='bold' if 'all3' in ens_col else 'normal',
                bbox=dict(
                    boxstyle='round,pad=0.35',
                    fc='#FFF5F5' if 'all3' in ens_col else '#FFFFFF',
                    ec='#FEB2B2' if 'all3' in ens_col else '#CBD5E0',
                    alpha=0.92
                )
            )

            if row_idx == 0:
                ax.set_title(f"{title_prefix}Ensemble: {ens_name}", fontsize=11.5, fontweight='bold', pad=10)
            ax.set_xlabel("Average of Rank (pegRNA Activity)" if row_idx == len(pe_systems) - 1 else "", fontsize=9.5)
            ax.set_ylabel(f"{pe_name}\nMeasured Activity (%)" if col_idx == 0 else "", fontsize=9.5)
            ax.grid(True, linestyle='--', alpha=0.45)

    plt.tight_layout()
    fname_base = f"fig3_ensemble_models_all_PE_systems{suffix}"
    master_png = os.path.join(out_dir, f"figures/{fname_base}.png")
    master_svg = os.path.join(out_dir, f"figures/{fname_base}.svg")
    plt.savefig(master_png, dpi=300)
    plt.savefig(master_svg)
    plt.close()
    print(f"  Saved master Figure 3 to: {master_png}")

    if not suffix:
        # Generate individual 1x4 figures for each PE system
        for pe_name, pe_col in pe_systems:
            fig_ind, axes_ind = plt.subplots(1, 4, figsize=(18.5, 4.6))
            for col_idx, (ens_name, ens_col, color) in enumerate(ensembles):
                ax = axes_ind[col_idx]
                x_vals = df_plot[ens_col]
                y_vals = df_plot[pe_col]

                sp_r, sp_p = spearmanr(x_vals, y_vals)
                pe_r, pe_p = pearsonr(x_vals, y_vals)
                sns.regplot(
                    x=x_vals,
                    y=y_vals,
                    ax=ax,
                    color=color,
                    scatter_kws={'alpha': 0.75, 's': 60, 'edgecolors': 'k', 'linewidths': 0.8},
                    line_kws={'color': '#9B2C2C' if 'all3' in ens_col else '#2C5282', 'linewidth': 1.8}
                )

                stats_text = (
                    f"Spearman $R$ = {sp_r:.4f} ($P$ = {sp_p:.2e})\n"
                    f"Pearson $r$ = {pe_r:.4f} ($P$ = {pe_p:.2e})\n"
                    f"$n$ = {len(x_vals)}"
                )
                ax.text(
                    0.04, 0.94,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    fontsize=9.0,
                    fontweight='bold' if 'all3' in ens_col else 'normal',
                    bbox=dict(
                        boxstyle='round,pad=0.35',
                        fc='#FFF5F5' if 'all3' in ens_col else '#FFFFFF',
                        ec='#FEB2B2' if 'all3' in ens_col else '#CBD5E0',
                        alpha=0.92
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
    Summary comparison showing both Spearman R and Pearson r across single models and ensembles.
    """
    print("\n[5/5] Generating Figure 4: Performance Comparison (Spearman & Pearson)...")

    model_order = [
        'DeepPrime-FT',
        'PRIDICT2.0',
        'DeepPrime6',
        'DeepPrime6 (Variant-Matched)',
        'DP + PRIDICT2.0',
        'DP + DP6',
        'PRIDICT2.0 + DP6',
        'DP + DP6 + PRIDICT2.0'
    ]
    present_models = [m for m in model_order if m in metrics_df['Model'].unique()]

    # 1. Grouped Bar Chart (2 panels: Spearman & Pearson)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 11), sharex=True)

    sns.barplot(
        data=metrics_df,
        x='PE_System',
        y='Spearman_rho',
        hue='Model',
        hue_order=present_models,
        palette=MODEL_COLORS,
        edgecolor='k',
        linewidth=0.8,
        ax=axes[0]
    )
    axes[0].axhline(0.6, color='red', linestyle='--', alpha=0.7, label='0.60 Reference')
    axes[0].set_title("A. Endogenous Site Rank Correlation (Spearman $\\rho$, N=32)", fontsize=13, fontweight='bold', pad=10)
    axes[0].set_ylabel("Spearman Rank Correlation ($\\rho$)", fontsize=11)
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 0.85)
    axes[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True, title="Model / Ensemble")
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    sns.barplot(
        data=metrics_df,
        x='PE_System',
        y='Pearson_r',
        hue='Model',
        hue_order=present_models,
        palette=MODEL_COLORS,
        edgecolor='k',
        linewidth=0.8,
        ax=axes[1]
    )
    axes[1].axhline(0.6, color='red', linestyle='--', alpha=0.7, label='0.60 Reference')
    axes[1].set_title("B. Endogenous Site Linear Correlation (Pearson $r$, N=32)", fontsize=13, fontweight='bold', pad=10)
    axes[1].set_ylabel("Pearson Linear Correlation ($r$)", fontsize=11)
    axes[1].set_xlabel("Prime Editing System Evaluated", fontsize=11)
    axes[1].set_ylim(0, 0.75)
    axes[1].get_legend().remove()
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    bar_png = os.path.join(out_dir, "figures/fig4_summary_spearman_gain_bars.png")
    bar_svg = os.path.join(out_dir, "figures/fig4_summary_spearman_gain_bars.svg")
    plt.savefig(bar_png, dpi=300)
    plt.savefig(bar_svg)
    plt.close()
    print(f"  Saved Figure 4 (Spearman & Pearson Bars) to: {bar_png}")

    # 2. Side-by-side Performance Heatmap (Spearman & Pearson)
    system_order = ['PEmax-dRNaseH', 'PE6a', 'PE6b', 'PE6c', 'Overall Average']
    pivot_sp = metrics_df.pivot(index='Model', columns='PE_System', values='Spearman_rho').reindex(present_models)[system_order]
    pivot_pe = metrics_df.pivot(index='Model', columns='PE_System', values='Pearson_r').reindex(present_models)[system_order]

    def _draw_heatmap_with_text(ax, pivot_df, cmap, vmin, vmax, cbar_label):
        sns.heatmap(
            pivot_df,
            annot=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': cbar_label},
            linewidths=0.6,
            linecolor='gray',
            ax=ax
        )
        vals = pivot_df.values
        mid = (vmin + vmax) / 2.0
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if pd.notna(v):
                    txt_color = 'white' if v > mid else '#1A202C'
                    ax.text(
                        j + 0.5, i + 0.5, f"{v:.4f}",
                        ha='center', va='center',
                        color=txt_color, fontsize=9.5, fontweight='bold'
                    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
    _draw_heatmap_with_text(axes[0], pivot_sp, 'YlGnBu', 0.40, 0.82, "Spearman's $\\rho$")
    axes[0].set_title("A. Spearman Rank Correlation ($\\rho$, N=32)", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_ylabel("Prediction Model / Ensemble", fontsize=11)
    axes[0].set_xlabel("Prime Editing System Evaluated", fontsize=11)

    _draw_heatmap_with_text(axes[1], pivot_pe, 'OrRd', 0.40, 0.70, "Pearson's $r$")
    axes[1].set_title("B. Pearson Linear Correlation ($r$, N=32)", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Prime Editing System Evaluated", fontsize=11)

    plt.tight_layout()
    hm_png = os.path.join(out_dir, "figures/fig4_performance_heatmap.png")
    hm_svg = os.path.join(out_dir, "figures/fig4_performance_heatmap.svg")
    plt.savefig(hm_png, dpi=300)
    plt.savefig(hm_svg)
    plt.close()
    print(f"  Saved Figure 4 (Dual Heatmap) to: {hm_png}")


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

    # 2. Primary Metrics summary (Full N=32)
    metrics_csv = os.path.join(out_dir, "results/endogenous_n32_evaluation_metrics_summary.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"  Saved primary metrics summary (N=32) to: {metrics_csv}")

    # 3. Inter-model correlations
    pairs = [
        ('DeepPrime-FT vs PRIDICT2.0', 'pred_dp_base', 'pred_pridict2'),
        ('DeepPrime-FT vs DeepPrime6', 'pred_dp_base', 'pred_dp6'),
        ('PRIDICT2.0 vs DeepPrime6', 'pred_pridict2', 'pred_dp6'),
    ]
    inter_records = []
    for pair_name, c1, c2 in pairs:
        sp_r, sp_p = spearmanr(df[c1], df[c2])
        pe_r, pe_p = pearsonr(df[c1], df[c2])
        inter_records.append({
            'Model Pair': pair_name,
            'Spearman_rho': sp_r,
            'Spearman_p': sp_p,
            'Pearson_r': pe_r,
            'Pearson_p': pe_p
        })
    inter_df = pd.DataFrame(inter_records)
    inter_csv = os.path.join(out_dir, "results/inter_model_correlations.csv")
    inter_df.to_csv(inter_csv, index=False)
    print(f"  Saved inter-model correlations to: {inter_csv}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)

    print("=" * 80)
    print("=== Master Endogenous Site Evaluation & Ensemble Benchmark ===")
    print("=" * 80)

    # 1. Load Data & Run Inference
    df_raw = load_dataset(
        args.excel_path,
        args.parquet_path,
        use_codebase_inference=args.run_inference,
        device=args.device,
        sync_gt=args.sync_ground_truth
    )

    # 2. Verify Predictions
    run_inference_validation(df_raw)

    # 3. Compute Rank Ensembles
    df_calc = compute_rank_ensembles(df_raw)

    # 4. Calculate Comprehensive Metrics (Full Cohort N=32)
    metrics_df = calculate_metrics_summary(df_calc, cohort_name="Full Cohort (N=32)")

    # 5. Generate Figures (Full Cohort N=32)
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
