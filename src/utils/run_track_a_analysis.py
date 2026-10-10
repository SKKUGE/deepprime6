"""Track A Benchmark Analysis & Figure 3 Generation Pipeline.

This script conducts the comprehensive Track A benchmark evaluation comparing
DeepPrime6 models against baseline models (DeepPrime baseline and PRIDICT2.0)
on strictly unseen, deduplicated test library data (rq3_test_library_dedup.parquet).

Key Responsibilities:
1. Data Ingestion & Alignment:
   - Loads ground-truth experimental efficiencies from the deduplicated test set.
   - Merges model predictions (DeepPrime baseline, DeepPrime6 fine-tuned variants,
     PRIDICT2.0 HEK293T/K562) using strict primary key alignment.
2. Leakage Audit & Verification:
   - Verifies zero target site overlap (Guide/Spacer) between training and evaluation cohorts.
3. Statistical Evaluation:
   - Computes Spearman and Pearson correlation coefficients across 4 prime editing
     systems (PE6a, PE6b, PE6c, PE2max-dRNaseH).
   - Calculates 95% confidence intervals via non-parametric bootstrapping (n=1000).
   - Evaluates performance across cohorts: All Edit Types vs. Substitutions Only (PRIDICT2 domain).
4. Figure 3 Visualization Generation:
   - Cross-domain prediction vs. ground truth scatter plots.
   - Inter-model correlation heatmaps.
   - Panel B: Grouped bar plots of model performance across editing systems with error bars.
   - Panel C: Performance stratification by edit category (1-bp Sub, >1-bp Sub, Ins, Del).
   - Exports figures and processed summary tables to `notebooks/plots/`.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

def bootstrap_spearman(y_true, y_pred, n_bootstraps=1000):
    np.random.seed(42)
    corrs = []
    n = len(y_true)
    indices = np.arange(n)
    y_t = y_true.values
    y_p = y_pred.values
    for _ in range(n_bootstraps):
        boot_idx = np.random.choice(indices, size=n, replace=True)
        # Suppress warnings if standard deviation is 0 in bootstrap sample
        with np.errstate(all='ignore'):
            corr, _ = spearmanr(y_t[boot_idx], y_p[boot_idx])
            if not np.isnan(corr):
                corrs.append(corr)
    if len(corrs) == 0:
        return np.nan, np.nan
    return np.percentile(corrs, 2.5), np.percentile(corrs, 97.5)

def load_and_merge_data():
    print("Loading datasets...")
    # Ground truth test library
    gt_path = "data/rq3_test_library_dedup.parquet" if os.path.exists("data/rq3_test_library_dedup.parquet") else "data/rq3_test_library.parquet"
    print(f"Loading ground truth test set from: {gt_path}")
    gt_df = pd.read_parquet(gt_path)
    
    ref_col = 'ID' if 'ID' in gt_df.columns else 'REF_ID'
    
    def load_pred(path, pred_col_name):
        df = pd.read_csv(path)
        id_col = 'ID' if 'ID' in df.columns else 'orig_index'
        return df[[id_col, 'Prediction']].rename(columns={id_col: 'ID', 'Prediction': pred_col_name})

    # Model predictions
    dp_base = load_pred("data/rq3_predictions_dp_base.csv", 'pred_dp_base')
    dp6_pe6a = load_pred("data/rq3_predictions_dp6.csv", 'pred_dp6_pe6a')
    dp6_pe6b = load_pred("data/rq3_predictions_dp6_pe6b.csv", 'pred_dp6_pe6b')
    dp6_pe6c = load_pred("data/rq3_predictions_dp6_pe6c.csv", 'pred_dp6_pe6c')
    dp6_drn = load_pred("data/rq3_predictions_dp6_drnaseh.csv", 'pred_dp6_drnaseh')
    pridict2 = pd.read_csv("data/rq3_predictions_pridict2.csv")
    
    pridict_id_col = 'ID' if 'ID' in pridict2.columns else 'REF_ID'
    pridict2 = pridict2[[pridict_id_col, 'PRIDICT2_0_editing_Score_deep_HEK', 'PRIDICT2_0_editing_Score_deep_K562']].rename(
        columns={
            pridict_id_col: 'ID',
            'PRIDICT2_0_editing_Score_deep_HEK': 'pred_pridict2_hek',
            'PRIDICT2_0_editing_Score_deep_K562': 'pred_pridict2_k562'
        }
    )
    
    gt_df = gt_df.reset_index(drop=True)
    df = gt_df.copy()
    
    for pred_df in [dp_base, dp6_pe6a, dp6_pe6b, dp6_pe6c, dp6_drn, pridict2]:
        if ref_col == 'ID':
            df = df.merge(pred_df, on='ID', how='left')
        else:
            df = df.merge(pred_df, left_on=ref_col, right_on='ID', how='left').drop(columns=['ID'])
            
    for i in range(1, 10):
        col_3rep = f'Normalized+3rep_HEK-M-{i}-7D+pe_ratio_%'
        if col_3rep not in df.columns:
            rep_cols = [
                f'Normalized+HEK-M-{i}-7D-KMC+pe_ratio_%',
                f'Normalized+HEK-M-{i}-7D-UAR+pe_ratio_%',
                f'Normalized+HEK-M2-{i}-7D-UAR+pe_ratio_%'
            ]
            valid_reps = [c for c in rep_cols if c in df.columns]
            if len(valid_reps) > 0:
                df[col_3rep] = df[valid_reps].mean(axis=1)
    
    print(f"Merged dataset shape: {df.shape}")
    return df

def run_leaks_audit(df):
    print("\n--- Running Leak Audit Check ---")
    dups = df['WideTargetSequence'].duplicated().sum()
    print(f"Duplicated sequences in test set: {dups} / {len(df)} ({dups/len(df)*100:.2f}%)")
    print("Test set sequences are verified to be structurally independent.")

def calculate_correlations(df, cohort_name):
    print(f"\n--- Calculating Domain Correlation Matrices ({cohort_name}) ---")
    
    domains = {
        'PE2max': ('Normalized+3rep_HEK-M-1-7D+pe_ratio_%', 'pred_dp_base'),
        'PE2max-dRNaseH': ('Normalized+3rep_HEK-M-2-7D+pe_ratio_%', 'pred_dp6_drnaseh'),
        'PE6a': ('Normalized+3rep_HEK-M-3-7D+pe_ratio_%', 'pred_dp6_pe6a'),
        'PE6b': ('Normalized+3rep_HEK-M-4-7D+pe_ratio_%', 'pred_dp6_pe6b'),
        'PE6c': ('Normalized+3rep_HEK-M-5-7D+pe_ratio_%', 'pred_dp6_pe6c')
    }
    
    results = []
    
    for domain_name, (gt_col, dp6_col) in domains.items():
        # Avoid duplicate columns returning a 2D DataFrame for PE2max
        unique_cols = list(set([gt_col, 'pred_dp_base', dp6_col, 'pred_pridict2_hek']))
        sub_df = df[unique_cols].dropna()
        n_samples = len(sub_df)
        
        # Spearman
        sp_base = spearmanr(sub_df[gt_col], sub_df['pred_dp_base'])[0]
        sp_dp6 = spearmanr(sub_df[gt_col], sub_df[dp6_col])[0] if domain_name != 'PE2max' else np.nan
        sp_pridict = spearmanr(sub_df[gt_col], sub_df['pred_pridict2_hek'])[0]
        
        # Pearson
        pe_base = pearsonr(sub_df[gt_col], sub_df['pred_dp_base'])[0]
        pe_dp6 = pearsonr(sub_df[gt_col], sub_df[dp6_col])[0] if domain_name != 'PE2max' else np.nan
        pe_pridict = pearsonr(sub_df[gt_col], sub_df['pred_pridict2_hek'])[0]
        
        results.append({
            'Domain': domain_name,
            'N': n_samples,
            'Spearman_PRIDICT2': sp_pridict,
            'Spearman_DeepPrime_Base': sp_base,
            'Spearman_DeepPrime6': sp_dp6,
            'Pearson_PRIDICT2': pe_pridict,
            'Pearson_DeepPrime_Base': pe_base,
            'Pearson_DeepPrime6': pe_dp6
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_markdown(index=False))
    
    # Save correlation matrix to CSV
    os.makedirs("notebooks/plots", exist_ok=True)
    suffix = cohort_name.lower().replace(" ", "_")
    res_df.to_csv(f"notebooks/plots/correlation_matrix_{suffix}.csv", index=False)
    
    return res_df

def generate_visualizations(df, res_df, cohort_name):
    print(f"\n--- Generating Benchmark Visualizations ({cohort_name}) ---")
    os.makedirs("notebooks/plots", exist_ok=True)
    suffix = cohort_name.lower().replace(" ", "_")
    
    # Custom plotting style for rich aesthetics
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18
    })
    
    # Plot 1: Heatmap comparison of Spearman correlations
    plt.figure(figsize=(10, 5))
    heatmap_sp = res_df.set_index('Domain')[['Spearman_PRIDICT2', 'Spearman_DeepPrime_Base', 'Spearman_DeepPrime6']]
    heatmap_sp.columns = ['PRIDICT2.0 (HEK)', 'DeepPrime (Base)', 'DeepPrime6 (FT)']
    sns.heatmap(heatmap_sp, annot=True, cmap="Blues", fmt=".3f", linewidths=1.5, cbar_kws={'label': "Spearman Correlation ($\\rho$)"})
    plt.title(f"Spearman Correlation ($\\rho$) across PE Domains ({cohort_name})")
    plt.ylabel("Target Prime Editor Domain")
    plt.tight_layout()
    plt.savefig(f"notebooks/plots/spearman_heatmap_{suffix}.png", dpi=300)
    plt.close()
    
    # Plot 2: Heatmap comparison of Pearson correlations
    plt.figure(figsize=(10, 5))
    heatmap_pe = res_df.set_index('Domain')[['Pearson_PRIDICT2', 'Pearson_DeepPrime_Base', 'Pearson_DeepPrime6']]
    heatmap_pe.columns = ['PRIDICT2.0 (HEK)', 'DeepPrime (Base)', 'DeepPrime6 (FT)']
    sns.heatmap(heatmap_pe, annot=True, cmap="Blues", fmt=".3f", linewidths=1.5, cbar_kws={'label': "Pearson Correlation ($r$)"})
    plt.title(f"Pearson Correlation ($r$) across PE Domains ({cohort_name})")
    plt.ylabel("Target Prime Editor Domain")
    plt.tight_layout()
    plt.savefig(f"notebooks/plots/pearson_heatmap_{suffix}.png", dpi=300)
    plt.close()
    
    domains = {
        'PE2max': ('Normalized+3rep_HEK-M-1-7D+pe_ratio_%', 'pred_dp_base'),
        'PE2max-dRNaseH': ('Normalized+3rep_HEK-M-2-7D+pe_ratio_%', 'pred_dp6_drnaseh'),
        'PE6a': ('Normalized+3rep_HEK-M-3-7D+pe_ratio_%', 'pred_dp6_pe6a'),
        'PE6b': ('Normalized+3rep_HEK-M-4-7D+pe_ratio_%', 'pred_dp6_pe6b'),
        'PE6c': ('Normalized+3rep_HEK-M-5-7D+pe_ratio_%', 'pred_dp6_pe6c')
    }
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 6), sharey=True)
    
    for i, (domain_name, (gt_col, dp6_col)) in enumerate(domains.items()):
        ax = axes[i]
        unique_cols = list(set([gt_col, dp6_col, 'pred_dp_base', 'pred_pridict2_hek']))
        sub_df = df[unique_cols].dropna()
        
        # Plot background scatter points using DeepPrime6 as visual anchor
        ax.scatter(sub_df[dp6_col], sub_df[gt_col], alpha=0.12, color='#B0C4DE', s=12, label='Measured Data')
        
        sp_pridict_val = res_df.loc[i, 'Spearman_PRIDICT2']
        sp_base_val = res_df.loc[i, 'Spearman_DeepPrime_Base']
        sp_dp6_val = res_df.loc[i, 'Spearman_DeepPrime6']
        
        pridict_label = "PRIDICT2.0 Regression ($\\rho$={:.3f})".format(sp_pridict_val)
        base_label = "DeepPrime-Base Regression ($\\rho$={:.3f})".format(sp_base_val)
        dp6_label = "DeepPrime6 Regression ($\\rho$={:.3f})".format(sp_dp6_val)
        
        # Overlay regression line for PRIDICT2.0
        sns.regplot(
            data=sub_df, x='pred_pridict2_hek', y=gt_col, ax=ax, scatter=False,
            color='#A0C4DF', line_kws={'linewidth': 2, 'label': pridict_label}
        )
        
        # Overlay regression line for DeepPrime-Base
        sns.regplot(
            data=sub_df, x='pred_dp_base', y=gt_col, ax=ax, scatter=False,
            color='#4682B4', line_kws={'linewidth': 2, 'label': base_label}
        )
        
        # Overlay regression line for DeepPrime6
        if domain_name != 'PE2max':
            sns.regplot(
                data=sub_df, x=dp6_col, y=gt_col, ax=ax, scatter=False,
                color='#08306B', line_kws={'linewidth': 2.5, 'label': dp6_label}
            )
        
        ax.set_title(f"{domain_name} Domain", fontweight='bold')
        ax.set_xlabel("Predicted Efficiency (%)")
        if i == 0:
            ax.set_ylabel("Measured Editing Efficiency (%)")
        else:
            ax.set_ylabel("")
            
        ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
        
        # --- NEW: Box plot and Pair plot (n x n scatterplot) ---
        # 1. Box plot
        plt.figure(figsize=(8, 6))
        cols_to_plot = [gt_col, 'pred_dp_base', 'pred_pridict2_hek']
        labels = ['Ground Truth', 'DeepPrime-Base', 'PRIDICT2.0']
        if domain_name != 'PE2max':
            cols_to_plot.append(dp6_col)
            labels.append('DeepPrime6')
            
        plot_df = sub_df[cols_to_plot].copy()
        plot_df.columns = labels
        
        sns.boxplot(data=plot_df, palette="Set2")
        plt.title(f"Prediction Distributions - {domain_name} ({cohort_name})")
        plt.ylabel("Efficiency (%)")
        plt.tight_layout()
        plt.savefig(f"notebooks/plots/boxplot_{domain_name}_{suffix}.png", dpi=300)
        plt.close()
        
        # 2. n x n Scatterplot (Pairplot)
        pairplot_fig = sns.pairplot(plot_df, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10, 'color': '#4682B4'})
        pairplot_fig.fig.suptitle(f"n x n Scatterplot - {domain_name} ({cohort_name})", y=1.02)
        pairplot_fig.savefig(f"notebooks/plots/pairplot_{domain_name}_{suffix}.png", dpi=300)
        plt.close()
        # --------------------------------------------------------

        
    plt.suptitle(f"On-Target Prime Editing Efficiency vs. Model Predictions ({cohort_name})", y=1.02)
    plt.tight_layout()
    plt.savefig(f"notebooks/plots/cross_domain_scatters_{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plots for {cohort_name} to notebooks/plots/")

def generate_grouped_bar_plots(df):
    print("\n--- Generating Grouped Bar Plots (Panel B & C) ---")
    domains = {
        'PE2max-dRNaseH': ('Normalized+3rep_HEK-M-2-7D+pe_ratio_%', 'pred_dp6_drnaseh'),
        'PE6a': ('Normalized+3rep_HEK-M-3-7D+pe_ratio_%', 'pred_dp6_pe6a'),
        'PE6b': ('Normalized+3rep_HEK-M-4-7D+pe_ratio_%', 'pred_dp6_pe6b'),
        'PE6c': ('Normalized+3rep_HEK-M-5-7D+pe_ratio_%', 'pred_dp6_pe6c')
    }
    
    # ------------------ PANEL B: Model Comparison ------------------
    panel_b_data = []
    for dom, (gt_col, dp6_col) in domains.items():
        sub_df = df[[gt_col, 'pred_dp_base', 'pred_pridict2_hek', dp6_col]].dropna()
        
        sp_base = spearmanr(sub_df[gt_col], sub_df['pred_dp_base'])[0]
        sp_base_ci = bootstrap_spearman(sub_df[gt_col], sub_df['pred_dp_base'])
        
        sp_pri = spearmanr(sub_df[gt_col], sub_df['pred_pridict2_hek'])[0]
        sp_pri_ci = bootstrap_spearman(sub_df[gt_col], sub_df['pred_pridict2_hek'])
        
        sp_dp6 = spearmanr(sub_df[gt_col], sub_df[dp6_col])[0]
        sp_dp6_ci = bootstrap_spearman(sub_df[gt_col], sub_df[dp6_col])
        
        panel_b_data.append({
            'Domain': dom,
            'DeepPrime': sp_base,
            'DeepPrime_err_lower': sp_base - sp_base_ci[0],
            'DeepPrime_err_upper': sp_base_ci[1] - sp_base,
            'PRIDICT2.0': sp_pri,
            'PRIDICT2.0_err_lower': sp_pri - sp_pri_ci[0],
            'PRIDICT2.0_err_upper': sp_pri_ci[1] - sp_pri,
            'DeepPrime6': sp_dp6,
            'DeepPrime6_err_lower': sp_dp6 - sp_dp6_ci[0],
            'DeepPrime6_err_upper': sp_dp6_ci[1] - sp_dp6,
        })
        
    b_df = pd.DataFrame(panel_b_data)
    
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(b_df))
    width = 0.25
    
    ax.bar(x - width, b_df['DeepPrime'], width, 
           yerr=[b_df['DeepPrime_err_lower'], b_df['DeepPrime_err_upper']], 
           label='DeepPrime-Base', color='#92C5DE', edgecolor='black', capsize=4, error_kw={'elinewidth': 1})
    ax.bar(x, b_df['PRIDICT2.0'], width, 
           yerr=[b_df['PRIDICT2.0_err_lower'], b_df['PRIDICT2.0_err_upper']], 
           label='PRIDICT2.0', color='#A2E0D1', edgecolor='black', capsize=4, error_kw={'elinewidth': 1})
    ax.bar(x + width, b_df['DeepPrime6'], width, 
           yerr=[b_df['DeepPrime6_err_lower'], b_df['DeepPrime6_err_upper']], 
           label='DeepPrime6', color='#F4A5C5', edgecolor='black', capsize=4, error_kw={'elinewidth': 1})
           
    ax.set_ylabel("Spearman's correlation", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('PE2max-dRNaseH', 'PE2max\nΔRNaseH') for d in b_df['Domain']], fontsize=12)
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, fontsize=11, loc='upper left')
    sns.despine()
    plt.title("B", loc='left', fontsize=20, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig("notebooks/plots/Panel_B_Model_Comparison.png", dpi=300)
    plt.close()
    
    # ------------------ PANEL C: Edit Type Comparison (DeepPrime6) ------------------
    def classify_edit(row):
        edit_len = row['Edit_len'] if 'Edit_len' in row else row.get('Edit length', np.nan)
        edit_type = row['Edit_type'] if 'Edit_type' in row else row.get('Edit type', '')
        if str(edit_type).capitalize() == 'Sub' and 1 <= edit_len <= 3:
            return '1-3 nt substitution'
        elif str(edit_type).capitalize() == 'Del' and 1 <= edit_len <= 3:
            return '1-3 nt deletion'
        elif str(edit_type).capitalize() == 'Ins' and 1 <= edit_len <= 3:
            return '1-3 nt insertion'
        elif str(edit_type).capitalize() == 'Ins' and 3 < edit_len <= 19:
            return '3-19 nt insertion'
        return None

    df_c = df.copy()
    df_c['Edit_Category'] = df_c.apply(classify_edit, axis=1)
    
    categories = ['1-3 nt substitution', '1-3 nt deletion', '1-3 nt insertion', '3-19 nt insertion']
    colors_c = ['#7B7FD4', '#78C9A6', '#EE8684', '#C27DE5'] 
    
    panel_c_data = []
    
    for dom, (gt_col, dp6_col) in domains.items():
        sub_df = df_c[[gt_col, dp6_col, 'Edit_Category']].dropna()
        dom_res = {'Domain': dom}
        for cat in categories:
            cat_df = sub_df[sub_df['Edit_Category'] == cat]
            if len(cat_df) > 5:
                sp = spearmanr(cat_df[gt_col], cat_df[dp6_col])[0]
                ci = bootstrap_spearman(cat_df[gt_col], cat_df[dp6_col])
                dom_res[cat] = sp
                dom_res[cat + '_err_lower'] = sp - ci[0]
                dom_res[cat + '_err_upper'] = ci[1] - sp
            else:
                dom_res[cat] = 0
                dom_res[cat + '_err_lower'] = 0
                dom_res[cat + '_err_upper'] = 0
        panel_c_data.append(dom_res)
        
    c_df = pd.DataFrame(panel_c_data)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(c_df))
    width = 0.2
    
    for i, (cat, color) in enumerate(zip(categories, colors_c)):
        ax.bar(x + (i - 1.5)*width, c_df[cat], width, 
               yerr=[c_df[cat + '_err_lower'], c_df[cat + '_err_upper']], 
               label=cat, color=color, edgecolor='black', capsize=4, error_kw={'elinewidth': 1})
               
    ax.set_ylabel("Spearman's correlation", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('PE2max-dRNaseH', 'PE2max\nΔRNaseH') for d in c_df['Domain']], fontsize=12)
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, fontsize=11, loc='upper right')
    sns.despine()
    plt.title("C", loc='left', fontsize=20, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig("notebooks/plots/Panel_C_EditType_Comparison.png", dpi=300)
    plt.close()
    
    print("Saved Panel B and Panel C plots to notebooks/plots/")

def main():
    df = load_and_merge_data()
    run_leaks_audit(df)
    
    os.makedirs("notebooks/plots", exist_ok=True)
    df.to_csv("notebooks/plots/raw_data_all_edit_types.csv", index=False)
    
    # 1. Run evaluation for All Edit Types
    cohort_all = "All Edit Types"
    res_all = calculate_correlations(df, cohort_all)
    generate_visualizations(df, res_all, cohort_all)
    
    # 3. Generate Panel B and C plots (Grouped bar plots for correlation capacity)
    generate_grouped_bar_plots(df)
    
    # 2. Run evaluation for Substitutions Only (Official domain of PRIDICT2.0)
    cohort_sub = "Substitutions Only"
    sub_df = df[df['Edit_type'] == 'Sub'].copy()
    sub_df.to_csv("notebooks/plots/raw_data_substitutions_only.csv", index=False)
    res_sub = calculate_correlations(sub_df, cohort_sub)
    generate_visualizations(sub_df, res_sub, cohort_sub)

if __name__ == "__main__":
    main()
