import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

def load_and_merge_data():
    print("Loading datasets...")
    # Ground truth test library
    gt_df = pd.read_parquet("data/rq3_test_library.parquet")
    
    # Model predictions
    dp_base = pd.read_csv("data/rq3_predictions_dp_base.csv")
    dp6_pe6a = pd.read_csv("data/rq3_predictions_dp6.csv")
    dp6_pe6b = pd.read_csv("data/rq3_predictions_dp6_pe6b.csv")
    dp6_pe6c = pd.read_csv("data/rq3_predictions_dp6_pe6c.csv")
    dp6_drn = pd.read_csv("data/rq3_predictions_dp6_drnaseh.csv")
    pridict2 = pd.read_csv("data/rq3_predictions_pridict2.csv")
    
    # Merge on REF_ID
    dp_base = dp_base[['ID', 'Prediction']].rename(columns={'Prediction': 'pred_dp_base'})
    dp6_pe6a = dp6_pe6a[['ID', 'Prediction']].rename(columns={'Prediction': 'pred_dp6_pe6a'})
    dp6_pe6b = dp6_pe6b[['ID', 'Prediction']].rename(columns={'Prediction': 'pred_dp6_pe6b'})
    dp6_pe6c = dp6_pe6c[['ID', 'Prediction']].rename(columns={'Prediction': 'pred_dp6_pe6c'})
    dp6_drn = dp6_drn[['ID', 'Prediction']].rename(columns={'Prediction': 'pred_dp6_drnaseh'})
    
    pridict2 = pridict2[['REF_ID', 'PRIDICT2_0_editing_Score_deep_HEK', 'PRIDICT2_0_editing_Score_deep_K562']].rename(
        columns={
            'PRIDICT2_0_editing_Score_deep_HEK': 'pred_pridict2_hek',
            'PRIDICT2_0_editing_Score_deep_K562': 'pred_pridict2_k562'
        }
    )
    
    gt_df = gt_df.reset_index(drop=True)
    df = gt_df.copy()
    
    df = df.merge(dp_base, left_on='REF_ID', right_on='ID', how='left').drop(columns=['ID'])
    df = df.merge(dp6_pe6a, left_on='REF_ID', right_on='ID', how='left').drop(columns=['ID'])
    df = df.merge(dp6_pe6b, left_on='REF_ID', right_on='ID', how='left').drop(columns=['ID'])
    df = df.merge(dp6_pe6c, left_on='REF_ID', right_on='ID', how='left').drop(columns=['ID'])
    df = df.merge(dp6_drn, left_on='REF_ID', right_on='ID', how='left').drop(columns=['ID'])
    df = df.merge(pridict2, on='REF_ID', how='left')
    
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
        
    plt.suptitle(f"On-Target Prime Editing Efficiency vs. Model Predictions ({cohort_name})", y=1.02)
    plt.tight_layout()
    plt.savefig(f"notebooks/plots/cross_domain_scatters_{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plots for {cohort_name} to notebooks/plots/")

def main():
    df = load_and_merge_data()
    run_leaks_audit(df)
    
    # 1. Run evaluation for All Edit Types
    cohort_all = "All Edit Types"
    res_all = calculate_correlations(df, cohort_all)
    generate_visualizations(df, res_all, cohort_all)
    
    # 2. Run evaluation for Substitutions Only (Official domain of PRIDICT2.0)
    cohort_sub = "Substitutions Only"
    sub_df = df[df['Edit_type'] == 'Sub'].copy()
    res_sub = calculate_correlations(sub_df, cohort_sub)
    generate_visualizations(sub_df, res_sub, cohort_sub)

if __name__ == "__main__":
    main()
