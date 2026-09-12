import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    input_file = "data/small_library/pe6to7_predictions.parquet"
    output_file = "data/small_library/prediction_correlations.png"
    
    if not os.path.exists(input_file):
        print(f"Waiting for {input_file} to be generated...")
        return
        
    df = pd.read_parquet(input_file)
    
    # Grid of 3 rows (PE6-2avg, PE6-3avg, PE6-4avg) x 3 cols (DP-base, DP6-specific, PRIDICT2)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    rows_exp = ['PE6-2avg', 'PE6-3avg', 'PE6-4avg']
    dp6_models = ['pred_dp6_pe6a', 'pred_dp6_pe6b', 'pred_dp6_pe6c']
    dp6_labels = ['DeepPrime6 (PE6a)', 'DeepPrime6 (PE6b)', 'DeepPrime6 (PE6c)']
    
    for i, exp_col in enumerate(rows_exp):
        # 1. DP-base vs exp
        ax = axes[i, 0]
        sns.scatterplot(data=df, x='pred_dp_base', y=exp_col, ax=ax, alpha=0.5, s=15)
        valid = ~df['pred_dp_base'].isna() & ~df[exp_col].isna()
        r = df.loc[valid, 'pred_dp_base'].corr(df.loc[valid, exp_col], method='pearson')
        rho = df.loc[valid, 'pred_dp_base'].corr(df.loc[valid, exp_col], method='spearman')
        ax.set_title(f"DeepPrime-base vs {exp_col}\nr={r:.3f}, rho={rho:.3f}")
        ax.set_xlabel("Predicted Efficiency (DP-base)")
        ax.set_ylabel(f"Experimental ({exp_col})")
        
        # 2. DP6 specific vs exp
        ax = axes[i, 1]
        dp6_col = dp6_models[i]
        dp6_label = dp6_labels[i]
        sns.scatterplot(data=df, x=dp6_col, y=exp_col, ax=ax, alpha=0.5, s=15)
        valid = ~df[dp6_col].isna() & ~df[exp_col].isna()
        r = df.loc[valid, dp6_col].corr(df.loc[valid, exp_col], method='pearson')
        rho = df.loc[valid, dp6_col].corr(df.loc[valid, exp_col], method='spearman')
        ax.set_title(f"{dp6_label} vs {exp_col}\nr={r:.3f}, rho={rho:.3f}")
        ax.set_xlabel(f"Predicted Efficiency ({dp6_label})")
        ax.set_ylabel(f"Experimental ({exp_col})")
        
        # 3. PRIDICT2 vs exp
        ax = axes[i, 2]
        sns.scatterplot(data=df, x='pred_pridict2', y=exp_col, ax=ax, alpha=0.5, s=15)
        valid = ~df['pred_pridict2'].isna() & ~df[exp_col].isna()
        r = df.loc[valid, 'pred_pridict2'].corr(df.loc[valid, exp_col], method='pearson')
        rho = df.loc[valid, 'pred_pridict2'].corr(df.loc[valid, exp_col], method='spearman')
        ax.set_title(f"PRIDICT2.0 vs {exp_col}\nr={r:.3f}, rho={rho:.3f}")
        ax.set_xlabel("Predicted Efficiency (PRIDICT2.0)")
        ax.set_ylabel(f"Experimental ({exp_col})")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Successfully generated 3x3 correlation plots at {output_file}")

if __name__ == "__main__":
    main()
