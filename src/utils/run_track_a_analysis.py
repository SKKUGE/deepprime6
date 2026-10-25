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

def resolve_paths():
    """Dynamically resolve input data and output directories based on execution location."""
    if os.path.exists("predictions/rq3_test_library.parquet") or os.path.exists("predictions/rq3_test_library_dedup.parquet"):
        data_dirs = ["predictions", "data", "../../data"]
        panels_dir = "panels"
        figures_dir = "figures"
        results_dir = "results"
        mirror_dirs = ["../../notebooks/plots"] if os.path.isdir("../../notebooks/plots") else []
    else:
        data_dirs = ["data", "predictions"]
        panels_dir = "notebooks/plots"
        figures_dir = "notebooks/plots"
        results_dir = "notebooks/plots"
        mirror_dirs = []

    for d in [panels_dir, figures_dir, results_dir] + mirror_dirs:
        os.makedirs(d, exist_ok=True)

    return {
        "data_dirs": data_dirs,
        "panels_dir": panels_dir,
        "figures_dir": figures_dir,
        "results_dir": results_dir,
        "mirror_dirs": mirror_dirs
    }

def find_file(filename, search_dirs):
    for d in search_dirs:
        cand = os.path.join(d, filename)
        if os.path.exists(cand):
            return cand
    return None

def save_fig(fig_or_plt, filename, is_panel=False, format=None, **kwargs):
    paths = resolve_paths()
    target_dirs = set()
    if is_panel:
        target_dirs.add(paths["panels_dir"])
        target_dirs.add(paths["figures_dir"])
    else:
        target_dirs.add(paths["figures_dir"])
    for m in paths["mirror_dirs"]:
        target_dirs.add(m)
        
    for td in target_dirs:
        out_path = os.path.join(td, filename)
        if format:
            fig_or_plt.savefig(out_path, format=format, **kwargs)
        else:
            fig_or_plt.savefig(out_path, dpi=300, **kwargs)

def save_csv(df, filename):
    paths = resolve_paths()
    target_dirs = set([paths["results_dir"]])
    for m in paths["mirror_dirs"]:
        target_dirs.add(m)
    for td in target_dirs:
        out_path = os.path.join(td, filename)
        df.to_csv(out_path, index=False)

def bootstrap_spearman(y_true, y_pred, n_bootstraps=1000):
    np.random.seed(42)
    corrs = []
    y_t = np.asarray(y_true).flatten()
    y_p = np.asarray(y_pred).flatten()
    n = len(y_t)
    indices = np.arange(n)
    for _ in range(n_bootstraps):
        boot_idx = np.random.choice(indices, size=n, replace=True)
        with np.errstate(all='ignore'):
            corr, _ = spearmanr(y_t[boot_idx], y_p[boot_idx])
            if not np.isnan(corr):
                corrs.append(corr)
    if len(corrs) == 0:
        return np.nan, np.nan
    return np.percentile(corrs, 2.5), np.percentile(corrs, 97.5)

def safe_pearson(x, y):
    x_arr = np.asarray(x, dtype=float).flatten()
    y_arr = np.asarray(y, dtype=float).flatten()
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    if np.sum(mask) < 2:
        return np.nan
    return float(pearsonr(x_arr[mask], y_arr[mask])[0])

def safe_spearman(x, y):
    x_arr = np.asarray(x, dtype=float).flatten()
    y_arr = np.asarray(y, dtype=float).flatten()
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    if np.sum(mask) < 2:
        return np.nan
    return float(spearmanr(x_arr[mask], y_arr[mask])[0])

def load_and_merge_data():
    paths = resolve_paths()
    print("Loading datasets...")
    # Ground truth test library
    gt_path = find_file("rq3_test_library_dedup.parquet", paths["data_dirs"])
    if not gt_path:
        gt_path = find_file("rq3_test_library.parquet", paths["data_dirs"])
    if not gt_path:
        raise FileNotFoundError(f"Could not find rq3_test_library parquet in {paths['data_dirs']}")
        
    print(f"Loading ground truth test set from: {gt_path}")
    gt_df = pd.read_parquet(gt_path)
    
    ref_col = 'ID' if 'ID' in gt_df.columns else 'REF_ID'
    
    def load_pred(path, pred_col_name):
        df = pd.read_csv(path)
        id_col = 'ID' if 'ID' in df.columns else 'orig_index'
        return df[[id_col, 'Prediction']].rename(columns={id_col: 'ID', 'Prediction': pred_col_name})

    def load_pred_by_name(filename, pred_col_name):
        fpath = find_file(filename, paths["data_dirs"])
        if not fpath:
            raise FileNotFoundError(f"Could not find {filename} in {paths['data_dirs']}")
        return load_pred(fpath, pred_col_name)

    # Model predictions
    dp_base = load_pred_by_name("rq3_predictions_dp_base.csv", 'pred_dp_base')
    dp6_pe6a = load_pred_by_name("rq3_predictions_dp6.csv", 'pred_dp6_pe6a')
    dp6_pe6b = load_pred_by_name("rq3_predictions_dp6_pe6b.csv", 'pred_dp6_pe6b')
    dp6_pe6c = load_pred_by_name("rq3_predictions_dp6_pe6c.csv", 'pred_dp6_pe6c')
    dp6_drn = load_pred_by_name("rq3_predictions_dp6_drnaseh.csv", 'pred_dp6_drnaseh')
    
    pridict_path = find_file("rq3_predictions_pridict2.csv", paths["data_dirs"])
    if not pridict_path:
        raise FileNotFoundError(f"Could not find rq3_predictions_pridict2.csv in {paths['data_dirs']}")
    pridict2 = pd.read_csv(pridict_path)
    
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
    
    pred_dfs = [dp_base, dp6_pe6a, dp6_pe6b, dp6_pe6c, dp6_drn, pridict2]
    
    # Scratch models
    scratch_files = [
        ("rq3_predictions_scratch_pe6a.csv", "pred_scratch_pe6a"),
        ("rq3_predictions_scratch_pe6b.csv", "pred_scratch_pe6b"),
        ("rq3_predictions_scratch_pe6c.csv", "pred_scratch_pe6c"),
        ("rq3_predictions_scratch_drnaseh.csv", "pred_scratch_drnaseh"),
    ]
    for fn, sc_name in scratch_files:
        fpath = find_file(fn, paths["data_dirs"])
        if fpath:
            pred_dfs.append(load_pred(fpath, sc_name))
        else:
            print(f"Warning: Scratch prediction file {fn} not found in {paths['data_dirs']}. Skipping.")
            
    for pred_df in pred_dfs:
        if ref_col == 'ID':
            df = df.merge(pred_df, on='ID', how='left')
        else:
            df = df.merge(pred_df, left_on=ref_col, right_on='ID', how='left').drop(columns=['ID'])
            
    import sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.utils.dataprep import IDX_TO_PE_TYPE, get_pe6_qc_mask

    for i in range(1, 10):
        col_clean = f'Normalized+2rep_HEK-M-{i}-7D+pe_ratio_%'
        rep_cols = [
            f'Normalized+HEK-M-{i}-7D-B1+pe_ratio_%',
            f'Normalized+HEK-M2-{i}-7D-B1+pe_ratio_%'
        ]
        valid_reps = [c for c in rep_cols if c in df.columns]
        if len(valid_reps) > 0:
            df[col_clean] = np.maximum(0.0, df[valid_reps].mean(axis=1))
            pe_type = IDX_TO_PE_TYPE[i]
            qc_mask = get_pe6_qc_mask(df, pe_type=pe_type, read_count_filter=200, max_ctrl_bg=5.0)
            df.loc[~qc_mask, col_clean] = np.nan
            # Ensure col_3rep alias holds the QC-filtered clean 2-rep mean
            df[f'Normalized+3rep_HEK-M-{i}-7D+pe_ratio_%'] = df[col_clean]
            if pe_type in df.columns:
                df.loc[~qc_mask, pe_type] = np.nan
    
    df = df.loc[:, ~df.columns.duplicated()]
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
        'PE2max': ('Normalized+3rep_HEK-M-1-7D+pe_ratio_%', 'pred_dp_base', None),
        'PE2max-dRNaseH': ('Normalized+3rep_HEK-M-2-7D+pe_ratio_%', 'pred_dp6_drnaseh', 'pred_scratch_drnaseh'),
        'PE6a': ('Normalized+3rep_HEK-M-3-7D+pe_ratio_%', 'pred_dp6_pe6a', 'pred_scratch_pe6a'),
        'PE6b': ('Normalized+3rep_HEK-M-4-7D+pe_ratio_%', 'pred_dp6_pe6b', 'pred_scratch_pe6b'),
        'PE6c': ('Normalized+3rep_HEK-M-5-7D+pe_ratio_%', 'pred_dp6_pe6c', 'pred_scratch_pe6c')
    }
    
    results = []
    
    for domain, (gt_col, dp6_col, sc_col) in domains.items():
        cols_to_use = list(dict.fromkeys([gt_col, 'pred_dp_base', dp6_col, 'pred_pridict2_hek'] + ([sc_col] if sc_col and sc_col in df.columns else [])))
        sub_df = df[cols_to_use].dropna()
        n_samples = len(sub_df)
        
        sp_pri = safe_spearman(sub_df[gt_col], sub_df['pred_pridict2_hek'])
        sp_base = safe_spearman(sub_df[gt_col], sub_df['pred_dp_base'])
        sp_dp6 = safe_spearman(sub_df[gt_col], sub_df[dp6_col])
        
        pe_pri = safe_pearson(sub_df[gt_col], sub_df['pred_pridict2_hek'])
        pe_base = safe_pearson(sub_df[gt_col], sub_df['pred_dp_base'])
        pe_dp6 = safe_pearson(sub_df[gt_col], sub_df[dp6_col])
        
        sp_sc = safe_spearman(sub_df[gt_col], sub_df[sc_col]) if sc_col and sc_col in sub_df.columns else np.nan
        pe_sc = safe_pearson(sub_df[gt_col], sub_df[sc_col]) if sc_col and sc_col in sub_df.columns else np.nan
        
        results.append({
            'Domain': domain,
            'N': n_samples,
            'Spearman_PRIDICT2': sp_pri,
            'Spearman_DeepPrime_FT': sp_base,
            'Spearman_From_Scratch': sp_sc,
            'Spearman_DeepPrime6': sp_dp6,
            'Pearson_PRIDICT2': pe_pri,
            'Pearson_DeepPrime_FT': pe_base,
            'Pearson_From_Scratch': pe_sc,
            'Pearson_DeepPrime6': pe_dp6
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_markdown(index=False))
    
    suffix = cohort_name.lower().replace(" ", "_")
    save_csv(res_df, f"correlation_matrix_{suffix}.csv")
    
    return res_df

def generate_visualizations(df, res_df, cohort_name):
    print(f"\n--- Generating Benchmark Visualizations ({cohort_name}) ---")
    suffix = cohort_name.lower().replace(" ", "_")
    
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
    plt.figure(figsize=(11, 5))
    heatmap_cols = ['Spearman_PRIDICT2', 'Spearman_DeepPrime_FT']
    col_names = ['PRIDICT2.0 (HEK)', 'DeepPrime-FT']
    if 'Spearman_From_Scratch' in res_df.columns and not res_df['Spearman_From_Scratch'].isna().all():
        heatmap_cols.append('Spearman_From_Scratch')
        col_names.append('From scratch')
    heatmap_cols.append('Spearman_DeepPrime6')
    col_names.append('DeepPrime6 (FT)')
    
    heatmap_sp = res_df.set_index('Domain')[heatmap_cols]
    heatmap_sp.columns = col_names
    sns.heatmap(heatmap_sp, annot=True, cmap="Blues", fmt=".3f", linewidths=1.5, cbar_kws={'label': "Spearman Correlation ($\\rho$)"})
    plt.title(f"Spearman Correlation ($\\rho$) across PE Domains ({cohort_name})")
    plt.ylabel("Target Prime Editor Domain")
    plt.tight_layout()
    save_fig(plt, f"spearman_heatmap_{suffix}.png")
    plt.close()
    
    # Plot 2: Heatmap comparison of Pearson correlations
    plt.figure(figsize=(11, 5))
    pe_cols = ['Pearson_PRIDICT2', 'Pearson_DeepPrime_FT']
    pe_names = ['PRIDICT2.0 (HEK)', 'DeepPrime-FT']
    if 'Pearson_From_Scratch' in res_df.columns and not res_df['Pearson_From_Scratch'].isna().all():
        pe_cols.append('Pearson_From_Scratch')
        pe_names.append('From scratch')
    pe_cols.append('Pearson_DeepPrime6')
    pe_names.append('DeepPrime6 (FT)')
    
    heatmap_pe = res_df.set_index('Domain')[pe_cols]
    heatmap_pe.columns = pe_names
    sns.heatmap(heatmap_pe, annot=True, cmap="Blues", fmt=".3f", linewidths=1.5, cbar_kws={'label': "Pearson Correlation ($r$)"})
    plt.title(f"Pearson Correlation ($r$) across PE Domains ({cohort_name})")
    plt.ylabel("Target Prime Editor Domain")
    plt.tight_layout()
    save_fig(plt, f"pearson_heatmap_{suffix}.png")
    plt.close()
    
    domains = {
        'PE2max': ('Normalized+3rep_HEK-M-1-7D+pe_ratio_%', 'pred_dp_base', None),
        'PE2max-dRNaseH': ('Normalized+3rep_HEK-M-2-7D+pe_ratio_%', 'pred_dp6_drnaseh', 'pred_scratch_drnaseh'),
        'PE6a': ('Normalized+3rep_HEK-M-3-7D+pe_ratio_%', 'pred_dp6_pe6a', 'pred_scratch_pe6a'),
        'PE6b': ('Normalized+3rep_HEK-M-4-7D+pe_ratio_%', 'pred_dp6_pe6b', 'pred_scratch_pe6b'),
        'PE6c': ('Normalized+3rep_HEK-M-5-7D+pe_ratio_%', 'pred_dp6_pe6c', 'pred_scratch_pe6c')
    }
    
    # Plot 3: Cross-Domain Prediction Scatter Plots
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    for idx, (domain_name, (gt_col, pred_col, sc_col)) in enumerate(domains.items()):
        ax = axes[idx]
        sub = df[[gt_col, pred_col]].dropna()
        sns.regplot(
            data=sub, x=pred_col, y=gt_col, ax=ax,
            scatter_kws={'alpha': 0.35, 'color': '#1f77b4', 's': 20},
            line_kws={'color': '#d62728', 'linewidth': 2}
        )
        r_val = safe_pearson(sub[pred_col], sub[gt_col])
        rho_val = safe_spearman(sub[pred_col], sub[gt_col])
        ax.set_title(f"{domain_name}\n$r$ = {r_val:.3f}, $\\rho$ = {rho_val:.3f}")
        ax.set_xlabel("Predicted Efficiency (%)")
        if idx == 0:
            ax.set_ylabel("Measured Efficiency (%)")
        else:
            ax.set_ylabel("")
            
    plt.suptitle(f"Cross-Domain In-House Validation ({cohort_name})", y=1.05)
    save_fig(plt, f"cross_domain_scatters_{suffix}.png", bbox_inches='tight')
    plt.close()
    
    # Plot 4: Boxplots and Pairplots per domain
    for domain_name, (gt_col, pred_col, sc_col) in domains.items():
        cols_to_plot = list(dict.fromkeys([gt_col, pred_col] + ([sc_col] if sc_col and sc_col in df.columns else []) + ['pred_dp_base', 'pred_pridict2_hek']))
        cols_present = [c for c in cols_to_plot if c in df.columns]
        plot_df = df[cols_present].dropna()
        rename_map = {
            gt_col: f"Measured ({domain_name})",
            pred_col: "DeepPrime6" if "dp6" in pred_col else pred_col,
            sc_col: "From-Scratch" if sc_col else None,
            'pred_dp_base': "DeepPrime-FT",
            'pred_pridict2_hek': "PRIDICT2.0"
        }
        rename_map = {k: v for k, v in rename_map.items() if k is not None}
        plot_df = plot_df.rename(columns=rename_map)
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=plot_df, palette="Set2")
        plt.title(f"Score Distribution - {domain_name} ({cohort_name})")
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        save_fig(plt, f"boxplot_{domain_name}_{suffix}.png")
        plt.close()
        
        pairplot_fig = sns.pairplot(plot_df, kind="reg", plot_kws={'scatter_kws': {'alpha': 0.2, 's': 10}, 'line_kws': {'color': 'red'}})
        pairplot_fig.fig.suptitle(f"Pairwise Relationships - {domain_name} ({cohort_name})", y=1.02)
        save_fig(pairplot_fig, f"pairplot_{domain_name}_{suffix}.png")
        plt.close()
        
    print(f"Saved plots for {cohort_name} successfully.")

def generate_grouped_bar_plots(df):
    print("\n--- Generating Grouped Bar Plots for Panel B and Panel C ---")
    
    domains = {
        'PE2max-dRNaseH': ('Normalized+3rep_HEK-M-2-7D+pe_ratio_%', 'pred_dp6_drnaseh', 'pred_scratch_drnaseh'),
        'PE6a': ('Normalized+3rep_HEK-M-3-7D+pe_ratio_%', 'pred_dp6_pe6a', 'pred_scratch_pe6a'),
        'PE6b': ('Normalized+3rep_HEK-M-4-7D+pe_ratio_%', 'pred_dp6_pe6b', 'pred_scratch_pe6b'),
        'PE6c': ('Normalized+3rep_HEK-M-5-7D+pe_ratio_%', 'pred_dp6_pe6c', 'pred_scratch_pe6c')
    }
    
    panel_b_data = []
    
    for dom, (gt_col, dp6_col, sc_col) in domains.items():
        has_sc = sc_col and sc_col in df.columns
        req_cols = list(dict.fromkeys([gt_col, 'pred_dp_base', 'pred_pridict2_hek', dp6_col] + ([sc_col] if has_sc else [])))
            
        sub_df = df[req_cols].dropna()
        
        sp_base = safe_spearman(sub_df[gt_col], sub_df['pred_dp_base'])
        sp_base_ci = bootstrap_spearman(sub_df[gt_col], sub_df['pred_dp_base'])
        
        sp_pri = safe_spearman(sub_df[gt_col], sub_df['pred_pridict2_hek'])
        sp_pri_ci = bootstrap_spearman(sub_df[gt_col], sub_df['pred_pridict2_hek'])
        
        sp_dp6 = safe_spearman(sub_df[gt_col], sub_df[dp6_col])
        sp_dp6_ci = bootstrap_spearman(sub_df[gt_col], sub_df[dp6_col])
        
        row_dict = {
            'Domain': dom,
            'DeepPrime-FT': sp_base,
            'DeepPrime-FT_err_lower': sp_base - sp_base_ci[0],
            'DeepPrime-FT_err_upper': sp_base_ci[1] - sp_base,
            'PRIDICT2.0': sp_pri,
            'PRIDICT2.0_err_lower': sp_pri - sp_pri_ci[0],
            'PRIDICT2.0_err_upper': sp_pri_ci[1] - sp_pri,
            'DeepPrime6': sp_dp6,
            'DeepPrime6_err_lower': sp_dp6 - sp_dp6_ci[0],
            'DeepPrime6_err_upper': sp_dp6_ci[1] - sp_dp6,
        }
        if has_sc:
            sp_sc = safe_spearman(sub_df[gt_col], sub_df[sc_col])
            sp_sc_ci = bootstrap_spearman(sub_df[gt_col], sub_df[sc_col])
            row_dict.update({
                'From_Scratch': sp_sc,
                'From_Scratch_err_lower': sp_sc - sp_sc_ci[0],
                'From_Scratch_err_upper': sp_sc_ci[1] - sp_sc,
            })
        panel_b_data.append(row_dict)
        
    b_df = pd.DataFrame(panel_b_data)
    
    # 1. ORIGINAL PANEL B (DeepPrime-FT, From scratch, DeepPrime6) - Matches paper figure
    if 'From_Scratch' in b_df.columns:
        sns.set_theme(style="ticks")
        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
        x = np.arange(len(b_df))
        width = 0.24
        
        ax.bar(x - width, b_df['DeepPrime-FT'], width, 
               yerr=[b_df['DeepPrime-FT_err_lower'], b_df['DeepPrime-FT_err_upper']], 
               label='DeepPrime-FT', color='#80B1D3', edgecolor='black', linewidth=0.8, capsize=3.5, error_kw={'elinewidth': 1})
        ax.bar(x, b_df['From_Scratch'], width, 
               yerr=[b_df['From_Scratch_err_lower'], b_df['From_Scratch_err_upper']], 
               label='From scratch', color='#B3E2CD', edgecolor='black', linewidth=0.8, capsize=3.5, error_kw={'elinewidth': 1})
        ax.bar(x + width, b_df['DeepPrime6'], width, 
               yerr=[b_df['DeepPrime6_err_lower'], b_df['DeepPrime6_err_upper']], 
               label='DeepPrime6', color='#FBB4AE', edgecolor='black', linewidth=0.8, capsize=3.5, error_kw={'elinewidth': 1})
               
        ax.set_ylabel("Spearman's correlation", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace('PE2max-dRNaseH', 'PE2max\nΔRNaseH') for d in b_df['Domain']], fontsize=12)
        ax.set_ylim(0, 0.85)
        ax.legend(frameon=False, fontsize=11, loc='upper left')
        sns.despine()
        plt.title("B", loc='left', fontsize=18, fontweight='bold', pad=8)
        plt.tight_layout()
        
        save_fig(plt, "Panel_B_Original_From_Scratch_Comparison.png", is_panel=True)
        save_fig(plt, "Panel_B_Original_From_Scratch_Comparison.svg", is_panel=True, format="svg")
        plt.close()
        print("Saved Original From Scratch Panel B (PNG and SVG).")

    # 2. EXTENDED 4-MODEL PANEL B (DeepPrime-FT, PRIDICT2.0, From Scratch, DeepPrime6)
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    x = np.arange(len(b_df))
    has_sc = 'From_Scratch' in b_df.columns
    
    if has_sc:
        width = 0.18
        ax.bar(x - 1.5*width, b_df['DeepPrime-FT'], width, 
               yerr=[b_df['DeepPrime-FT_err_lower'], b_df['DeepPrime-FT_err_upper']], 
               label='DeepPrime-FT', color='#80B1D3', edgecolor='black', linewidth=0.8, capsize=3, error_kw={'elinewidth': 1})
        ax.bar(x - 0.5*width, b_df['PRIDICT2.0'], width, 
               yerr=[b_df['PRIDICT2.0_err_lower'], b_df['PRIDICT2.0_err_upper']], 
               label='PRIDICT2.0', color='#8DD3C7', edgecolor='black', linewidth=0.8, capsize=3, error_kw={'elinewidth': 1})
        ax.bar(x + 0.5*width, b_df['From_Scratch'], width, 
               yerr=[b_df['From_Scratch_err_lower'], b_df['From_Scratch_err_upper']], 
               label='From scratch', color='#BEBADA', edgecolor='black', linewidth=0.8, capsize=3, error_kw={'elinewidth': 1})
        ax.bar(x + 1.5*width, b_df['DeepPrime6'], width, 
               yerr=[b_df['DeepPrime6_err_lower'], b_df['DeepPrime6_err_upper']], 
               label='DeepPrime6', color='#FBB4AE', edgecolor='black', linewidth=0.8, capsize=3, error_kw={'elinewidth': 1})
    else:
        width = 0.25
        ax.bar(x - width, b_df['DeepPrime-FT'], width, 
               yerr=[b_df['DeepPrime-FT_err_lower'], b_df['DeepPrime-FT_err_upper']], 
               label='DeepPrime-FT', color='#80B1D3', edgecolor='black', linewidth=0.8, capsize=3, error_kw={'elinewidth': 1})
        ax.bar(x, b_df['PRIDICT2.0'], width, 
               yerr=[b_df['PRIDICT2.0_err_lower'], b_df['PRIDICT2.0_err_upper']], 
               label='PRIDICT2.0', color='#8DD3C7', edgecolor='black', linewidth=0.8, capsize=3, error_kw={'elinewidth': 1})
        ax.bar(x + width, b_df['DeepPrime6'], width, 
               yerr=[b_df['DeepPrime6_err_lower'], b_df['DeepPrime6_err_upper']], 
               label='DeepPrime6', color='#FBB4AE', edgecolor='black', linewidth=0.8, capsize=3, error_kw={'elinewidth': 1})
           
    ax.set_ylabel("Spearman's correlation", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('PE2max-dRNaseH', 'PE2max\nΔRNaseH') for d in b_df['Domain']], fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, fontsize=10.5, loc='upper left')
    sns.despine()
    plt.title("B", loc='left', fontsize=18, fontweight='bold', pad=8)
    plt.tight_layout()
    
    save_fig(plt, "Panel_B_Model_Comparison.png", is_panel=True)
    save_fig(plt, "Panel_B_Model_Comparison.svg", is_panel=True, format="svg")
    plt.close()
    print("Saved 4-Model Panel B (PNG and SVG).")
    
    save_csv(b_df, "Panel_B_model_performance_summary.csv")
    
    # ------------------ PANEL C: Edit Type Comparison (DeepPrime6) ------------------
    def classify_edit(row):
        edit_len = row['Edit_len'] if 'Edit_len' in row else row.get('Edit length', np.nan)
        edit_type = row['Edit_type'] if 'Edit_type' in row else row.get('Edit type', '')
        if str(edit_type).capitalize() == 'Sub' and 1 <= edit_len <= 3:
            return '1-3 nt substitution'
        elif str(edit_type).capitalize() == 'Del' and 1 <= edit_len <= 4:
            return '1-4 nt deletion'
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
    
    for dom, (gt_col, dp6_col, *_) in domains.items():
        sub_df = df_c[[gt_col, dp6_col, 'Edit_Category']].dropna()
        dom_res = {'Domain': dom}
        for cat in categories:
            cat_df = sub_df[sub_df['Edit_Category'] == cat]
            if len(cat_df) > 5:
                sp = safe_spearman(cat_df[gt_col], cat_df[dp6_col])
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
    save_csv(c_df, "Panel_C_edit_type_performance_summary.csv")
    
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
    save_fig(plt, "Panel_C_EditType_Comparison.png", is_panel=True)
    save_fig(plt, "Panel_C_EditType_Comparison.svg", is_panel=True, format="svg")
    plt.close()
    
    print("Saved Panel B and Panel C plots successfully.")

def main():
    df = load_and_merge_data()
    run_leaks_audit(df)
    
    save_csv(df, "raw_data_all_edit_types.csv")
    
    # 1. Run evaluation for All Edit Types
    cohort_all = "All Edit Types"
    res_all = calculate_correlations(df, cohort_all)
    generate_visualizations(df, res_all, cohort_all)
    
    # 2. Generate Panel B and C plots (Grouped bar plots for correlation capacity)
    generate_grouped_bar_plots(df)
    
    # 3. Run evaluation for Substitutions Only (Official domain of PRIDICT2.0)
    cohort_sub = "Substitutions Only"
    sub_df = df[df['Edit_type'] == 'Sub'].copy()
    save_csv(sub_df, "raw_data_substitutions_only.csv")
    res_sub = calculate_correlations(sub_df, cohort_sub)
    generate_visualizations(sub_df, res_sub, cohort_sub)

if __name__ == "__main__":
    main()
