import pandas as pd
import numpy as np
import os
import sys
import subprocess
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure imports work from project root
sys.path.append("/home/work/workdir/deepprime6-genomebiol-revision")

def run_deepprime_predictions(df, output_dir):
    print("Preparing features for DeepPrime/DeepPrime6...")
    
    # Required features
    standard_features = [
        "PBS_len", "RTT_len", "RT-PBS_len", "Edit_pos", "Edit_len", "RHA_len",
        "type_sub", "type_ins", "type_del", "Tm1_PBS", "Tm2_RTT_cTarget_sameLength", "Tm3_RTT_cTarget_replaced", 
        "Tm4_cDNA_PAM-oppositeTarget", "Tm5_RTT_cDNA", "deltaTm_Tm4-Tm2",
        "GC_count_PBS", "GC_count_RTT", "GC_count_RT-PBS", "GC_contents_PBS", "GC_contents_RTT", "GC_contents_RT-PBS",
        "MFE_RT-PBS-polyT", "MFE_Spacer", "DeepSpCas9_score"
    ]
    
    biofeatures_raw = df[standard_features].copy()
    
    # Load normalization stats from DeepPrime base model
    dp_ckpt_dir = "src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428"
    norm_mean = pd.read_csv(os.path.join(dp_ckpt_dir, "mean.csv"), index_col=0).squeeze()
    norm_std = pd.read_csv(os.path.join(dp_ckpt_dir, "std.csv"), index_col=0).squeeze()
    
    feature_rename_map = {
        "PBS_len": "PBSlen", "RTT_len": "RTlen", "RT-PBS_len": "RT-PBSlen",
        "Tm1_PBS": "Tm1", "Tm2_RTT_cTarget_sameLength": "Tm2", "Tm3_RTT_cTarget_replaced": "Tm2new",
        "Tm5_RTT_cDNA": "Tm3", "Tm4_cDNA_PAM-oppositeTarget": "Tm4", "deltaTm_Tm4-Tm2": "TmD",
        "GC_count_PBS": "nGCcnt1", "GC_count_RTT": "nGCcnt2", "GC_count_RT-PBS": "nGCcnt3",
        "GC_contents_PBS": "fGCcont1", "GC_contents_RTT": "fGCcont2", "GC_contents_RT-PBS": "fGCcont3",
        "MFE_RT-PBS-polyT": "MFE3", "MFE_Spacer": "MFE4",
    }
    biofeatures_renamed = biofeatures_raw.rename(columns=feature_rename_map)
    dp_feature_names = norm_mean.index.tolist()
    biofeatures_ordered = biofeatures_renamed[dp_feature_names]
    
    norm_biofeatures = (biofeatures_ordered - norm_mean) / norm_std
    norm_biofeatures = norm_biofeatures.fillna(0)
    
    feat_df = df[['ID', 'Target', 'Masked_EditSeq'] + standard_features].copy()
    for col in norm_biofeatures.columns:
        feat_df[col] = norm_biofeatures[col]
        
    pe_types = [
        "PEmax", "PEmaxdRNaseH", "PE6a(+PEmaxCas9)", "PE6b(+PEmaxCas9)",
        "PE6c(+PEmaxCas9)", "PE6d(+PEmaxCas9)", "PE6e(+dRNaseH)", "PE6f(+dRNaseH)", "PE6g(+dRNaseH)"
    ]
    for pe_type in pe_types:
        feat_df[pe_type] = 0.0
        
    temp_input = os.path.join(output_dir, "temp_dp_input.parquet")
    feat_df.to_parquet(temp_input, index=False)
    
    # 1. Predict DeepPrime-base (ensemble of 20 models)
    print("Running DeepPrime-base predictions...")
    dp_base_preds = []
    for i in range(20):
        ckpt_path = os.path.join(dp_ckpt_dir, f"final_model_{i}.pt")
        temp_output = os.path.join(output_dir, f"temp_dp_base_{i}.csv")
        cmd = [
            "python", "src/predict.py",
            "trainer=gpu",
            f"data.csv_path={temp_input}",
            f"ckpt_path={ckpt_path}",
            f"model.prediction_save_path={temp_output}",
            "data.skip_preprocessing=True",
            "data.batch_size=8192",
            "data.dataloader.num_workers=0"
        ]
        subprocess.run(cmd, check=True)
        preds = pd.read_csv(temp_output)
        dp_base_preds.append(preds['Prediction'].values)
        os.remove(temp_output)
    
    df['pred_dp_base'] = np.mean(dp_base_preds, axis=0)
    
    # 2. Predict DeepPrime6 (PE6a, PE6b, PE6c)
    print("Running DeepPrime6 predictions...")
    dp6_ckpt_dir = "src/models/weights/DeepPrime6-weights"
    dp6_models = {
        "PE6a": "pe6a_mainft.ckpt",
        "PE6b": "pe6b_mainft.ckpt",
        "PE6c": "pe6c_mainft.ckpt"
    }
    
    pe_type_mapping = {
        "PE6a": "PE6a(+PEmaxCas9)",
        "PE6b": "PE6b(+PEmaxCas9)",
        "PE6c": "PE6c(+PEmaxCas9)",
    }
    
    dp6_preds_all = []
    for model_name, ckpt_name in dp6_models.items():
        ckpt_path = os.path.join(dp6_ckpt_dir, ckpt_name)
        temp_output = os.path.join(output_dir, f"temp_dp6_{model_name}.csv")
        target_pe_type = pe_type_mapping[model_name]
        
        cmd = [
            "python", "src/predict.py",
            "trainer=gpu",
            f"data.csv_path={temp_input}",
            f"ckpt_path={ckpt_path}",
            f"data.datafilter.PE_types=[\"{target_pe_type}\"]",
            f"model.prediction_save_path={temp_output}",
            "data.skip_preprocessing=True",
            "data.batch_size=8192",
            "data.dataloader.num_workers=0"
        ]
        subprocess.run(cmd, check=True)
        preds = pd.read_csv(temp_output)
        df[f'pred_dp6_{model_name.lower()}'] = preds['Prediction'].values
        dp6_preds_all.append(preds['Prediction'].values)
        os.remove(temp_output)
        
    df['pred_dp6'] = np.mean(dp6_preds_all, axis=0)
    
    os.remove(temp_input)
    return df

def run_pridict2_predictions(df):
    print("Running PRIDICT2.0 predictions...")
    pridict2_dir = os.path.abspath("PRIDICT2")
    sys.path.append(pridict2_dir)
    from src.utils.predict_pridict2 import load_pridict_model, deeppridict, compute_average_predictions, extract_single_pegRNA_features
    
    # Load PRIDICT2 models
    pridict_models = load_pridict_model([0])
    
    preds_pridict2_hek = np.zeros(len(df))
    
    pr_input_df = df.copy()
    # Format Edit_type to PRIDICT2's expectations
    def get_pr_edit_type(row):
        if row['Edit_type'] == 'sub': return 'Sub'
        elif row['Edit_type'] == 'del': return 'Del'
        elif row['Edit_type'] == 'ins': return 'Ins'
        else: return 'Sub'
    pr_input_df['Edit_type'] = df.apply(get_pr_edit_type, axis=1)
    
    pr_input_df['REF'] = df['OriginalBase']
    pr_input_df['ALT'] = df['EditedBase']
    pr_input_df['nick_position'] = df['Nicking']
    pr_input_df['WT_context'] = df['WildTypeSequence']
    pr_input_df['REF_ID'] = df['ID']
    
    from src.utils.predict_pridict2 import load_pridict_model, deeppridict, compute_average_predictions, extract_single_pegRNA_features
    
    # Load PRIDICT2 models
    pridict_models = load_pridict_model([0])
    
    # We will initialize with np.nan and update based on ID
    df['pred_pridict2'] = np.nan
    
    features_list_pr = []
    valid_ids = []
    
    # Process all rows (batch processing logic could be kept, but for simplicity of merging, we process all at once since the dataset is small)
    print("Running PRIDICT2.0 predictions...")
    for idx, row in pr_input_df.iterrows():
        try:
            feat = extract_single_pegRNA_features(row)
            if feat is not None:
                features_list_pr.append(feat)
                valid_ids.append(row['ID'])
        except Exception as e:
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
        tmp = [all_avg_preds[model_id] for model_id in all_avg_preds]
        tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
        agg_df = compute_average_predictions(tmp_df, grp_cols=['seq_id', 'dataset_name'])
        
        # Using HEK as default representative dataset
        hek_cond = agg_df['dataset_name'] == 'HEK'
        agg_hek = agg_df[hek_cond].sort_values('seq_id')
        
        # Make sure we assign based on original ID mapped back from sequence_name
        for _, r in agg_hek.iterrows():
            v_idx = int(r['seq_id'])
            ref_id = pegdataframe.iloc[v_idx]['sequence_name']
            df.loc[df['ID'] == ref_id, 'pred_pridict2'] = r['pred_averageedited'] * 100
            
    print(f"PRIDICT2: Processed {len(features_list_pr)}/{len(df)} rows successfully.")
    
    return df

def plot_correlations(df, output_path):
    print("Generating correlation plots...")
    
    # We will plot PE6-2avg against the predictions
    targets = ['pred_dp_base', 'pred_dp6', 'pred_pridict2']
    titles = ['DeepPrime-base', 'DeepPrime6 (Ensemble)', 'PRIDICT2.0']
    
    experimental_col = 'PE6-2avg'
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (target, title) in enumerate(zip(targets, titles)):
        sns.scatterplot(data=df, x=target, y=experimental_col, ax=axes[idx], alpha=0.5, s=15)
        
        valid_mask = ~df[target].isna() & ~df[experimental_col].isna()
        r = df.loc[valid_mask, target].corr(df.loc[valid_mask, experimental_col], method='pearson')
        rho = df.loc[valid_mask, target].corr(df.loc[valid_mask, experimental_col], method='spearman')
        
        axes[idx].set_title(f"{title} vs {experimental_col}\nPearson: {r:.3f}, Spearman: {rho:.3f}")
        axes[idx].set_xlabel("Predicted Efficiency")
        axes[idx].set_ylabel(f"Experimental Efficiency ({experimental_col})")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plots to {output_path}")

def main():
    input_file = "data/small_library/pe6to7_with_avgs.parquet"
    output_dir = "data/small_library"
    output_file = os.path.join(output_dir, "pe6to7_predictions.parquet")
    plot_file = os.path.join(output_dir, "prediction_correlations.png")
    
    df = pd.read_parquet(input_file)
    print(f"Loaded dataset with {len(df)} rows")
    
    df = run_deepprime_predictions(df, output_dir)
    df = run_pridict2_predictions(df)
    
    df.to_parquet(output_file, index=False)
    print(f"Saved updated dataset to {output_file}")
    
    plot_correlations(df, plot_file)

if __name__ == "__main__":
    main()
