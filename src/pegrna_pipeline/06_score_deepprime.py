import pandas as pd
import argparse
import os
import yaml
import torch
import numpy as np
import sys
import glob
import subprocess

# Ensure we can import from src
sys.path.append("/home/work/workdir/deepprime6-genomebiol-revision")
from src.data.components.pe6_preprocess_data import preprocess_data

# Ensure PRIDICT2 path for PE6DeepPrimeModule
pridict2_dir = os.path.abspath("PRIDICT2")
sys.path.append(pridict2_dir)
from src.models.pe6_module_dp_only import PE6DeepPrimeModule

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 6: Score pegRNAs using DeepPrime(base)")
    parser.add_argument("--input", required=True, help="Input pegrna_designs_qc_pass.parquet")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--model-versions", required=True, help="Path to model_versions.yaml")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    return parser.parse_args()

def map_to_legacy_features(df):
    """Maps new pipeline columns to legacy ML columns expected by pe6_preprocess_data."""
    ldf = df.copy()
    ldf['WideTargetSequence'] = ldf['WT_context']
    ldf['ContextSeqUsed'] = 'WideTargetSequence'
    ldf['Guide'] = ldf['spacer']
    ldf['leading G'] = ""
    ldf['PBS'] = ldf['PBS_sequence']
    ldf['RTT'] = ldf['RTT_sequence']
    ldf['Edit length'] = ldf['ALT'].str.len()
    
    # We need Edit position relative to RTT.
    # Edit position is 1-indexed distance from the nick.
    # In the new pipeline, nick_pos is the index of the nick. The edit in the forward context is at index 100.
    # (or in reverse context at len - 100 - len(ref)). 
    # Edit_pos = Edit start - Nicking + 1
    # We can calculate it directly:
    context_window = 100
    edit_starts = []
    for _, row in ldf.iterrows():
        strand = row['pegRNA_strand']
        if strand == '+':
            edit_starts.append(context_window)
        else:
            edit_starts.append(len(row['WT_context']) - context_window - len(row['REF']))
    
    ldf['Edit_start_idx'] = edit_starts
    ldf['Edit position'] = ldf['Edit_start_idx'] - ldf['nick_position'] + 1
    ldf['Nicking'] = df['nick_position'].astype(int)
    
    # Map Edit type
    def get_legacy_edit_type(row):
        ref, alt = row['REF'], row['ALT']
        if len(ref) == len(alt):
            return 'sub'
        elif len(ref) > len(alt):
            return 'del'
        else:
            return 'ins'
    ldf['Edit_type'] = ldf.apply(get_legacy_edit_type, axis=1)
    
    # For GuideStart, GuideEnd, Nicking, pe6_preprocess_data expects us to provide Nicking or it recalculates.
    # It recalculates 'Nicking' using find_all_indices(Guide) - 3. 
    # We will just supply 'Nicking' and hopefully it uses it. Actually `calculate_guide_features` overwrites it.
    # This is fine since our spacer perfectly matches the WT context.
    return ldf

def one_hot_encode_sequence(sequence):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "X": 4, "N": 4}
    map_seq = [mapping[i] for i in sequence.upper()]
    arr_seq = np.eye(5)[map_seq]
    return np.delete(arr_seq, -1, axis=1)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_parquet(args.input)
    if len(df) == 0:
        print("Empty input. Saving empty output.")
        df['pred_dp_base'] = []
        df.to_parquet(os.path.join(args.output_dir, "pegrna_designs_dp_scored.parquet"), index=False)
        return
        
    with open(args.model_versions, "r") as f:
        mv = yaml.safe_load(f)
        
    dp_ckpt_dir = mv['models']['deepprime']['checkpoint']
    
    print("Mapping columns to legacy formats...")
    legacy_df = map_to_legacy_features(df)
    
    print("Extracting bio-features for DeepPrime...")
    feat_df = preprocess_data(data=legacy_df)
    
    # Prepare standard features
    standard_features = [
        "PBS_len", "RTT_len", "RT-PBS_len", "Edit_pos", "Edit_len", "RHA_len",
        "type_sub", "type_ins", "type_del", "Tm1_PBS", "Tm2_RTT_cTarget_sameLength", "Tm3_RTT_cTarget_replaced", 
        "Tm4_cDNA_PAM-oppositeTarget", "Tm5_RTT_cDNA", "deltaTm_Tm4-Tm2",
        "GC_count_PBS", "GC_count_RTT", "GC_count_RT-PBS", "GC_contents_PBS", "GC_contents_RTT", "GC_contents_RT-PBS",
        "MFE_RT-PBS-polyT", "MFE_Spacer", "DeepSpCas9_score"
    ]
    
    biofeatures_raw = feat_df[standard_features].copy()
    
    # Normalize features using the provided DP_variant mean and std
    norm_mean = pd.read_csv(os.path.join(dp_ckpt_dir, "mean.csv"), index_col=0).squeeze()
    norm_std = pd.read_csv(os.path.join(dp_ckpt_dir, "std.csv"), index_col=0).squeeze()
    
    # Match columns carefully. The mean.csv uses old DP names.
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
    
    for col in norm_biofeatures.columns:
        feat_df[col] = norm_biofeatures[col]
        
    # Add dummy target labels since PE6DeepPrimeDataset checks for them
    pe_types = [
        "PEmax", "PEmaxdRNaseH", "PE6a(+PEmaxCas9)", "PE6b(+PEmaxCas9)",
        "PE6c(+PEmaxCas9)", "PE6d(+PEmaxCas9)", "PE6e(+dRNaseH)", "PE6f(+dRNaseH)", "PE6g(+dRNaseH)"
    ]
    for pe_type in pe_types:
        feat_df[pe_type] = 0.0
        
    # We must ensure there's an orig_index or ID column for predict.py to output
    feat_df['ID'] = df['run_id'].values
        
    temp_input = os.path.join(args.output_dir, "temp_deepprime_input.parquet")
    temp_output = os.path.join(args.output_dir, "temp_deepprime_scores.csv")
    feat_df.to_parquet(temp_input, index=False)
    
    print("Running DeepPrime predictions via src/predict.py...")
    cmd = [
        "python", "src/predict.py",
        "trainer=gpu",
        f"data.csv_path={temp_input}",
        "ckpt_path=null",
        f"model.prediction_save_path={temp_output}",
        "data.skip_preprocessing=True",
        "data.batch_size=8192",
        "data.dataloader.num_workers=0"
    ]
    subprocess.run(cmd, check=True)
    
    # Load the output predictions
    print("Loading predictions...")
    preds = pd.read_csv(temp_output)
    
    # The output has 'Prediction' and 'orig_index' or 'ID' depending on the model config
    # We can rely on the fact that predict.py preserves row order for skip_preprocessing=True
    # But just in case, we can map using the ID/orig_index if it was saved.
    # Actually predict.py saves: Target, Prediction, orig_index, PE_type
    if 'Prediction' in preds.columns:
        df['pred_dp_base'] = preds['Prediction'].values
    else:
        raise ValueError("Prediction column not found in predict.py output")
        
    # Cleanup temps
    try:
        os.remove(temp_input)
        os.remove(temp_output)
    except:
        pass
    
    print(f"Saving {len(df)} predictions to pegrna_designs_dp_scored.parquet...")
    df.to_parquet(os.path.join(args.output_dir, "pegrna_designs_dp_scored.parquet"), index=False)
    
    # We can also save the extracted biofeatures so Phase 7 doesn't have to recompute them!
    # Because both DP and DP6 use the exact same feature extraction logic.
    feat_df.to_parquet(os.path.join(args.output_dir, "pegrna_designs_biofeatures.parquet"), index=False)

if __name__ == "__main__":
    main()
