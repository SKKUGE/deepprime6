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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from Bio.Seq import Seq, reverse_complement
from src.data.components.pe6_preprocess_data import preprocess_data

# Ensure PRIDICT2 path for PE6DeepPrimeModule
pridict2_dir = os.path.abspath("PRIDICT2")
if pridict2_dir not in sys.path:
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
    ldf['WideTargetSequence'] = [
        str(reverse_complement(row['WT_context'])) if row['pegRNA_strand'] == '-' else row['WT_context']
        for _, row in ldf.iterrows()
    ]
    ldf['ContextSeqUsed'] = 'WideTargetSequence'
    ldf['Guide'] = ldf['spacer']
    ldf['leading G'] = ""
    ldf['PBS'] = ldf['PBS_sequence']
    ldf['RTT'] = ldf['RTT_sequence']

    def get_legacy_edit_len(row):
        ref = "" if pd.isna(row['REF']) else str(row['REF'])
        alt = "" if pd.isna(row['ALT']) else str(row['ALT'])
        if len(ref) == len(alt):
            return len(alt)
        return abs(len(ref) - len(alt))

    ldf['Edit length'] = ldf.apply(get_legacy_edit_len, axis=1)
    
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
    
    # Keep raw biofeatures in feat_df so that src/predict.py normalizes once using mean.csv/std.csv,
    # and Phase 7 (07_score_deepprime6.py) normalizes once using pe6_mean.csv/pe6_std.csv.
    pe_types = [
        "PEmax", "PEmaxdRNaseH", "PE6a(+PEmaxCas9)", "PE6b(+PEmaxCas9)",
        "PE6c(+PEmaxCas9)", "PE6d(+PEmaxCas9)", "PE6e(+dRNaseH)", "PE6f(+dRNaseH)", "PE6g(+dRNaseH)"
    ]
    for pe_type in pe_types:
        feat_df[pe_type] = 0.0
        
    # Ensure a unique ID column for predict.py output
    feat_df['ID'] = df['design_id'].values if 'design_id' in df.columns else df['run_id'].values
        
    temp_input = os.path.join(args.output_dir, "temp_deepprime_input.parquet")
    temp_output = os.path.join(args.output_dir, "temp_deepprime_scores.csv")
    feat_df.to_parquet(temp_input, index=False)
    
    print("Running DeepPrime predictions via src/predict.py...")
    cmd = [
        sys.executable, "src/predict.py",
        "trainer=gpu",
        f"data.csv_path={temp_input}",
        "ckpt_path=null",
        f"data.dataset.norm_mean_path={os.path.join(dp_ckpt_dir, 'mean.csv')}",
        f"data.dataset.norm_std_path={os.path.join(dp_ckpt_dir, 'std.csv')}",
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
