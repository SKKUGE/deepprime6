import pandas as pd
import argparse
import os
import yaml
import numpy as np
import sys

# Ensure we can import from PRIDICT2
pridict2_dir = os.path.abspath("PRIDICT2")
sys.path.append(pridict2_dir)
from src.utils.predict_pridict2 import load_pridict_model, deeppridict, compute_average_predictions, extract_single_pegRNA_features

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 8: Score pegRNAs using PRIDICT2.0")
    parser.add_argument("--input", required=True, help="Input pegrna_designs_dp6_scored.parquet")
    parser.add_argument("--features", required=True, help="Input pegrna_designs_biofeatures.parquet generated from Phase 6")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--model-versions", required=True, help="Path to model_versions.yaml")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Args parsed.")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Reading input dataframe...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows.")
    if len(df) == 0:
        print("Empty input. Saving empty output.")
        df['pred_pridict2'] = []
        df.to_parquet(os.path.join(args.output_dir, "pegrna_designs_pridict2_scored.parquet"), index=False)
        return
        
    feat_df = pd.read_parquet(args.features)
    
    with open(args.model_versions, "r") as f:
        mv = yaml.safe_load(f)
        
    # We load PRIDICT2 model (it uses ONNX and its own internal load logic)
    print("Loading PRIDICT2.0 models...")
    pridict_models = load_pridict_model(run_ids=[0])
    
    # Map to PRIDICT2 input format
    pr_input_df = pd.DataFrame()
    pr_input_df['REF_ID'] = df['design_id']
    pr_input_df['WildTypeSequence'] = df['WT_context']
    pr_input_df['PrimeEditedSequence'] = df['Edited_context']
    # PRIDICT2 expects 'Sub', 'Ins', 'Del'. The feat_df already mapped this to some extent?
    # Actually, in PRIDICT2 prediction loop they often hardcode 'Sub' or use proper. 
    # Let's derive it from length
    def get_pr_edit_type(row):
        ref, alt = row['REF'], row['ALT']
        if len(ref) == len(alt): return 'Sub'
        elif len(ref) > len(alt): return 'Del'
        else: return 'Ins'
    pr_input_df['Edit_type'] = df.apply(get_pr_edit_type, axis=1)
    
    pr_input_df['Guide'] = df['spacer']
    pr_input_df['PBS'] = df['PBS_sequence']
    pr_input_df['RTT'] = df['RTT_sequence']
    
    print("Running PRIDICT2.0 feature parsing and inference in chunks...")
    num_samples = len(df)
    pr_chunk_size = 50000
    preds_pridict2_hek = np.zeros(num_samples)
    
    for chunk_start in range(0, num_samples, pr_chunk_size):
        chunk_end = min(chunk_start + pr_chunk_size, num_samples)
        chunk_df = pr_input_df.iloc[chunk_start:chunk_end].reset_index(drop=True)
        
        # Parse features
        features_list_pr = []
        valid_indices = []
        for idx, row in chunk_df.iterrows():
            try:
                feat = extract_single_pegRNA_features(row)
                if feat is not None:
                    features_list_pr.append(feat)
                    valid_indices.append(idx)
            except Exception:
                pass
                
        if len(features_list_pr) > 0:
            pegdataframe = pd.DataFrame(features_list_pr)
            # Fill missing keys if any failed (should be rare for valid rows)
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
                    
            # Call deeppridict
            all_avg_preds = deeppridict(pegdataframe, pridict_models)
            tmp = [all_avg_preds[model_id] for model_id in all_avg_preds]
            tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
            agg_df = compute_average_predictions(tmp_df, grp_cols=['seq_id', 'dataset_name'])
            
            # We only want the representative score (HEK293T) as configured in model_versions.yaml
            hek_cond = agg_df['dataset_name'] == 'HEK'
            
            # Assign back to valid indices
            agg_hek = agg_df[hek_cond].sort_values('seq_id')
            if len(agg_hek) == len(valid_indices):
                for v_idx, score in zip(valid_indices, agg_hek['pred_averageedited'].values * 100):
                    preds_pridict2_hek[chunk_start + v_idx] = score
            else:
                # Fallback if somehow length mismatches, though it shouldn't
                pass
        
        print(f"Processed {chunk_end}/{num_samples} rows for PRIDICT2.0...")
        
    df['pred_pridict2'] = preds_pridict2_hek
    
    print(f"Saving {len(df)} predictions to pegrna_designs_pridict2_scored.parquet...")
    df.to_parquet(os.path.join(args.output_dir, "pegrna_designs_pridict2_scored.parquet"), index=False)

if __name__ == "__main__":
    main()
