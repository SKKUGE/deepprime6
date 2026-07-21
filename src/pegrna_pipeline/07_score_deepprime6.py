import pandas as pd
import argparse
import os
import yaml
import torch
import numpy as np
import sys

# Ensure we can import from src
sys.path.append("/home/work/workdir/deepprime6-genomebiol-revision")
from src.data.components.pe6_preprocess_data import preprocess_data

# Ensure PRIDICT2 path for PE6DeepPrimeModule
pridict2_dir = os.path.abspath("PRIDICT2")
sys.path.append(pridict2_dir)
from src.models.pe6_module_dp_only import PE6DeepPrimeModule

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 7: Score pegRNAs using DeepPrime6")
    parser.add_argument("--input", required=True, help="Input pegrna_designs_dp_scored.parquet")
    parser.add_argument("--features", required=True, help="Input pegrna_designs_biofeatures.parquet generated from Phase 6")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--model-versions", required=True, help="Path to model_versions.yaml")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    return parser.parse_args()

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
        df['pred_dp6'] = []
        df.to_parquet(os.path.join(args.output_dir, "pegrna_designs_dp6_scored.parquet"), index=False)
        return
        
    feat_df = pd.read_parquet(args.features)
        
    with open(args.model_versions, "r") as f:
        mv = yaml.safe_load(f)
        
    dp6_checkpoints = mv['models']['deepprime6'].get('checkpoints', {})
    
    # Add dummy target labels since PE6DeepPrimeDataset checks for them
    pe_types = [
        "PEmax", "PEmaxdRNaseH", "PE6a(+PEmaxCas9)", "PE6b(+PEmaxCas9)",
        "PE6c(+PEmaxCas9)", "PE6d(+PEmaxCas9)", "PE6e(+dRNaseH)", "PE6f(+dRNaseH)", "PE6g(+dRNaseH)"
    ]
    for pe_type in pe_types:
        feat_df[pe_type] = 0.0
        
    feat_df['ID'] = df['run_id'].values
        
    temp_input = os.path.join(args.output_dir, "temp_dp6_input.parquet")
    feat_df.to_parquet(temp_input, index=False)
    
    print("Running DeepPrime6 predictions via src/predict.py...")
    import subprocess
    
    model_preds = {}
    
    # PE type mapping corresponding to each model
    pe_type_mapping = {
        "PE6a": "PE6a(+PEmaxCas9)",
        "PE6b": "PE6b(+PEmaxCas9)",
        "PE6c": "PE6c(+PEmaxCas9)",
        "PEmaxdRNaseH": "PEmaxdRNaseH"
    }
    
    for model_name, ckpt_path in dp6_checkpoints.items():
        print(f"Scoring {model_name}...")
        temp_output = os.path.join(args.output_dir, f"temp_dp6_scores_{model_name}.csv")
        target_pe_type = pe_type_mapping.get(model_name, "PE6b(+PEmaxCas9)")
        
        cmd = [
            "python", "src/predict.py",
            "trainer=gpu",
            f"data.csv_path={temp_input}",
            f"ckpt_path={ckpt_path}",
            f"data.datafilter.PE_types=[\"{target_pe_type}\"]",
            f"model.prediction_save_path={temp_output}",
            "data.skip_preprocessing=True",
            "data.batch_size=8192"
        ]
        subprocess.run(cmd, check=True)
        
        preds = pd.read_csv(temp_output)
        if 'Prediction' in preds.columns:
            df[f'pred_dp6_{model_name.lower()}'] = preds['Prediction'].values
            model_preds[model_name] = preds['Prediction'].values
        else:
            raise ValueError(f"Prediction column not found in predict.py output for {model_name}")
            
        try:
            os.remove(temp_output)
        except:
            pass
            
    # Calculate ensemble average
    all_preds = np.array(list(model_preds.values()))
    df['pred_dp6'] = np.mean(all_preds, axis=0)
        
    # Cleanup temps
    try:
        os.remove(temp_input)
    except:
        pass
    
    print(f"Saving {len(df)} predictions to pegrna_designs_dp6_scored.parquet...")
    df.to_parquet(os.path.join(args.output_dir, "pegrna_designs_dp6_scored.parquet"), index=False)

if __name__ == "__main__":
    main()
