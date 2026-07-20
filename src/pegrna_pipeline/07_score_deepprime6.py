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
        
    dp6_ckpt_path = mv['models']['deepprime6']['checkpoint']
    
    # Add dummy target labels since PE6DeepPrimeDataset checks for them
    pe_types = [
        "PEmax", "PEmaxdRNaseH", "PE6a(+PEmaxCas9)", "PE6b(+PEmaxCas9)",
        "PE6c(+PEmaxCas9)", "PE6d(+PEmaxCas9)", "PE6e(+dRNaseH)", "PE6f(+dRNaseH)", "PE6g(+dRNaseH)"
    ]
    for pe_type in pe_types:
        feat_df[pe_type] = 0.0
        
    feat_df['ID'] = df['run_id'].values
        
    temp_input = os.path.join(args.output_dir, "temp_dp6_input.parquet")
    temp_output = os.path.join(args.output_dir, "temp_dp6_scores.csv")
    feat_df.to_parquet(temp_input, index=False)
    
    print("Running DeepPrime6 predictions via src/predict.py...")
    import subprocess
    cmd = [
        "python", "src/predict.py",
        "trainer=gpu",
        f"data.csv_path={temp_input}",
        f"ckpt_path={dp6_ckpt_path}",
        "data.datafilter.PE_types=[\"PE6b(+PEmaxCas9)\"]",
        f"model.prediction_save_path={temp_output}",
        "data.skip_preprocessing=True",
        "data.batch_size=8192"
    ]
    subprocess.run(cmd, check=True)
    
    # Load the output predictions
    print("Loading DeepPrime6 predictions...")
    preds = pd.read_csv(temp_output)
    
    if 'Prediction' in preds.columns:
        df['pred_dp6'] = preds['Prediction'].values
    else:
        raise ValueError("Prediction column not found in predict.py output")
        
    # Cleanup temps
    try:
        os.remove(temp_input)
        os.remove(temp_output)
    except:
        pass
    
    print(f"Saving {len(df)} predictions to pegrna_designs_dp6_scored.parquet...")
    df.to_parquet(os.path.join(args.output_dir, "pegrna_designs_dp6_scored.parquet"), index=False)

if __name__ == "__main__":
    main()
