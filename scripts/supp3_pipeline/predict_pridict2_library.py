import os
import sys
import pandas as pd
import numpy as np
import time
import argparse
from multiprocessing import Pool

# Set up python paths to import from cloned PRIDICT2
pridict2_dir = os.path.abspath("PRIDICT2")
sys.path.append(pridict2_dir)

import torch
import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pridict.pridictv2.utilities import *
from pridict.pridictv2.dataset import *
from pridict.pridictv2.predict_outcomedistrib import *
from trained_models.DeepCas9_TestCode import runprediction

# Re-use helper functions from project's predict_pridict2.py
from src.utils.predict_pridict2 import (
    extract_single_pegRNA_features,
    load_pridict_model,
    deeppridict,
    compute_average_predictions
)

def process_chunk(chunk_df):
    features = []
    for idx, row in chunk_df.iterrows():
        try:
            feat = extract_single_pegRNA_features(row)
            if feat is not None:
                # Store original raw CSV index (stored in 'index' column) to map back
                feat['orig_index'] = int(row['index'])
                features.append(feat)
        except Exception:
            pass
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory for input data")
    parser.add_argument("--output-dir", default="data/predictions", help="Directory for output data")
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    print(f"Using input data directory: {data_dir}")
    print(f"Using output data directory: {output_dir}")
    parquet_path = os.path.join(output_dir, "preprocessed_pegrna_prediction.parquet")
    print(f"Loading preprocessed dataset from {parquet_path}...")
    df_filtered = pd.read_parquet(parquet_path)
    
    df_filtered['index'] = df_filtered['ID'].astype(int)
    df_filtered['REF_ID'] = df_filtered['ID']
    print(f"Loaded preprocessed dataset rows: {len(df_filtered)}")
    
    num_cores = 32
    print("Loading PRIDICT2.0 trained models (using run 0)...")
    models_list = load_pridict_model(run_ids=[0])
    
    # Process in chunks of 100,000 to survive server restarts
    chunk_size = 100000
    num_chunks = int(np.ceil(len(df_filtered) / chunk_size))
    print(f"Total chunks to process: {num_chunks}")
    
    os.makedirs(output_dir, exist_ok=True)
    completed_dfs = []
    
    for i in range(num_chunks):
        chunk_path = os.path.join(output_dir, f"predictions_pridict2_chunk_{i}.csv")
        if os.path.exists(chunk_path):
            print(f"Chunk {i} already processed. Loading from cache...")
            completed_dfs.append(pd.read_csv(chunk_path))
            continue
            
        print(f"\nProcessing chunk {i+1}/{num_chunks}...")
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df_filtered))
        df_chunk = df_filtered.iloc[start_idx:end_idx].copy()
        
        # Run parallel feature extraction on this chunk
        t0 = time.time()
        df_split = np.array_split(df_chunk, num_cores)
        with Pool(num_cores) as p:
            results = p.map(process_chunk, df_split)
        features_list = [item for sublist in results for item in sublist]
        print(f"Feature extraction for chunk {i} completed in {time.time() - t0:.2f}s.")
        
        if len(features_list) == 0:
            print(f"Warning: No features extracted for chunk {i}.")
            continue
            
        pegdataframe = pd.DataFrame(features_list)
        
        # Run GPU PRIDICT2.0 model prediction on this chunk
        t1 = time.time()
        all_avg_preds = deeppridict(pegdataframe, models_list)
        print(f"PRIDICT2.0 prediction for chunk {i} completed in {time.time() - t1:.2f}s.")
        
        # Aggregate predictions across models
        tmp = [all_avg_preds[model_id] for model_id in all_avg_preds]
        tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
        agg_df = compute_average_predictions(tmp_df, grp_cols=['seq_id', 'dataset_name'])
        
        cell_types = ['HEK', 'K562']
        for cell_type in cell_types:
            cond = agg_df['dataset_name'] == cell_type
            sub_df = agg_df[cond].copy()
            score_map = sub_df.set_index('seq_id')['pred_averageedited'] * 100
            pegdataframe[f'PRIDICT2_Score_{cell_type}'] = pegdataframe.index.map(score_map)
            
        output_chunk = pegdataframe.copy()
        output_chunk['ID'] = output_chunk['orig_index'].astype(str)
        output_chunk['variant_id'] = output_chunk['sequence_name']
        
        # Reset index to create a sequential 0, 1, 2... index
        output_chunk = output_chunk.reset_index(drop=True).reset_index(drop=False)
        
        output_cols = [
            'index', 'ID', 'variant_id', 'PRIDICT2_Score_HEK', 'PRIDICT2_Score_K562'
        ]
        output_chunk = output_chunk[output_cols].copy()
        
        output_chunk.to_csv(chunk_path, index=False)
        completed_dfs.append(output_chunk)
        
    print("\nAll chunks processed. Concatenating results...")
    final_df = pd.concat(completed_dfs, axis=0, ignore_index=True)
    
    # Re-generate sequential index for concatenated dataframe to be 100% correct
    final_df['index'] = range(len(final_df))
    
    output_path = os.path.join(output_dir, "predictions_pridict2.csv")
    final_df.to_csv(output_path, index=False)
    
    # Clean up intermediate chunk files
    for i in range(num_chunks):
        try:
            os.remove(os.path.join(output_dir, f"predictions_pridict2_chunk_{i}.csv"))
        except Exception:
            pass
            
    print("Done!")

if __name__ == "__main__":
    main()
