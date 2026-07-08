import pandas as pd
import json
import time
import os
import argparse
from src.data.components.pe6_preprocess_data import preprocess_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory for input data")
    parser.add_argument("--output-dir", default="data/predictions", help="Directory for output data")
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    print(f"Using input data directory: {data_dir}")
    print(f"Using output data directory: {output_dir}")
    print("Loading original pegRNA dataset...")
    df = pd.read_csv(os.path.join(data_dir, "MFE_randompeg_RHA30_0.71M_result.csv"))
    
    print("Loading resolved genomic details...")
    with open(os.path.join(data_dir, "resolved_pegrna_genomic_details.json")) as f:
        resolved = json.load(f)
        
    print(f"Total resolved unique variant IDs: {len(resolved)}")
    
    # Filter df to only include successfully resolved variants
    # Use exact raw ID matching
    df_filtered = df[df['ID'].astype(str).isin(resolved.keys())].copy()
    print(f"Filtered dataset rows: {len(df_filtered)}")
    
    # Extract 80nt target sequence from wt_pridict_200 (nick is at index 100)
    df_filtered['WideTargetSequence'] = df_filtered['ID'].astype(str).map(lambda x: resolved[x]['wt_pridict_200'][75:155])
    
    # The spacer is at index 25 - 17 to 25 + 3 = 8:28 of the 80nt window
    df_filtered['Guide'] = df_filtered['WideTargetSequence'].str.slice(8, 28)
    df_filtered['leading G'] = ""
    df_filtered['PBS'] = df_filtered['PBS_pegRNA_DNA']
    df_filtered['RTT'] = df_filtered['RTT_template_DNA']
    
    # Map other resolved details
    edit_type_map = {k: v['edit_type'] for k, v in resolved.items()}
    edit_len_map = {k: v['edit_len'] for k, v in resolved.items()}
    edit_pos_map = {k: v['edit_pos'] for k, v in resolved.items()}
    
    df_filtered['Edit_type'] = df_filtered['ID'].astype(str).map(edit_type_map)
    df_filtered['Edit_len'] = df_filtered['ID'].astype(str).map(edit_len_map)
    df_filtered['Edit_pos'] = df_filtered['ID'].astype(str).map(edit_pos_map)
    
    df_filtered['PBS_len'] = df_filtered['PBSlen']
    df_filtered['RTT_len'] = df_filtered['RTlen']
    
    # Nicking index is at 25 in the 80nt window
    df_filtered['Nicking'] = 25
    df_filtered['ContextSeqUsed'] = "WideTargetSequence"
    
    # Cast ID to string
    df_filtered['ID'] = df_filtered['ID'].astype(str)
    
    # Keep only columns that are needed
    cols = [
        'ID', 'WideTargetSequence', 'Guide', 'leading G', 'PBS', 'RTT',
        'Edit_type', 'Edit_len', 'Edit_pos', 'PBS_len', 'RTT_len', 'Nicking', 'ContextSeqUsed'
    ]
    df_filtered = df_filtered[cols].copy()
    
    print("Running project's native preprocess_data...")
    t0 = time.time()
    processed_df = preprocess_data(data=df_filtered)
    print(f"Preprocessing completed in {time.time() - t0:.2f} seconds.")
    
    # Add dummy target labels since PE6DeepPrimeDataset checks for them
    pe_types = [
        "PEmax",
        "PEmaxdRNaseH",
        "PE6a(+PEmaxCas9)",
        "PE6b(+PEmaxCas9)",
        "PE6c(+PEmaxCas9)",
        "PE6d(+PEmaxCas9)",
        "PE6e(+dRNaseH)",
        "PE6f(+dRNaseH)",
        "PE6g(+dRNaseH)"
    ]
    for pe_type in pe_types:
        processed_df[pe_type] = 0.0
        
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "preprocessed_pegrna_prediction.parquet")
    print(f"Saving preprocessed dataset to {output_path}...")
    processed_df.to_parquet(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
