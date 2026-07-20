import pandas as pd
import argparse
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 9: Normalize scores (Percentile ranking)")
    parser.add_argument("--input", required=True, help="Input pegrna_designs_pridict2_scored.parquet")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_parquet(args.input)
    if len(df) == 0:
        print("Empty input. Saving empty output.")
        df.to_parquet(os.path.join(args.output_dir, "pegrna_designs_scored_normalized.parquet"), index=False)
        return
        
    print("Calculating percentile ranks for model scores...")
    
    # Higher score = better efficiency, so we want percentile such that 1.0 is the best score.
    if 'pred_dp_base' in df.columns:
        df['Percentile_DP_Base'] = df['pred_dp_base'].rank(pct=True, ascending=True)
    else:
        df['Percentile_DP_Base'] = 0.5
        
    if 'pred_pridict2' in df.columns:
        df['Percentile_PRIDICT2'] = df['pred_pridict2'].rank(pct=True, ascending=True)
    else:
        df['Percentile_PRIDICT2'] = 0.5
        
    print("Setting PRIDICT2 as Representative Score, DP Base as Auxiliary...")
    df['Representative_Percentile'] = df['Percentile_PRIDICT2']
    
    # Disagreement is the absolute difference between the percentiles
    df['Disagreement_Score'] = np.abs(df['Percentile_PRIDICT2'] - df['Percentile_DP_Base'])
    
    # Keep run_id and mutation_key to ensure data lineage is intact
    if 'run_id' not in df.columns:
        df['run_id'] = "pilot_run_v1"
        
    out_path = os.path.join(args.output_dir, "pegrna_designs_scored_normalized.parquet")
    print(f"Saving {len(df)} normalized designs to {out_path}...")
    df.to_parquet(out_path, index=False)
    
if __name__ == "__main__":
    main()
