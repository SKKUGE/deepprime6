import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 13: Plot Model Score Scatter")
    parser.add_argument("--input", default="data/pilot_output/pegrna_designs_scored_normalized.parquet", help="Input pegrna_designs_scored_normalized.parquet")
    parser.add_argument("--output-dir", default="data/pilot_output", help="Directory to store outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_parquet(args.input)
    if len(df) == 0:
        print("Empty input. No plot to generate.")
        return
        
    print(f"Generating scatter plots for {len(df)} variants...")
    
    # 1. Percentile Scatter Plot (PRIDICT2 vs DP_Base)
    if 'Percentile_PRIDICT2' in df.columns and 'Percentile_DP_Base' in df.columns:
        plt.figure(figsize=(8, 8))
        plt.scatter(df['Percentile_DP_Base'], df['Percentile_PRIDICT2'], alpha=0.1, s=1, color='blue')
        plt.title('Percentile Ranking: PRIDICT2 vs DeepPrime Base')
        plt.xlabel('DeepPrime Base (Percentile)')
        plt.ylabel('PRIDICT2 (Percentile)')
        plt.plot([0, 1], [0, 1], 'r--', lw=1)  # y=x line
        
        out_path = os.path.join(args.output_dir, "scatter_percentiles_pridict2_vs_dp_base.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved percentile scatter plot to {out_path}")
        
    # 2. Raw Score Plot (using Hexbin for density visibility instead of scatter)
    if 'pred_pridict2' in df.columns and 'pred_dp_base' in df.columns:
        plt.figure(figsize=(8, 8))
        hb = plt.hexbin(df['pred_dp_base'], df['pred_pridict2'], gridsize=50, cmap='viridis', mincnt=1, bins='log')
        cb = plt.colorbar(hb, label='log10(N)')
        plt.title('Raw Scores Density: PRIDICT2 vs DeepPrime Base')
        plt.xlabel('DeepPrime Base (Raw Score)')
        plt.ylabel('PRIDICT2 (Raw Score)')
        
        out_path = os.path.join(args.output_dir, "density_raw_scores_pridict2_vs_dp_base.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved raw score density plot to {out_path}")
        
    # 3. Export Source Data to CSV
    csv_path = os.path.join(args.output_dir, "plot_source_data.csv")
    cols_to_save = ['run_id', 'mutation_key', 'design_id', 'spacer', 'PBS_sequence', 'RTT_sequence']
    for col in ['pred_dp_base', 'pred_pridict2', 'Percentile_DP_Base', 'Percentile_PRIDICT2']:
        if col in df.columns:
            cols_to_save.append(col)
            
    # Keep only columns that exist
    cols_to_save = [c for c in cols_to_save if c in df.columns]
    
    if len(cols_to_save) > 0:
        print(f"Exporting plot source data to {csv_path}...")
        df[cols_to_save].to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()
