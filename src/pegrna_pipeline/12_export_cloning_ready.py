import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 12: Export cloning-ready table")
    parser.add_argument("--input", required=True, help="Input final_candidates_validated.parquet")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    return parser.parse_args()

def generate_oligo_sequence(row):
    # Depending on the vector and cloning strategy (e.g. Golden Gate with BsaI / BsmBI)
    # A standard pegRNA oligo structure might look like:
    # 5' - BsaI site - Spacer - scaffold - RTT - PBS - Term - BsaI site - 3'
    # For now, we will export the components clearly so users can append whatever adapter they need.
    return row['spacer'] + "-scaffold-" + row['RTT_sequence'] + row['PBS_sequence']

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_parquet(args.input)
    if len(df) == 0:
        print("Empty input. Saving empty output.")
        pd.DataFrame().to_csv(os.path.join(args.output_dir, "cloning_ready_pegrnas.csv"), index=False)
        return
        
    print(f"Exporting {len(df)} validated candidates for cloning...")
    
    df['Oligo_Sequence_Placeholder'] = df.apply(generate_oligo_sequence, axis=1)
    
    # Select columns to export
    export_cols = [
        'design_id', 'mutation_key', 'run_id', 'Name', 'chromosome', 'pos', 'REF', 'ALT', 'strand',
        'spacer', 'PAM', 'nick_position', 'PBS_sequence', 'PBS_length', 
        'RTT_sequence', 'RTT_length', 'pegRNA_extension', 'pegRNA_strand',
        'pred_dp_base', 'pred_dp6', 'pred_pridict2',
        'Representative_Percentile', 'Disagreement_Score', 'Efficacy_Category', 'Disagreement_Subcategory',
        'Oligo_Sequence_Placeholder'
    ]
    
    # Include all available columns in export_cols
    available_cols = [c for c in export_cols if c in df.columns]
    df_export = df[available_cols].copy()
    
    out_csv = os.path.join(args.output_dir, "cloning_ready_pegrnas.csv")
    df_export.to_csv(out_csv, index=False)
    
    # Save a summary report
    summary = df_export.groupby(['Efficacy_Category', 'Disagreement_Subcategory']).size().unstack(fill_value=0)
    summary_path = os.path.join(args.output_dir, "final_selection_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Final Selection Summary\n\n")
        f.write(f"Total PegRNAs Selected: {len(df)}\n\n")
        f.write("## Distribution by Stratum\n")
        f.write(summary.to_markdown() + "\n")
        
    print(f"Successfully exported {len(df)} pegRNAs to {out_csv}")
    print(f"Summary report written to {summary_path}")
    
if __name__ == "__main__":
    main()
