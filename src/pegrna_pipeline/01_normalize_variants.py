import pandas as pd
import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Normalize variants and filter for 1-3bp mutations")
    parser.add_argument("--input", required=True, help="Input CSV file (clinvar_fix.csv)")
    parser.add_argument("--fasta", required=True, help="Reference FASTA file")
    parser.add_argument("--run-id", required=True, help="Pipeline run identifier")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    return parser.parse_args()

def parse_ref_alt(wt, ed):
    ref = ""
    alt = ""
    edit_pos = -1
    for i in range(min(len(wt), len(ed))):
        if wt[i] != ed[i]:
            edit_pos = i
            break
            
    if edit_pos != -1:
        ref = wt[edit_pos]
        alt = ed[edit_pos]
        if len(wt) > len(ed):
            diff = len(wt) - len(ed)
            ref = wt[edit_pos:edit_pos+diff+1]
            alt = ed[edit_pos]
        elif len(ed) > len(wt):
            diff = len(ed) - len(wt)
            ref = wt[edit_pos]
            alt = ed[edit_pos:edit_pos+diff+1]
    return ref, alt

def get_edit_type_and_length(ref, alt):
    edit_length = len(alt) if len(alt) > len(ref) else len(ref)
    if len(ref) == len(alt):
        mut_type = 'Sub'
    elif len(ref) > len(alt):
        mut_type = 'Del'
    else:
        mut_type = 'Ins'
    return mut_type, edit_length

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading input file: {args.input}")
    df = pd.read_csv(args.input)
    
    if 'Name' not in df.columns:
        raise ValueError("Missing required column: Name")
        
    normalized_records = []
    
    for idx, row in df.iterrows():
        name = row['Name']
        wt = str(row['4']).upper()
        ed = str(row['5']).upper()
        
        ref, alt = parse_ref_alt(wt, ed)
        mut_type, edit_length = get_edit_type_and_length(ref, alt)
        
        # Filter for 1-3bp edits
        if edit_length > 3:
            continue
            
        normalized_records.append({
            'run_id': args.run_id,
            'mutation_key': name,
            'WT_context': wt,
            'Edited_context': ed,
            'REF': ref,
            'ALT': alt,
            'chromosome': 'unknown',
            'pos': 0,
            'strand': '+',
            'edit_type': mut_type,
            'edit_length': edit_length
        })
        
    normalized = pd.DataFrame(normalized_records)
    
    print(f"Normalized {len(normalized)} valid 1-3bp variants out of {len(df)}")
    
    output_path = os.path.join(args.output_dir, "normalized_variants.parquet")
    normalized.to_parquet(output_path, index=False)
    
    with open(os.path.join(args.output_dir, "normalization_report.json"), "w") as f:
        json.dump({
            "total_input": len(df),
            "total_valid_1_3bp": len(normalized)
        }, f, indent=4)

if __name__ == "__main__":
    main()
