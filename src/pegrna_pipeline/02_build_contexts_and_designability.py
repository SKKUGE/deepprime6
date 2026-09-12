import pandas as pd
import argparse
import os
import json
import yaml
from Bio.Seq import Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Build contexts and assess designability")
    parser.add_argument("--input", required=True, help="Input variants_eligible_1to3bp.parquet")
    parser.add_argument("--fasta", required=True, help="Path to reference FASTA (hg38)")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    return parser.parse_args()

def reverse_complement(seq):
    return str(Seq(seq).reverse_complement())

def find_pams(sequence, pam_pattern="NGG"):
    pams = []
    if pam_pattern == "NGG":
        for i in range(len(sequence) - 2):
            if sequence[i+1:i+3] == "GG":
                pams.append(i)
    return pams

def assess_designability(wt_fwd, ref, alt, config):
    # Simplified PAM availability check
    # Assumes the edit is centrally located (around index len(wt_fwd)//2)
    pams_fwd = find_pams(wt_fwd, config['design_parameters']['pam'])
    pams_rev = find_pams(reverse_complement(wt_fwd), config['design_parameters']['pam'])
    
    rtt_max = config['design_parameters']['rtt_length_max']
    valid_pams = 0
    edit_idx = len(wt_fwd) // 2
    
    for pam_pos in pams_fwd:
        nick_pos = pam_pos - 3
        dist_to_edit = edit_idx - nick_pos
        if 0 < dist_to_edit <= rtt_max:
            valid_pams += 1
            
    for pam_pos in pams_rev:
        nick_pos = pam_pos - 3
        edit_pos_rev = len(wt_fwd) - edit_idx - len(ref)
        dist_to_edit = edit_pos_rev - nick_pos
        if 0 < dist_to_edit <= rtt_max:
            valid_pams += 1
            
    return valid_pams > 0

def main():
    import pysam
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_parquet(args.input)
    fasta = pysam.FastaFile(args.fasta)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    records = []
    designable_count = 0
    
    for _, row in df.iterrows():
        chrom = row['chromosome']
        pos = row['pos']
        ref = row['REF']
        alt = row['ALT']
        strand = row['strand']
        
        # We already have WT_context and Edited_context from phase 1
        wt_ctx = str(row['WT_context']).upper()
        ed_ctx = str(row['Edited_context']).upper()
        
        if chrom != 'unknown':
            # Could re-extract from fasta here if needed, but we already have 121nt contexts.
            pass
            
        # Pad context to 200bp with the edit exactly at index 100
        edit_pos = -1
        for i in range(min(len(wt_ctx), len(ed_ctx))):
            if wt_ctx[i] != ed_ctx[i]:
                edit_pos = i
                break
        
        if edit_pos != -1:
            pad_left = 100 - edit_pos
            if pad_left > 0:
                wt_ctx = ("N" * pad_left) + wt_ctx
                ed_ctx = ("N" * pad_left) + ed_ctx
            elif pad_left < 0:
                wt_ctx = wt_ctx[-pad_left:]
                ed_ctx = ed_ctx[-pad_left:]
                
            pad_right_wt = 200 - len(wt_ctx)
            if pad_right_wt > 0:
                wt_ctx = wt_ctx + ("N" * pad_right_wt)
            elif pad_right_wt < 0:
                wt_ctx = wt_ctx[:200]
                
            pad_right_ed = 200 - len(ed_ctx)
            # The edited context might have a different length if it's an indel,
            # but usually models prefer the same length or dynamic handling.
            # We'll pad ed_ctx based on ref/alt diff. 
            # Actually, just pad it to 200 + (len(alt) - len(ref))
            target_ed_len = 200 + len(alt) - len(ref)
            pad_right_ed = target_ed_len - len(ed_ctx)
            if pad_right_ed > 0:
                ed_ctx = ed_ctx + ("N" * pad_right_ed)
            elif pad_right_ed < 0:
                ed_ctx = ed_ctx[:target_ed_len]
                
        designable = False
        # For the pilot, 'N' is present due to padding, so we bypass 'N' not in wt_ctx
        # We just need it to be long enough.
        if len(wt_ctx) >= 200:
            # We assume edit is roughly in the middle
            has_valid_pam = assess_designability(wt_ctx, ref, alt, config)
            if has_valid_pam:
                designable = True
                
        r = row.to_dict()
        r['WT_context'] = wt_ctx
        r['Edited_context'] = ed_ctx
        r['designable'] = designable
        
        if designable:
            designable_count += 1
            
        records.append(r)
        
    df_out = pd.DataFrame(records)
    df_out.to_parquet(os.path.join(args.output_dir, "variants_designability.parquet"), index=False)
    
    qc_summary = {
        "total_input": len(df),
        "designable": designable_count,
        "undesignable": len(df) - designable_count
    }
    
    with open(os.path.join(args.output_dir, "designability_qc_summary.json"), "w") as f:
        json.dump(qc_summary, f, indent=4)
        
    print(f"Phase 2 completed. Designable mutations: {designable_count}/{len(df)}")

if __name__ == "__main__":
    main()
