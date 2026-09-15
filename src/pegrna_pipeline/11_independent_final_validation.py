import pandas as pd
import argparse
import os
from Bio import Align
from Bio.Seq import Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 11: Independent final validation of candidates")
    parser.add_argument("--input", required=True, help="Input final_candidates.parquet")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    return parser.parse_args()

def reverse_complement(seq):
    return str(Seq(seq).reverse_complement())

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_parquet(args.input)
    if len(df) == 0:
        print("Empty input. Saving empty output.")
        df.to_parquet(os.path.join(args.output_dir, "final_candidates_validated.parquet"), index=False)
        return
        
    print(f"Running independent validation on {len(df)} candidates...")
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    
    validated = []
    
    for idx, row in df.iterrows():
        spacer = row['spacer']
        pam = row['PAM']
        pbs = row['PBS_sequence']
        rtt = row['RTT_sequence']
        wt = row['WT_context']
        ed = row['Edited_context']
        strand = row['pegRNA_strand']
        
        # Validation 1: Spacer + PAM in WT
        target_seq = spacer + pam
        # Depending on strand, it will be fwd or rc
        fwd_score = aligner.score(wt.upper(), target_seq.upper())
        rc_score = aligner.score(wt.upper(), reverse_complement(target_seq).upper())
        
        has_target = target_seq.upper() in wt.upper() or reverse_complement(target_seq).upper() in wt.upper()
        
        # Validation 2: pegRNA Extension (RTT + PBS) properly matches the edited target flap
        # The pegRNA extension is RTT + PBS.
        # RTT is the reverse complement of the edited target strand flap.
        # PBS is the reverse complement of the 3' flap of the target strand.
        # So RC(RTT + PBS) = RC(PBS) + RC(RTT) = Target Flap + Edited Flap.
        # This sequence should perfectly match a subregion of the Edited Context!
        
        target_ed = ed.upper() if strand == '+' else reverse_complement(ed).upper()
        expected_extension = reverse_complement(rtt + pbs).upper()
        has_ext = expected_extension in target_ed
        
        if has_target and has_ext:
            validated.append(row)
            
    df_validated = pd.DataFrame(validated)
    print(f"Validation complete: {len(df_validated)} out of {len(df)} passed.")
    
    out_path = os.path.join(args.output_dir, "final_candidates_validated.parquet")
    df_validated.to_parquet(out_path, index=False)
    
if __name__ == "__main__":
    main()
