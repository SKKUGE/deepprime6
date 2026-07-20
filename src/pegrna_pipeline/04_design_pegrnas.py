import pandas as pd
import argparse
import os
import yaml
import hashlib
from Bio.Seq import Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: Design pegRNAs from scratch")
    parser.add_argument("--input", required=True, help="Input mutation_panel_primary.parquet")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    return parser.parse_args()

def reverse_complement(seq):
    return str(Seq(seq).reverse_complement())

def generate_design_id(mutation_key, spacer, pam, nick_pos, pbs, rtt):
    hash_input = f"{mutation_key}_{spacer}_{pam}_{nick_pos}_{pbs}_{rtt}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def find_pams(sequence, pam_pattern="NGG"):
    pams = []
    if pam_pattern == "NGG":
        for i in range(len(sequence) - 2):
            if sequence[i+1:i+3] == "GG":
                pams.append(i)
    return pams

def extract_target_features(wt_ctx, ed_ctx, ref, alt, strand, pbs_len, rtt_len):
    """
    Given the WT and Edited context (which are strand-aware),
    generate Spacer, PAM, Nick, PBS, RTT for all valid PAMs.
    
    The edit is at index 100 in the context (assuming context_window=100).
    """
    context_window = 100
    
    # We must find SpCas9 NGG PAMs on BOTH strands of the strand-aware context?
    # No, usually we just search for SpCas9 PAMs in the forward and reverse complement of the target context.
    # Actually, the user expects 'strand-aware' meaning WT context is already in the orientation of the gene/variant.
    # To find all Cas9 sites, we search the (+) and (-) strand of this context.
    
    pams_fwd = find_pams(wt_ctx)
    rc_wt_ctx = reverse_complement(wt_ctx)
    pams_rev = find_pams(rc_wt_ctx)
    
    designs = []
    
    # Process forward strand PAMs
    for pam_start in pams_fwd:
        nick_pos = pam_start - 3
        
        # Spacer is 20bp before PAM
        spacer_start = pam_start - 20
        if spacer_start < 0:
            continue
            
        spacer = wt_ctx[spacer_start:pam_start]
        pam = wt_ctx[pam_start:pam_start+3]
        
        # Calculate distance to edit
        # For forward strand, nick_pos is the index in wt_ctx.
        # The edit starts at `context_window`.
        edit_start = context_window
        edit_end = context_window + len(ref)
        
        # RTT must span from nick_pos to past the edit_end
        if nick_pos >= edit_end:
            continue # Edit is upstream of nick, cannot be encoded by RTT extending downstream
            
        required_rtt_len = edit_end - nick_pos
        if required_rtt_len > rtt_len:
            continue # Edit is too far downstream
            
        # Extract PBS (reverse complement of the sequence upstream of nick)
        pbs_start = nick_pos - pbs_len
        if pbs_start < 0:
            continue
        pbs_seq_target = wt_ctx[pbs_start:nick_pos]
        pbs = reverse_complement(pbs_seq_target)
        
        # Extract RTT (reverse complement of the edited sequence downstream of nick)
        # We need the edited sequence from nick_pos to nick_pos + rtt_len
        # Because we're designing the pegRNA, RTT is complementary to the target strand
        # Wait, the RTT on the pegRNA is complementary to the edited *non-target* strand?
        # Actually, RTT acts as a template for the 3' flap (target strand).
        # So RTT sequence is the same as the edited NON-target strand.
        # RTT on the pegRNA is reverse-complement of the edited target strand flap.
        # Which means RTT is EXACTLY the edited sequence (since it's a guide RNA, U -> T).
        # Let's align carefully: pegRNA spacer binds target strand.
        # pegRNA extension (PBS+RTT) is at the 3' end.
        # PBS binds the 3' flap of target strand (so PBS is reverse complement of target strand).
        # RTT is the template for synthesizing the new target strand flap.
        # So RTT must be the reverse complement of the newly synthesized target strand.
        # Wait, the newly synthesized strand IS the target strand.
        # Therefore, RTT is the reverse complement of the EDITED target strand.
        # Let's verify: In literature, RTT is usually the same sequence as the edited non-target strand, 
        # which means it's the reverse complement of the edited target strand.
        
        # Actually, the convention for RTT in many tools is just the reverse complement of the edited target strand flap.
        rtt_target = ed_ctx[nick_pos : nick_pos + rtt_len]
        if len(rtt_target) < rtt_len:
            continue
            
        rtt = reverse_complement(rtt_target)
        
        designs.append({
            'spacer': spacer,
            'PAM': pam,
            'nick_position': nick_pos,
            'PBS_sequence': pbs,
            'PBS_length': pbs_len,
            'RTT_sequence': rtt,
            'RTT_length': rtt_len,
            'pegRNA_strand': '+'
        })
        
    # Process reverse strand PAMs
    for pam_start in pams_rev:
        nick_pos = pam_start - 3
        
        spacer_start = pam_start - 20
        if spacer_start < 0:
            continue
            
        spacer = rc_wt_ctx[spacer_start:pam_start]
        pam = rc_wt_ctx[pam_start:pam_start+3]
        
        # For reverse strand, the edit is located at a different index
        rc_ed_ctx = reverse_complement(ed_ctx)
        
        # We need to find the equivalent edit position in the RC context
        # The edit in the forward context is at `context_window` and has length `len(ref)`.
        # In the RC context, the edit starts at `len(wt_ctx) - context_window - len(ref)`
        edit_start = len(wt_ctx) - context_window - len(ref)
        edit_end = edit_start + len(alt) # In edited context, the edit has length `len(alt)`
        
        if nick_pos >= edit_end:
            continue
            
        required_rtt_len = edit_end - nick_pos
        if required_rtt_len > rtt_len:
            continue
            
        pbs_start = nick_pos - pbs_len
        if pbs_start < 0:
            continue
            
        pbs_seq_target = rc_wt_ctx[pbs_start:nick_pos]
        pbs = reverse_complement(pbs_seq_target)
        
        rtt_target = rc_ed_ctx[nick_pos : nick_pos + rtt_len]
        if len(rtt_target) < rtt_len:
            continue
            
        rtt = reverse_complement(rtt_target)
        
        designs.append({
            'spacer': spacer,
            'PAM': pam,
            'nick_position': nick_pos, # Note: this is position on the relative strand
            'PBS_sequence': pbs,
            'PBS_length': pbs_len,
            'RTT_sequence': rtt,
            'RTT_length': rtt_len,
            'pegRNA_strand': '-'
        })
        
    return designs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_parquet(args.input)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    pbs_min = config['design_parameters']['pbs_length_min']
    pbs_max = config['design_parameters']['pbs_length_max']
    rtt_min = config['design_parameters']['rtt_length_min']
    rtt_max = config['design_parameters']['rtt_length_max']
    
    records = []
    
    for _, row in df.iterrows():
        base_dict = row.to_dict()
        
        # Generate all combinations
        for pbs_len in range(pbs_min, pbs_max + 1):
            for rtt_len in range(rtt_min, rtt_max + 1):
                designs = extract_target_features(
                    row['WT_context'], row['Edited_context'], 
                    row['REF'], row['ALT'], row['strand'],
                    pbs_len, rtt_len
                )
                
                for d in designs:
                    # Merge dictionaries
                    rec = {**base_dict, **d}
                    
                    # Generate design_id
                    d_id = generate_design_id(
                        rec['mutation_key'], rec['spacer'], rec['PAM'], 
                        rec['nick_position'], rec['PBS_sequence'], rec['RTT_sequence']
                    )
                    rec['design_id'] = d_id
                    rec['pegRNA_extension'] = rec['RTT_sequence'] + rec['PBS_sequence']
                    
                    records.append(rec)
                    
    df_out = pd.DataFrame(records)
    print(f"Generated {len(df_out)} total valid pegRNA designs for {len(df)} mutations.")
    
    df_out.to_parquet(os.path.join(args.output_dir, "pegrna_designs_raw.parquet"), index=False)

if __name__ == "__main__":
    main()
