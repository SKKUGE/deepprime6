import pandas as pd
import json
import time
import os
import argparse
from src.data.components.pe6_preprocess_data import preprocess_data

# Fast reverse complement using translate
RC_TRANS = str.maketrans('ATGCNatgcn', 'TACGNtacgn')
def fast_rc(seq):
    return seq.translate(RC_TRANS)[::-1]

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
    with open(os.path.join(data_dir, "ncbi_cache", "resolved_pegrna_genomic_details.json")) as f:
        resolved = json.load(f)
        
    print(f"Total resolved unique variant IDs: {len(resolved)}")
    
    # Filter df to only include successfully resolved variants
    df_filtered = df[df['ID'].astype(str).isin(resolved.keys())].copy()
    print(f"Filtered dataset rows: {len(df_filtered)}")
    
    # Perform pre-inference biological QC and dynamic context reconstruction
    records = []
    print("Performing pre-inference biological QC and dynamic context reconstruction...")
    t_start = time.time()
    
    for idx, row in df_filtered.iterrows():
        aid = str(row['ID'])
        if aid not in resolved:
            continue
            
        detail = resolved[aid]
        pbs = str(row['PBS_pegRNA_DNA']).upper()
        rc_pbs = fast_rc(pbs)
        rtt_peg = str(row['RTT_template_DNA']).upper()
        rc_rtt = fast_rc(rtt_peg)
        
        # Decide which sequence to search in (prefer wide 2000bp cache if available)
        if 'wt_genomic_wide' in detail:
            wt_full = detail['wt_genomic_wide'].upper()
        else:
            wt_full = detail.get('wt_pridict_200', '').upper()
            
        if not wt_full:
            continue
            
        # Search for PBS reverse complement
        pos_fwd = wt_full.find(rc_pbs)
        
        # Search on reverse strand (since wt_genomic_wide is forward strand only)
        rc_wt_full = fast_rc(wt_full)
        pos_rc = rc_wt_full.find(rc_pbs)
        
        best_nick = None
        best_strand = None
        best_offset = 999
        
        # 1. Forward strand SpCas9 search
        if pos_fwd != -1:
            nick_fwd = pos_fwd + len(rc_pbs)
            # Find closest NGG PAM
            for i in range(len(wt_full) - 23):
                pam = wt_full[i+20 : i+23]
                if pam[1:3] == 'GG':
                    nick_cand = i + 20
                    offset = abs(nick_cand - nick_fwd)
                    if offset < best_offset:
                        best_offset = offset
                        best_nick = nick_cand
                        best_strand = '+'
                        
        # 2. Reverse strand SpCas9 search
        if pos_rc != -1:
            nick_rev = pos_rc + len(rc_pbs)
            # Find closest NGG PAM on reverse complement
            for i in range(len(rc_wt_full) - 23):
                pam = rc_wt_full[i+20 : i+23]
                if pam[1:3] == 'GG':
                    nick_cand = i + 20
                    offset = abs(nick_cand - nick_rev)
                    if offset < best_offset:
                        best_offset = offset
                        best_nick = nick_cand
                        best_strand = '-'
                        
        # Pre-inference QC: Only keep designs with a valid Cas9 site (offset <= 6)
        if best_nick is None or best_offset > 6:
            continue
            
        # Extract target strand genomic sequence
        target_seq = wt_full if best_strand == '+' else rc_wt_full
        
        # Extract 80nt target sequence centered around the nick site (nick at index 25 in 80nt)
        start_80 = best_nick - 25
        end_80 = best_nick + 55
        
        padded_80 = target_seq
        offset_pad80 = 0
        if start_80 < 0:
            pad_len = abs(start_80)
            padded_80 = ("N" * pad_len) + padded_80
            offset_pad80 = pad_len
            start_80 = 0
            end_80 += offset_pad80
            
        if end_80 > len(padded_80):
            pad_len = end_80 - len(padded_80)
            padded_80 = padded_80 + ("N" * pad_len)
            
        wt_80 = padded_80[start_80:end_80]
        guide = wt_80[8:28]
        
        # Extract 200nt WildTypeSequence centered around the nick site (nick at index 100)
        start_200 = best_nick - 100
        end_200 = best_nick + 100
        
        padded_200 = target_seq
        offset_pad200 = 0
        if start_200 < 0:
            pad_len = abs(start_200)
            padded_200 = ("N" * pad_len) + padded_200
            offset_pad200 = pad_len
            start_200 = 0
            end_200 += offset_pad200
            
        if end_200 > len(padded_200):
            pad_len = end_200 - len(padded_200)
            padded_200 = padded_200 + ("N" * pad_len)
            
        wt_200_centered = padded_200[start_200:end_200]
        
        # Calculate WT RTT length (aligned RTT region)
        rtt_flank = rc_rtt[-12:] if len(rc_rtt) >= 12 else rc_rtt
        idx_flank = target_seq.find(rtt_flank, best_nick)
        if idx_flank != -1:
            wt_rtt_end = idx_flank + len(rtt_flank)
            wt_rtt_len = wt_rtt_end - best_nick
        else:
            edit_type = detail.get('edit_type', 'Sub')
            edit_len = detail.get('edit_len', 0)
            if edit_type == 'Ins':
                wt_rtt_len = len(rc_rtt) - edit_len
            elif edit_type == 'Del':
                wt_rtt_len = len(rc_rtt) + edit_len
            else:
                wt_rtt_len = len(rc_rtt)
                
        # Construct PrimeEditedSequence centered at index 100
        ed_200_centered = wt_200_centered[:100] + rc_rtt + wt_200_centered[100 + wt_rtt_len:]
        
        records.append({
            'orig_index': idx,
            'WideTargetSequence': wt_80,
            'Guide': guide,
            'WildTypeSequence': wt_200_centered,
            'PrimeEditedSequence': ed_200_centered,
            'Edit_type': detail.get('edit_type', 'Sub'),
            'Edit_len': detail.get('edit_len', 1),
            'Edit_pos': detail.get('edit_pos', 1),
        })
        
    print(f"Biological QC and context extraction completed in {time.time() - t_start:.2f} seconds.")
    print(f"Selected {len(records)} / {len(df_filtered)} valid pegRNA designs (QC pass rate: {len(records)/len(df_filtered)*100:.2f}%).")
    
    if not records:
        print("Error: No designs passed the biological QC. Cannot proceed.")
        return
        
    qc_df = pd.DataFrame(records)
    
    # Map back original inputs for the selected valid designs
    valid_indices = qc_df['orig_index']
    df_valid = df.loc[valid_indices].copy()
    df_valid['WideTargetSequence'] = qc_df['WideTargetSequence'].values
    df_valid['Guide'] = qc_df['Guide'].values
    df_valid['WildTypeSequence'] = qc_df['WildTypeSequence'].values
    df_valid['PrimeEditedSequence'] = qc_df['PrimeEditedSequence'].values
    df_valid['Edit_type'] = qc_df['Edit_type'].values
    df_valid['Edit_len'] = qc_df['Edit_len'].values
    df_valid['Edit_pos'] = qc_df['Edit_pos'].values
    
    df_valid['leading G'] = ""
    df_valid['PBS'] = df_valid['PBS_pegRNA_DNA']
    df_valid['RTT'] = df_valid['RTT_template_DNA']
    df_valid['PBS_len'] = df_valid['PBSlen']
    df_valid['RTT_len'] = df_valid['RTlen']
    df_valid['Nicking'] = 25
    df_valid['ContextSeqUsed'] = "WideTargetSequence"
    
    # Store the original raw row index as string in the ID column for mapping predictions back
    df_valid['ID'] = df_valid.index.astype(str)
    
    # Keep columns for pre-processing
    cols = [
        'ID', 'WideTargetSequence', 'Guide', 'leading G', 'PBS', 'RTT',
        'Edit_type', 'Edit_len', 'Edit_pos', 'PBS_len', 'RTT_len', 'Nicking', 'ContextSeqUsed',
        'WildTypeSequence', 'PrimeEditedSequence'
    ]
    df_valid = df_valid[cols].copy()
    
    print("Running project's native preprocess_data...")
    t0 = time.time()
    processed_df = preprocess_data(data=df_valid)
    print(f"Preprocessing completed in {time.time() - t0:.2f} seconds.")
    
    # Add dummy target labels since PE6DeepPrimeDataset checks for them
    pe_types = [
        "PEmax", "PEmaxdRNaseH", "PE6a(+PEmaxCas9)", "PE6b(+PEmaxCas9)",
        "PE6c(+PEmaxCas9)", "PE6d(+PEmaxCas9)", "PE6e(+dRNaseH)", "PE6f(+dRNaseH)", "PE6g(+dRNaseH)"
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
