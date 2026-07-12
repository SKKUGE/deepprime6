import pandas as pd
import json
import os
from Bio.Seq import Seq

def find_true_spacer_and_pam(wt_200, pbs, rc_pbs, approx_nick):
    best_offset = 999
    best_spacer = None
    best_pam = None
    best_strand = None
    
    for i in range(len(wt_200) - 23):
        # 1. Forward strand SpCas9 site: spacer = wt_200[i:i+20], PAM = wt_200[i+20:i+23]
        pam_fwd = wt_200[i+20 : i+23]
        if pam_fwd[1:3] == "GG":
            nick_fwd = i + 20
            offset = abs(nick_fwd - approx_nick)
            if offset < best_offset:
                best_offset = offset
                best_spacer = wt_200[i : i+20]
                best_pam = pam_fwd
                best_strand = '+'
                
        # 2. Reverse strand SpCas9 site: PAM = wt_200[i:i+3] (CC)
        pam_rev = wt_200[i : i+3]
        if pam_rev[0:2] == "CC":
            nick_rev = i + 3
            offset = abs(nick_rev - approx_nick)
            if offset < best_offset:
                best_offset = offset
                best_spacer = str(Seq(wt_200[i+3 : i+23]).reverse_complement())
                best_pam = str(Seq(pam_rev).reverse_complement())
                best_strand = '-'
                
    if best_offset <= 6:
        return best_spacer, best_pam, best_strand, best_offset
    return None, None, None, best_offset

def main():
    candidates_path = "data/predictions/selected_50_validation_candidates.csv"
    resolved_path = "data/ncbi_cache/resolved_pegrna_genomic_details.json"
    
    cand_df = pd.read_csv(candidates_path)
    with open(resolved_path) as f:
        resolved = json.load(f)
        
    passed = 0
    failed = 0
    
    for idx, row in cand_df.iterrows():
        raw_id = str(row['ID'])
        pbs = str(row['PBS_pegRNA_DNA']).upper()
        rc_pbs = str(Seq(pbs).reverse_complement())
        
        if raw_id not in resolved:
            print(f"[-] Candidate {idx} (ID {raw_id}): Not in resolved details")
            failed += 1
            continue
            
        detail = resolved[raw_id]
        wt_200 = detail['wt_pridict_200'].upper()
        
        # Calculate approx nick from PBS
        pos_fwd = wt_200.find(pbs)
        pos_rc = wt_200.find(rc_pbs)
        
        if pos_rc != -1:
            approx_nick = pos_rc + len(pbs)
        elif pos_fwd != -1:
            approx_nick = pos_fwd
        else:
            print(f"[-] Candidate {idx} (ID {raw_id}): PBS/rcPBS not found in wt_200")
            failed += 1
            continue
            
        spacer, pam, strand, offset = find_true_spacer_and_pam(wt_200, pbs, rc_pbs, approx_nick)
        
        if spacer:
            passed += 1
            print(f"[+] Candidate {idx} (ID {raw_id}) PASSED: Spacer={spacer}, PAM={pam}, Strand={strand}, Offset={offset}, Gene={detail.get('gene', 'unknown')}, Transcript={detail.get('transcript')}")
        else:
            failed += 1
            print(f"[-] Candidate {idx} (ID {raw_id}) FAILED: No active SpCas9 site found near PBS boundary (min offset = {offset})")
            
    print(f"\n=== Evaluation Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

if __name__ == '__main__':
    main()
