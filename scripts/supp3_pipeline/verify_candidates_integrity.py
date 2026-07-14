import pandas as pd
import numpy as np
import os

def run_checks(df, label, is_parquet=False):
    print(f"\n==========================================")
    print(f"Running integrity checks on: {label}")
    print(f"==========================================")
    
    total = len(df)
    print(f"Total entries: {total}")
    
    # 1. Guide search in WideTargetSequence and PAM validation
    pam_passed = 0
    edit_passed = 0
    both_passed = 0
    
    details = []
    
    for idx, row in df.iterrows():
        # ID is either the row index (if parquet) or the ID column
        row_id = str(row.get('ID', idx))
        
        # Guide and target sequence
        guide = str(row.get('Guide', '')).upper()
        wt_80 = str(row.get('WideTargetSequence', '')).upper()
        
        if not guide or not wt_80:
            details.append({
                'id': row_id,
                'guide': guide,
                'has_guide': False,
                'pam': None,
                'has_pam': False,
                'edit_len': 0,
                'has_edit': False
            })
            continue
            
        # Find guide start
        guide_start = wt_80.find(guide)
        
        has_pam = False
        pam = None
        if guide_start != -1:
            guide_end = guide_start + len(guide) - 1
            # PAM starts at guide_end + 1
            pam_start = guide_end + 1
            pam_end = pam_start + 3
            if pam_end <= len(wt_80):
                pam = wt_80[pam_start:pam_end]
                if len(pam) == 3 and pam[1:3] == 'GG':
                    has_pam = True
        
        # Check Edit encoding
        edit_len = row.get('Edit_len', 0)
        # Fallback to compare WT and PE sequence if Edit_len is missing or 0
        wt_seq = str(row.get('WildTypeSequence', '')).upper()
        pe_seq = str(row.get('PrimeEditedSequence', '')).upper()
        has_edit = (edit_len > 0)
        
        # If we have wild-type and prime-edited sequences, they should differ at the edit site
        if wt_seq and pe_seq:
            # Masked_EditSeq in FINAL_v6 has 'x' padding, clean it for comparison
            wt_clean = wt_seq.replace('X', '').replace('N', '')
            pe_clean = pe_seq.replace('X', '').replace('N', '').replace('x', '')
            if wt_clean and pe_clean and wt_clean != pe_clean:
                has_edit = True
                
        if has_pam:
            pam_passed += 1
        if has_edit:
            edit_passed += 1
        if has_pam and has_edit:
            both_passed += 1
            
        details.append({
            'id': row_id,
            'guide': guide,
            'has_guide': (guide_start != -1),
            'pam': pam,
            'has_pam': has_pam,
            'edit_len': edit_len,
            'has_edit': has_edit
        })
        
    print(f"1. PAM immediately downstream of Guide matches NGG: {pam_passed} / {total} ({pam_passed/total*100:.2f}%)")
    print(f"2. RTT encodes edit (edit_len > 0 / sequence diff): {edit_passed} / {total} ({edit_passed/total*100:.2f}%)")
    print(f"3. BOTH PAM and Edit encoding pass: {both_passed} / {total} ({both_passed/total*100:.2f}%)")
    
    return pd.DataFrame(details)

def main():
    # 1. Check preprocessed database parquet
    parquet_path = 'data/predictions/preprocessed_pegrna_prediction.parquet'
    if os.path.exists(parquet_path):
        prep_df = pd.read_parquet(parquet_path)
        # Running check on a subset or full
        details_prep = run_checks(prep_df, "Full Preprocessed Database Parquet")
        
        # Print details for ID 902 and 8327 specifically
        print("\n=== Specific Audit for ID=902 and ID=8327 (from Preprocessed Parquet) ===")
        for target_id in ['902', '8327']:
            match = details_prep[details_prep['id'] == target_id]
            if not match.empty:
                info = match.iloc[0]
                print(f"ID {target_id}:")
                print(f"  Guide: {info['guide']}")
                print(f"  PAM downstream of Guide: {info['pam']} (Valid NGG? {info['has_pam']})")
                print(f"  RTT edit encoding: edit_len={info['edit_len']} (Has edit? {info['has_edit']})")
            else:
                print(f"ID {target_id} not found in preprocessed parquet details.")
    else:
        print(f"Preprocessed parquet not found at {parquet_path}")

    # 2. Check current selected candidates
    cand_path = 'data/predictions/selected_50_validation_candidates.csv'
    if os.path.exists(cand_path):
        # We need to map WideTargetSequence and other columns back from prep_df if not present
        # In selected_50_validation_candidates.csv, WideTargetSequence, Guide, WildTypeSequence, PrimeEditedSequence, Edit_len are already present
        cand_df = pd.read_csv(cand_path)
        details_cand = run_checks(cand_df, "New Selected 50 Candidates (selected_50_validation_candidates.csv)")
    else:
        print(f"Selected candidates CSV not found at {cand_path}")

    # 3. Check buggy/old candidate spreadsheet
    f6_path = 'data/FINAL_v6_ALL_INFO.csv'
    if os.path.exists(f6_path):
        f6_df = pd.read_csv(f6_path)
        details_f6 = run_checks(f6_df, "Old/Buggy Candidate Spreadsheet (FINAL_v6_ALL_INFO.csv)")
    else:
        print(f"Old candidate spreadsheet not found at {f6_path}")

if __name__ == '__main__':
    main()
