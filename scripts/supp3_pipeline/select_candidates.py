import pandas as pd
import os
import argparse
import json
from Bio.Seq import Seq

def find_true_spacer_and_pam(wt_200, pbs):
    rc_pbs = str(Seq(pbs).reverse_complement())
    pos_fwd = wt_200.find(pbs)
    pos_rc = wt_200.find(rc_pbs)
    
    if pos_rc != -1:
        approx_nick = pos_rc + len(pbs)
    elif pos_fwd != -1:
        approx_nick = pos_fwd
    else:
        return None, None, None, 999
        
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

def is_valid_design(row, resolved):
    var_id = str(row['ID']).strip()
    if var_id not in resolved:
        # If not resolved yet, we allow it (it will be checked/resolved later)
        return True
    
    wt_200 = resolved[var_id].get('wt_pridict_200', '').upper()
    if not wt_200:
        return True
        
    pbs = str(row['PBS_pegRNA_DNA']).upper()
    
    # Check if the design has a valid SpCas9 site within 6bp of the nick
    _, _, _, offset = find_true_spacer_and_pam(wt_200, pbs)
    if offset <= 6:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory for input data")
    parser.add_argument("--output-dir", default="data/predictions", help="Directory for output data")
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    # Load resolved genomic details cache if it exists
    resolved_path = os.path.join(data_dir, "ncbi_cache", "resolved_pegrna_genomic_details.json")
    resolved = {}
    if os.path.exists(resolved_path):
        try:
            with open(resolved_path) as f:
                resolved = json.load(f)
            print(f"Loaded {len(resolved)} resolved cache entries for candidate design verification.")
        except Exception as e:
            print(f"Warning: Failed to load resolved cache: {e}")

    print(f"Using input data directory: {data_dir}")
    print(f"Using output data directory: {output_dir}")
    print("Loading predictions and raw dataset...")
    raw = pd.read_csv(os.path.join(data_dir, "MFE_randompeg_RHA30_0.71M_result.csv"))
    
    # Load all prediction files from output_dir
    dp_base = pd.read_csv(os.path.join(output_dir, "pegrna_predictions.csv"))
    pe6a = pd.read_csv(os.path.join(output_dir, "predictions_pe6a.csv"))
    pe6b = pd.read_csv(os.path.join(output_dir, "predictions_pe6b.csv"))
    pe6c = pd.read_csv(os.path.join(output_dir, "predictions_pe6c.csv"))
    pemaxdrnaseh = pd.read_csv(os.path.join(output_dir, "predictions_pemaxdrnaseh.csv"))
    pridict = pd.read_csv(os.path.join(output_dir, "predictions_pridict2.csv"))
    
    # Map predictions back to raw using the original indices
    print("Mapping model scores to pegRNAs...")
    # Convert 'ID' column (which holds the original raw index as a string) to integer index
    dp_base['orig_index'] = dp_base['ID'].astype(int)
    pe6a['orig_index'] = pe6a['ID'].astype(int)
    pe6b['orig_index'] = pe6b['ID'].astype(int)
    pe6c['orig_index'] = pe6c['ID'].astype(int)
    pemaxdrnaseh['orig_index'] = pemaxdrnaseh['ID'].astype(int)
    pridict['orig_index'] = pridict['ID'].astype(int)
    
    raw['Score_DP_Base'] = raw.index.map(dp_base.set_index('orig_index')['Prediction'])
    raw['Score_PE6a'] = raw.index.map(pe6a.set_index('orig_index')['Prediction'])
    raw['Score_PE6b'] = raw.index.map(pe6b.set_index('orig_index')['Prediction'])
    raw['Score_PE6c'] = raw.index.map(pe6c.set_index('orig_index')['Prediction'])
    raw['Score_PEmax_dRNaseH'] = raw.index.map(pemaxdrnaseh.set_index('orig_index')['Prediction'])
    raw['Score_PRIDICT_HEK'] = raw.index.map(pridict.set_index('orig_index')['PRIDICT2_Score_HEK'])
    raw['Score_PRIDICT_K562'] = raw.index.map(pridict.set_index('orig_index')['PRIDICT2_Score_K562'])
    
    # Map preprocessed sequence context columns back to raw dataframe
    parquet_path = os.path.join(output_dir, "preprocessed_pegrna_prediction.parquet")
    if os.path.exists(parquet_path):
        print("Mapping preprocessed genomic sequence contexts back to candidates...")
        prep_df = pd.read_parquet(parquet_path)
        prep_df['orig_index'] = prep_df['ID'].astype(int)
        raw['WideTargetSequence'] = raw.index.map(prep_df.set_index('orig_index')['WideTargetSequence'])
        raw['Guide'] = raw.index.map(prep_df.set_index('orig_index')['Guide'])
        raw['WildTypeSequence'] = raw.index.map(prep_df.set_index('orig_index')['WildTypeSequence'])
        raw['PrimeEditedSequence'] = raw.index.map(prep_df.set_index('orig_index')['PrimeEditedSequence'])
        raw['Edit_type'] = raw.index.map(prep_df.set_index('orig_index')['Edit_type'])
        raw['Edit_len'] = raw.index.map(prep_df.set_index('orig_index')['Edit_len'])
        raw['Edit_pos'] = raw.index.map(prep_df.set_index('orig_index')['Edit_pos'])
    
    # Map resolved genomic context coordinates and gene info
    raw['chr'] = raw['ID'].astype(str).map(lambda x: resolved.get(x, {}).get('chr', 'unknown'))
    raw['strand'] = raw['ID'].astype(str).map(lambda x: resolved.get(x, {}).get('strand', 'unknown'))
    raw['transcript'] = raw['ID'].astype(str).map(lambda x: resolved.get(x, {}).get('transcript', 'unknown'))
    raw['gene'] = raw['ID'].astype(str).map(lambda x: resolved.get(x, {}).get('gene', 'unknown'))

    # Keep only successfully scored pegRNAs
    df = raw.dropna(subset=['Score_DP_Base', 'Score_PE6a', 'Score_PRIDICT_HEK']).copy()
    print(f"Total pegRNAs with complete predictions: {len(df)}")
    
    # Calculate representative model scores
    df['DP_Base'] = df['Score_DP_Base']
    df['DP6_Avg'] = df[['Score_PE6a', 'Score_PE6b', 'Score_PE6c', 'Score_PEmax_dRNaseH']].mean(axis=1)
    df['PRIDICT2_Avg'] = df[['Score_PRIDICT_HEK', 'Score_PRIDICT_K562']].mean(axis=1)
    
    # Compute percentile ranks (0 to 1) across the entire library
    print("Calculating percentile ranks...")
    df['Percentile_DP_Base'] = df['DP_Base'].rank(pct=True)
    df['Percentile_DP6'] = df['DP6_Avg'].rank(pct=True)
    df['Percentile_PRIDICT2'] = df['PRIDICT2_Avg'].rank(pct=True)
    
    # Calculate average percentile and disagreement score (standard deviation)
    percentile_cols = ['Percentile_DP_Base', 'Percentile_DP6', 'Percentile_PRIDICT2']
    df['Average_Percentile'] = df[percentile_cols].mean(axis=1)
    df['Disagreement_Score'] = df[percentile_cols].std(axis=1)
    

            
    # Track selected raw IDs to ensure 100% unique ClinVar variants
    # Since ID is unique for each variant (both normal and clinic are separate), we map on ID!
    # Cast ID to string
    df['ID'] = df['ID'].astype(str)
    selected_variants = set()
    selected_rows = []
    
    def get_candidate(candidates_pool, sort_by_cols, ascending_list, category, subcategory):
        for idx, row in candidates_pool.sort_values(sort_by_cols, ascending=ascending_list).iterrows():
            var_id = row['ID']
            if var_id not in selected_variants:
                if not is_valid_design(row, resolved):
                    continue
                selected_variants.add(var_id)
                row_dict = row.to_dict()
                row_dict['orig_index'] = idx # Store original raw index
                row_dict['Category'] = category
                row_dict['Subcategory'] = subcategory
                selected_rows.append(row_dict)
                return True
        return False

    print("Selecting 50 validation candidates...")
    
    # 1. Highly Efficient (15 candidates)
    # Consensus (10): highest avg percentile, lowest disagreement
    high_pool = df[df['Average_Percentile'] > 0.667]
    for _ in range(10):
        available = high_pool[~high_pool['ID'].isin(selected_variants)]
        get_candidate(available, ['Average_Percentile', 'Disagreement_Score'], [False, True], 'Highly Efficient', 'Consensus')
        
    # Disagreement (5): highest disagreement in high percentile pool
    for _ in range(5):
        available = high_pool[~high_pool['ID'].isin(selected_variants)]
        get_candidate(available, ['Disagreement_Score', 'Average_Percentile'], [False, False], 'Highly Efficient', 'Disagreement')
        
    # 2. Modest (20 candidates)
    # Consensus (14): avg percentile closest to 0.5, lowest disagreement
    mod_pool = df[(df['Average_Percentile'] > 0.333) & (df['Average_Percentile'] <= 0.667)].copy()
    mod_pool['Dist_to_Median'] = (mod_pool['Average_Percentile'] - 0.5).abs()
    for _ in range(14):
        available = mod_pool[~mod_pool['ID'].isin(selected_variants)]
        get_candidate(available, ['Dist_to_Median', 'Disagreement_Score'], [True, True], 'Modest', 'Consensus')
        
    # Disagreement (6): highest disagreement in modest percentile pool
    for _ in range(6):
        available = mod_pool[~mod_pool['ID'].isin(selected_variants)]
        get_candidate(available, ['Disagreement_Score', 'Dist_to_Median'], [False, True], 'Modest', 'Disagreement')
        
    # 3. Inefficient (15 candidates)
    # Consensus (10): lowest avg percentile, lowest disagreement
    low_pool = df[df['Average_Percentile'] <= 0.333]
    for _ in range(10):
        available = low_pool[~low_pool['ID'].isin(selected_variants)]
        get_candidate(available, ['Average_Percentile', 'Disagreement_Score'], [True, True], 'Inefficient', 'Consensus')
        
    # Disagreement (5): highest disagreement in low percentile pool
    for _ in range(5):
        available = low_pool[~low_pool['ID'].isin(selected_variants)]
        get_candidate(available, ['Disagreement_Score', 'Average_Percentile'], [False, True], 'Inefficient', 'Disagreement')
        
    selected_df = pd.DataFrame(selected_rows)
    print(f"Successfully selected {len(selected_df)} validation candidates.")
    
    # Verification: Pairwise Pearson Correlation of percentiles in the selected 50 candidates
    print("\n=== Verification: Pairwise Pearson Correlation of Percentile Scores ===")
    corr_matrix = selected_df[percentile_cols].corr(method='pearson')
    print(corr_matrix)
    
    # Save the selected candidates
    os.makedirs(output_dir, exist_ok=True)
    selected_df.to_csv(os.path.join(output_dir, "selected_50_validation_candidates.csv"), index=False)
    print(f"\nSaved selected candidates to: {os.path.join(output_dir, 'selected_50_validation_candidates.csv')}")
    
    # Output the summary list
    cols_summary = [
        'ID', 'Category', 'Subcategory', 'PBSlen', 'RTlen', 
        'Percentile_DP_Base', 'Percentile_DP6', 'Percentile_PRIDICT2', 
        'Average_Percentile', 'Disagreement_Score'
    ]
    print("\n=== Summary Table of Selected 50 Candidates ===")
    print(selected_df[cols_summary].to_string(index=False))

if __name__ == "__main__":
    main()
