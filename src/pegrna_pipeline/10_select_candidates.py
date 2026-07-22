import pandas as pd
import argparse
import os
import random
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 10: Select final candidates")
    parser.add_argument("--input", required=True, help="Input pegrna_designs_scored_normalized.parquet")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    return parser.parse_args()

def classify_stratum(row):
    pct = row['Representative_Percentile']
    dis = row['Disagreement_Score']
    
    # Efficacy Category
    if pct > 0.667:
        cat = 'Highly_Efficient'
    elif pct > 0.333:
        cat = 'Modest'
    else:
        cat = 'Inefficient'
        
    # Disagreement Subcategory
    if dis > 0.20:
        subcat = 'Disagreement'
    else:
        subcat = 'Consensus'
        
    return cat, subcat

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    quota_he = config.get('scoring', {}).get('selection_quota', {}).get('highly_efficient', 15)
    quota_mo = config.get('scoring', {}).get('selection_quota', {}).get('modest', 20)
    quota_ie = config.get('scoring', {}).get('selection_quota', {}).get('inefficient', 15)
    
    df = pd.read_parquet(args.input)
    if len(df) == 0:
        print("Empty input. Saving empty output.")
        df.to_parquet(os.path.join(args.output_dir, "final_candidates.parquet"), index=False)
        return
        
    print("Classifying pegRNAs into strata...")
    # Calculate strata
    strata = df.apply(classify_stratum, axis=1)
    df['Efficacy_Category'] = [s[0] for s in strata]
    df['Disagreement_Subcategory'] = [s[1] for s in strata]
    
    selected_indices = set()
    
    # 1. Primary Selection: Best pegRNA for EVERY mutation
    print("Selecting 1 primary pegRNA per mutation (Max PRIDICT2 score, Min Disagreement)...")
    df_sorted = df.sort_values(['Representative_Percentile', 'Disagreement_Score'], ascending=[False, True])
    primary_selected = df_sorted.drop_duplicates(subset=['mutation_key'], keep='first')
    
    # 2. Secondary Selection: Sample other strata to fulfill diversity/uniqueness requirements
    primary_spacers = primary_selected.set_index('mutation_key')['spacer'].to_dict()
    available = df.drop(index=primary_selected.index).copy()
    
    def is_unique_spacer(row):
        mut = row['mutation_key']
        if mut in primary_spacers:
            return row['spacer'] != primary_spacers[mut]
        return True
        
    available['is_unique'] = available.apply(is_unique_spacer, axis=1)
    available = available[available['is_unique'] == True]
    
    def secondary_priority(row):
        score = 0
        if row['Disagreement_Subcategory'] == 'Disagreement': score += 100
        if row['Efficacy_Category'] == 'Modest': score += 50
        elif row['Efficacy_Category'] == 'Inefficient': score += 25
        return score
        
    available['sec_priority'] = available.apply(secondary_priority, axis=1)
    available = available.sort_values(['sec_priority', 'Disagreement_Score', 'Representative_Percentile'], ascending=[False, False, False])
    secondary_selected = available.drop_duplicates(subset=['mutation_key'], keep='first')
    
    # Combine primary and secondary as our candidate pool
    pool = pd.concat([primary_selected, secondary_selected])
    
    # Remove duplicate mutations (keep first, which is the primary/higher score one)
    pool = pool.drop_duplicates(subset=['mutation_key'], keep='first')
    
    # Subsample to exact quotas
    final_dfs = []
    
    he_pool = pool[pool['Efficacy_Category'] == 'Highly_Efficient']
    mo_pool = pool[pool['Efficacy_Category'] == 'Modest']
    ie_pool = pool[pool['Efficacy_Category'] == 'Inefficient']
    
    print(f"Sampling exact quota: HE={quota_he}, MO={quota_mo}, IE={quota_ie}")
    
    if len(he_pool) >= quota_he:
        final_dfs.append(he_pool.sample(n=quota_he, random_state=42))
    else:
        final_dfs.append(he_pool)
        
    if len(mo_pool) >= quota_mo:
        final_dfs.append(mo_pool.sample(n=quota_mo, random_state=42))
    else:
        final_dfs.append(mo_pool)
        
    if len(ie_pool) >= quota_ie:
        final_dfs.append(ie_pool.sample(n=quota_ie, random_state=42))
    else:
        final_dfs.append(ie_pool)
        
    final_df = pd.concat(final_dfs)
    
    print(f"Selected {len(final_df)} final candidates.")
    print("Distribution of selected pegRNAs:")
    print(pd.crosstab(final_df['Efficacy_Category'], final_df['Disagreement_Subcategory']))
    
    out_path = os.path.join(args.output_dir, "final_candidates.parquet")
    final_df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    main()
