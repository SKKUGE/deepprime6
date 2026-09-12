import pandas as pd
import argparse
import os
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: Select Mutation Panel")
    parser.add_argument("--input", required=True, help="Input variants_designability.parquet")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_parquet(args.input)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    target_count = config['target_panel'].get('mutation_count', 1000)
    seed = config['target_panel'].get('random_seed', 42)
    
    # Filter to only designable mutations
    df_valid = df[df['designable'] == True].copy()
    
    # Ensure unique mutation_key
    df_valid = df_valid.drop_duplicates(subset=['mutation_key'])
    
    # Separate Indels and SNV/MNVs
    df_indel = df_valid[df_valid['edit_type'].isin(['Ins', 'Del'])].copy()
    df_sub = df_valid[df_valid['edit_type'].isin(['Sub', 'MNV'])].copy()
    
    selected_indels = df_indel
    if len(df_indel) > target_count:
        # If indels exceed the quota, we must stratify and reduce them
        # (Simplified stratification by edit_type and length for now)
        selected_indels = df_indel.groupby(['edit_type', 'edit_length']).sample(
            frac=target_count/len(df_indel), random_state=seed, replace=False
        )
        # Ensure we hit exactly target_count if frac rounding missed it
        if len(selected_indels) > target_count:
            selected_indels = selected_indels.sample(n=target_count, random_state=seed)
        elif len(selected_indels) < target_count:
            shortfall = target_count - len(selected_indels)
            remaining = df_indel[~df_indel['mutation_key'].isin(selected_indels['mutation_key'])]
            added = remaining.sample(n=min(shortfall, len(remaining)), random_state=seed)
            selected_indels = pd.concat([selected_indels, added])
            
        # Save original full indel list as requested
        df_indel.to_parquet(os.path.join(args.output_dir, "mutation_panel_all_indels_reserve.parquet"), index=False)
    
    remaining_quota = target_count - len(selected_indels)
    
    selected_subs = pd.DataFrame()
    if remaining_quota > 0 and not df_sub.empty:
        # Stratify SNV/MNV to avoid bias
        # For a robust stratification, we might group by strand and chromosome
        strat_cols = ['strand', 'chromosome']
        # If any stratification groups are too small, sample gracefully
        try:
            selected_subs = df_sub.groupby(strat_cols).sample(
                frac=min(1.0, remaining_quota/len(df_sub)), random_state=seed
            )
            # Adjust exactly to remaining_quota
            if len(selected_subs) > remaining_quota:
                selected_subs = selected_subs.sample(n=remaining_quota, random_state=seed)
            elif len(selected_subs) < remaining_quota:
                shortfall = remaining_quota - len(selected_subs)
                remaining = df_sub[~df_sub['mutation_key'].isin(selected_subs['mutation_key'])]
                added = remaining.sample(n=min(shortfall, len(remaining)), random_state=seed)
                selected_subs = pd.concat([selected_subs, added])
        except ValueError:
            # Fallback if groupby fails due to missing groups
            selected_subs = df_sub.sample(n=min(remaining_quota, len(df_sub)), random_state=seed)

    final_panel = pd.concat([selected_indels, selected_subs]).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"Selected {len(final_panel)} mutations (Target: {target_count})")
    print(f"  Indels: {len(selected_indels)}")
    print(f"  SNV/MNV: {len(selected_subs)}")
    
    final_panel.to_parquet(os.path.join(args.output_dir, "mutation_panel_primary.parquet"), index=False)
    
    # Generate Selection Report
    report_path = os.path.join(args.output_dir, "mutation_panel_selection_report.md")
    with open(report_path, "w") as f:
        f.write("# Mutation Panel Selection Report\n\n")
        f.write(f"- Total designable available: {len(df_valid)}\n")
        f.write(f"- Target Quota: {target_count}\n")
        f.write(f"- Selected Indels: {len(selected_indels)}\n")
        f.write(f"- Selected SNV/MNV: {len(selected_subs)}\n")
        f.write(f"- Final Panel Size: {len(final_panel)}\n\n")
        
        f.write("## Edit Type Distribution\n")
        f.write(final_panel['edit_type'].value_counts().to_markdown() + "\n\n")
        
        f.write("## Strand Distribution\n")
        f.write(final_panel['strand'].value_counts().to_markdown() + "\n\n")

if __name__ == "__main__":
    main()
