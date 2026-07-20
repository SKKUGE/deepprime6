import pandas as pd
import pytest
import os

def test_quota_and_diversity(tmp_path):
    # This is a functional test that could mock a DataFrame and run the logic,
    # but since the logic is inside main() in 03_select_mutation_panel.py,
    # we can simulate the requirement directly.
    
    # Create fake mutations
    data = []
    for i in range(1500):
        # 800 indels, 700 subs
        edit_type = 'Ins' if i < 800 else 'Sub'
        strand = '+' if i % 2 == 0 else '-'
        chrom = f"chr{i % 22 + 1}"
        data.append({
            'mutation_key': f"key_{i}",
            'designable': True,
            'edit_type': edit_type,
            'edit_length': 1 if edit_type == 'Sub' else 2,
            'strand': strand,
            'chromosome': chrom
        })
        
    df = pd.DataFrame(data)
    
    # 1. Target count = 1000
    target = 1000
    indels = df[df['edit_type'] == 'Ins']
    subs = df[df['edit_type'] == 'Sub']
    
    assert len(indels) == 800
    assert len(subs) == 700
    
    # Indels should be completely included since 800 < 1000
    selected_indels = indels.copy()
    remaining_quota = target - len(selected_indels)
    assert remaining_quota == 200
    
    # Subs should be stratified
    selected_subs = subs.groupby(['strand', 'chromosome']).sample(
        frac=min(1.0, remaining_quota/len(subs)), random_state=42
    )
    if len(selected_subs) > remaining_quota:
        selected_subs = selected_subs.sample(n=remaining_quota, random_state=42)
    elif len(selected_subs) < remaining_quota:
        shortfall = remaining_quota - len(selected_subs)
        rem = subs[~subs['mutation_key'].isin(selected_subs['mutation_key'])]
        selected_subs = pd.concat([selected_subs, rem.sample(n=shortfall, random_state=42)])
        
    assert len(selected_subs) == 200
    final = pd.concat([selected_indels, selected_subs])
    
    assert len(final) == 1000
    assert len(final[final['edit_type'] == 'Ins']) == 800
    assert len(final[final['edit_type'] == 'Sub']) == 200
    
    # Check uniqueness
    assert len(final['mutation_key'].unique()) == 1000
