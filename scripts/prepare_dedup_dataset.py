#!/usr/bin/env python3
"""
Prepares the deduplicated PE6 dataset:
- Groups by pegRNA identity: Guide + RTT + PBS + linker + motif + WideTargetSequence + Edit_type + Edit length + Edit position
- Efficiency labels: Mean of valid barcode replicates (since background normalization / ratio is applied)
- Read counts: Sum (add) of barcode replicates (since reads reflect cumulative physical sequencing depth)
- Other categorical/sequence annotations: First
Saves to data/processed_data_dedup.parquet and updates rq3_test_library_dedup.parquet.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SRC_PARQUET = "/home/work/workdir/pe6-deep/data/pe6/processed_data.parquet"
OUTPUT_PARQUET = "data/processed_data_dedup.parquet"
RQ3_TEST_OUTPUT = "data/rq3_test_library_dedup.parquet"

def main():
    print(f"Loading raw processed data from {SRC_PARQUET}...")
    df = pd.read_parquet(SRC_PARQUET)
    print(f"Loaded raw data: {df.shape}")

    peg_key = [
        "Guide", "RTT", "PBS", "linker", "motif",
        "WideTargetSequence", "Edit_type", "Edit length", "Edit position"
    ]
    avail_key = [c for c in peg_key if c in df.columns]
    print(f"Grouping keys: {avail_key}")

    eff_cols = [c for c in df.columns if "comb_vis+" in c or "Normalized+" in c or "pe_ratio_%" in c]
    rc_cols = [c for c in df.columns if "read_counts" in c or "read_count" in c.lower()]

    print(f"Number of efficiency columns: {len(eff_cols)}")
    print(f"Number of read count columns: {len(rc_cols)}")

    agg_dict = {}
    for col in df.columns:
        if col in avail_key:
            continue
        elif col in eff_cols:
            agg_dict[col] = "mean"
        elif col in rc_cols:
            agg_dict[col] = "sum"
        else:
            agg_dict[col] = "first"

    print("Aggregating duplicates across barcode copies...")
    df_dedup = df.groupby(avail_key, as_index=False).agg(agg_dict)
    print(f"Deduplicated dataset shape: {df_dedup.shape} (from {df.shape[0]} rows)")

    # Save deduplicated parquet
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    df_dedup.to_parquet(OUTPUT_PARQUET)
    print(f"Saved deduplicated dataset to {OUTPUT_PARQUET}")

    # Generate split to verify zero leakage and export test set
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(splitter_test.split(df_dedup, groups=df_dedup["Guide"]))
    test_df = df_dedup.iloc[test_idx].copy()
    test_df.to_parquet(RQ3_TEST_OUTPUT)
    print(f"Saved strictly unseen test library to {RQ3_TEST_OUTPUT}: {test_df.shape}")

if __name__ == "__main__":
    main()
