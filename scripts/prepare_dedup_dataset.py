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
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

_local_src = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data", "processed_data.parquet")
_pe6_deep_src = "/home/work/workdir/pe6-deep/data/pe6/processed_data.parquet"
DEFAULT_SRC = os.environ.get("PE6_RAW_PARQUET", _local_src if os.path.exists(_local_src) else _pe6_deep_src)

def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate PE6 dataset by unique pegRNA design.")
    parser.add_argument("--src_parquet", type=str, default=DEFAULT_SRC, help="Path to raw processed parquet file.")
    parser.add_argument("--output_parquet", type=str, default="data/processed_data_dedup.parquet", help="Path to save deduplicated dataset.")
    parser.add_argument("--test_output", type=str, default="data/rq3_test_library_dedup.parquet", help="Path to export clean test split.")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading raw processed data from {args.src_parquet}...")
    df = pd.read_parquet(args.src_parquet)
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

    # Update target columns strictly according to EFF_PAIR (derived from AVG_PAIR in src/utils/dataprep.py):
    # - PEmax ~ PE6d: 2 UAR replicates (HEK-M-*-UAR + HEK-M2-*-UAR) / 2.0 (KMC excluded due to noise)
    # - PE6e ~ PE6g: 2 replicates (HEK-M-*-KMC + HEK-M-*-UAR) / 2.0 (M2 missing; KMC correlation r > 0.85)
    print("Updating training target columns strictly using EFF_PAIR from src/utils/dataprep.py...")
    from src.utils.dataprep import EFF_PAIR
    for col_name, eff_cols in EFF_PAIR.items():
        valid_cols = [c for c in eff_cols if c in df_dedup.columns]
        if len(valid_cols) > 0:
            df_dedup[col_name] = np.maximum(0.0, df_dedup[valid_cols].mean(axis=1))
            print(f"  {col_name}: updated from {valid_cols} (mean={df_dedup[col_name].mean():.3f})")

    # Save deduplicated parquet
    os.makedirs(os.path.dirname(args.output_parquet) or ".", exist_ok=True)
    df_dedup.to_parquet(args.output_parquet)
    print(f"Saved deduplicated dataset to {args.output_parquet}")

    # Generate split to verify zero leakage and export test set
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(splitter_test.split(df_dedup, groups=df_dedup["Guide"]))
    test_df = df_dedup.iloc[test_idx].copy()
    os.makedirs(os.path.dirname(args.test_output), exist_ok=True)
    test_df.to_parquet(args.test_output)
    csv_test_output = args.test_output.replace(".parquet", ".csv")
    test_df.to_csv(csv_test_output, index=False)
    print(f"Saved strictly unseen test library to {args.test_output} and {csv_test_output}: {test_df.shape}")

    # Synchronize to data/rq3_test_library.* as well
    test_df.to_parquet("data/rq3_test_library.parquet")
    test_df.to_csv("data/rq3_test_library.csv", index=False)
    print("Synchronized data/rq3_test_library.parquet and data/rq3_test_library.csv")

if __name__ == "__main__":
    main()
