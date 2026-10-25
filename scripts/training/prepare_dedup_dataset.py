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

def find_project_root():
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, ".project-root")) or (
            os.path.exists(os.path.join(curr, "src")) and os.path.exists(os.path.join(curr, "configs"))
        ):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.getcwd())

_repo_root = find_project_root()
os.chdir(_repo_root)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_local_src = os.path.join(_repo_root, "data", "processed_data.parquet")
DEFAULT_SRC = os.environ.get("PE6_RAW_PARQUET", _local_src)

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
        "WideTargetSequence", "Edit_type",
        "Edit length", "Edit position",
        "Edit_len", "Edit_pos",
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
    # - PEmax ~ PE6d: 2 B1 replicates (HEK-M-*-B1 + HEK-M2-*-B1) / 2.0 
    # - PE6e ~ PE6g: 2 replicates (HEK-M-*-B2 + HEK-M-*-B1) / 2.0 (M2 missing; B2 correlation r > 0.85)
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

    # Generate canonical train/val/test splits matching PE6DataModule
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(
        splitter_test.split(df_dedup, y=df_dedup.get("Target"), groups=df_dedup["Guide"])
    )
    df_train_val = df_dedup.iloc[train_val_idx]
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_rel_idx, val_rel_idx = next(
        splitter_val.split(df_train_val, y=df_train_val.get("Target"), groups=df_train_val["Guide"])
    )
    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

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

    # Export full deduplicated CSV with split labels and per-editor QC-filtered CSVs
    from src.utils.dataprep import get_pe6_qc_mask

    df_with_splits = df_dedup.copy()
    df_with_splits["split"] = ""
    df_with_splits.iloc[train_idx, df_with_splits.columns.get_loc("split")] = "train"
    df_with_splits.iloc[val_idx, df_with_splits.columns.get_loc("split")] = "val"
    df_with_splits.iloc[test_idx, df_with_splits.columns.get_loc("split")] = "test"

    splits_csv_path = "data/processed_data_dedup_with_splits.csv"
    df_with_splits.to_csv(splits_csv_path, index=False)
    print(f"Saved {splits_csv_path}: {df_with_splits.shape}")

    pe_export_map = [
        ("PE2max_dRNaseH", "PEmaxdRNaseH"),
        ("PE6a", "PE6a(+PEmaxCas9)"),
        ("PE6b", "PE6b(+PEmaxCas9)"),
        ("PE6c", "PE6c(+PEmaxCas9)"),
        ("PE2max", "PEmax"),
    ]

    qc_dir = "data/qc_filtered"
    os.makedirs(qc_dir, exist_ok=True)
    df_with_qc_and_splits = df_with_splits.copy()

    for short_name, pe_col in pe_export_map:
        qc_col = f"qc_pass_{short_name}"
        mask = get_pe6_qc_mask(df_with_splits, pe_col, read_count_filter=200, max_ctrl_bg=5.0)
        df_with_qc_and_splits[qc_col] = mask

        # Export per-editor QC-filtered CSV from clean base df_with_splits (no cross-editor qc_pass columns)
        df_pe_filtered = df_with_splits.loc[mask].copy()
        df_pe_filtered[qc_col] = True
        out_pe_csv = os.path.join(qc_dir, f"model_input_qc_pass_{short_name}.csv")
        df_pe_filtered.to_csv(out_pe_csv, index=False)
        vc = df_pe_filtered["split"].value_counts().to_dict()
        print(
            f"Saved {out_pe_csv}: n={len(df_pe_filtered):,} "
            f"(train={vc.get('train', 0):,}, val={vc.get('val', 0):,}, test={vc.get('test', 0):,})"
        )

    qc_splits_csv_path = "data/processed_data_dedup_with_qc_and_splits.csv"
    df_with_qc_and_splits.to_csv(qc_splits_csv_path, index=False)
    print(f"Saved {qc_splits_csv_path}: {df_with_qc_and_splits.shape}")

if __name__ == "__main__":
    main()
