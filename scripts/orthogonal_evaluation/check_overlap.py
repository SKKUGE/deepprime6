#!/usr/bin/env python3
"""
Orthogonal Benchmark: Overlap & Leakage Audit Tool
Author: DeepPrime6 Project

Purpose:
Audits target-level (WideTargetSequence) and pegRNA design-level overlap between:
1. Our full library (`pe6_full_library.csv`) & actual training set
2. DeepPrime Generalization Test Set (Table S2-1) & DeepPrime Pretraining Set
3. PRIDICT2.0 23k dataset (Train/Val vs. Fold 0 Test)
"""

import os
import sys
import pandas as pd
import numpy as np

def find_project_root():
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, ".project-root")) or (
            os.path.exists(os.path.join(curr, "PRIDICT2")) and os.path.exists(os.path.join(curr, "src"))
        ):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.getcwd())

PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
pridict_path = os.path.join(PROJECT_ROOT, "PRIDICT2")
if pridict_path not in sys.path:
    sys.path.insert(0, pridict_path)

from pridict.pridictv2.dataset import get_stratified_partitions


def main():
    print("Loading datasets for overlap analysis...", flush=True)
    
    # 1. Our full library & our training set
    pe6_full = pd.read_csv("data/Library design/pairwise_library_design/FULL_LIBRARY/results/pe6_full_library.csv")
    rq3_test = pd.read_parquet("data/rq3_test_library.parquet")
    test_ref_ids = set(rq3_test["REF_ID"])
    pe6_train = pe6_full[~pe6_full["REF_ID"].isin(test_ref_ids)].copy()
    
    print(f"Our Full Library (pe6_full_library.csv): {len(pe6_full):,} rows", flush=True)
    print(f"Our Internal Test Set (rq3_test_library): {len(rq3_test):,} rows", flush=True)
    print(f"Our Actual Training Set (pe6_train): {len(pe6_train):,} rows", flush=True)

    # Sets of WideTargetSequence
    pe6_full_wts = set(pe6_full["WideTargetSequence"].dropna().str.upper())
    pe6_train_wts = set(pe6_train["WideTargetSequence"].dropna().str.upper())
    
    # Sets of (Guide, WideTargetSequence)
    pe6_full_guide_wts = set(zip(pe6_full["Guide"].dropna().str.upper(), pe6_full["WideTargetSequence"].dropna().str.upper()))
    pe6_train_guide_wts = set(zip(pe6_train["Guide"].dropna().str.upper(), pe6_train["WideTargetSequence"].dropna().str.upper()))

    print("\n=== 1. Checking DeepPrime Official Test Set ===", flush=True)
    dp_test_path = "data/Library design/pairwise_library_design/OnTarget/fetching_from_DPlt=3nt;6K/dst/Generalization_Test/Table_S2_1_Test_Set.csv"
    dp_test = pd.read_csv(dp_test_path)
    col_wts_dp = [c for c in dp_test.columns if c.startswith("Wide target sequence")][0]
    dp_wts_series = dp_test[col_wts_dp].str.upper()
    dp_guide_series = dp_wts_series.str[4:24]
    
    # Overlap by WideTargetSequence
    dp_in_full_wts = dp_wts_series.isin(pe6_full_wts)
    dp_in_train_wts = dp_wts_series.isin(pe6_train_wts)
    
    print(f"DeepPrime official test set total: {len(dp_test):,}", flush=True)
    print(f"Overlap with our Full Library by WideTargetSequence: {dp_in_full_wts.sum():,} rows", flush=True)
    print(f"Overlap with our Actual Training Set by WideTargetSequence: {dp_in_train_wts.sum():,} rows", flush=True)

    dp_guide_wts_series = list(zip(dp_guide_series, dp_wts_series))
    dp_in_full_guide_wts = pd.Series([x in pe6_full_guide_wts for x in dp_guide_wts_series])
    dp_in_train_guide_wts = pd.Series([x in pe6_train_guide_wts for x in dp_guide_wts_series])
    print(f"Overlap with our Full Library by (Guide + Target): {dp_in_full_guide_wts.sum():,} rows", flush=True)
    print(f"Overlap with our Actual Training Set by (Guide + Target): {dp_in_train_guide_wts.sum():,} rows", flush=True)

    print("\n=== 2. Checking PRIDICT2.0 Official Test Set ===", flush=True)
    pridict_df = pd.read_csv("PRIDICT2/dataset/proc_v2/data_23k_v1.csv")
    dparts = get_stratified_partitions(pridict_df["grp_id"].values, num_folds=5, valid_set_portion=0.1, random_state=42)
    p_test_idx = dparts[0]["test"]
    p_train_idx = list(dparts[0]["train"]) + list(dparts[0]["validation"])
    p_test = pridict_df.iloc[p_test_idx].copy()
    
    print(f"PRIDICT2.0 Fold 0 Test Set total: {len(p_test):,}", flush=True)
    
    p_test_wts = p_test["wide_initial_target"].str.slice(5, 79).str.upper()
    p_test_guide = p_test["wide_initial_target"].str.slice(9, 29).str.upper()

    p_in_full_wts = p_test_wts.isin(pe6_full_wts)
    p_in_train_wts = p_test_wts.isin(pe6_train_wts)
    print(f"Overlap with our Full Library by WideTargetSequence: {p_in_full_wts.sum():,} rows", flush=True)
    print(f"Overlap with our Actual Training Set by WideTargetSequence: {p_in_train_wts.sum():,} rows", flush=True)

    p_guide_wts = list(zip(p_test_guide, p_test_wts))
    p_in_full_gw = pd.Series([x in pe6_full_guide_wts for x in p_guide_wts])
    p_in_train_gw = pd.Series([x in pe6_train_guide_wts for x in p_guide_wts])
    print(f"Overlap with our Full Library by (Guide + Target): {p_in_full_gw.sum():,} rows", flush=True)
    print(f"Overlap with our Actual Training Set by (Guide + Target): {p_in_train_gw.sum():,} rows", flush=True)

    # Check DeepPrime pretraining overlap with PRIDICT2 test set
    dp_train_wts_path = "data/Library design/pairwise_library_design/OnTarget/fetching_from_DPlt=3nt;6K/src/Table S2-1.csv"
    print("\n=== Checking DeepPrime-Base pretraining overlap with PRIDICT2 test set ===", flush=True)
    dp_all = pd.read_csv(dp_train_wts_path, usecols=["Data set name", col_wts_dp])
    dp_train_wts = set(dp_all[dp_all["Data set name"] == "Train"][col_wts_dp].str.upper())
    p_in_dp_train = p_test_wts.isin(dp_train_wts)
    print(f"PRIDICT2 test set rows in DeepPrime-Base training set: {p_in_dp_train.sum():,} rows", flush=True)

    # Check PRIDICT2 training set overlap with DeepPrime test set
    print("\n=== Checking PRIDICT2 training set overlap with DeepPrime test set ===", flush=True)
    p_train_wts = set(pridict_df.iloc[p_train_idx]["wide_initial_target"].str.slice(5, 79).str.upper())
    dp_in_p_train = dp_wts_series.isin(p_train_wts)
    print(f"DeepPrime test set rows in PRIDICT2.0 training set: {dp_in_p_train.sum():,} rows", flush=True)


if __name__ == "__main__":
    main()
