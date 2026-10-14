#!/usr/bin/env python3
"""
Orthogonal Benchmark: Step 1 - Stratified Target-Level Data Filtering & Leakage Elimination Module
Author: DeepPrime6 Project

Purpose:
1. Loads official benchmark test sets:
   - DeepPrime Table S2-1 Generalization Test Set (N = 28,883)
   - PRIDICT2.0 23k Fold 0 Test Set (N = 4,592)
2. Implements strict WideTargetSequence (Target Locus) Stratification:
   - Mirrors GroupShuffleSplit / Target-level stratification used during model training.
   - Purges any target locus seen in our library (pe6_full + small_library, forward and reverse complement).
   - Purges any target locus seen in baseline pretraining sets (DeepPrime-base Train, PRIDICT2.0 Train/Val).
   - For PRIDICT2, groups all pegRNAs by `grp_id` (genomic target locus) and excludes the entire `grp_id` if any pegRNA targets a trained locus.
3. Merges pegRNA design metadata required for cross-model inference.
4. Saves strictly unseen evaluation datasets and overlap summaries to:
   - `data/orthogonal_testset/`
   - `experiments/20261009_orthogonal_benchmark_leakage_stratification/results/`
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from Bio.Seq import reverse_complement

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

OUTPUT_DIR = "data/orthogonal_testset"
EXP_DIR = "experiments/20261009_orthogonal_benchmark_leakage_stratification/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)


def make_design_sig(wts, etype, elen, epos, pbslen, rtlen):
    """Generates a canonical signature string for an exact pegRNA design."""
    try:
        return f"{str(wts).upper()};{str(etype).lower()};{int(float(elen))};{int(float(epos))};{int(float(pbslen))};{int(float(rtlen))}"
    except Exception:
        return f"{str(wts).upper()};{str(etype).lower()};{elen};{epos};{pbslen};{rtlen}"


def main():
    print("==========================================================================", flush=True)
    print("=== Step 1: Stratified Target-Level Filtering & Leakage Elimination ===", flush=True)
    print("==========================================================================", flush=True)

    # 1. Collect all our PE6 targets (pe6_full + small_library)
    pe6_full_path = "data/Library design/pairwise_library_design/FULL_LIBRARY/results/pe6_full_library.csv"
    print(f"Loading our full library from {pe6_full_path}...", flush=True)
    pe6_full = pd.read_csv(pe6_full_path)
    
    small_lib_path = "data/small_library/pe6to7_with_avgs.parquet"
    pe6_small = pd.read_parquet(small_lib_path) if os.path.exists(small_lib_path) else pd.DataFrame()

    pe6_targets_74 = set()
    pe6_non74_targets = []
    
    for s in pe6_full["WideTargetSequence"].dropna().str.upper():
        if len(s) == 74:
            pe6_targets_74.add(s)
            pe6_targets_74.add(reverse_complement(s))
        else:
            pe6_non74_targets.append(s)
            pe6_non74_targets.append(reverse_complement(s))
            
    if not pe6_small.empty and "Target" in pe6_small.columns:
        for s in pe6_small["Target"].dropna().str.upper():
            if len(s) == 74:
                pe6_targets_74.add(s)
                pe6_targets_74.add(reverse_complement(s))

    pe6_full_designs = set([
        make_design_sig(r["WideTargetSequence"], r["Edit_type"], r["Edit length"], r["Edit position"], r["PBS_length"], r["RT_length"])
        for _, r in pe6_full.iterrows()
        if pd.notna(r["WideTargetSequence"]) and pd.notna(r["Edit length"])
    ])
    print(f"  PE6 full rows: {len(pe6_full):,} | unique 74nt targets (incl. RC): {len(pe6_targets_74):,} | non-74nt targets: {len(pe6_non74_targets):,}", flush=True)

    # 2. DeepPrime-Base Pretraining Set (Table S2-1 Train)
    dp_train_path = "data/Library design/pairwise_library_design/OnTarget/fetching_from_DPlt=3nt;6K/src/Table S2-1.csv"
    print(f"Loading DeepPrime-base pretraining data from {dp_train_path}...", flush=True)
    dp_all = pd.read_csv(dp_train_path, low_memory=False)
    col_wts_dp_all = [c for c in dp_all.columns if c.startswith("Wide target sequence")][0]
    dp_train_targets_74 = set()
    for s in dp_all[dp_all["Data set name"] == "Train"][col_wts_dp_all].str.upper():
        dp_train_targets_74.add(s)
        dp_train_targets_74.add(reverse_complement(s))
    print(f"  DeepPrime-base train targets (incl. RC): {len(dp_train_targets_74):,}", flush=True)

    # 3. PRIDICT2.0 Pretraining Set (all train/val folds)
    pridict_full_path = "PRIDICT2/dataset/proc_v2/data_23k_v1.csv"
    print(f"Loading PRIDICT2.0 dataset from {pridict_full_path}...", flush=True)
    pridict_df = pd.read_csv(pridict_full_path)
    dparts = get_stratified_partitions(pridict_df["grp_id"].values, num_folds=5, valid_set_portion=0.1, random_state=42)
    p_test_indices = dparts[0]["test"]
    p_train_indices = list(dparts[0]["train"]) + list(dparts[0]["validation"])
    
    pr_train_74 = set()
    for t in pridict_df.iloc[p_train_indices]["wide_initial_target"].str.upper():
        for i in range(len(t) - 74 + 1):
            k = t[i:i+74]
            pr_train_74.add(k)
            pr_train_74.add(reverse_complement(k))
    print(f"  PRIDICT2.0 train 74-mer targets (incl. RC): {len(pr_train_74):,}", flush=True)

    # ---------------------------------------------------------
    # 4. Filter DeepPrime Official Test Set
    # ---------------------------------------------------------
    print("\n--- Processing DeepPrime Official Generalization Test Set ---", flush=True)
    dp_test_path = "data/Library design/pairwise_library_design/OnTarget/fetching_from_DPlt=3nt;6K/dst/Generalization_Test/Table_S2_1_Test_Set.csv"
    dp_test = pd.read_csv(dp_test_path)
    col_wts_dp = [c for c in dp_test.columns if c.startswith("Wide target sequence")][0]
    col_ets_dp = [c for c in dp_test.columns if c.startswith("Edited target sequence")][0]

    dp_test["ID"] = [f"DP_TEST_{i}" for i in range(len(dp_test))]
    dp_test["WTS_clean"] = dp_test[col_wts_dp].str.upper()
    dp_test["ETS_clean"] = dp_test[col_ets_dp].str.upper()

    def get_dp_edit_type(r):
        if r["type_sub"]: return "sub"
        if r["type_ins"]: return "ins"
        if r["type_del"]: return "del"
        return "sub"
    dp_test["Edit_type_clean"] = dp_test.apply(get_dp_edit_type, axis=1)

    dp_test["design_sig"] = (
        dp_test["WTS_clean"] + ";" +
        dp_test["Edit_type_clean"] + ";" +
        dp_test["Edit_len"].astype(int).astype(str) + ";" +
        dp_test["Edit_pos"].astype(int).astype(str) + ";" +
        dp_test["PBSlen"].astype(int).astype(str) + ";" +
        dp_test["RTlen"].astype(int).astype(str)
    )

    # Mark stratified overlap flags
    dp_test["in_pe6_target"] = dp_test["WTS_clean"].isin(pe6_targets_74)
    dp_test["in_pe6_exact_pegrna"] = dp_test["design_sig"].isin(pe6_full_designs)
    dp_test["in_dp_train_target"] = dp_test["WTS_clean"].isin(dp_train_targets_74)
    dp_test["in_pridict2_train_target"] = dp_test["WTS_clean"].isin(pr_train_74)

    # Substring check for non-74nt pe6 targets (if any)
    pe6_short_targets = set([s for s in pe6_non74_targets if len(s) < 74])
    unmatched_wts = set(dp_test[~dp_test["in_pe6_target"]]["WTS_clean"].unique())
    wts_hit_map = {w: any(s in w for s in pe6_short_targets) for w in unmatched_wts}
    dp_test["in_pe6_non74"] = dp_test["WTS_clean"].map(lambda w: wts_hit_map.get(w, False))

    dp_leakage_mask = dp_test["in_pe6_target"] | dp_test["in_pe6_non74"] | dp_test["in_dp_train_target"] | dp_test["in_pridict2_train_target"]
    dp_test_filtered = dp_test[~dp_leakage_mask].copy().reset_index(drop=True)

    print(f"  Total DeepPrime Test rows: {len(dp_test):,}", flush=True)
    print(f"  pe6 target overlap (forward + RC + non-74): {(dp_test['in_pe6_target'] | dp_test['in_pe6_non74']).sum():,}", flush=True)
    print(f"  pe6 exact pegRNA design overlap: {dp_test['in_pe6_exact_pegrna'].sum():,}", flush=True)
    print(f"  DeepPrime-base train target overlap (RC): {dp_test['in_dp_train_target'].sum():,}", flush=True)
    print(f"  PRIDICT2.0 train target 74-mer overlap: {dp_test['in_pridict2_train_target'].sum():,}", flush=True)
    print(f"  => Strictly Unseen DeepPrime Test rows retained: {len(dp_test_filtered):,} (excluded {dp_leakage_mask.sum():,})", flush=True)

    # Merge Table S1 metadata to get Guide, PBS, RTT, and edit context
    print("  Merging Table S1-ClinVar metadata for pegRNA sequence reconstruction...", flush=True)
    s1_path = "data/Library design/pairwise_library_design/OnTarget/fetching_from_DPlt=3nt;6K/src/Table S1-ClinVar.csv"
    s1 = pd.read_csv(s1_path)
    s1["Edit type"] = s1["Edit type"].str.capitalize()
    col_wts_s1 = [c for c in s1.columns if c.startswith("Wide target sequence")][0]

    keys_dp = ["PBSlen", "RTlen", "Edit_len", "Edit_pos", "Edit_type_clean", "WTS_clean"]
    keys_s1 = ["PBS length", "RTT length", "Edit length", "Edit position", "Edit type", col_wts_s1]

    dp_test_filtered["merge_key"] = (
        dp_test_filtered["PBSlen"].astype(str) + ";" +
        dp_test_filtered["RTlen"].astype(str) + ";" +
        dp_test_filtered["Edit_len"].astype(str) + ";" +
        dp_test_filtered["Edit_pos"].astype(str) + ";" +
        dp_test_filtered["Edit_type_clean"].astype(str).str.lower() + ";" +
        dp_test_filtered["WTS_clean"].astype(str).str.upper()
    )
    s1["merge_key"] = (
        s1["PBS length"].astype(str) + ";" +
        s1["RTT length"].astype(str) + ";" +
        s1["Edit length"].astype(str) + ";" +
        s1["Edit position"].astype(str) + ";" +
        s1["Edit type"].astype(str).str.lower() + ";" +
        s1[col_wts_s1].astype(str).str.upper()
    )
    s1_dedup = s1.drop_duplicates(subset=["merge_key"])

    dp_test_filtered = dp_test_filtered.merge(
        s1_dedup[["merge_key", "3' extension sequence of pegRNA", "Guide sequence ((G/g)N19)",
                  "Unedited (gRNA binding -2 ~ RTT binding +2)", "Prime-edited (gRNA binding -2 ~ RTT binding +2)"]],
        on="merge_key",
        how="left"
    )
    print(f"  Matched pegRNA metadata for {dp_test_filtered['Guide sequence ((G/g)N19)'].notna().sum():,} / {len(dp_test_filtered):,} filtered rows.", flush=True)

    dp_filtered_out_path = os.path.join(OUTPUT_DIR, "deepprime_test_filtered.csv")
    dp_test_filtered.to_csv(dp_filtered_out_path, index=False)
    dp_test_filtered.to_csv(os.path.join(EXP_DIR, "deepprime_test_filtered.csv"), index=False)
    print(f"  Saved filtered DeepPrime test set to: {dp_filtered_out_path}", flush=True)

    # ---------------------------------------------------------
    # 5. Filter PRIDICT2.0 Official Test Set
    # ---------------------------------------------------------
    print("\n--- Processing PRIDICT2.0 Official Fold 0 Test Set ---", flush=True)
    p_test = pridict_df.iloc[p_test_indices].copy().reset_index(drop=True)
    p_test["ID"] = [f"PR_TEST_{i}" for i in range(len(p_test))]
    p_test["WTS_clean"] = p_test["wide_initial_target"].str.slice(5, 79).str.upper()

    # Check all sliding 74-mers against PE6 and DeepPrime train targets
    p_in_pe6 = []
    p_in_dp_train = []
    for _, r in p_test.iterrows():
        t = r["wide_initial_target"].upper()
        kmers = set()
        for i in range(len(t) - 74 + 1):
            k = t[i:i+74]
            kmers.add(k)
            kmers.add(reverse_complement(k))
        p_in_pe6.append(bool(kmers.intersection(pe6_targets_74)))
        p_in_dp_train.append(bool(kmers.intersection(dp_train_targets_74)))

    p_test["hit_pe6_target"] = p_in_pe6
    p_test["hit_dp_train_target"] = p_in_dp_train

    # Stratified target locus exclusion: If any pegRNA in a grp_id hits a trained locus, exclude the whole grp_id
    bad_grps = set(p_test[p_test["hit_pe6_target"] | p_test["hit_dp_train_target"]]["grp_id"])
    p_test["in_bad_grp_id"] = p_test["grp_id"].isin(bad_grps)

    p_test_filtered = p_test[~p_test["in_bad_grp_id"]].copy().reset_index(drop=True)

    print(f"  Total PRIDICT2 Test rows: {len(p_test):,}", flush=True)
    print(f"  Rows directly matching PE6 targets (sliding 74-mer forward + RC): {sum(p_in_pe6):,}", flush=True)
    print(f"  Rows directly matching DeepPrime train targets (sliding 74-mer forward + RC): {sum(p_in_dp_train):,}", flush=True)
    print(f"  Total rows excluded via Target Locus Grouping (grp_id level): {p_test['in_bad_grp_id'].sum():,}", flush=True)
    print(f"  => Strictly Unseen PRIDICT2 Test rows retained: {len(p_test_filtered):,}", flush=True)

    p_filtered_out_path = os.path.join(OUTPUT_DIR, "pridict2_test_filtered.csv")
    p_test_filtered.to_csv(p_filtered_out_path, index=False)
    p_test_filtered.to_csv(os.path.join(EXP_DIR, "pridict2_test_filtered.csv"), index=False)
    print(f"  Saved filtered PRIDICT2 test set to: {p_filtered_out_path}", flush=True)

    # ---------------------------------------------------------
    # 6. Overlap and Exclusion Summary Table
    # ---------------------------------------------------------
    overlap_summary = pd.DataFrame([
        {
            "Test Set Name": "DeepPrime Official Generalization Test Set",
            "Total Original Rows": len(dp_test),
            "pe6 Target Locus Overlap (Forward + RC)": int((dp_test["in_pe6_target"] | dp_test["in_pe6_non74"]).sum()),
            "pe6 Exact pegRNA Overlap": int(dp_test["in_pe6_exact_pegrna"].sum()),
            "PRIDICT2.0 Train Target Overlap": int(dp_test["in_pridict2_train_target"].sum()),
            "DeepPrime-base Train Target Overlap": int(dp_test["in_dp_train_target"].sum()),
            "Total Excluded (Target Stratification)": int(dp_leakage_mask.sum()),
            "Final Strictly Unseen Test Set (N)": len(dp_test_filtered)
        },
        {
            "Test Set Name": "PRIDICT2.0 Official Fold 0 Test Set",
            "Total Original Rows": len(p_test),
            "pe6 Target Locus Overlap (Forward + RC)": int(sum(p_in_pe6)),
            "pe6 Exact pegRNA Overlap": 0,
            "PRIDICT2.0 Train Target Overlap": 0,
            "DeepPrime-base Train Target Overlap": int(sum(p_in_dp_train)),
            "Total Excluded (Target Stratification)": int(p_test["in_bad_grp_id"].sum()),
            "Final Strictly Unseen Test Set (N)": len(p_test_filtered)
        }
    ])

    summary_out_path = os.path.join(OUTPUT_DIR, "overlap_exclusion_summary.csv")
    overlap_summary.to_csv(summary_out_path, index=False)
    overlap_summary.to_csv(os.path.join(EXP_DIR, "overlap_exclusion_summary.csv"), index=False)
    print(f"\nSaved overlap exclusion summary to: {summary_out_path}", flush=True)
    print("\n" + overlap_summary.to_string(index=False))

    print("\n[Complete] Stratified filtering stage completed successfully!", flush=True)


if __name__ == "__main__":
    main()
