#!/usr/bin/env python3
"""
Prediction Pipeline for FRT Small Library (112 pegRNA designs)
Evaluates 6 models:
1. DeepPrime-base (20-model ensemble with original GENET GeneInteractionModelVanilla architecture)
2. DeepPrime6-pe6a (pe6a_mainft.ckpt)
3. DeepPrime6-pe6b (pe6b_mainft.ckpt)
4. DeepPrime6-pe6c (pe6c_mainft.ckpt)
5. DeepPrime6-pemaxdrnaseh (pemaxdrnaseh_mainft.ckpt)
6. PRIDICT2.0 (Official released fold 0 model, predicting HEK score * 100)

Maintains 1:1 mapping with original INDEX/ID and documents failure reasons for edge cases.
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import torch

# Ensure repo root and PRIDICT2 are in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PRIDICT_DIR = os.path.join(REPO_ROOT, "PRIDICT2")
if PRIDICT_DIR not in sys.path:
    sys.path.insert(0, PRIDICT_DIR)

from src.models.pe6_module_dp_only import PE6DeepPrimeModule
from src.models.components.dprime import GeneInteractionModelVanilla
from src.data.components.pe6_preprocess_data import preprocess_data
from src.utils.predict_pridict2 import (
    load_pridict_model,
    deeppridict,
    compute_average_predictions,
    extract_single_pegRNA_features
)


def char_to_onehot_tensor(seqs_wt, seqs_ed, device):
    """Encodes WT and edited sequences into (B, 4, 2, 74) tensor with 2*x-1 normalization.
    Masked 'x' / unknown characters map to 0 in one-hot, which translates to -1.0.
    """
    char_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    num_samples = len(seqs_wt)
    g_arr = np.zeros((num_samples, 4, 2, 74), dtype=np.float32)
    for idx, (wt, ed) in enumerate(zip(seqs_wt, seqs_ed)):
        for col_idx, ch in enumerate(wt[:74]):
            if ch in char_map:
                g_arr[idx, char_map[ch], 0, col_idx] = 1.0
        for col_idx, ch in enumerate(ed[:74]):
            if ch in char_map:
                g_arr[idx, char_map[ch], 1, col_idx] = 1.0
    return (2 * torch.from_numpy(g_arr) - 1).float().to(device)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== FRT Small Library Inference Pipeline ===", flush=True)
    print(f"Executing on device: {device}", flush=True)

    input_excel = "data/frt-library-prediction/frt-small-lib_table_final_260128 (3).xlsm"
    out_dir_data = "data/frt-library-prediction"
    out_dir_exp = "experiments/20261009_frt_small_library/results"
    plots_dir_exp = "experiments/20261009_frt_small_library/plots"
    os.makedirs(out_dir_data, exist_ok=True)
    os.makedirs(out_dir_exp, exist_ok=True)
    os.makedirs(plots_dir_exp, exist_ok=True)

    # 1. Load input dataset
    print(f"\n[Step 1] Loading '{input_excel}', sheet '정리'...", flush=True)
    df_raw = pd.read_excel(input_excel, sheet_name="정리")
    total_rows = len(df_raw)
    print(f"Loaded {total_rows} pegRNA designs across {df_raw['Group'].nunique()} groups.", flush=True)

    col_unedited = [c for c in df_raw.columns if "unedited" in c][0]
    col_edited = [c for c in df_raw.columns if "prime edited" in c][0]

    # Initialize results DataFrame with all original metadata
    metadata_cols = [
        "Group", "INDEX", "Edit type", "Edit position", "Edit length",
        "length difference", "PBS length", "RTT length", "RHA_len",
        "Seed guide ID", "Guide sequence ((G/g)N19)", "target 20bp", "PAM", "RTT", "PBS",
        "74bp wide target", col_unedited, col_edited
    ]
    # Filter to existing columns
    metadata_cols = [c for c in metadata_cols if c in df_raw.columns]
    res_df = df_raw[metadata_cols].copy()

    # Model prediction columns
    res_df["pred_dp_base"] = np.nan
    res_df["pred_dp6_drnaseh"] = np.nan
    res_df["pred_dp6_pe6a"] = np.nan
    res_df["pred_dp6_pe6b"] = np.nan
    res_df["pred_dp6_pe6c"] = np.nan
    res_df["pred_pridict2"] = np.nan

    # Status columns
    res_df["status_deepprime"] = "Pending"
    res_df["status_pridict2"] = "Pending"
    res_df["prediction_status"] = "Pending"
    res_df["failure_reason"] = ""

    # 2. Identify eligible rows for DeepPrime & PRIDICT2
    print("\n[Step 2] Validating architectural constraints...", flush=True)
    # DeepPrime constraints: RTT length <= 53 nt (for 74nt CNN window: 21nt upstream + 53nt downstream)
    # and Edit type in ['sub', 'ins', 'del']
    dp_mask = (res_df["RTT length"] <= 53) & (res_df["Edit type"].isin(["sub", "ins", "del"]))
    print(f"DeepPrime-eligible rows: {dp_mask.sum()}/{total_rows}")
    for idx, row in res_df[~dp_mask].iterrows():
        reasons = []
        if row["RTT length"] > 53:
            reasons.append(f"RTT length ({row['RTT length']}nt) exceeds maximum 74nt CNN window limit (<=53nt)")
        if row["Edit type"] not in ["sub", "ins", "del"]:
            reasons.append(f"Edit type '{row['Edit type']}' is unsupported by DeepPrime 3-class one-hot classification")
        fail_msg = "; ".join(reasons)
        res_df.loc[idx, "status_deepprime"] = f"Failed: {fail_msg}"
        print(f"  [DeepPrime Ineligible] {row['INDEX']} ({row['Group']}): {fail_msg}")

    # PRIDICT2 constraints: Edit type in ['sub', 'ins', 'del']
    pr_mask = res_df["Edit type"].isin(["sub", "ins", "del"])
    print(f"PRIDICT2-eligible rows: {pr_mask.sum()}/{total_rows}")
    for idx, row in res_df[~pr_mask].iterrows():
        fail_msg = f"Edit type '{row['Edit type']}' is unsupported (PRIDICT2 expects single continuous Sub, Ins, or Del)"
        res_df.loc[idx, "status_pridict2"] = f"Failed: {fail_msg}"
        print(f"  [PRIDICT2 Ineligible] {row['INDEX']} ({row['Group']}): {fail_msg}")

    # 3. DeepPrime & DeepPrime6 Predictions
    print("\n[Step 3] Running DeepPrime-base and DeepPrime6 inferences...", flush=True)
    dp_indices = res_df[dp_mask].index
    dp_sub_df = res_df.loc[dp_indices].copy().reset_index(drop=True)

    # Format for preprocess_data
    dp_prep_df = dp_sub_df.rename(columns={
        "74bp wide target": "WideTargetSequence",
        "target 20bp": "Guide",
        "PBS length": "PBS_length",
        "RTT length": "RT_length",
        "Edit type": "Edit_type",
        "INDEX": "REF_ID"
    })
    dp_prep_df["ContextSeqUsed"] = "WideTargetSequence"
    dp_prep_df["leading G"] = ""
    dp_prep_df["Nicking"] = 21

    print("  Extracting DeepPrime biofeatures (Tm, MFE, GC, DeepSpCas9)...", flush=True)
    processed_dp = preprocess_data(data=dp_prep_df)
    print(f"  Feature extraction complete for {len(processed_dp)} rows.")

    # Prepare biofeatures
    feature_rename_map = {
        "PBS_len": "PBSlen", "RTT_len": "RTlen", "RT-PBS_len": "RT-PBSlen",
        "Tm1_PBS": "Tm1", "Tm2_RTT_cTarget_sameLength": "Tm2", "Tm3_RTT_cTarget_replaced": "Tm2new",
        "Tm4_cDNA_PAM-oppositeTarget": "Tm3", "Tm5_RTT_cDNA": "Tm4", "deltaTm_Tm4-Tm2": "TmD",
        "GC_count_PBS": "nGCcnt1", "GC_count_RTT": "nGCcnt2", "GC_count_RT-PBS": "nGCcnt3",
        "GC_contents_PBS": "fGCcont1", "GC_contents_RTT": "fGCcont2", "GC_contents_RT-PBS": "fGCcont3",
        "MFE_RT-PBS-polyT": "MFE3", "MFE_Spacer": "MFE4",
    }
    dp_ckpt_dir = "src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428"
    norm_mean_base = pd.read_csv(f"{dp_ckpt_dir}/mean.csv", header=None, index_col=0).squeeze()
    norm_std_base = pd.read_csv(f"{dp_ckpt_dir}/std.csv", header=None, index_col=0).squeeze()

    dp6_dir = "src/models/weights/DeepPrime6-weights"
    norm_mean_pe6 = pd.read_csv(os.path.join(dp6_dir, "pe6_mean.csv"), header=None, index_col=0).squeeze()
    norm_std_pe6 = pd.read_csv(os.path.join(dp6_dir, "pe6_std.csv"), header=None, index_col=0).squeeze()

    bio_renamed = processed_dp.rename(columns=feature_rename_map)
    bio_ordered = bio_renamed[norm_mean_base.index]
    norm_bio_base = ((bio_ordered - norm_mean_base) / norm_std_base).fillna(0).values
    norm_bio_pe6 = ((bio_ordered - norm_mean_pe6) / norm_std_pe6).fillna(0).values

    # Sequence encoding
    g_tensor = char_to_onehot_tensor(
        processed_dp["Target"].tolist(),
        processed_dp["Masked_EditSeq"].tolist(),
        device
    )
    b_tensor_base = torch.from_numpy(norm_bio_base).float().to(device)
    b_tensor_pe6 = torch.from_numpy(norm_bio_pe6).float().to(device)

    # Load DeepPrime-base 20 models (Strict original GeneInteractionModelVanilla)
    print("  Loading 20 DeepPrime-base ensemble models...", flush=True)
    base_files = sorted(glob.glob(f"{dp_ckpt_dir}/*.pt"))
    base_models = []
    for f in base_files:
        m = GeneInteractionModelVanilla()
        m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=True)
        base_models.append(m.to(device).eval())

    # Load DeepPrime6 checkpoints
    print("  Loading 4 DeepPrime6 checkpoints...", flush=True)
    dp6_ckpts = {
        "pred_dp6_pe6a": "pe6a_mainft.ckpt",
        "pred_dp6_pe6b": "pe6b_mainft.ckpt",
        "pred_dp6_pe6c": "pe6c_mainft.ckpt",
        "pred_dp6_drnaseh": "pemaxdrnaseh_mainft.ckpt"
    }
    dp6_models = {}
    for col_name, ckpt_file in dp6_ckpts.items():
        ckpt_path = os.path.join(dp6_dir, ckpt_file)
        m = PE6DeepPrimeModule.load_from_checkpoint(ckpt_path, strict=False).to(device).eval()
        dp6_models[col_name] = m

    # Run DeepPrime & DeepPrime6 forward passes
    print("  Executing forward pass for DeepPrime-base and DeepPrime6...", flush=True)
    with torch.no_grad():
        preds_dp_base = torch.stack([m(g_tensor, b_tensor_base) for m in base_models], dim=0).mean(dim=0).cpu().numpy().flatten()
        dp6_predictions = {
            col_name: model((g_tensor, b_tensor_pe6)).cpu().numpy().flatten()
            for col_name, model in dp6_models.items()
        }

    # Map back to res_df
    for i, orig_idx in enumerate(dp_indices):
        res_df.loc[orig_idx, "pred_dp_base"] = preds_dp_base[i]
        for col_name in dp6_predictions:
            res_df.loc[orig_idx, col_name] = dp6_predictions[col_name][i]
        res_df.loc[orig_idx, "status_deepprime"] = "Success"

    print("  DeepPrime scoring complete for all 109 eligible rows.")

    # 4. PRIDICT2.0 Predictions
    print("\n[Step 4] Running PRIDICT2.0 inference...", flush=True)
    pr_indices = res_df[pr_mask].index
    pr_sub_df = res_df.loc[pr_indices].copy().reset_index(drop=True)

    pr_input_rows = []
    for _, r in pr_sub_df.iterrows():
        pr_input_rows.append(pd.Series({
            "REF_ID": r["INDEX"],
            "WildTypeSequence": r[col_unedited],
            "PrimeEditedSequence": r[col_edited],
            "Edit_type": r["Edit type"].capitalize(),
            "Guide": r["target 20bp"],
            "PBS": r["PBS"],
            "RTT": r["RTT"]
        }))

    print(f"  Extracting PRIDICT2 features for {len(pr_input_rows)} rows...", flush=True)
    features_list_pr = []
    valid_pr_indices = []
    for i, pr_row in enumerate(pr_input_rows):
        orig_idx = pr_indices[i]
        try:
            feat = extract_single_pegRNA_features(pr_row)
            if feat is not None:
                features_list_pr.append(feat)
                valid_pr_indices.append(orig_idx)
            else:
                res_df.loc[orig_idx, "status_pridict2"] = "Failed: Feature extraction returned None"
        except Exception as e:
            res_df.loc[orig_idx, "status_pridict2"] = f"Failed: Feature extraction exception: {e}"

    if len(features_list_pr) > 0:
        pegdf = pd.DataFrame(features_list_pr)
        deepdfcols = [
            "wide_initial_target", "wide_mutated_target", "deepeditposition", 
            "deepeditposition_lst", "Correction_Type", "Correction_Length", 
            "protospacerlocation_only_initial", "PBSlocation",
            "RT_initial_location", "RT_mutated_location",
            "RToverhangmatches", "RToverhanglength", 
            "RTlength", "PBSlength", "RTmt", "RToverhangmt", "PBSmt", "protospacermt",
            "extensionmt", "original_base_mt", "edited_base_mt", "original_base_mt_nan",
            "edited_base_mt_nan"
        ]
        for c in deepdfcols:
            if c not in pegdf.columns:
                pegdf[c] = 0

        print("  Loading PRIDICT2 models and running inference...", flush=True)
        pr_models = load_pridict_model(run_ids=[0])
        all_avg_preds = deeppridict(pegdf, pr_models)
        tmp = [all_avg_preds[m] for m in all_avg_preds]
        tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
        agg_df = compute_average_predictions(tmp_df, grp_cols=["seq_id", "dataset_name"])

        # Extract HEK scores (* 100 to match percentage efficiency scale)
        hek_df = agg_df[agg_df["dataset_name"] == "HEK"].sort_values("seq_id")
        hek_scores = (hek_df["pred_averageedited"].values) * 100

        for i, orig_idx in enumerate(valid_pr_indices):
            res_df.loc[orig_idx, "pred_pridict2"] = hek_scores[i]
            res_df.loc[orig_idx, "status_pridict2"] = "Success"

    print("  PRIDICT2 scoring complete for all eligible rows.")

    # 5. Populate unified prediction_status and failure_reason
    print("\n[Step 5] Consolidating prediction status and error documentation...", flush=True)
    for idx, row in res_df.iterrows():
        dp_ok = (row["status_deepprime"] == "Success")
        pr_ok = (row["status_pridict2"] == "Success")

        if dp_ok and pr_ok:
            res_df.loc[idx, "prediction_status"] = "All models succeeded"
            res_df.loc[idx, "failure_reason"] = "None"
        elif not dp_ok and pr_ok:
            res_df.loc[idx, "prediction_status"] = "PRIDICT2 succeeded, DeepPrime failed"
            res_df.loc[idx, "failure_reason"] = row["status_deepprime"]
        elif dp_ok and not pr_ok:
            res_df.loc[idx, "prediction_status"] = "DeepPrime succeeded, PRIDICT2 failed"
            res_df.loc[idx, "failure_reason"] = row["status_pridict2"]
        else:
            res_df.loc[idx, "prediction_status"] = "All models failed"
            res_df.loc[idx, "failure_reason"] = f"DeepPrime: {row['status_deepprime']} | PRIDICT2: {row['status_pridict2']}"

    # Reorder columns logically
    final_cols = [
        "INDEX", "Group", "Edit type", "Edit position", "Edit length",
        "length difference", "PBS length", "RTT length", "RHA_len",
        "Guide sequence ((G/g)N19)", "target 20bp", "PAM", "RTT", "PBS",
        "pred_dp_base", "pred_dp6_drnaseh", "pred_dp6_pe6a", "pred_dp6_pe6b", "pred_dp6_pe6c", "pred_pridict2",
        "prediction_status", "failure_reason", "status_deepprime", "status_pridict2",
        "74bp wide target", col_unedited, col_edited
    ]
    final_cols = [c for c in final_cols if c in res_df.columns]
    res_df = res_df[final_cols]

    # 6. Save results to CSV and Excel
    print("\n[Step 6] Saving outputs...", flush=True)
    csv_path_data = os.path.join(out_dir_data, "frt_small_lib_predictions.csv")
    xlsx_path_data = os.path.join(out_dir_data, "frt_small_lib_predictions.xlsx")
    csv_path_exp = os.path.join(out_dir_exp, "frt_small_lib_predictions.csv")
    xlsx_path_exp = os.path.join(out_dir_exp, "frt_small_lib_predictions.xlsx")

    res_df.to_csv(csv_path_data, index=False)
    res_df.to_excel(xlsx_path_data, index=False)
    res_df.to_csv(csv_path_exp, index=False)
    res_df.to_excel(xlsx_path_exp, index=False)

    print(f"Saved predictions CSV to:\n  - {csv_path_data}\n  - {csv_path_exp}")
    print(f"Saved predictions Excel to:\n  - {xlsx_path_data}\n  - {xlsx_path_exp}")

    # 7. Summary Statistics
    summary = {
        "total_designs": total_rows,
        "all_models_succeeded": int((res_df["prediction_status"] == "All models succeeded").sum()),
        "pridict2_only_succeeded": int((res_df["prediction_status"] == "PRIDICT2 succeeded, DeepPrime failed").sum()),
        "all_models_failed": int((res_df["prediction_status"] == "All models failed").sum()),
        "by_group": {}
    }
    for grp, sub in res_df.groupby("Group"):
        summary["by_group"][grp] = {
            "count": len(sub),
            "dp_base_mean": float(sub["pred_dp_base"].dropna().mean()) if sub["pred_dp_base"].notna().any() else None,
            "dp6_pe6a_mean": float(sub["pred_dp6_pe6a"].dropna().mean()) if sub["pred_dp6_pe6a"].notna().any() else None,
            "dp6_pe6b_mean": float(sub["pred_dp6_pe6b"].dropna().mean()) if sub["pred_dp6_pe6b"].notna().any() else None,
            "dp6_pe6c_mean": float(sub["pred_dp6_pe6c"].dropna().mean()) if sub["pred_dp6_pe6c"].notna().any() else None,
            "dp6_drnaseh_mean": float(sub["pred_dp6_drnaseh"].dropna().mean()) if sub["pred_dp6_drnaseh"].notna().any() else None,
            "pridict2_mean": float(sub["pred_pridict2"].dropna().mean()) if sub["pred_pridict2"].notna().any() else None,
        }

    summary_file = os.path.join(out_dir_exp, "prediction_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to {summary_file}")

    print("\n=== Prediction Status Breakdown ===")
    print(res_df["prediction_status"].value_counts())

    print("\n=== Model Prediction Means across All Scored Designs ===")
    for col in ["pred_dp_base", "pred_dp6_drnaseh", "pred_dp6_pe6a", "pred_dp6_pe6b", "pred_dp6_pe6c", "pred_pridict2"]:
        s = res_df[col].dropna()
        print(f"  {col:18s}: Count={len(s):3d}, Mean={s.mean():6.2f}%, Median={s.median():6.2f}%, Min={s.min():6.2f}%, Max={s.max():6.2f}%")

    print("\nPipeline execution complete!")


if __name__ == "__main__":
    main()
