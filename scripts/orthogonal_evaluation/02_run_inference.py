#!/usr/bin/env python3
"""
Orthogonal Benchmark: Step 2 - GPU Inference & Performance Evaluation Module

Purpose:
1. Loads the pre-filtered, strictly unseen benchmark datasets:
   - `data/orthogonal_testset/deepprime_test_filtered.csv`
   - `data/orthogonal_testset/pridict2_test_filtered.csv`
2. Executes GPU batch inference for 6 target models:
   - DeepPrime-base (20-model ensemble)
   - PRIDICT2.0 (Official released fold 0 model)
   - DeepPrime6-pe6a (pe6a_mainft.ckpt)
   - DeepPrime6-pe6b (pe6b_mainft.ckpt)
   - DeepPrime6-pe6c (pe6c_mainft.ckpt)
   - DeepPrime6-pemaxdrnaseh (pemaxdrnaseh_mainft.ckpt)
3. Computes Spearman rho, Pearson r, and sample size N against ground-truth measured PE efficiency.
4. Saves predictions and performance comparison tables to:
   - `data/orthogonal_testset/`
   - `experiments/20261009_orthogonal_benchmark_leakage_stratification/results/`
   - `notebooks/plots/orthogonal_benchmark/`
"""

import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

# Ensure project root is active working directory and in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
pridict_path = os.path.join(PROJECT_ROOT, "PRIDICT2")
if pridict_path not in sys.path:
    sys.path.insert(0, pridict_path)

from Bio.Seq import reverse_complement
from src.models.pe6_module_dp_only import PE6DeepPrimeModule
from src.models.components.dprime import GeneInteractionModelVanilla
from src.utils.predict_pridict2 import (
    load_pridict_model,
    deeppridict,
    compute_average_predictions,
    extract_single_pegRNA_features
)
from src.data.components.pe6_preprocess_data import preprocess_data

OUTPUT_DIR = "data/orthogonal_testset"
EXP_DIR = "experiments/20261009_orthogonal_benchmark_leakage_stratification/results"
PLOTS_DIR = "notebooks/plots/orthogonal_benchmark"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "PBSlen", "RTlen", "RT-PBSlen", "Edit_pos", "Edit_len", "RHA_len",
    "type_sub", "type_ins", "type_del", "Tm1", "Tm2", "Tm2new", "Tm3", "Tm4", "TmD",
    "nGCcnt1", "nGCcnt2", "nGCcnt3", "fGCcont1", "fGCcont2", "fGCcont3",
    "MFE3", "MFE4", "DeepSpCas9_score"
]


def load_deepprime_base_models(weights_dir, device):
    """Loads all 20 pretrained checkpoints of the DeepPrime-base ensemble."""
    files = sorted(glob.glob(os.path.join(weights_dir, "*.pt")))
    if not files:
        raise FileNotFoundError(f"No checkpoint files (.pt) found in {weights_dir}")
    models = []
    for f in files:
        m = GeneInteractionModelVanilla()
        m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=True)
        models.append(m.to(device).eval())
    return models


def predict_deepprime_base(models, g, b):
    """Ensemble prediction averaging across 20 DeepPrime-base models."""
    return torch.stack([m(g, b) for m in models], dim=0).mean(dim=0)


def char_to_onehot_tensor(seqs_wt, seqs_ed, device):
    """Encodes WT and edited sequences into (B, 4, 2, 74) tensor with 2*x-1 normalization."""
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


def run_deepprime_inference(seqs_wt, seqs_ed, biofeatures_df, dp_base_models, dp6_models, norm_mean, norm_std, device, batch_size=4096):
    """Runs DeepPrime-base and all DeepPrime6 models in GPU batches."""
    num_samples = len(seqs_wt)
    b_norm = (biofeatures_df[FEATURE_NAMES] - norm_mean) / norm_std
    b_tensor = torch.from_numpy(b_norm.fillna(0).values).float().to(device)
    
    preds_dp_base = np.zeros(num_samples, dtype=np.float32)
    preds_pe6a = np.zeros(num_samples, dtype=np.float32)
    preds_pe6b = np.zeros(num_samples, dtype=np.float32)
    preds_pe6c = np.zeros(num_samples, dtype=np.float32)
    preds_drnaseh = np.zeros(num_samples, dtype=np.float32)
    
    print(f"  Running DeepPrime & DeepPrime6 forward passes on {num_samples:,} samples...", flush=True)
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            wt_sub = seqs_wt[start_idx:end_idx]
            ed_sub = seqs_ed[start_idx:end_idx]
            g_batch = char_to_onehot_tensor(wt_sub, ed_sub, device)
            b_batch = b_tensor[start_idx:end_idx]
            
            # DeepPrime Base (20 models average)
            preds_dp_base[start_idx:end_idx] = predict_deepprime_base(dp_base_models, g_batch, b_batch).cpu().numpy().flatten()
            
            # DeepPrime6 Models
            preds_pe6a[start_idx:end_idx] = dp6_models["pe6a"]((g_batch, b_batch)).cpu().numpy().flatten()
            preds_pe6b[start_idx:end_idx] = dp6_models["pe6b"]((g_batch, b_batch)).cpu().numpy().flatten()
            preds_pe6c[start_idx:end_idx] = dp6_models["pe6c"]((g_batch, b_batch)).cpu().numpy().flatten()
            preds_drnaseh[start_idx:end_idx] = dp6_models["drnaseh"]((g_batch, b_batch)).cpu().numpy().flatten()
            
    return {
        "pred_dp_base": preds_dp_base,
        "pred_dp6_pe6a": preds_pe6a,
        "pred_dp6_pe6b": preds_pe6b,
        "pred_dp6_pe6c": preds_pe6c,
        "pred_dp6_drnaseh": preds_drnaseh
    }


def compute_metrics(y_true, y_pred):
    """Computes Spearman rho, Pearson r, and sample size N."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[mask]
    yp = y_pred[mask]
    n = len(yt)
    if n < 2:
        return np.nan, np.nan, n
    rho, _ = spearmanr(yp, yt)
    r, _ = pearsonr(yp, yt)
    return rho, r, n


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing Inference on device: {device}", flush=True)

    # 1. Load Pretrained Weights & Checkpoints
    print("\n--- 1. Loading Pretrained Weights & Checkpoints ---", flush=True)
    norm_mean = pd.read_csv("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/mean.csv", index_col=0, header=None).squeeze()
    norm_std = pd.read_csv("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/std.csv", index_col=0, header=None).squeeze()

    print("  Loading DeepPrime-base 20-model ensemble...", flush=True)
    dp_base_models = load_deepprime_base_models("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428", device)

    print("  Loading DeepPrime6 4 checkpoints...", flush=True)
    ckpt_dir = "src/models/weights/DeepPrime6-weights"
    dp6_models = {
        "pe6a": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6a_mainft.ckpt"), strict=False).to(device).eval(),
        "pe6b": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6b_mainft.ckpt"), strict=False).to(device).eval(),
        "pe6c": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6c_mainft.ckpt"), strict=False).to(device).eval(),
        "drnaseh": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pemaxdrnaseh_mainft.ckpt"), strict=False).to(device).eval(),
    }

    print("  Loading PRIDICT2.0 model...", flush=True)
    pr_models = load_pridict_model(run_ids=[0])

    # ---------------------------------------------------------
    # 2. Evaluate on Filtered DeepPrime Test Set
    # ---------------------------------------------------------
    print("\n=======================================================", flush=True)
    print("=== 2. Evaluating on Filtered DeepPrime Test Set ===", flush=True)
    print("=======================================================", flush=True)
    dp_filtered_path = os.path.join(OUTPUT_DIR, "deepprime_test_filtered.csv")
    if not os.path.exists(dp_filtered_path):
        raise FileNotFoundError(f"Filtered file not found: {dp_filtered_path}. Run 01_filter_testsets.py first.")
    
    dp_test = pd.read_csv(dp_filtered_path)
    print(f"  Loaded filtered DeepPrime test set: {len(dp_test):,} rows (100% strictly unseen)", flush=True)

    # DeepPrime and DeepPrime6 predictions
    dp_preds = run_deepprime_inference(
        dp_test["WTS_clean"].tolist(),
        dp_test["ETS_clean"].tolist(),
        dp_test[FEATURE_NAMES],
        dp_base_models,
        dp6_models,
        norm_mean,
        norm_std,
        device
    )
    for k, v in dp_preds.items():
        dp_test[k] = v

    # PRIDICT2.0 prediction on DeepPrime test set
    print("  Extracting PRIDICT2 features for filtered DeepPrime rows...", flush=True)
    pr_features_list = []
    for i, r in dp_test.iterrows():
        w = r["WTS_clean"]
        u = r["Unedited (gRNA binding -2 ~ RTT binding +2)"]
        p = r["Prime-edited (gRNA binding -2 ~ RTT binding +2)"]
        u_idx = w.find(u) if pd.notna(u) else -1
        if u_idx != -1:
            edited = w[:u_idx] + p + w[u_idx + len(u):]
        else:
            edited = w
        ext = r["3' extension sequence of pegRNA"] if pd.notna(r["3' extension sequence of pegRNA"]) else ""
        rt_len = int(r["RTlen"])
        pbs_seq = ext[rt_len:] if len(ext) >= rt_len else ""
        rtt_seq = ext[:rt_len] if len(ext) >= rt_len else ""
        guide_seq = "G" + str(r["Guide sequence ((G/g)N19)"])
        
        pr_row = {
            "REF_ID": r["ID"],
            "WildTypeSequence": w,
            "PrimeEditedSequence": edited,
            "Edit_type": r["Edit_type_clean"].capitalize(),
            "Guide": guide_seq,
            "PBS": pbs_seq,
            "RTT": rtt_seq
        }
        try:
            feat = extract_single_pegRNA_features(pr_row)
            if feat is not None:
                pr_features_list.append(feat)
            else:
                pr_features_list.append({"sequence_name": r["ID"]})
        except Exception:
            pr_features_list.append({"sequence_name": r["ID"]})

    pr_peg_df = pd.DataFrame(pr_features_list)
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
        if c not in pr_peg_df.columns:
            pr_peg_df[c] = 0

    print("  Running PRIDICT2.0 forward pass on filtered DeepPrime rows...", flush=True)
    pr_preds_res = deeppridict(pr_peg_df, pr_models)
    tmp = [pr_preds_res[m] for m in pr_preds_res]
    tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
    agg_df = compute_average_predictions(tmp_df, grp_cols=["seq_id", "dataset_name"])
    hek_df = agg_df[agg_df["dataset_name"] == "HEK"].sort_values("seq_id").reset_index(drop=True)
    dp_test["pred_pridict2_hek"] = hek_df["pred_averageedited"].values * 100

    # Save DeepPrime predictions
    dp_out_cols = [
        "ID", "WTS_clean", "ETS_clean", "Edit_type_clean", "Edit_pos", "Edit_len", "PBSlen", "RTlen",
        "Measured_PE_efficiency", "pred_dp_base", "pred_pridict2_hek", "pred_dp6_pe6a", "pred_dp6_pe6b", "pred_dp6_pe6c", "pred_dp6_drnaseh"
    ]
    dp_test_out = dp_test[dp_out_cols].copy()
    for d in [OUTPUT_DIR, EXP_DIR, PLOTS_DIR]:
        dp_test_out.to_csv(os.path.join(d, "deepprime_test_predictions.csv"), index=False)
        dp_test_out.to_csv(os.path.join(d, "deepprime_test_raw_predictions.csv"), index=False)
    print(f"  Saved filtered predictions to {OUTPUT_DIR}, {EXP_DIR}, and {PLOTS_DIR}", flush=True)

    # ---------------------------------------------------------
    # 3. Evaluate on Filtered PRIDICT2.0 Test Set
    # ---------------------------------------------------------
    print("\n=======================================================", flush=True)
    print("=== 3. Evaluating on Filtered PRIDICT2.0 Test Set ===", flush=True)
    print("=======================================================", flush=True)
    p_filtered_path = os.path.join(OUTPUT_DIR, "pridict2_test_filtered.csv")
    if not os.path.exists(p_filtered_path):
        raise FileNotFoundError(f"Filtered file not found: {p_filtered_path}. Run 01_filter_testsets.py first.")

    p_test = pd.read_csv(p_filtered_path)
    print(f"  Loaded filtered PRIDICT2 test set: {len(p_test):,} rows (100% strictly unseen)", flush=True)

    # PRIDICT2.0 prediction
    print("  Running PRIDICT2.0 forward pass...", flush=True)
    p_test_input = p_test.copy()
    p_test_input["seq_id"] = list(range(len(p_test_input)))
    res_pr = deeppridict(p_test_input, pr_models)
    tmp_p = [res_pr[m] for m in res_pr]
    tmp_df_p = pd.concat(tmp_p, axis=0, ignore_index=True)
    agg_df_p = compute_average_predictions(tmp_df_p, grp_cols=["seq_id", "dataset_name"])
    p_test["pred_pridict2_hek"] = agg_df_p.loc[agg_df_p["dataset_name"] == "HEK", "pred_averageedited"].values * 100
    p_test["Measured_PE_efficiency"] = p_test["HEKaverageedited_clamped"].values * 100

    # DeepPrime & DeepPrime6 predictions via preprocess_data
    print("  Converting PRIDICT2 test rows to DeepPrime biofeatures...", flush=True)
    prep_records = []
    for i, r in p_test.iterrows():
        p_loc = json.loads(r["PBSlocation"])
        rt_loc = json.loads(r["RT_initial_location"])
        seq_wt = r["wide_initial_target"]
        seq_mut = r["wide_mutated_target"]
        nick = p_loc[1]
        edit_pos = int(r["deepeditposition"]) - nick    # DeepPrime way: The relative position from the nick
        prep_records.append({
            "WideTargetSequence": seq_wt[5:79],
            "Guide": seq_wt[9:29],
            "leading G": "",
            "PBS": reverse_complement(seq_wt[p_loc[0]:p_loc[1]]),
            "RTT": reverse_complement(seq_mut[rt_loc[0]:rt_loc[1]]),
            "Edit length": int(r["Correction_Length"]),
            "Edit position": edit_pos,
            "Edit_type": "sub" if r["Correction_Type"] == "Replacement" else ("ins" if r["Correction_Type"] == "Insertion" else "del"),
            "Nicking": 21,
            "ContextSeqUsed": "WideTargetSequence"
        })
    p_prep_df = preprocess_data(data=pd.DataFrame(prep_records))

    feature_rename_map = {
        "PBS_len": "PBSlen", "RTT_len": "RTlen", "RT-PBS_len": "RT-PBSlen",
        "Tm1_PBS": "Tm1", "Tm2_RTT_cTarget_sameLength": "Tm2", "Tm3_RTT_cTarget_replaced": "Tm2new",
        "Tm5_RTT_cDNA": "Tm3", "Tm4_cDNA_PAM-oppositeTarget": "Tm4", "deltaTm_Tm4-Tm2": "TmD",
        "GC_count_PBS": "nGCcnt1", "GC_count_RTT": "nGCcnt2", "GC_count_RT-PBS": "nGCcnt3",
        "GC_contents_PBS": "fGCcont1", "GC_contents_RTT": "fGCcont2", "GC_contents_RT-PBS": "fGCcont3",
        "MFE_RT-PBS-polyT": "MFE3", "MFE_Spacer": "MFE4",
    }
    p_prep_renamed = p_prep_df.rename(columns=feature_rename_map)

    p_dp_preds = run_deepprime_inference(
        p_prep_renamed["Target"].str.upper().tolist(),
        p_prep_renamed["Masked_EditSeq"].str.upper().tolist(),
        p_prep_renamed[FEATURE_NAMES],
        dp_base_models,
        dp6_models,
        norm_mean,
        norm_std,
        device
    )
    for k, v in p_dp_preds.items():
        p_test[k] = v

    # Save PRIDICT2 predictions
    pr_out_cols = [
        "ID", "seq_id", "grp_id", "Correction_Type", "Correction_Length", "deepeditposition", "PBSlength", "RTlength",
        "Measured_PE_efficiency", "pred_dp_base", "pred_pridict2_hek", "pred_dp6_pe6a", "pred_dp6_pe6b", "pred_dp6_pe6c", "pred_dp6_drnaseh"
    ]
    p_test_out = p_test[pr_out_cols].copy()
    for d in [OUTPUT_DIR, EXP_DIR, PLOTS_DIR]:
        p_test_out.to_csv(os.path.join(d, "pridict2_test_predictions.csv"), index=False)
        p_test_out.to_csv(os.path.join(d, "pridict2_test_raw_predictions.csv"), index=False)
    print(f"  Saved filtered predictions to {OUTPUT_DIR}, {EXP_DIR}, and {PLOTS_DIR}", flush=True)

    # ---------------------------------------------------------
    # 4. Performance Metrics Comparison
    # ---------------------------------------------------------
    print("\n--- 4. Computing Metrics on Strictly Unseen Filtered Datasets ---", flush=True)
    models_to_eval = [
        ("DeepPrime-base", "pred_dp_base"),
        ("PRIDICT2.0", "pred_pridict2_hek"),
        ("DeepPrime6-pe6a", "pred_dp6_pe6a"),
        ("DeepPrime6-pe6b", "pred_dp6_pe6b"),
        ("DeepPrime6-pe6c", "pred_dp6_pe6c"),
        ("DeepPrime6-pemaxdrnaseh", "pred_dp6_drnaseh")
    ]

    # DeepPrime Performance Table
    dp_perf_rows = []
    yt_dp = dp_test["Measured_PE_efficiency"].values
    for m_name, pred_col in models_to_eval:
        yp = dp_test[pred_col].values
        rho, r, n = compute_metrics(yt_dp, yp)
        dp_perf_rows.append({
            "Evaluation Benchmark": "DeepPrime Official Generalization Test Set",
            "Condition": "Strict Target Locus Stratification",
            "Model": m_name,
            "Spearman": round(rho, 4),
            "Pearson": round(r, 4),
            "N": n
        })
    dp_perf_df = pd.DataFrame(dp_perf_rows)
    dp_perf_df.to_csv(os.path.join(OUTPUT_DIR, "deepprime_test_performance.csv"), index=False)
    dp_perf_df.to_csv(os.path.join(EXP_DIR, "deepprime_test_performance.csv"), index=False)

    # PRIDICT2 Performance Table
    pr_perf_rows = []
    yt_pr = p_test["Measured_PE_efficiency"].values
    for m_name, pred_col in models_to_eval:
        yp = p_test[pred_col].values
        rho, r, n = compute_metrics(yt_pr, yp)
        pr_perf_rows.append({
            "Evaluation Benchmark": "PRIDICT2.0 Official Fold 0 Test Set",
            "Condition": "Strict Target Locus Stratification",
            "Model": m_name,
            "Spearman": round(rho, 4),
            "Pearson": round(r, 4),
            "N": n
        })
    pr_perf_df = pd.DataFrame(pr_perf_rows)
    pr_perf_df.to_csv(os.path.join(OUTPUT_DIR, "pridict2_test_performance.csv"), index=False)
    pr_perf_df.to_csv(os.path.join(EXP_DIR, "pridict2_test_performance.csv"), index=False)

    # Combined Performance Summary for plots directory
    perf_summary_combined = pd.concat([dp_perf_df, pr_perf_df], ignore_index=True)
    perf_summary_combined.to_csv(os.path.join(PLOTS_DIR, "performance_summary.csv"), index=False)
    perf_summary_combined.to_csv(os.path.join(EXP_DIR, "performance_summary.csv"), index=False)

    # Print Results
    print("\n=======================================================", flush=True)
    print("=== FINAL ORTHOGONAL BENCHMARK PERFORMANCE TABLES ===", flush=True)
    print("=======================================================", flush=True)
    print("\n[DeepPrime Generalization Test Set (Strictly Unseen)]")
    print(dp_perf_df[["Model", "Spearman", "Pearson", "N"]].to_string(index=False))
    print("\n[PRIDICT2.0 Fold 0 Test Set (Strictly Unseen)]")
    print(pr_perf_df[["Model", "Spearman", "Pearson", "N"]].to_string(index=False))
    print("\n[Complete] Inference and evaluation completed successfully!", flush=True)


if __name__ == "__main__":
    main()
