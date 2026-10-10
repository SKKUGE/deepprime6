#!/usr/bin/env python3
"""
Compute all model predictions on the deduplicated test set (data/rq3_test_library_dedup.parquet).
Models:
1. DeepPrime-base (Vanilla 20-model ensemble)
2. DeepPrime6-pe6a (pe6a_mainft.ckpt)
3. DeepPrime6-pe6b (pe6b_mainft.ckpt)
4. DeepPrime6-pe6c (pe6c_mainft.ckpt)
5. DeepPrime6-drnaseh (pemaxdrnaseh_mainft.ckpt)
6. PRIDICT2.0 (Official HEK & K562 scores)
"""

import os
import sys
import glob
import time
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.pe6_module_dp_only import PE6DeepPrimeModule
from src.models.components.dprime import GeneInteractionModelVanilla
from src.utils.predict_pridict2 import (
    load_pridict_model,
    deeppridict,
    compute_average_predictions,
    extract_single_pegRNA_features
)

FEATURE_NAMES = [
    "PBSlen", "RTlen", "RT-PBSlen", "Edit_pos", "Edit_len", "RHA_len",
    "type_sub", "type_ins", "type_del", "Tm1", "Tm2", "Tm2new", "Tm3", "Tm4", "TmD",
    "nGCcnt1", "nGCcnt2", "nGCcnt3", "fGCcont1", "fGCcont2", "fGCcont3",
    "MFE3", "MFE4", "DeepSpCas9_score"
]

feature_rename_map = {
    "PBS_len": "PBSlen", "RTT_len": "RTlen", "RT-PBS_len": "RT-PBSlen",
    "Tm1_PBS": "Tm1", "Tm2_RTT_cTarget_sameLength": "Tm2", "Tm3_RTT_cTarget_replaced": "Tm2new",
    "Tm4_cDNA_PAM-oppositeTarget": "Tm3", "Tm5_RTT_cDNA": "Tm4", "deltaTm_Tm4-Tm2": "TmD",
    "GC_count_PBS": "nGCcnt1", "GC_count_RTT": "nGCcnt2", "GC_count_RT-PBS": "nGCcnt3",
    "GC_contents_PBS": "fGCcont1", "GC_contents_RTT": "fGCcont2", "GC_contents_RT-PBS": "fGCcont3",
    "MFE_RT-PBS-polyT": "MFE3", "MFE_Spacer": "MFE4",
}

def load_deepprime_base_models(weights_dir, device):
    files = sorted(glob.glob(os.path.join(weights_dir, "*.pt")))
    models = []
    for f in files:
        m = GeneInteractionModelVanilla()
        m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=True)
        models.append(m.to(device).eval())
    return models

def predict_deepprime_base(models, g, b):
    return torch.stack([m(g, b) for m in models], dim=0).mean(dim=0)

def char_to_onehot_tensor(seqs_wt, seqs_ed, device):
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
    print(f"Executing Inference on device: {device}", flush=True)

    test_path = "data/rq3_test_library_dedup.parquet"
    print(f"Loading deduplicated test set from {test_path}...", flush=True)
    df = pd.read_parquet(test_path)
    print(f"Loaded {len(df):,} test sequences.", flush=True)

    norm_mean = pd.read_csv("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/mean.csv", index_col=0, header=None).squeeze()
    norm_std = pd.read_csv("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/std.csv", index_col=0, header=None).squeeze()

    print("Loading DeepPrime-base 20 models...", flush=True)
    dp_base_models = load_deepprime_base_models("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428", device)

    print("Loading DeepPrime6 4 checkpoints...", flush=True)
    ckpt_dir = "src/models/weights/DeepPrime6-weights"
    dp6_models = {
        "pe6a": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6a_mainft.ckpt"), strict=False).to(device).eval(),
        "pe6b": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6b_mainft.ckpt"), strict=False).to(device).eval(),
        "pe6c": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6c_mainft.ckpt"), strict=False).to(device).eval(),
        "drnaseh": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pemaxdrnaseh_mainft.ckpt"), strict=False).to(device).eval(),
    }

    # Prepare features for DeepPrime inference
    df_renamed = df.rename(columns=feature_rename_map)
    b_norm = (df_renamed[FEATURE_NAMES] - norm_mean) / norm_std
    b_tensor = torch.from_numpy(b_norm.fillna(0).values).float().to(device)
    g_batch = char_to_onehot_tensor(
        df_renamed["Target"].str.upper().tolist(),
        df_renamed["Masked_EditSeq"].str.upper().tolist(),
        device
    )

    print("Running forward passes for DeepPrime models...", flush=True)
    with torch.no_grad():
        pred_base = predict_deepprime_base(dp_base_models, g_batch, b_tensor).cpu().numpy().flatten()
        pred_pe6a = dp6_models["pe6a"]((g_batch, b_tensor)).cpu().numpy().flatten()
        pred_pe6b = dp6_models["pe6b"]((g_batch, b_tensor)).cpu().numpy().flatten()
        pred_pe6c = dp6_models["pe6c"]((g_batch, b_tensor)).cpu().numpy().flatten()
        pred_drnaseh = dp6_models["drnaseh"]((g_batch, b_tensor)).cpu().numpy().flatten()

    # Save individual prediction CSVs matching Track A format
    id_col = df["ID"].values
    
    def save_pred(pred_arr, filepath):
        out_df = pd.DataFrame({"ID": id_col, "orig_index": id_col, "Prediction": pred_arr})
        out_df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}", flush=True)

    save_pred(pred_base, "data/rq3_predictions_dp_base.csv")
    save_pred(pred_pe6a, "data/rq3_predictions_dp6.csv")
    save_pred(pred_pe6b, "data/rq3_predictions_dp6_pe6b.csv")
    save_pred(pred_pe6c, "data/rq3_predictions_dp6_pe6c.csv")
    save_pred(pred_drnaseh, "data/rq3_predictions_dp6_drnaseh.csv")

    # Now run PRIDICT2.0 prediction on deduplicated test set
    print("\n--- Running PRIDICT2.0 Inference on deduplicated test set ---", flush=True)
    pr_models = load_pridict_model(run_ids=[0])

    features_list = []
    skipped_indices = []
    t0 = time.time()
    for idx, row in df.iterrows():
        try:
            r_dict = row.to_dict()
            r_dict["REF_ID"] = str(r_dict["ID"])
            feat = extract_single_pegRNA_features(r_dict)
            if feat is not None:
                features_list.append(feat)
            else:
                skipped_indices.append(idx)
        except Exception as e:
            skipped_indices.append(idx)
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df)} sequences (matched {len(features_list)})...", flush=True)

    print(f"Feature extraction done in {time.time() - t0:.2f}s. Matched: {len(features_list)}/{len(df)}")
    pegdataframe = pd.DataFrame(features_list)

    all_avg_preds = deeppridict(pegdataframe, pr_models)
    tmp = [all_avg_preds[m] for m in all_avg_preds]
    tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
    agg_df = compute_average_predictions(tmp_df, grp_cols=["seq_id", "dataset_name"])

    for cell_type in ["HEK", "K562"]:
        cond = agg_df["dataset_name"] == cell_type
        pegdataframe[f"PRIDICT2_0_editing_Score_deep_{cell_type}"] = agg_df.loc[cond, "pred_averageedited"].values * 100

    pr_out = df.merge(
        pegdataframe[["sequence_name", "PRIDICT2_0_editing_Score_deep_HEK", "PRIDICT2_0_editing_Score_deep_K562"]],
        left_on="ID",
        right_on="sequence_name",
        how="left"
    )
    pr_out["REF_ID"] = pr_out["ID"]
    pr_out.to_csv("data/rq3_predictions_pridict2.csv", index=False)
    print("Saved PRIDICT2.0 predictions to data/rq3_predictions_pridict2.csv", flush=True)

    # Copy all prediction files to experiments/20261008_panelB_base_architecture_revalidation/predictions/
    exp_pred_dir = "experiments/20261008_panelB_base_architecture_revalidation/predictions"
    os.makedirs(exp_pred_dir, exist_ok=True)
    for fn in [
        "rq3_predictions_dp_base.csv", "rq3_predictions_dp6.csv", "rq3_predictions_dp6_pe6b.csv",
        "rq3_predictions_dp6_pe6c.csv", "rq3_predictions_dp6_drnaseh.csv", "rq3_predictions_pridict2.csv"
    ]:
        os.system(f"cp data/{fn} {exp_pred_dir}/{fn}")
    df.to_parquet(f"{exp_pred_dir}/rq3_test_library.parquet")
    df.to_csv(f"{exp_pred_dir}/rq3_test_library.csv", index=False)
    print(f"All predictions successfully synced to {exp_pred_dir}!", flush=True)

if __name__ == "__main__":
    main()
