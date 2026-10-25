#!/usr/bin/env python3
"""
Figure 3 Evaluation Script: Main Library Strictly Unseen Test Set Performance

Evaluates:
- Ground-truth measured efficiency on N=2,325 strictly unseen test pegRNAs
- DeepPrime-base (20-model ensemble)
- DeepPrime6-PEmaxdRNaseH (pemaxdrnaseh_mainft.ckpt)
- DeepPrime6-PE6a (pe6a_mainft.ckpt)
- DeepPrime6-PE6b (pe6b_mainft.ckpt)
- DeepPrime6-PE6c (pe6c_mainft.ckpt)

Generates:
- High-resolution Figure 3 publication plots (PNG & SVG)
- Evaluation summary table & predictions CSV
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.pe6_module_dp_only import PE6DeepPrimeModule
from src.models.components.dprime import GeneInteractionModelVanilla

FEATURE_NAMES = [
    "PBSlen", "RTlen", "RT-PBSlen", "Edit_pos", "Edit_len", "RHA_len",
    "type_sub", "type_ins", "type_del", "Tm1", "Tm2", "Tm2new", "Tm3", "Tm4", "TmD",
    "nGCcnt1", "nGCcnt2", "nGCcnt3", "fGCcont1", "fGCcont2", "fGCcont3",
    "MFE3", "MFE4", "DeepSpCas9_score"
]

FEATURE_RENAME_MAP = {
    "PBS_len": "PBSlen", "RTT_len": "RTlen", "RT-PBS_len": "RT-PBSlen",
    "Tm1_PBS": "Tm1", "Tm2_RTT_cTarget_sameLength": "Tm2",
    "Tm3_RTT_cTarget_replaced": "Tm2new",
    "Tm4_cDNA_PAM-oppositeTarget": "Tm3",
    "Tm5_RTT_cDNA": "Tm4",
    "deltaTm_Tm4-Tm2": "TmD",
    "GC_count_PBS": "nGCcnt1", "GC_count_RTT": "nGCcnt2", "GC_count_RT-PBS": "nGCcnt3",
    "GC_contents_PBS": "fGCcont1", "GC_contents_RTT": "fGCcont2", "GC_contents_RT-PBS": "fGCcont3",
    "MFE_RT-PBS-polyT": "MFE3", "MFE_Spacer": "MFE4",
}

PE_TARGET_COLS = {
    "PEmaxdRNaseH": ["Normalized+3rep_HEK-M-2-7D+pe_ratio_%", "PEmaxdRNaseH", "comb_vis+HEK-M-2-7D%"],
    "PE6a": ["Normalized+3rep_HEK-M-3-7D+pe_ratio_%", "PE6a(+PEmaxCas9)", "comb_vis+HEK-M-3-7D%"],
    "PE6b": ["Normalized+3rep_HEK-M-4-7D+pe_ratio_%", "PE6b(+PEmaxCas9)", "comb_vis+HEK-M-4-7D%"],
    "PE6c": ["Normalized+3rep_HEK-M-5-7D+pe_ratio_%", "PE6c(+PEmaxCas9)", "comb_vis+HEK-M-5-7D%"],
}

def get_target_values(df, col_candidates):
    if isinstance(col_candidates, str):
        col_candidates = [col_candidates]
    for c in col_candidates:
        if c in df.columns:
            return df[c].values
    raise KeyError(f"None of candidate columns {col_candidates} found in dataframe!")

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
        for col_idx, ch in enumerate(str(wt)[:74]):
            if ch in char_map: g_arr[idx, char_map[ch], 0, col_idx] = 1.0
        for col_idx, ch in enumerate(str(ed)[:74]):
            if ch in char_map: g_arr[idx, char_map[ch], 1, col_idx] = 1.0
    return (2 * torch.from_numpy(g_arr) - 1).float().to(device)

def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    if n < 2: return np.nan, np.nan, np.nan, n
    r, _ = pearsonr(yp, yt)
    rho, _ = spearmanr(yp, yt)
    mse = mean_squared_error(yt, yp)
    return r, rho, mse, n

def main():
    print("==========================================================================", flush=True)
    print("=== Figure 3: Main Library Test Set Performance Evaluation ===", flush=True)
    print("==========================================================================", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    test_path = "data/rq3_test_library_dedup.parquet"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}!")

    df_test = pd.read_parquet(test_path)
    print(f"Loaded strictly unseen test set: {len(df_test):,} rows", flush=True)

    from src.utils.dataprep import IDX_TO_PE_TYPE, get_pe6_qc_mask

    # Ensure clean 2-rep B1 average columns exist and apply canonical QC filter (matching PE6DeepPrimeDataset)
    for i in range(1, 10):
        c3 = f"Normalized+3rep_HEK-M-{i}-7D+pe_ratio_%"
        r2 = f"Normalized+HEK-M-{i}-7D-B1+pe_ratio_%"
        r3 = f"Normalized+HEK-M2-{i}-7D-B1+pe_ratio_%"
        reps = [c for c in [r2, r3] if c in df_test.columns]
        if reps:
            df_test[c3] = np.maximum(0.0, df_test[reps].mean(axis=1))
            pe_type = IDX_TO_PE_TYPE[i]
            qc_mask = get_pe6_qc_mask(df_test, pe_type=pe_type, read_count_filter=200, max_ctrl_bg=5.0)
            df_test.loc[~qc_mask, c3] = np.nan
            if pe_type in df_test.columns:
                df_test.loc[~qc_mask, pe_type] = np.nan

    # Prepare features
    # 1. Base normalization for DeepPrime-base
    norm_mean_base = pd.read_csv("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/mean.csv", index_col=0, header=None).squeeze()
    norm_std_base = pd.read_csv("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/std.csv", index_col=0, header=None).squeeze()

    # 2. Canonical PE6 normalization for DeepPrime6 checkpoints
    pe6_ckpt_dir = "src/models/weights/DeepPrime6-weights"
    norm_mean_pe6 = pd.read_csv(os.path.join(pe6_ckpt_dir, "pe6_mean.csv"), index_col=0, header=None).squeeze()
    norm_std_pe6 = pd.read_csv(os.path.join(pe6_ckpt_dir, "pe6_std.csv"), index_col=0, header=None).squeeze()

    from src.utils.inference import prepare_biofeatures
    b_tensor_base = prepare_biofeatures(df_test, norm_mean_base, norm_std_base, device)
    b_tensor_pe6 = prepare_biofeatures(df_test, norm_mean_pe6, norm_std_pe6, device)

    wt_seqs = df_test["Target"].tolist() if "Target" in df_test.columns else df_test["WideTargetSequence"].tolist()
    ed_seqs = df_test["Masked_EditSeq"].tolist() if "Masked_EditSeq" in df_test.columns else wt_seqs

    # Load models
    print("Loading DeepPrime-base ensemble (20 models)...", flush=True)
    dp_base_models = load_deepprime_base_models("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428", device)

    print("Loading DeepPrime6 4 checkpoints...", flush=True)
    ckpt_dir = "src/models/weights/DeepPrime6-weights"
    dp6_models = {
        "PEmaxdRNaseH": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pemaxdrnaseh_mainft.ckpt"), strict=False).to(device).eval(),
        "PE6a": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6a_mainft.ckpt"), strict=False).to(device).eval(),
        "PE6b": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6b_mainft.ckpt"), strict=False).to(device).eval(),
        "PE6c": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6c_mainft.ckpt"), strict=False).to(device).eval(),
    }

    # Batch GPU Inference
    batch_size = 2048
    num_samples = len(df_test)
    preds_dp_base = np.zeros(num_samples, dtype=np.float32)
    preds_dp6 = {k: np.zeros(num_samples, dtype=np.float32) for k in dp6_models}

    print("Running GPU forward passes on test set...", flush=True)
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            g_batch = char_to_onehot_tensor(wt_seqs[start_idx:end_idx], ed_seqs[start_idx:end_idx], device)
            b_batch_base = b_tensor_base[start_idx:end_idx]
            b_batch_pe6 = b_tensor_pe6[start_idx:end_idx]

            preds_dp_base[start_idx:end_idx] = predict_deepprime_base(dp_base_models, g_batch, b_batch_base).cpu().numpy().flatten()
            for k, m in dp6_models.items():
                preds_dp6[k][start_idx:end_idx] = m((g_batch, b_batch_pe6)).cpu().numpy().flatten()

    df_test["Pred_DeepPrime_Base"] = preds_dp_base
    for k, v in preds_dp6.items():
        df_test[f"Pred_DeepPrime6_{k}"] = v

    # Performance Evaluation Table
    results = []
    print("\n--- Model Performance Summary on Strictly Unseen Main Library Test Set ---", flush=True)
    for model_name, target_candidates in PE_TARGET_COLS.items():
        y_true = get_target_values(df_test, target_candidates)

        # Base model metrics
        r_base, rho_base, mse_base, n_base = compute_metrics(y_true, preds_dp_base)
        # DeepPrime6 fine-tuned metrics
        r_dp6, rho_dp6, mse_dp6, n_dp6 = compute_metrics(y_true, preds_dp6[model_name])

        delta_r = r_dp6 - r_base
        results.append({
            "PE System": model_name,
            "DeepPrime-FT (Pearson r)": round(r_base, 4),
            "DeepPrime-FT (Spearman rho)": round(rho_base, 4),
            "DeepPrime6 (Pearson r)": round(r_dp6, 4),
            "DeepPrime6 (Spearman rho)": round(rho_dp6, 4),
            "Delta Pearson r": round(delta_r, 4),
            "N (Valid)": n_dp6
        })

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False), flush=True)

    out_csv = "figures/evaluation/Figure3_main_library_test_performance.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    for extra_csv in [
        "notebooks/plots/Figure3_main_library_test_performance.csv",
    ]:
        os.makedirs(os.path.dirname(extra_csv), exist_ok=True)
        df_res.to_csv(extra_csv, index=False)
    print(f"\nSaved metrics summary to: {out_csv}", flush=True)

    # -------------------------------------------------------------
    # Plotting Figure 3 (Multi-panel Publication Style)
    # -------------------------------------------------------------
    sns.set_theme(style="ticks", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes = axes.flatten()

    palette = {"PEmaxdRNaseH": "#1f77b4", "PE6a": "#ff7f0e", "PE6b": "#2ca02c", "PE6c": "#d62728"}

    for idx, (model_name, target_candidates) in enumerate(PE_TARGET_COLS.items()):
        ax = axes[idx]
        y_true = get_target_values(df_test, target_candidates)
        y_pred = preds_dp6[model_name]

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        yt, yp = y_true[mask], y_pred[mask]
        r, rho, _, n = compute_metrics(yt, yp)

        color = palette.get(model_name, "#333333")
        sns.regplot(
            x=yt, y=yp, ax=ax,
            scatter_kws={"alpha": 0.45, "s": 22, "color": color, "edgecolor": "none"},
            line_kws={"color": "black", "linewidth": 2.0}
        )

        ax.set_title(f"{model_name}\nPearson $r = {r:.3f}$ | Spearman $\\rho = {rho:.3f}$ (N={n:,})", fontsize=14, weight="bold")
        ax.set_xlabel("Measured Editing Efficiency (%)", fontsize=12)
        ax.set_ylabel("DeepPrime6 Predicted Efficiency (%)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Figure 3: DeepPrime6 Accuracy on Strictly Unseen Main Library Test Set (Zero Leakage)", fontsize=16, weight="bold", y=0.99)
    plt.tight_layout()

    out_png = "figures/evaluation/Figure3_main_library_test_performance.png"
    out_svg = "figures/evaluation/Figure3_main_library_test_performance.svg"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    for extra_dir in [
        "notebooks/plots",
    ]:
        os.makedirs(extra_dir, exist_ok=True)
        plt.savefig(os.path.join(extra_dir, "Figure3_main_library_test_performance.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(extra_dir, "Figure3_main_library_test_performance.svg"), format="svg", bbox_inches="tight")
    plt.close()

    print(f"Saved Figure 3 plots to:\n  - {out_png}\n  - {out_svg}", flush=True)

if __name__ == "__main__":
    main()
