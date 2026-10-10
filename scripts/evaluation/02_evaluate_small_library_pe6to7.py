#!/usr/bin/env python3
"""
Small Library Orthogonal Benchmark Script: PE6 to PE7 Generalization

Evaluates:
- Small Library dataset (pe6to7_with_avgs.parquet, N=1,960)
- PE6 vs PE7 measured efficiencies
- DeepPrime-base (20-ensemble) predictions
- DeepPrime6 (PEmaxdRNaseH, PE6b, PE6c) fine-tuned predictions

Generates:
- High-resolution orthogonal correlation plots (PNG & SVG)
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
from src.data.components.pe6_preprocess_data import preprocess_data

FEATURE_NAMES = [
    "PBSlen", "RTlen", "RT-PBSlen", "Edit_pos", "Edit_len", "RHA_len",
    "type_sub", "type_ins", "type_del", "Tm1", "Tm2", "Tm2new", "Tm3", "Tm4", "TmD",
    "nGCcnt1", "nGCcnt2", "nGCcnt3", "fGCcont1", "fGCcont2", "fGCcont3",
    "MFE3", "MFE4", "DeepSpCas9_score"
]

FEATURE_RENAME_MAP = {
    "PBS_len": "PBSlen", "RTT_len": "RTlen", "RT-PBS_len": "RT-PBSlen",
    "Tm1_PBS": "Tm1", "Tm2_RTT_cTarget_sameLength": "Tm2",
    "Tm3_RTT_cTarget_replaced": "Tm2new", "Tm4_cDNA_PAM-oppositeTarget": "Tm3",
    "Tm5_RTT_cDNA": "Tm4", "deltaTm_Tm4-Tm2": "TmD",
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
    print("=== Small Library Orthogonal Benchmark: PE6 & PE7 Generalization ===", flush=True)
    print("==========================================================================", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    data_path = "data/small_library/pe6to7_with_avgs.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Small library file not found: {data_path}")

    raw_df = pd.read_parquet(data_path)
    print(f"Loaded small library: {len(raw_df):,} rows", flush=True)

    # Feature preprocessing
    processed_cache = "data/small_library/pe6to7_processed.parquet"
    if os.path.exists(processed_cache):
        print(f"Loading pre-computed biofeatures from {processed_cache}...", flush=True)
        df = pd.read_parquet(processed_cache)
    else:
        print("Computing biofeatures for small library...", flush=True)
        df = preprocess_data(data=raw_df)
        df.to_parquet(processed_cache, index=False)

    norm_mean = pd.read_csv("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/mean.csv", index_col=0, header=None).squeeze()
    norm_std = pd.read_csv("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/std.csv", index_col=0, header=None).squeeze()

    bio_df = df.rename(columns=FEATURE_RENAME_MAP)
    b_norm = (bio_df[FEATURE_NAMES] - norm_mean) / norm_std
    b_tensor = torch.from_numpy(b_norm.fillna(0).values).float().to(device)

    wt_seqs = df["Target"].tolist() if "Target" in df.columns else df["WideTargetSequence"].tolist()
    ed_seqs = df["Masked_EditSeq"].tolist() if "Masked_EditSeq" in df.columns else wt_seqs

    # Load models
    dp_base_models = load_deepprime_base_models("src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428", device)
    ckpt_dir = "src/models/weights/DeepPrime6-weights"
    dp6_models = {
        "PEmaxdRNaseH": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pemaxdrnaseh_mainft.ckpt"), strict=False).to(device).eval(),
        "PE6a": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6a_mainft.ckpt"), strict=False).to(device).eval(),
        "PE6b": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6b_mainft.ckpt"), strict=False).to(device).eval(),
        "PE6c": PE6DeepPrimeModule.load_from_checkpoint(os.path.join(ckpt_dir, "pe6c_mainft.ckpt"), strict=False).to(device).eval(),
    }

    # Predict
    num_samples = len(df)
    batch_size = 2048
    preds_dp_base = np.zeros(num_samples, dtype=np.float32)
    preds_dp6 = {k: np.zeros(num_samples, dtype=np.float32) for k in dp6_models}

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            g_batch = char_to_onehot_tensor(wt_seqs[start_idx:end_idx], ed_seqs[start_idx:end_idx], device)
            b_batch = b_tensor[start_idx:end_idx]

            preds_dp_base[start_idx:end_idx] = predict_deepprime_base(dp_base_models, g_batch, b_batch).cpu().numpy().flatten()
            for k, m in dp6_models.items():
                preds_dp6[k][start_idx:end_idx] = m((g_batch, b_batch)).cpu().numpy().flatten()

    df["Pred_DeepPrime_Base"] = preds_dp_base
    for k, v in preds_dp6.items():
        df[f"Pred_DeepPrime6_{k}"] = v

    def resolve_col(candidates):
        if isinstance(candidates, str):
            candidates = [candidates]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Compare on PE6 and PE7 measured averages
    eval_pairs = [
        ("PEmaxdRNaseH (PE6-2)", ["PEmaxdRNaseH-avg", "PE6-2avg"], "PEmaxdRNaseH"),
        ("PE6b (PE6-3)", ["PE6b-avg", "PE6-3avg"], "PE6b"),
        ("PE6c (PE6-4)", ["PE6c-avg", "PE6-4avg"], "PE6c"),
        ("PE7maxdRNaseH (PE7-2)", ["PE7maxdRNaseH-avg", "PE7-2avg"], "PEmaxdRNaseH"),
        ("PE7b (PE7-3)", ["PE7b-avg", "PE7-3avg"], "PE6b"),
        ("PE7c (PE7-4)", ["PE7c-avg", "PE7-4avg"], "PE6c"),
    ]

    results = []
    valid_eval_pairs = []
    for test_label, meas_candidates, model_key in eval_pairs:
        col = resolve_col(meas_candidates)
        if col is not None:
            valid_eval_pairs.append((test_label, col, model_key))
            y_meas = df[col].values
            r_base, rho_base, _, n = compute_metrics(y_meas, preds_dp_base)
            r_dp6, rho_dp6, _, _ = compute_metrics(y_meas, preds_dp6[model_key])

            results.append({
                "Evaluation Target": test_label,
                "Model Evaluated": model_key,
                "DeepPrime-base (Pearson r)": round(r_base, 4),
                "DeepPrime-base (Spearman rho)": round(rho_base, 4),
                "DeepPrime6 (Pearson r)": round(r_dp6, 4),
                "DeepPrime6 (Spearman rho)": round(rho_dp6, 4),
                "Delta Pearson r": round(r_dp6 - r_base, 4),
                "N (Valid)": n
            })

    df_res = pd.DataFrame(results)
    print("\n--- Small Library (PE6 & PE7) Benchmark Results ---", flush=True)
    print(df_res.to_string(index=False), flush=True)

    out_csv = "figures/evaluation/Small_library_orthogonal_benchmark.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved summary to {out_csv}", flush=True)

    # Plotting
    sns.set_theme(style="ticks", font_scale=1.1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    axes = axes.flatten()

    for idx, (test_label, col, model_key) in enumerate(valid_eval_pairs):
        ax = axes[idx]
        yt = df[col].values
        yp = preds_dp6[model_key]
        r, rho, _, n = compute_metrics(yt, yp)

        sns.regplot(
            x=yt, y=yp, ax=ax,
            scatter_kws={"alpha": 0.4, "s": 20, "color": "#1f77b4" if "PE6" in test_label else "#d62728"},
            line_kws={"color": "black", "linewidth": 2.0}
        )
        ax.set_title(f"{test_label}\nDeepPrime6 {model_key} ($r={r:.3f}, \\rho={rho:.3f}, N={n}$)", fontsize=13, weight="bold")
        ax.set_xlabel("Measured Average Efficiency (%)", fontsize=11)
        ax.set_ylabel("DeepPrime6 Predicted (%)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Small Library Benchmark: DeepPrime6 Accuracy Across PE6 & PE7 Systems", fontsize=16, weight="bold", y=0.99)
    plt.tight_layout()

    out_png = "figures/evaluation/Small_library_orthogonal_benchmark.png"
    out_svg = "figures/evaluation/Small_library_orthogonal_benchmark.svg"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved plots to:\n  - {out_png}\n  - {out_svg}", flush=True)

if __name__ == "__main__":
    main()
