#!/usr/bin/env python3
"""
Cross-Benchmark 4-Model Evaluation & Aggregated Pearson Correlation Heatmap.

Evaluates 4 model families (8 prediction columns):
  1. DP-PE2        : DeepPrime (PE2, HEK293T, 5-model ensemble, exp(pred) - 1)
  2. DP-FT         : DeepPrime-FT (PE2max + epegRNA opti, 20-model ensemble, exp(pred) - 1)
  3. PRIDICT2.0    : PRIDICT2.0 (Official Fold 0 HEK293T score * 100)
  4. DeepPrime6    : 4 fine-tuned PE6 models + ensemble average
       - DP6-dRNaseH (pemaxdrnaseh_mainft.ckpt)
       - PE6a        (pe6a_mainft.ckpt)
       - PE6b        (pe6b_mainft.ckpt)
       - PE6c        (pe6c_mainft.ckpt)
       - DP6-avg     (mean of the 4 DeepPrime6 predictions)

Across 7 strictly target-stratified / held-out test datasets (in 3 groups):
  Group 1 (DeepPrime Cell paper test sets, target-leakage filtered):
    - DP, PE2                  (n = 7,243)
    - DP-FT (PE2max-epegRNA)   (n = 389)
  Group 2 (PRIDICT2.0 Fold 0 test set, target-leakage filtered):
    - PRIDICT2.0               (n = 4,311)
  Group 3 (DeepPrime6 Main Library deduplicated held-out test set, QC-passed):
    - DP6-PEmax-dRNaseH        (n = 1,539)
    - DP6-PE6a                 (n = 1,520)
    - DP6-PE6b                 (n = 1,556)
    - DP6-PE6c                 (n = 1,500)

Outputs:
  - Per-testset unified prediction tables (CSV)
  - Per-model-family prediction tables (4 CSVs across all test datasets)
  - Aggregated Pearson correlation matrix CSV + detailed Pearson/Spearman/N CSV
  - Publication-ready grouped heatmap (PNG & SVG)
"""

import glob
import os
import shutil
import sys
from typing import Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.matrix as sm
import torch
from Bio.Seq import reverse_complement
from scipy.stats import pearsonr, spearmanr

# Patch Seaborn 0.12.2 annotation truncation bug on Matplotlib >= 3.8
def _patched_annotate_heatmap(self, ax, mesh):
    mesh.update_scalarmappable()
    height, width = self.annot_data.shape
    xpos, ypos = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    array = mesh.get_array()
    array_iter = array.flat if hasattr(array, "flat") else np.ravel(array)
    for x, y, m, color, val in zip(
        xpos.flat, ypos.flat, array_iter, mesh.get_facecolors(), self.annot_data.flat
    ):
        if m is not np.ma.masked:
            lum = sm.relative_luminance(color)
            text_color = ".15" if lum > 0.408 else "w"
            annotation = ("{:" + self.fmt + "}").format(val)
            text_kwargs = dict(color=text_color, ha="center", va="center")
            if self.annot_kws:
                text_kwargs.update(self.annot_kws)
            ax.text(x, y, annotation, **text_kwargs)


sm._HeatMapper._annotate_heatmap = _patched_annotate_heatmap


def find_project_root() -> str:
    """Finds the repository root directory containing src/ and PRIDICT2/."""
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, ".project-root")) or (
            os.path.exists(os.path.join(curr, "src"))
            and os.path.exists(os.path.join(curr, "PRIDICT2"))
        ):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.getcwd())


PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
PRIDICT2_DIR = os.path.join(PROJECT_ROOT, "PRIDICT2")
if PRIDICT2_DIR not in sys.path:
    sys.path.insert(0, PRIDICT2_DIR)

from genet.models import LoadModel
from pridict.pridictv2.dataset import get_stratified_partitions
from src.utils.dataprep import IDX_TO_PE_TYPE, get_pe6_qc_mask
from src.utils.inference import (
    FEATURE_RENAME_MAP,
    char_to_onehot_tensor,
    load_deepprime6_models,
    load_deepprime_base_models,
    predict_deepprime6,
    predict_deepprime_base,
    prepare_biofeatures,
)
from src.utils.predict_pridict2 import (
    compute_average_predictions,
    deeppridict,
    extract_single_pegRNA_features,
    load_pridict_model,
)

OUT_DIR = "data/orthogonal_testset/cross_benchmark_4models"
PLOT_MIRROR_DIRS = [
    "experiments/20261009_orthogonal_benchmark_leakage_stratification/panels",
    "notebooks/plots/orthogonal_benchmark",
    "figures/evaluation",
]

PRED_COLS = [
    "pred_dp_pe2",
    "pred_dp_ft",
    "pred_pridict2",
    "pred_dp6_drnaseh",
    "pred_dp6_pe6a",
    "pred_dp6_pe6b",
    "pred_dp6_pe6c",
    "pred_dp6_avg",
]

MODEL_DISPLAY_COLS = [
    "DP-PE2",
    "DP-FT",
    "PRIDICT2.0",
    "DP6-dRNaseH",
    "PE6a",
    "PE6b",
    "PE6c",
    "DP6-avg",
]

PRIDICT2_DEEP_COLS = [
    "wide_initial_target",
    "wide_mutated_target",
    "deepeditposition",
    "deepeditposition_lst",
    "Correction_Type",
    "Correction_Length",
    "protospacerlocation_only_initial",
    "PBSlocation",
    "RT_initial_location",
    "RT_mutated_location",
    "RToverhangmatches",
    "RToverhanglength",
    "RTlength",
    "PBSlength",
    "RTmt",
    "RToverhangmt",
    "PBSmt",
    "protospacermt",
    "extensionmt",
    "original_base_mt",
    "edited_base_mt",
    "original_base_mt_nan",
    "edited_base_mt_nan",
]


def load_genet_pe2_models(device: torch.device) -> Tuple[list, pd.Series, pd.Series]:
    """Loads the 5-model DeepPrime (PE2, HEK293T) ensemble and its normalization stats."""
    mdir = LoadModel("DeepPrime", "PE2", "HEK293T").model_dir
    mean_s = (
        pd.read_csv(os.path.join(mdir, "mean_231124.csv"), header=None, index_col=0)
        .squeeze()
        .rename(index=FEATURE_RENAME_MAP)
    )
    std_s = (
        pd.read_csv(os.path.join(mdir, "std_231124.csv"), header=None, index_col=0)
        .squeeze()
        .rename(index=FEATURE_RENAME_MAP)
    )
    models = load_deepprime_base_models(weights_dir=mdir, device=device)
    return models, mean_s, std_s


def build_strict_leakage_target_sets() -> Tuple[set, List[str], set, set]:
    """Constructs the exact target-locus exclusion sets matching 01_filter_testsets.py."""
    pe6_full = pd.read_csv(
        "data/library_design/pairwise_library_design/FULL_LIBRARY/results/pe6_full_library.csv"
    )
    small_lib_path = "data/small_library/pe6to7_with_avgs.parquet"
    pe6_small = pd.read_parquet(small_lib_path) if os.path.exists(small_lib_path) else pd.DataFrame()

    pe6_targets_74 = set()
    pe6_non74 = []
    for s in pe6_full["WideTargetSequence"].dropna().str.upper():
        if len(s) == 74:
            pe6_targets_74.add(s)
            pe6_targets_74.add(reverse_complement(s))
        else:
            pe6_non74.append(s)
            pe6_non74.append(reverse_complement(s))
    if not pe6_small.empty and "Target" in pe6_small.columns:
        for s in pe6_small["Target"].dropna().str.upper():
            if len(s) == 74:
                pe6_targets_74.add(s)
                pe6_targets_74.add(reverse_complement(s))

    dp_all = pd.read_csv(
        "data/library_design/pairwise_library_design/OnTarget/fetching_from_DPlt=3nt;6K/src/Table S2-1.csv",
        low_memory=False,
    )
    col_wts_dp = [c for c in dp_all.columns if c.startswith("Wide target sequence")][0]
    dp_train_targets_74 = set()
    for s in dp_all[dp_all["Data set name"] == "Train"][col_wts_dp].str.upper():
        dp_train_targets_74.add(s)
        dp_train_targets_74.add(reverse_complement(s))

    pr_df = pd.read_csv("PRIDICT2/dataset/proc_v2/data_23k_v1.csv")
    dparts = get_stratified_partitions(
        pr_df["grp_id"].values, num_folds=5, valid_set_portion=0.1, random_state=42
    )
    p_train_indices = list(dparts[0]["train"]) + list(dparts[0]["validation"])
    pr_train_74 = set()
    for t in pr_df.iloc[p_train_indices]["wide_initial_target"].str.upper():
        for i in range(len(t) - 74 + 1):
            k = t[i : i + 74]
            pr_train_74.add(k)
            pr_train_74.add(reverse_complement(k))

    return pe6_targets_74, pe6_non74, dp_train_targets_74, pr_train_74


def prepare_filtered_pe2max_e_testset() -> pd.DataFrame:
    """Loads Table S2 Sheet 7 (PE2max-e, n=589) and applies strict target-locus filtering."""
    cached_path = os.path.join(OUT_DIR, "pe2max_e_test_filtered.csv")
    if os.path.exists(cached_path):
        df = pd.read_csv(cached_path)
        print(f"  Loaded cached filtered PE2max-e testset: {len(df):,} rows from {cached_path}", flush=True)
        return df

    src_parquet = "data/orthogonal_testset/supp_fig7_testsets/pe2max_e_test_589.parquet"
    df_raw = pd.read_parquet(src_parquet).reset_index(drop=True)
    wt_col = [c for c in df_raw.columns if c.startswith("Wide target sequence")][0]
    ed_col = [c for c in df_raw.columns if c.startswith("Edited target sequence")][0]

    df_raw["ID"] = [f"DP_PE2MAX_E_TEST_{i}" for i in range(len(df_raw))]
    df_raw["WTS_clean"] = df_raw[wt_col].str.upper()
    df_raw["ETS_clean"] = df_raw[ed_col].str.upper()

    def _edit_type(r):
        if r["type_sub"] == 1:
            return "sub"
        if r["type_ins"] == 1:
            return "ins"
        if r["type_del"] == 1:
            return "del"
        return "sub"

    df_raw["Edit_type_clean"] = df_raw.apply(_edit_type, axis=1)

    pe6_targets_74, pe6_non74, dp_train_targets_74, pr_train_74 = build_strict_leakage_target_sets()
    pe6_short = {s for s in pe6_non74 if len(s) < 74}

    in_pe6 = df_raw["WTS_clean"].isin(pe6_targets_74)
    in_pe6_non74 = df_raw["WTS_clean"].map(lambda w: any(s in w for s in pe6_short))
    in_dp_train = df_raw["WTS_clean"].isin(dp_train_targets_74)
    in_pr_train = df_raw["WTS_clean"].isin(pr_train_74)

    leak_mask = in_pe6 | in_pe6_non74 | in_dp_train | in_pr_train
    df_filt = df_raw[~leak_mask].copy().reset_index(drop=True)
    print(
        f"  Filtered PE2max-e testset: {len(df_raw):,} -> {len(df_filt):,} "
        f"(excluded {leak_mask.sum():,})",
        flush=True,
    )
    df_filt.to_csv(cached_path, index=False)
    return df_filt


def run_pridict2_from_74nt_df(df: pd.DataFrame) -> np.ndarray:
    """Reconstructs pegRNA inputs from 74bp WT/Edited sequences and runs PRIDICT2.0 (HEK * 100)."""
    pr_models = load_pridict_model(run_ids=[0])
    features_list = []
    failed_indices = set()

    for row_idx, (_, r) in enumerate(df.iterrows()):
        wt = str(r["WTS_clean"]).upper()
        ed = str(r["ETS_clean"]).upper()
        pbs_len = int(r["PBSlen"])
        rt_len = int(r["RTlen"])
        edit_len = int(r["Edit_len"])
        etype = str(r["Edit_type_clean"]).lower()
        if etype == "sub":
            wt_span = rt_len
        elif etype == "ins":
            wt_span = rt_len - edit_len
        elif etype == "del":
            wt_span = rt_len + edit_len
        else:
            raise ValueError(f"Unsupported Edit_type: {etype}")

        nick = 21
        pbs = reverse_complement(wt[nick - pbs_len : nick])
        rtt = reverse_complement(ed[nick : nick + rt_len])
        edited = wt[:nick] + ed[nick : nick + rt_len] + wt[nick + wt_span :]
        guide = "G" + wt[5:24]

        pr_row = {
            "REF_ID": r["ID"],
            "WildTypeSequence": wt,
            "PrimeEditedSequence": edited,
            "Edit_type": etype.capitalize(),
            "Guide": guide,
            "PBS": pbs,
            "RTT": rtt,
        }
        try:
            feat = extract_single_pegRNA_features(pr_row)
            if feat is not None:
                features_list.append(feat)
            else:
                failed_indices.add(row_idx)
                features_list.append({"sequence_name": r["ID"]})
        except Exception:
            failed_indices.add(row_idx)
            features_list.append({"sequence_name": r["ID"]})

    pr_peg_df = pd.DataFrame(features_list)
    for col in PRIDICT2_DEEP_COLS:
        if col not in pr_peg_df.columns:
            pr_peg_df[col] = 0

    pr_preds_res = deeppridict(pr_peg_df, pr_models)
    tmp_df = pd.concat([pr_preds_res[m] for m in pr_preds_res], axis=0, ignore_index=True)
    agg_df = compute_average_predictions(tmp_df, grp_cols=["seq_id", "dataset_name"])
    hek_df = agg_df[agg_df["dataset_name"] == "HEK"].sort_values("seq_id").reset_index(drop=True)
    scores = hek_df["pred_averageedited"].to_numpy(dtype=np.float64) * 100.0
    if failed_indices:
        for idx in failed_indices:
            scores[idx] = np.nan
    return scores


def run_dp_models_for_df(
    wt_seqs: List[str],
    ed_seqs: List[str],
    bio_df: pd.DataFrame,
    dp_pe2_bundle: Tuple[list, pd.Series, pd.Series],
    dp_ft_models: list,
    dp6_models: dict,
    device: torch.device,
    batch_size: int = 2048,
) -> Dict[str, np.ndarray]:
    """Runs DP-PE2 (exp-1), DP-FT (exp-1), and 4 DeepPrime6 models + DP6-avg."""
    pe2_models, mean_pe2, std_pe2 = dp_pe2_bundle
    g_tensor = char_to_onehot_tensor(wt_seqs, ed_seqs, device)

    b_pe2 = prepare_biofeatures(bio_df, mean_pe2, std_pe2, device=device)
    b_ft = prepare_biofeatures(bio_df, "base", device=device)
    b_pe6 = prepare_biofeatures(bio_df, "pe6", device=device)

    raw_pe2 = predict_deepprime_base(pe2_models, g_tensor, b_pe2, batch_size=batch_size)
    raw_ft = predict_deepprime_base(dp_ft_models, g_tensor, b_ft, batch_size=batch_size)
    preds_dp6 = predict_deepprime6(dp6_models, g_tensor, b_pe6, batch_size=batch_size)

    exp_pe2 = np.exp(raw_pe2.astype(np.float64)) - 1.0
    exp_ft = np.exp(raw_ft.astype(np.float64)) - 1.0

    dp6_drn = preds_dp6["drnaseh"].astype(np.float64)
    dp6_6a = preds_dp6["pe6a"].astype(np.float64)
    dp6_6b = preds_dp6["pe6b"].astype(np.float64)
    dp6_6c = preds_dp6["pe6c"].astype(np.float64)
    dp6_avg = (dp6_drn + dp6_6a + dp6_6b + dp6_6c) / 4.0

    return {
        "pred_dp_pe2_raw_log": raw_pe2.astype(np.float64),
        "pred_dp_pe2": exp_pe2,
        "pred_dp_ft_raw_log": raw_ft.astype(np.float64),
        "pred_dp_ft": exp_ft,
        "pred_dp6_drnaseh": dp6_drn,
        "pred_dp6_pe6a": dp6_6a,
        "pred_dp6_pe6b": dp6_6b,
        "pred_dp6_pe6c": dp6_6c,
        "pred_dp6_avg": dp6_avg,
    }


def compute_pearson_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, int]:
    """Computes Pearson r, Spearman rho, and valid N after NaN masking."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[mask]
    yp = y_pred[mask]
    n = int(len(yt))
    if n < 2:
        return np.nan, np.nan, n
    r_val, _ = pearsonr(yp, yt)
    rho_val, _ = spearmanr(yp, yt)
    return float(r_val), float(rho_val), n


def export_model_family_tables(dataset_frames: List[Tuple[str, pd.DataFrame, str]]) -> None:
    """Exports 4 separate prediction tables—one per model family—across all 7 test datasets."""
    common_records = []
    for ds_label, df, gt_col in dataset_frames:
        sub = df[df[gt_col].notna()].copy()
        part = pd.DataFrame(
            {
                "Test_dataset": ds_label,
                "ID": sub["ID"].astype(str).values,
                "Measured_PE_efficiency": sub[gt_col].astype(float).values,
                "pred_dp_pe2": sub["pred_dp_pe2"].astype(float).values,
                "pred_dp_pe2_raw_log": sub["pred_dp_pe2_raw_log"].astype(float).values,
                "pred_dp_ft": sub["pred_dp_ft"].astype(float).values,
                "pred_dp_ft_raw_log": sub["pred_dp_ft_raw_log"].astype(float).values,
                "pred_pridict2": sub["pred_pridict2"].astype(float).values,
                "pred_dp6_drnaseh": sub["pred_dp6_drnaseh"].astype(float).values,
                "pred_dp6_pe6a": sub["pred_dp6_pe6a"].astype(float).values,
                "pred_dp6_pe6b": sub["pred_dp6_pe6b"].astype(float).values,
                "pred_dp6_pe6c": sub["pred_dp6_pe6c"].astype(float).values,
                "pred_dp6_avg": sub["pred_dp6_avg"].astype(float).values,
            }
        )
        common_records.append(part)

    master_df = pd.concat(common_records, ignore_index=True)

    family_specs = {
        "predictions_model_DeepPrime_PE2.csv": [
            "Test_dataset",
            "ID",
            "Measured_PE_efficiency",
            "pred_dp_pe2",
            "pred_dp_pe2_raw_log",
        ],
        "predictions_model_DeepPrime_FT_epeg_opti.csv": [
            "Test_dataset",
            "ID",
            "Measured_PE_efficiency",
            "pred_dp_ft",
            "pred_dp_ft_raw_log",
        ],
        "predictions_model_PRIDICT2_0.csv": [
            "Test_dataset",
            "ID",
            "Measured_PE_efficiency",
            "pred_pridict2",
        ],
        "predictions_model_DeepPrime6_4models.csv": [
            "Test_dataset",
            "ID",
            "Measured_PE_efficiency",
            "pred_dp6_drnaseh",
            "pred_dp6_pe6a",
            "pred_dp6_pe6b",
            "pred_dp6_pe6c",
            "pred_dp6_avg",
        ],
    }

    for fname, cols in family_specs.items():
        fpath = os.path.join(OUT_DIR, fname)
        master_df[cols].to_csv(fpath, index=False)
        print(f"  Exported model-family prediction table: {fpath} ({len(master_df):,} rows)", flush=True)


def plot_grouped_heatmap(
    corr_df: pd.DataFrame,
    out_png: str,
    out_svg: str,
    cbar_label: str = "Pearson's correlation (r)",
    vmin_val: float = 0.35,
    vmax_val: float = 0.80,
) -> None:
    """Plots the 3-block grouped correlation heatmap matching the target layout."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    groups = [
        ["DP, PE2", "DP-FT (PE2max-epegRNA)"],
        ["PRIDICT2.0"],
        ["DP6-PEmax-dRNaseH", "DP6-PE6a", "DP6-PE6b", "DP6-PE6c"],
    ]

    corr_plot_df = corr_df.copy()
    corr_plot_df.index = [idx.replace(" (n =", "\n(n =") for idx in corr_df.index]
    row_lookup = {idx.split("\n")[0]: idx for idx in corr_plot_df.index}

    fig = plt.figure(figsize=(11.5, 6.8), dpi=300)
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=2,
        height_ratios=[2, 1, 4],
        width_ratios=[1.0, 0.035],
        hspace=0.14,
        wspace=0.05,
    )

    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    cax = fig.add_subplot(gs[:, 1])

    for g_idx, (ax, row_keys) in enumerate(zip(axes, groups)):
        full_keys = [row_lookup[k] for k in row_keys]
        sub_df = corr_plot_df.loc[full_keys, MODEL_DISPLAY_COLS]

        sns.heatmap(
            sub_df,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=vmin_val,
            vmax=vmax_val,
            cbar=(g_idx == 0),
            cbar_ax=(cax if g_idx == 0 else None),
            cbar_kws={"label": cbar_label} if g_idx == 0 else None,
            linewidths=0.6,
            linecolor="#E2E8F0",
            ax=ax,
            annot_kws={"size": 11, "weight": "semibold"},
        )

        # Vertical group dividers matching DP-PE2 // DP-FT // PRIDICT2.0 // DP6-*
        for x_sep in [1, 2, 3]:
            ax.axvline(x_sep, color="#1A202C", linewidth=1.6)
        # Subtle dashed separator before DP6-avg
        ax.axvline(7, color="#4A5568", linewidth=1.0, linestyle="--")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.1)

        ax.set_ylabel("")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10.5)

        if g_idx < 2:
            ax.set_xticks([])
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Prediction model", fontsize=13, labelpad=10, fontweight="bold")
            ax.set_xticklabels(MODEL_DISPLAY_COLS, rotation=32, ha="right", fontsize=11)

    cax.tick_params(labelsize=10.5)
    cax.set_ylabel(cbar_label, fontsize=12, labelpad=10)
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)

    fig.text(
        0.015,
        0.52,
        "Test dataset",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=13,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.26, right=0.90, top=0.96, bottom=0.18)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap: {out_png} & {out_svg}", flush=True)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running Cross-Benchmark 4-Model Pipeline on device: {device}", flush=True)

    # 1. Load models
    print("\n--- 1. Loading Models ---", flush=True)
    dp_pe2_bundle = load_genet_pe2_models(device)
    dp_ft_models = load_deepprime_base_models(device=device)
    dp6_models = load_deepprime6_models(device=device, is_scratch=False)

    # ---------------------------------------------------------
    # 2. Dataset 1: DP, PE2 (deepprime_test_filtered.csv, n=7,243)
    # ---------------------------------------------------------
    print("\n--- 2. Dataset 1: DP, PE2 (n=7,243) ---", flush=True)
    dp_filt = pd.read_csv("data/orthogonal_testset/deepprime_test_filtered.csv")
    dp_existing = pd.read_csv("data/orthogonal_testset/deepprime_test_predictions.csv")
    assert (dp_filt["ID"].values == dp_existing["ID"].values).all(), "ID mismatch in deepprime_test!"

    dp_res = run_dp_models_for_df(
        dp_filt["WTS_clean"].tolist(),
        dp_filt["ETS_clean"].tolist(),
        dp_filt,
        dp_pe2_bundle,
        dp_ft_models,
        dp6_models,
        device,
    )
    # Verify exact parity with cached DP-PE2 and DP6 predictions
    assert np.allclose(dp_res["pred_dp_pe2"], dp_existing["pred_dp_base"].values, atol=1e-4)
    assert np.allclose(dp_res["pred_dp6_pe6a"], dp_existing["pred_dp6_pe6a"].values, atol=1e-4)

    dp_table = dp_existing[
        [
            "ID",
            "WTS_clean",
            "ETS_clean",
            "Edit_type_clean",
            "Edit_pos",
            "Edit_len",
            "PBSlen",
            "RTlen",
            "Measured_PE_efficiency",
        ]
    ].copy()
    for col, vals in dp_res.items():
        dp_table[col] = vals
    dp_table["pred_pridict2"] = dp_existing["pred_pridict2_hek"].astype(np.float64).values
    dp_table_path = os.path.join(OUT_DIR, "pred_table_testset_dp_pe2_n7243.csv")
    dp_table.to_csv(dp_table_path, index=False)
    print(f"  Saved: {dp_table_path} ({len(dp_table):,} rows)", flush=True)

    # ---------------------------------------------------------
    # 3. Dataset 2: DP-FT (PE2max-epegRNA, filtered n=389)
    # ---------------------------------------------------------
    print("\n--- 3. Dataset 2: DP-FT (PE2max-epegRNA, filtered n=389) ---", flush=True)
    pe2max_e_df = prepare_filtered_pe2max_e_testset()
    pe2max_e_res = run_dp_models_for_df(
        pe2max_e_df["WTS_clean"].tolist(),
        pe2max_e_df["ETS_clean"].tolist(),
        pe2max_e_df,
        dp_pe2_bundle,
        dp_ft_models,
        dp6_models,
        device,
    )

    pe2max_e_table_path = os.path.join(OUT_DIR, "pred_table_testset_dp_ft_pe2max_e_n389.csv")
    if os.path.exists(pe2max_e_table_path):
        cached_e = pd.read_csv(pe2max_e_table_path)
        if len(cached_e) == len(pe2max_e_df) and "pred_pridict2" in cached_e.columns and cached_e["pred_pridict2"].notna().all():
            pr_scores_e = cached_e["pred_pridict2"].to_numpy(dtype=np.float64)
            print("  Reused cached PRIDICT2.0 scores for PE2max-e.", flush=True)
        else:
            pr_scores_e = run_pridict2_from_74nt_df(pe2max_e_df)
    else:
        pr_scores_e = run_pridict2_from_74nt_df(pe2max_e_df)

    pe2max_e_table = pe2max_e_df[
        [
            "ID",
            "WTS_clean",
            "ETS_clean",
            "Edit_type_clean",
            "Edit_pos",
            "Edit_len",
            "PBSlen",
            "RTlen",
            "Measured_PE_efficiency",
        ]
    ].copy()
    for col, vals in pe2max_e_res.items():
        pe2max_e_table[col] = vals
    pe2max_e_table["pred_pridict2"] = pr_scores_e
    pe2max_e_table.to_csv(pe2max_e_table_path, index=False)
    print(f"  Saved: {pe2max_e_table_path} ({len(pe2max_e_table):,} rows)", flush=True)

    # ---------------------------------------------------------
    # 4. Dataset 3: PRIDICT2.0 (pridict2_test_filtered.csv, n=4,311)
    # ---------------------------------------------------------
    print("\n--- 4. Dataset 3: PRIDICT2.0 (n=4,311) ---", flush=True)
    pr_existing = pd.read_csv("data/orthogonal_testset/pridict2_test_predictions.csv")
    pr_bio = pd.read_parquet("data/orthogonal_testset/pridict2_test_dp_biofeatures.parquet")
    assert len(pr_existing) == len(pr_bio), "Row mismatch in pridict2_test!"

    pr_res = run_dp_models_for_df(
        pr_bio["Target"].str.upper().tolist(),
        pr_bio["Masked_EditSeq"].str.upper().tolist(),
        pr_bio,
        dp_pe2_bundle,
        dp_ft_models,
        dp6_models,
        device,
    )
    assert np.allclose(pr_res["pred_dp_pe2"], pr_existing["pred_dp_base"].values, atol=1e-4)
    assert np.allclose(pr_res["pred_dp6_pe6a"], pr_existing["pred_dp6_pe6a"].values, atol=1e-4)

    pr_table = pr_existing[
        [
            "ID",
            "seq_id",
            "grp_id",
            "Correction_Type",
            "Correction_Length",
            "deepeditposition",
            "PBSlength",
            "RTlength",
            "Measured_PE_efficiency",
        ]
    ].copy()
    for col, vals in pr_res.items():
        pr_table[col] = vals
    pr_table["pred_pridict2"] = pr_existing["pred_pridict2_hek"].astype(np.float64).values
    pr_table_path = os.path.join(OUT_DIR, "pred_table_testset_pridict2_n4311.csv")
    pr_table.to_csv(pr_table_path, index=False)
    print(f"  Saved: {pr_table_path} ({len(pr_table):,} rows)", flush=True)

    # ---------------------------------------------------------
    # 5. Datasets 4-7: DeepPrime6 Main Library (rq3_test_library_with_pridict2.parquet)
    # ---------------------------------------------------------
    print("\n--- 5. Datasets 4-7: DeepPrime6 Main Library (QC-filtered per domain) ---", flush=True)
    rq3 = pd.read_parquet("data/main_library/test_split/rq3_test_library_with_pridict2.parquet")
    rq3_res = run_dp_models_for_df(
        rq3["Target"].str.upper().tolist(),
        rq3["Masked_EditSeq"].str.upper().tolist(),
        rq3,
        dp_pe2_bundle,
        dp_ft_models,
        dp6_models,
        device,
    )
    assert np.allclose(rq3_res["pred_dp_pe2"], rq3["pred_dp_base"].values, atol=1e-4)
    assert np.allclose(rq3_res["pred_dp6_pe6a"], rq3["pred_dp6_pe6a"].values, atol=1e-4)

    # Apply canonical 2-rep mean and strict QC mask matching 01_evaluate_figure3_main_library.py
    pe6_domains = [
        ("DP6-PEmax-dRNaseH", 2, "GT_PEmax_dRNaseH"),
        ("DP6-PE6a", 3, "GT_PE6a"),
        ("DP6-PE6b", 4, "GT_PE6b"),
        ("DP6-PE6c", 5, "GT_PE6c"),
    ]
    rq3_table = pd.DataFrame(
        {
            "ID": rq3["ID"].astype(str).values if "ID" in rq3.columns else rq3["REF_ID"].astype(str).values,
            "Target": rq3["Target"].astype(str).values,
            "Masked_EditSeq": rq3["Masked_EditSeq"].astype(str).values,
            "Edit_type": rq3["Edit_type"].astype(str).values,
            "Edit_len": rq3["Edit_len"].values if "Edit_len" in rq3.columns else rq3["Edit length"].values,
        }
    )
    for ds_name, m_idx, gt_col in pe6_domains:
        r2 = f"Normalized+HEK-M-{m_idx}-7D-B1+pe_ratio_%"
        r3 = f"Normalized+HEK-M2-{m_idx}-7D-B1+pe_ratio_%"
        reps = [c for c in [r2, r3] if c in rq3.columns]
        gt_vals = rq3[reps].clip(lower=0.0).mean(axis=1, skipna=False)
        qc_mask = get_pe6_qc_mask(rq3, pe_type=IDX_TO_PE_TYPE[m_idx], read_count_filter=200, max_ctrl_bg=5.0)
        gt_vals[~qc_mask] = np.nan
        rq3_table[gt_col] = gt_vals.to_numpy(dtype=np.float64)

    for col, vals in rq3_res.items():
        rq3_table[col] = vals
    rq3_table["pred_pridict2"] = rq3["pred_pridict2_hek"].to_numpy(dtype=np.float64)
    rq3_table_path = os.path.join(OUT_DIR, "pred_table_testset_dp6_main_library_n2325.csv")
    rq3_table.to_csv(rq3_table_path, index=False)
    print(f"  Saved: {rq3_table_path} ({len(rq3_table):,} rows)", flush=True)

    # ---------------------------------------------------------
    # 6. Export 4 Separate Model-Family Prediction Tables
    # ---------------------------------------------------------
    print("\n--- 6. Exporting 4 Model-Family Prediction Tables ---", flush=True)
    dataset_specs = [
        ("DP, PE2", dp_table, "Measured_PE_efficiency"),
        ("DP-FT (PE2max-epegRNA)", pe2max_e_table, "Measured_PE_efficiency"),
        ("PRIDICT2.0", pr_table, "Measured_PE_efficiency"),
        ("DP6-PEmax-dRNaseH", rq3_table, "GT_PEmax_dRNaseH"),
        ("DP6-PE6a", rq3_table, "GT_PE6a"),
        ("DP6-PE6b", rq3_table, "GT_PE6b"),
        ("DP6-PE6c", rq3_table, "GT_PE6c"),
    ]
    export_model_family_tables(dataset_specs)

    # ---------------------------------------------------------
    # 7. Compute Pearson & Spearman Correlation Aggregation & Heatmaps
    # ---------------------------------------------------------
    print("\n--- 7. Aggregating Pearson & Spearman Correlations ---", flush=True)
    pearson_rows = []
    spearman_rows = []
    detailed_rows = []

    for ds_label, df_ds, gt_col in dataset_specs:
        y_true = df_ds[gt_col].to_numpy(dtype=np.float64)
        valid_gt_n = int(np.sum(~np.isnan(y_true)))
        row_label = f"{ds_label} (n = {valid_gt_n:,})"
        p_row = {"Test dataset": row_label, "Test_dataset_raw": ds_label, "N": valid_gt_n}
        s_row = {"Test dataset": row_label, "Test_dataset_raw": ds_label, "N": valid_gt_n}

        for p_col, disp_name in zip(PRED_COLS, MODEL_DISPLAY_COLS):
            y_pred = df_ds[p_col].to_numpy(dtype=np.float64)
            r_val, rho_val, n_val = compute_pearson_spearman(y_true, y_pred)
            p_row[disp_name] = r_val
            s_row[disp_name] = rho_val
            detailed_rows.append(
                {
                    "Test_dataset": ds_label,
                    "Model": disp_name,
                    "Prediction_column": p_col,
                    "Pearson_r": round(r_val, 6),
                    "Spearman_rho": round(rho_val, 6),
                    "N": n_val,
                }
            )
        pearson_rows.append(p_row)
        spearman_rows.append(s_row)

    hm_df = pd.DataFrame(pearson_rows)
    hm_csv_path = os.path.join(OUT_DIR, "cross_benchmark_pearson_correlation_matrix.csv")
    hm_df.to_csv(hm_csv_path, index=False)

    sp_df = pd.DataFrame(spearman_rows)
    sp_csv_path = os.path.join(OUT_DIR, "cross_benchmark_spearman_correlation_matrix.csv")
    sp_df.to_csv(sp_csv_path, index=False)

    det_df = pd.DataFrame(detailed_rows)
    det_csv_path = os.path.join(OUT_DIR, "cross_benchmark_metrics_detailed.csv")
    det_df.to_csv(det_csv_path, index=False)

    for title, df_mat in [("PEARSON (r)", hm_df), ("SPEARMAN (rho)", sp_df)]:
        print(f"\n=== AGGREGATED {title} MATRIX ===")
        disp_table = df_mat[["Test_dataset_raw", "N"] + MODEL_DISPLAY_COLS].copy()
        for c in MODEL_DISPLAY_COLS:
            disp_table[c] = disp_table[c].map(lambda x: f"{x:.4f}")
        print(disp_table.to_string(index=False))

    out_png = os.path.join(OUT_DIR, "cross_benchmark_4models_pearson_heatmap.png")
    out_svg = os.path.join(OUT_DIR, "cross_benchmark_4models_pearson_heatmap.svg")
    plot_grouped_heatmap(
        hm_df.set_index("Test dataset")[MODEL_DISPLAY_COLS],
        out_png,
        out_svg,
        cbar_label="Pearson's correlation (r)",
        vmin_val=0.35,
        vmax_val=0.80,
    )

    out_sp_png = os.path.join(OUT_DIR, "cross_benchmark_4models_spearman_heatmap.png")
    out_sp_svg = os.path.join(OUT_DIR, "cross_benchmark_4models_spearman_heatmap.svg")
    plot_grouped_heatmap(
        sp_df.set_index("Test dataset")[MODEL_DISPLAY_COLS],
        out_sp_png,
        out_sp_svg,
        cbar_label="Spearman's correlation ($\\rho$)",
        vmin_val=0.40,
        vmax_val=0.85,
    )

    artifacts_to_mirror = [
        out_png,
        out_svg,
        out_sp_png,
        out_sp_svg,
        hm_csv_path,
        sp_csv_path,
        det_csv_path,
    ]
    for mirror_dir in PLOT_MIRROR_DIRS:
        os.makedirs(mirror_dir, exist_ok=True)
        for src_path in artifacts_to_mirror:
            shutil.copy2(src_path, os.path.join(mirror_dir, os.path.basename(src_path)))

    print("\n[Complete] All prediction tables, aggregated Pearson & Spearman correlations, and heatmaps generated!", flush=True)


if __name__ == "__main__":
    main()
