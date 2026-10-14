#!/usr/bin/env python3
"""
Unified DeepPrime & DeepPrime6 Canonical Inference Engine.

Single Source of Truth (SSOT) across all workflows for:
1. Biological feature schema (24 canonical features) and column name harmonization.
2. One-hot sequence tensor encoding for Target and Masked_EditSeq (range [-1, 1]).
3. Standardized domain biofeature scaling (Z-score normalization).
4. Model loaders with integrity verification:
   - DeepPrime-base (20-model ensemble with original architecture)
   - DeepPrime6 Fine-Tuned (4 models: pe6a, pe6b, pe6c, pemaxdrnaseh)
   - DeepPrime6 From-Scratch (4 models: pe6a, pe6b, pe6c, pemaxdrnaseh)
   - PRIDICT2.0 Deep Ensemble (HEK293T & K562 multi-task scores)
5. Vectorized, high-throughput batched GPU inference routines.
6. Comprehensive end-to-end DataFrame scoring pipeline.
"""

import os
import sys
import glob
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure project root is discoverable
def get_project_root() -> str:
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, ".project-root")) or (
            os.path.exists(os.path.join(curr, "src")) and os.path.exists(os.path.join(curr, "configs"))
        ):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.getcwd())

PROJECT_ROOT = get_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.pe6_module_dp_only import PE6DeepPrimeModule
from src.models.components.dprime import GeneInteractionModelVanilla

logger = logging.getLogger("src.utils.inference")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s][inference][%(levelname)s] - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ==============================================================================
# 1. Canonical Schema Constants & Configuration Loaders
# ==============================================================================

import yaml

def load_model_versions_config() -> Dict[str, Any]:
    """Loads config/model_versions.yaml if available, providing official SSOT model registry."""
    cfg_path = os.path.join(PROJECT_ROOT, "config", "model_versions.yaml")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not parse {cfg_path}: {e}")
    return {}

_MODEL_VER_CFG = load_model_versions_config()
_M_CFG = _MODEL_VER_CFG.get("models", {})

FEATURE_NAMES: List[str] = [
    "PBSlen", "RTlen", "RT-PBSlen", "Edit_pos", "Edit_len", "RHA_len",
    "type_sub", "type_ins", "type_del", "Tm1", "Tm2", "Tm2new", "Tm3", "Tm4", "TmD",
    "nGCcnt1", "nGCcnt2", "nGCcnt3", "fGCcont1", "fGCcont2", "fGCcont3",
    "MFE3", "MFE4", "DeepSpCas9_score"
]

FEATURE_RENAME_MAP: Dict[str, str] = {
    "PBS_len": "PBSlen",
    "RTT_len": "RTlen",
    "RT-PBS_len": "RT-PBSlen",
    "Tm1_PBS": "Tm1",
    "Tm2_RTT_cTarget_sameLength": "Tm2",
    "Tm3_RTT_cTarget_replaced": "Tm2new",
    "Tm4_cDNA_PAM-oppositeTarget": "Tm3",
    "Tm5_RTT_cDNA": "Tm4",
    "deltaTm_Tm4-Tm2": "TmD",
    "GC_count_PBS": "nGCcnt1",
    "GC_count_RTT": "nGCcnt2",
    "GC_count_RT-PBS": "nGCcnt3",
    "GC_contents_PBS": "fGCcont1",
    "GC_contents_RTT": "fGCcont2",
    "GC_contents_RT-PBS": "fGCcont3",
    "MFE_RT-PBS-polyT": "MFE3",
    "MFE_Spacer": "MFE4",
}

DEFAULT_DP_BASE_DIR = os.path.join(
    PROJECT_ROOT,
    _M_CFG.get("deepprime", {}).get("checkpoint", "src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428")
)
DEFAULT_DP6_FT_DIR = os.path.join(PROJECT_ROOT, "src/models/weights/DeepPrime6-weights")
DEFAULT_DP6_SCRATCH_DIR = os.path.join(PROJECT_ROOT, "src/models/weights/DeepPrime6-scratch-weights")

# ==============================================================================
# 2. Sequence & Feature Encoding
# ==============================================================================

def char_to_onehot_tensor(
    seqs_wt: List[str],
    seqs_ed: List[str],
    device: torch.device,
    seq_len: int = 74
) -> torch.Tensor:
    """Encodes WT and edited sequences into (N, 4, 2, seq_len) tensor normalized to [-1, 1]."""
    char_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    num_samples = len(seqs_wt)
    g_arr = np.zeros((num_samples, 4, 2, seq_len), dtype=np.float32)
    for idx, (wt, ed) in enumerate(zip(seqs_wt, seqs_ed)):
        wt_str = str(wt).upper()[:seq_len]
        ed_str = str(ed).upper()[:seq_len]
        for col_idx, ch in enumerate(wt_str):
            if ch in char_map:
                g_arr[idx, char_map[ch], 0, col_idx] = 1.0
        for col_idx, ch in enumerate(ed_str):
            if ch in char_map:
                g_arr[idx, char_map[ch], 1, col_idx] = 1.0
    return (2 * torch.from_numpy(g_arr) - 1).float().to(device)


def prepare_biofeatures(
    df: pd.DataFrame,
    norm_mean_or_type: Union[pd.Series, str] = "pe6",
    norm_std: Optional[Union[pd.Series, torch.device]] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Harmonizes column names and standardizes biofeatures to (N, 24) tensor.
    
    Supports:
      1. prepare_biofeatures(df, norm_mean_series, norm_std_series, device)
      2. prepare_biofeatures(df, norm_type="pe6"|"base", device=device)
    """
    # Extract exactly the 24 FEATURE_NAMES without risk of duplicate columns
    extracted_features = {}
    for f in FEATURE_NAMES:
        if f in df.columns:
            extracted_features[f] = df[f].values
        else:
            aliases = [k for k, target in FEATURE_RENAME_MAP.items() if target == f]
            found = False
            for a in aliases:
                if a in df.columns:
                    extracted_features[f] = df[a].values
                    found = True
                    break
            if not found:
                raise ValueError(f"Missing required biological feature: {f} (aliases: {aliases})")

    if isinstance(norm_mean_or_type, str):
        ntype = norm_mean_or_type.lower()
        if device is None and isinstance(norm_std, torch.device):
            device = norm_std
        if ntype in ["pe6", "dp6", "ft", "scratch"]:
            base_dir = DEFAULT_DP6_FT_DIR
            norm_mean = pd.read_csv(os.path.join(base_dir, "pe6_mean.csv"), index_col=0, header=None).squeeze()
            norm_std_s = pd.read_csv(os.path.join(base_dir, "pe6_std.csv"), index_col=0, header=None).squeeze()
        elif ntype in ["base", "dp_base"]:
            base_dir = DEFAULT_DP_BASE_DIR
            norm_mean = pd.read_csv(os.path.join(base_dir, "mean.csv"), index_col=0, header=None).squeeze()
            norm_std_s = pd.read_csv(os.path.join(base_dir, "std.csv"), index_col=0, header=None).squeeze()
        else:
            raise ValueError(f"Unknown norm_type: {norm_mean_or_type}")
    else:
        norm_mean = norm_mean_or_type
        norm_std_s = norm_std

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Standardize features using numpy broadcasting aligned to FEATURE_NAMES
    mean_vals = norm_mean[FEATURE_NAMES].astype(np.float32).values
    std_vals = norm_std_s[FEATURE_NAMES].astype(np.float32).values
    raw_vals = np.column_stack([extracted_features[f] for f in FEATURE_NAMES]).astype(np.float32)
    normed = (raw_vals - mean_vals) / std_vals
    normed = np.nan_to_num(normed, nan=0.0)
    return torch.from_numpy(normed).float().to(device)


# ==============================================================================
# 3. Model Loaders
# ==============================================================================

def load_deepprime_base_models(
    weights_dir: Optional[str] = None,
    device: Optional[torch.device] = None
) -> List[GeneInteractionModelVanilla]:
    """Loads all 20 pretrained checkpoints of DeepPrime-base (GeneInteractionModelVanilla)."""
    if weights_dir is None:
        weights_dir = DEFAULT_DP_BASE_DIR
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    files = sorted(glob.glob(os.path.join(weights_dir, "*.pt")))
    if not files:
        raise FileNotFoundError(f"No checkpoint files (.pt) found in {weights_dir}")

    models = []
    for f in files:
        m = GeneInteractionModelVanilla()
        sd = torch.load(f, map_location="cpu", weights_only=True)
        m.load_state_dict(sd, strict=True)
        models.append(m.to(device).eval())
    return models


def load_deepprime6_models(
    weights_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    is_scratch: bool = False
) -> Dict[str, PE6DeepPrimeModule]:
    """Loads DeepPrime6 models (pe6a, pe6b, pe6c, drnaseh)."""
    if weights_dir is None:
        weights_dir = DEFAULT_DP6_SCRATCH_DIR if is_scratch else DEFAULT_DP6_FT_DIR
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt_map = {
        "pe6a": "pe6a_scratch.ckpt" if is_scratch else "pe6a_mainft.ckpt",
        "pe6b": "pe6b_scratch.ckpt" if is_scratch else "pe6b_mainft.ckpt",
        "pe6c": "pe6c_scratch.ckpt" if is_scratch else "pe6c_mainft.ckpt",
        "drnaseh": "pemaxdrnaseh_scratch.ckpt" if is_scratch else "pemaxdrnaseh_mainft.ckpt",
    }

    models = {}
    for key, filename in ckpt_map.items():
        ckpt_path = os.path.join(weights_dir, filename)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        m = PE6DeepPrimeModule.load_from_checkpoint(ckpt_path, strict=False)
        models[key] = m.to(device).eval()
    return models


# ==============================================================================
# 4. Batched Inference Core
# ==============================================================================

def predict_deepprime_base(
    models: List[GeneInteractionModelVanilla],
    g_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    batch_size: int = 2048
) -> np.ndarray:
    """Runs 20-model ensemble inference on GPU in batches."""
    num_samples = len(g_tensor)
    preds = np.zeros(num_samples, dtype=np.float32)
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            g_b = g_tensor[start_idx:end_idx]
            b_b = b_tensor[start_idx:end_idx]
            # Average predictions across 20 base models
            batch_pred = torch.stack([m(g_b, b_b) for m in models], dim=0).mean(dim=0)
            preds[start_idx:end_idx] = batch_pred.cpu().numpy().flatten()
    return preds


def predict_deepprime6(
    models: Dict[str, PE6DeepPrimeModule],
    g_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    batch_size: int = 2048
) -> Dict[str, np.ndarray]:
    """Runs DeepPrime6 models (pe6a, pe6b, pe6c, drnaseh) in batches."""
    num_samples = len(g_tensor)
    preds_dict = {k: np.zeros(num_samples, dtype=np.float32) for k in models.keys()}
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            g_b = g_tensor[start_idx:end_idx]
            b_b = b_tensor[start_idx:end_idx]
            for key, m in models.items():
                batch_pred = m((g_b, b_b))
                preds_dict[key][start_idx:end_idx] = batch_pred.cpu().numpy().flatten()
    return preds_dict


def predict_pridict2_scores(
    df: pd.DataFrame,
    cell_types: List[str] = ["HEK", "K562"]
) -> Dict[str, np.ndarray]:
    """Runs PRIDICT2.0 official ensemble on pegRNA designs."""
    try:
        from src.utils.predict_pridict2 import (
            load_pridict_model,
            deeppridict,
            compute_average_predictions,
            extract_single_pegRNA_features
        )
    except ImportError:
        pridict_dir = os.path.join(PROJECT_ROOT, "PRIDICT2")
        if pridict_dir not in sys.path:
            sys.path.insert(0, pridict_dir)
        from src.utils.predict_pridict2 import (
            load_pridict_model,
            deeppridict,
            compute_average_predictions,
            extract_single_pegRNA_features
        )

    logger.info("Initializing PRIDICT2.0 model...")
    pr_models = load_pridict_model(run_ids=[0])

    features_list = []
    idx_map = []
    for pos_idx, (_, row) in enumerate(df.iterrows()):
        try:
            r_dict = row.to_dict()
            if "REF_ID" not in r_dict:
                r_dict["REF_ID"] = str(r_dict.get("ID", pos_idx))
            feat = extract_single_pegRNA_features(r_dict)
            if feat is not None:
                features_list.append(feat)
                idx_map.append(pos_idx)
        except Exception:
            pass

    results = {f"pred_pridict2_{ct.lower()}": np.full(len(df), np.nan, dtype=np.float32) for ct in cell_types}
    if not features_list:
        logger.warning("No valid pegRNA features could be extracted for PRIDICT2.0.")
        return results

    logger.info(f"Running PRIDICT2.0 deep inference on {len(features_list):,} sequences...")
    pegdataframe = pd.DataFrame(features_list)
    all_avg_preds = deeppridict(pegdataframe, pr_models)
    tmp = [all_avg_preds[m] for m in all_avg_preds]
    tmp_df = pd.concat(tmp, axis=0, ignore_index=True)
    agg_df = compute_average_predictions(tmp_df, grp_cols=["seq_id", "dataset_name"])

    for ct in cell_types:
        col_key = f"pred_pridict2_{ct.lower()}"
        cond = agg_df["dataset_name"] == ct
        scores = agg_df.loc[cond, "pred_averageedited"].values * 100.0
        # Map back to positional row indices
        for pos_idx, score in zip(idx_map, scores):
            results[col_key][pos_idx] = score

    return results


# ==============================================================================
# 5. High-Level Comprehensive Inference Pipeline
# ==============================================================================

def run_comprehensive_inference(
    df_or_path: Union[pd.DataFrame, str],
    include_base: bool = True,
    include_dp6_ft: bool = True,
    include_dp6_scratch: bool = False,
    include_pridict2: bool = False,
    device: Optional[torch.device] = None,
    batch_size: int = 2048,
    dp_base_dir: Optional[str] = None,
    dp6_ft_dir: Optional[str] = None,
    dp6_scratch_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Executes full multi-model inference on a given dataset.
    
    Accepts:
      - DataFrame or path to .parquet / .csv / .xlsm
    Returns:
      - Augmented DataFrame with standardized prediction columns:
        * pred_dp_base
        * pred_dp6_pe6a, pred_dp6_pe6b, pred_dp6_pe6c, pred_dp6_drnaseh
        * pred_scratch_pe6a, pred_scratch_pe6b, pred_scratch_pe6c, pred_scratch_drnaseh (optional)
        * pred_pridict2_hek, pred_pridict2_k562 (optional)
    """
    if isinstance(df_or_path, str):
        if df_or_path.endswith(".parquet"):
            df = pd.read_parquet(df_or_path)
        elif df_or_path.endswith(".csv"):
            df = pd.read_csv(df_or_path)
        elif df_or_path.endswith(".xlsm") or df_or_path.endswith(".xlsx"):
            df = pd.read_excel(df_or_path, sheet_name=0)
        else:
            raise ValueError(f"Unsupported file format: {df_or_path}")
    else:
        df = df_or_path.copy()

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting comprehensive inference on {len(df):,} sequences [Device: {device}]")

    # Resolve sequences
    wt_col = "Target" if "Target" in df.columns else ("WideTargetSequence" if "WideTargetSequence" in df.columns else None)
    if wt_col is None:
        raise ValueError("Could not find Target or WideTargetSequence column in input data.")
    ed_col = "Masked_EditSeq" if "Masked_EditSeq" in df.columns else wt_col

    wt_seqs = df[wt_col].astype(str).tolist()
    ed_seqs = df[ed_col].astype(str).tolist()
    g_tensor = char_to_onehot_tensor(wt_seqs, ed_seqs, device)

    # Resolve paths & normalization
    base_dir = dp_base_dir or DEFAULT_DP_BASE_DIR
    ft_dir = dp6_ft_dir or DEFAULT_DP6_FT_DIR
    scratch_dir = dp6_scratch_dir or DEFAULT_DP6_SCRATCH_DIR

    norm_mean_base = pd.read_csv(os.path.join(base_dir, "mean.csv"), index_col=0, header=None).squeeze()
    norm_std_base = pd.read_csv(os.path.join(base_dir, "std.csv"), index_col=0, header=None).squeeze()

    norm_mean_pe6 = pd.read_csv(os.path.join(ft_dir, "pe6_mean.csv"), index_col=0, header=None).squeeze()
    norm_std_pe6 = pd.read_csv(os.path.join(ft_dir, "pe6_std.csv"), index_col=0, header=None).squeeze()

    # 1. DeepPrime-base Inference
    if include_base:
        logger.info("Executing DeepPrime-base 20-ensemble inference...")
        b_base = prepare_biofeatures(df, norm_mean_base, norm_std_base, device)
        base_models = load_deepprime_base_models(base_dir, device)
        df["pred_dp_base"] = predict_deepprime_base(base_models, g_tensor, b_base, batch_size=batch_size)

    # 2. DeepPrime6 Fine-Tuning Inference
    if include_dp6_ft:
        logger.info("Executing DeepPrime6 (Fine-Tuning) inference...")
        b_pe6 = prepare_biofeatures(df, norm_mean_pe6, norm_std_pe6, device)
        ft_models = load_deepprime6_models(ft_dir, device, is_scratch=False)
        ft_preds = predict_deepprime6(ft_models, g_tensor, b_pe6, batch_size=batch_size)
        df["pred_dp6_pe6a"] = ft_preds["pe6a"]
        df["pred_dp6_pe6b"] = ft_preds["pe6b"]
        df["pred_dp6_pe6c"] = ft_preds["pe6c"]
        df["pred_dp6_drnaseh"] = ft_preds["drnaseh"]

    # 3. DeepPrime6 From-Scratch Inference
    if include_dp6_scratch:
        logger.info("Executing DeepPrime6 (From-Scratch) inference...")
        b_pe6 = prepare_biofeatures(df, norm_mean_pe6, norm_std_pe6, device)
        scratch_models = load_deepprime6_models(scratch_dir, device, is_scratch=True)
        scratch_preds = predict_deepprime6(scratch_models, g_tensor, b_pe6, batch_size=batch_size)
        df["pred_scratch_pe6a"] = scratch_preds["pe6a"]
        df["pred_scratch_pe6b"] = scratch_preds["pe6b"]
        df["pred_scratch_pe6c"] = scratch_preds["pe6c"]
        df["pred_scratch_drnaseh"] = scratch_preds["drnaseh"]

    # 4. PRIDICT2.0 Inference
    if include_pridict2:
        logger.info("Executing PRIDICT2.0 inference...")
        pr_scores = predict_pridict2_scores(df, cell_types=["HEK", "K562"])
        df["pred_pridict2_hek"] = pr_scores["pred_pridict2_hek"]
        df["pred_pridict2_k562"] = pr_scores["pred_pridict2_k562"]
        df["pred_pridict2"] = pr_scores["pred_pridict2_hek"]

    logger.info("Inference pipeline complete.")
    return df
