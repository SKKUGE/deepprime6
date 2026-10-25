"""Helpers to make sure DeepPrime-Base inference uses the *original* DeepPrime architecture.

The original DeepPrime (Yu et al., 2023; GENET weights ``final_model_*.pt``) is
``GeneInteractionModelVanilla``: no BatchNorm/ReLU after the last linear layer of ``d``.
``GeneInteractionModelOriginalImplementation`` (extra BN + ReLU in ``d``) is the architecture used to
*fine-tune* DeepPrime6 from the GENET weights and must not be used to evaluate DeepPrime-Base.
"""
from __future__ import annotations

import logging

from omegaconf import DictConfig

ORIGINAL_DEEPPRIME_TARGET = "src.models.components.dprime.GeneInteractionModelVanilla"


def use_original_architecture_for_baseline(cfg: DictConfig, log: logging.Logger | None = None) -> bool:
    """Switch ``cfg.model.feature_extractor`` to the original DeepPrime architecture.

    Intended for inference/evaluation runs that use raw DeepPrime (GENET) weights, i.e. runs without a
    Lightning checkpoint where ``cfg.model.model_weights.baseline`` is set. Training runs (which fine-tune
    from the baseline weights) must not call this.

    Returns:
        True if the config was changed (or already pointed to the original architecture).
    """
    model_cfg = cfg.get("model")
    if model_cfg is None:
        return False
    weights = model_cfg.get("model_weights")
    if weights is None or weights.get("baseline") is None:
        return False
    if "feature_extractor" not in model_cfg:
        return False

    previous = model_cfg.feature_extractor.get("_target_")
    model_cfg.feature_extractor._target_ = ORIGINAL_DEEPPRIME_TARGET
    if log is not None and previous != ORIGINAL_DEEPPRIME_TARGET:
        log.info(
            "Raw DeepPrime weights without a Lightning checkpoint: using the original DeepPrime "
            f"architecture ({ORIGINAL_DEEPPRIME_TARGET}) instead of {previous}"
        )
    return True


def resolve_default_normalization_paths(
    cfg: DictConfig, ckpt_path: str | None, log: logging.Logger | None = None
) -> None:
    """Populate ``cfg.data.dataset.norm_mean_path`` and ``norm_std_path`` if not explicitly set.

    - For Lightning checkpoints (``.ckpt``), resolves to ``pe6_mean.csv`` / ``pe6_std.csv`` in the
      checkpoint directory (falling back to ``src/models/weights/DeepPrime6-weights/``).
    - For DeepPrime-Base (``ckpt_path is None`` with ``baseline`` weights), resolves to
      ``mean.csv`` / ``std.csv`` in the baseline weights directory.
    """
    import os

    data_cfg = cfg.get("data")
    if data_cfg is None or "dataset" not in data_cfg:
        return
    ds_cfg = data_cfg.dataset
    if ds_cfg.get("norm_mean_path") and ds_cfg.get("norm_std_path"):
        return

    candidate_mean = None
    candidate_std = None

    if ckpt_path:
        ckpt_dir = os.path.dirname(ckpt_path)
        for d in (ckpt_dir, "src/models/weights/DeepPrime6-weights"):
            m = os.path.join(d, "pe6_mean.csv")
            s = os.path.join(d, "pe6_std.csv")
            if os.path.exists(m) and os.path.exists(s):
                candidate_mean, candidate_std = m, s
                break
    else:
        model_cfg = cfg.get("model")
        baseline = model_cfg.get("model_weights", {}).get("baseline") if model_cfg else None
        if baseline:
            base_dir = os.path.dirname(str(baseline))
            m = os.path.join(base_dir, "mean.csv")
            s = os.path.join(base_dir, "std.csv")
            if os.path.exists(m) and os.path.exists(s):
                candidate_mean, candidate_std = m, s

    if candidate_mean and candidate_std:
        ds_cfg.norm_mean_path = candidate_mean
        ds_cfg.norm_std_path = candidate_std
        if log is not None:
            log.info(f"Auto-resolved normalization stats: {candidate_mean}, {candidate_std}")

