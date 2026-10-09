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
