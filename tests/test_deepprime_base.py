"""Tests that DeepPrime-Base inference uses the original DeepPrime (GENET) architecture and weights."""

import functools
import glob

import pytest
import torch
from omegaconf import OmegaConf

from src.models.components.dprime import (
    GeneInteractionModelOriginalImplementation,
    GeneInteractionModelVanilla,
)
from src.models.pe6_module_dp_only import PE6DeepPrimeModule
from src.utils.deepprime_base import (
    ORIGINAL_DEEPPRIME_TARGET,
    resolve_default_normalization_paths,
    use_original_architecture_for_baseline,
)

WEIGHT_GLOB = "src/models/weights/DP_variant_293T_PE2max_epegRNA_Opti_220428/final_model_*.pt"
WEIGHT_FILES = sorted(glob.glob(WEIGHT_GLOB))
needs_weights = pytest.mark.skipif(len(WEIGHT_FILES) != 20, reason="DeepPrime base weights not available")


def _module(feature_extractor_cls, baseline):
    return PE6DeepPrimeModule(
        optimizer=None,
        scheduler=None,
        feature_extractor=functools.partial(feature_extractor_cls),
        dataprops=OmegaConf.create({"PE_class_list": ["PE6a(+PEmaxCas9)"]}),
        model_weights=OmegaConf.create({"baseline": baseline, "requires_grad": False}),
        compile=False,
        criterion=torch.nn.MSELoss,
    )


def _inputs(n=8):
    torch.manual_seed(0)
    g = torch.randn(n, 4, 2, 74)
    b = torch.randn(n, 24)
    return g, b


class TestOriginalArchitectureSwitch:
    def test_switches_when_baseline_weights_are_used(self):
        cfg = OmegaConf.create(
            {
                "model": {
                    "feature_extractor": {"_target_": "src.models.components.dprime.GeneInteractionModelOriginalImplementation"},
                    "model_weights": {"baseline": "weights/*.pt"},
                }
            }
        )
        assert use_original_architecture_for_baseline(cfg) is True
        assert cfg.model.feature_extractor._target_ == ORIGINAL_DEEPPRIME_TARGET

    def test_untouched_without_baseline_weights(self):
        target = "src.models.components.dprime.GeneInteractionModelOriginalImplementation"
        cfg = OmegaConf.create(
            {"model": {"feature_extractor": {"_target_": target}, "model_weights": {"baseline": None}}}
        )
        assert use_original_architecture_for_baseline(cfg) is False
        assert cfg.model.feature_extractor._target_ == target

    def test_resolve_default_normalization_paths_for_ckpt(self):
        cfg = OmegaConf.create(
            {
                "data": {"dataset": {"norm_mean_path": None, "norm_std_path": None}},
                "model": {"model_weights": {"baseline": WEIGHT_GLOB}},
            }
        )
        resolve_default_normalization_paths(cfg, "src/models/weights/DeepPrime6-weights/pe6a_mainft.ckpt")
        assert cfg.data.dataset.norm_mean_path.endswith("pe6_mean.csv")
        assert cfg.data.dataset.norm_std_path.endswith("pe6_std.csv")

    def test_resolve_default_normalization_paths_for_baseline(self):
        cfg = OmegaConf.create(
            {
                "data": {"dataset": {"norm_mean_path": None, "norm_std_path": None}},
                "model": {"model_weights": {"baseline": WEIGHT_GLOB}},
            }
        )
        resolve_default_normalization_paths(cfg, None)
        assert cfg.data.dataset.norm_mean_path.endswith("mean.csv")
        assert cfg.data.dataset.norm_std_path.endswith("std.csv")


@needs_weights
class TestBaselineLoading:
    def test_vanilla_loads_all_weights_exactly(self):
        module = _module(GeneInteractionModelVanilla, WEIGHT_GLOB)
        for ref_file, model in zip(WEIGHT_FILES, module.feature_extractor.models):
            ref = torch.load(ref_file, map_location="cpu", weights_only=True)
            state = model.state_dict()
            assert set(state.keys()) == set(ref.keys())
            for k, v in ref.items():
                assert torch.equal(state[k], v), k

    def test_module_matches_direct_genet_ensemble(self):
        module = _module(GeneInteractionModelVanilla, WEIGHT_GLOB).eval()
        direct = []
        for f in WEIGHT_FILES:
            m = GeneInteractionModelVanilla()
            m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=True)
            direct.append(m.eval())
        g, b = _inputs()
        with torch.no_grad():
            expected = torch.stack([m(g, b) for m in direct]).mean(0)
            actual = module((g, b))
        assert torch.allclose(expected, actual, atol=1e-6)

    def test_fine_tuning_architecture_still_loads_baseline(self):
        # DeepPrime6 fine-tuning starts from the same weights with extra BN layers; must keep working
        module = _module(GeneInteractionModelOriginalImplementation, WEIGHT_GLOB)
        assert len(module.feature_extractor.models) == 20

    def test_fine_tuning_architecture_differs_from_original(self):
        original = _module(GeneInteractionModelVanilla, WEIGHT_GLOB).eval()
        finetune_arch = _module(GeneInteractionModelOriginalImplementation, WEIGHT_GLOB).eval()
        g, b = _inputs()
        with torch.no_grad():
            assert not torch.allclose(original((g, b)), finetune_arch((g, b)), atol=1e-4)

    def test_missing_non_bn_weights_raise(self):
        model = GeneInteractionModelVanilla()
        sd = torch.load(WEIGHT_FILES[0], map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if k != "d.0.weight"}
        result = model.load_state_dict(sd, strict=False)
        with pytest.raises(RuntimeError, match="d.0.weight"):
            PE6DeepPrimeModule._verify_baseline_load(result, model, "dummy.pt")
