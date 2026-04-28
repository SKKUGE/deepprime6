"""Tests for model architecture components."""

import pytest
import torch

from src.models.components.blocks.feature_merger import ResidualFeatureMerger
from src.models.components.blocks.parallel import ParallelDeepPrimeModels
from src.models.components.blocks.residual_layer import FCResidualBlock, ResidualCNNBlock
from src.models.components.dprime import GeneInteractionModel


BATCH_SIZE = 4
SEQ_LEN = 74  # standard DeepPrime sequence window
NUM_BIOFEATURES = 24


# ---------------------------------------------------------------------------
# FCResidualBlock
# ---------------------------------------------------------------------------


class TestFCResidualBlock:
    def test_output_shape_same_dims(self):
        block = FCResidualBlock(input_dim=64, hidden_dim=64, output_dim=64)
        x = torch.randn(BATCH_SIZE, 64)
        out = block(x)
        assert out.shape == (BATCH_SIZE, 64)

    def test_output_shape_different_dims(self):
        block = FCResidualBlock(input_dim=32, hidden_dim=64, output_dim=128)
        x = torch.randn(BATCH_SIZE, 32)
        out = block(x)
        assert out.shape == (BATCH_SIZE, 128)

    def test_output_is_finite(self):
        block = FCResidualBlock(input_dim=16, hidden_dim=32, output_dim=16)
        x = torch.randn(BATCH_SIZE, 16)
        out = block(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# ResidualCNNBlock
# ---------------------------------------------------------------------------


class TestResidualCNNBlock:
    def test_output_shape_same_channels(self):
        block = ResidualCNNBlock(in_channels=32, out_channels=32, num_conv_layers=2)
        x = torch.randn(BATCH_SIZE, 32, 16)
        out = block(x)
        assert out.shape == (BATCH_SIZE, 32, 16)

    def test_output_shape_different_channels(self):
        block = ResidualCNNBlock(in_channels=16, out_channels=32, num_conv_layers=2)
        x = torch.randn(BATCH_SIZE, 16, 16)
        out = block(x)
        assert out.shape == (BATCH_SIZE, 32, 16)

    def test_output_is_finite(self):
        block = ResidualCNNBlock(in_channels=8, out_channels=8, num_conv_layers=3)
        x = torch.randn(BATCH_SIZE, 8, 20)
        out = block(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# ResidualFeatureMerger
# ---------------------------------------------------------------------------


class TestResidualFeatureMerger:
    def _make_merger(self, num_residual_blocks=3, final_avgpool=True):
        return ResidualFeatureMerger(
            num_residual_blocks=num_residual_blocks,
            num_conv_layers_per_block=2,
            layer_in_in_channels=2,
            layer_hidden_in_channels=16,
            adaptive_avgpool_output_size=4,
            final_avgpool=final_avgpool,
        )

    def test_output_is_1d_per_sample(self):
        merger = self._make_merger()
        x = torch.randn(BATCH_SIZE, 2, 16)
        out = merger(x)
        assert out.dim() == 2
        assert out.shape[0] == BATCH_SIZE

    def test_output_is_finite(self):
        merger = self._make_merger()
        x = torch.randn(BATCH_SIZE, 2, 16)
        out = merger(x)
        assert torch.isfinite(out).all()

    def test_no_final_avgpool(self):
        merger = self._make_merger(final_avgpool=False)
        x = torch.randn(BATCH_SIZE, 2, 16)
        out = merger(x)
        assert out.dim() == 2
        assert out.shape[0] == BATCH_SIZE


# ---------------------------------------------------------------------------
# GeneInteractionModel (DeepPrime backbone)
# ---------------------------------------------------------------------------


class TestGeneInteractionModel:
    """Tests the DeepPrime GeneInteractionModel feature extractor."""

    def _make_model(self):
        return GeneInteractionModel()

    def _make_inputs(self):
        # g: (batch, 4 channels, 2 rows, SEQ_LEN cols) — one-hot encoded sequences
        g = torch.randn(BATCH_SIZE, 4, 2, SEQ_LEN)
        # x: (batch, num_biofeatures)
        x = torch.randn(BATCH_SIZE, NUM_BIOFEATURES)
        return g, x

    def test_forward_output_shape(self):
        model = self._make_model()
        model.eval()
        g, x = self._make_inputs()
        with torch.no_grad():
            out = model(g, x)
        # Default head_in_features = 140 → output is (batch, 140)
        assert out.shape == (BATCH_SIZE, model.head_in_features)

    def test_forward_output_is_finite(self):
        model = self._make_model()
        model.eval()
        g, x = self._make_inputs()
        with torch.no_grad():
            out = model(g, x)
        assert torch.isfinite(out).all()

    def test_gradients_flow(self):
        model = self._make_model()
        model.train()
        g, x = self._make_inputs()
        out = model(g, x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# ParallelDeepPrimeModels
# ---------------------------------------------------------------------------


class TestParallelDeepPrimeModels:
    def test_ensemble_output_length(self):
        num_models = 3
        models = [GeneInteractionModel() for _ in range(num_models)]
        ensemble = ParallelDeepPrimeModels(models)
        ensemble.eval()

        g = torch.randn(BATCH_SIZE, 4, 2, SEQ_LEN)
        x = torch.randn(BATCH_SIZE, NUM_BIOFEATURES)
        with torch.no_grad():
            outputs = ensemble(g, x)

        assert len(outputs) == num_models

    def test_ensemble_output_shapes(self):
        num_models = 2
        models = [GeneInteractionModel() for _ in range(num_models)]
        ensemble = ParallelDeepPrimeModels(models)
        ensemble.eval()

        g = torch.randn(BATCH_SIZE, 4, 2, SEQ_LEN)
        x = torch.randn(BATCH_SIZE, NUM_BIOFEATURES)
        with torch.no_grad():
            outputs = ensemble(g, x)

        expected_features = models[0].head_in_features
        for out in outputs:
            assert out.shape == (BATCH_SIZE, expected_features)
