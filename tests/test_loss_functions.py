"""Tests for loss functions."""

import pytest
import torch

from src.models.components.loss.weighted_mse_loss import (
    BalancedMSELoss,
    ImprovedBalancedLoss,
    ScaledMSELoss,
)


BATCH_SIZE = 8
NUM_PE_CLASSES = 1  # single PE-type label column


def _make_targets(batch_size: int = BATCH_SIZE, num_classes: int = NUM_PE_CLASSES):
    """Create a target tensor matching the expected loss input format.

    The last three columns encode edit type flags (sub, ins, del).
    """
    labels = torch.rand(batch_size, num_classes) * 10  # simulated efficiency values
    edit_type = torch.zeros(batch_size, 3)
    # Assign each sample one of the three edit types
    for i in range(batch_size):
        edit_type[i, i % 3] = 1
    return torch.cat([labels, edit_type], dim=1)


# ---------------------------------------------------------------------------
# ScaledMSELoss
# ---------------------------------------------------------------------------


class TestScaledMSELoss:
    def test_returns_scalar(self):
        loss_fn = ScaledMSELoss()
        pred = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        target = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_loss_is_non_negative(self):
        loss_fn = ScaledMSELoss()
        pred = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        target = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_perfect_prediction_near_zero(self):
        loss_fn = ScaledMSELoss()
        values = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        loss = loss_fn(values, values)
        assert loss.item() < 1e-6

    def test_handles_nan_gracefully(self):
        loss_fn = ScaledMSELoss()
        pred = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        target = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        pred[0] = float("nan")
        target[1] = float("nan")
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# ImprovedBalancedLoss
# ---------------------------------------------------------------------------


class TestImprovedBalancedLoss:
    def test_returns_scalar(self):
        loss_fn = ImprovedBalancedLoss()
        pred = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        target = _make_targets()
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_loss_is_finite(self):
        loss_fn = ImprovedBalancedLoss()
        pred = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        target = _make_targets()
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_loss_is_non_negative(self):
        loss_fn = ImprovedBalancedLoss()
        pred = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        target = _make_targets()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_gradients_flow(self):
        loss_fn = ImprovedBalancedLoss()
        pred = torch.rand(BATCH_SIZE, NUM_PE_CLASSES, requires_grad=True)
        target = _make_targets()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# BalancedMSELoss
# ---------------------------------------------------------------------------


class TestBalancedMSELoss:
    def test_returns_scalar_with_scale(self):
        loss_fn = BalancedMSELoss(scale=True)
        pred = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        target = _make_targets()
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_returns_scalar_without_scale(self):
        loss_fn = BalancedMSELoss(scale=False)
        pred = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        target = _make_targets()
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_loss_is_finite(self):
        loss_fn = BalancedMSELoss()
        pred = torch.rand(BATCH_SIZE, NUM_PE_CLASSES)
        target = _make_targets()
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_gradients_flow(self):
        loss_fn = BalancedMSELoss()
        pred = torch.rand(BATCH_SIZE, NUM_PE_CLASSES, requires_grad=True)
        target = _make_targets()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
