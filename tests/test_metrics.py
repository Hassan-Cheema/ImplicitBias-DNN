"""
Unit tests for metrics module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

from src.models import LinearModel, ShallowMLP
from src.metrics import (
    compute_weight_norms,
    compute_margin,
    compute_accuracy,
    compute_gradient_norm,
    MetricsTracker,
    MetricsSnapshot
)
from src.data import generate_linearly_separable


class TestWeightNorms:
    """Tests for weight norm computation."""

    def test_l2_norm(self):
        """Test L2 norm computation."""
        model = LinearModel(input_dim=3, output_dim=1)
        # Set known weights
        with torch.no_grad():
            model.linear.weight.fill_(1.0)

        l2, linf = compute_weight_norms(model)

        # L2 norm of [1, 1, 1] = sqrt(3)
        assert abs(l2 - 3**0.5) < 1e-5
        assert abs(linf - 1.0) < 1e-5

    def test_linf_norm(self):
        """Test L-infinity norm computation."""
        model = LinearModel(input_dim=3, output_dim=1)
        with torch.no_grad():
            model.linear.weight.data = torch.tensor([[1.0, -5.0, 2.0]])

        _, linf = compute_weight_norms(model)
        assert abs(linf - 5.0) < 1e-5


class TestMargin:
    """Tests for margin computation."""

    def test_margin_positive(self):
        """Test margin is positive for well-separated data."""
        X, y = generate_linearly_separable(n_samples=50, dim=2, margin=2.0, seed=42)

        model = LinearModel(input_dim=2, output_dim=1)
        # Set weights to separate along x-axis
        with torch.no_grad():
            model.linear.weight.data = torch.tensor([[5.0, 0.0]])

        margin = compute_margin(model, X, y, normalize=False)
        # Should be positive for separable data with correct separator
        assert margin > 0

    def test_normalized_margin(self):
        """Test normalized margin is invariant to weight scaling."""
        X, y = generate_linearly_separable(n_samples=50, dim=2, margin=1.0, seed=42)

        model1 = LinearModel(input_dim=2, output_dim=1)
        model2 = LinearModel(input_dim=2, output_dim=1)

        with torch.no_grad():
            model1.linear.weight.data = torch.tensor([[1.0, 0.0]])
            model2.linear.weight.data = torch.tensor([[10.0, 0.0]])

        nm1 = compute_margin(model1, X, y, normalize=True)
        nm2 = compute_margin(model2, X, y, normalize=True)

        # Normalized margins should be similar
        assert abs(nm1 - nm2) < 0.5


class TestAccuracy:
    """Tests for accuracy computation."""

    def test_perfect_accuracy(self):
        """Test accuracy is 1.0 for perfect classifier."""
        X = torch.tensor([[1.0, 0], [-1.0, 0], [2.0, 0], [-2.0, 0]])
        y = torch.tensor([1.0, -1.0, 1.0, -1.0])

        model = LinearModel(input_dim=2, output_dim=1)
        with torch.no_grad():
            model.linear.weight.data = torch.tensor([[1.0, 0.0]])

        acc = compute_accuracy(model, X, y)
        assert abs(acc - 1.0) < 1e-5

    def test_random_accuracy(self):
        """Test accuracy is approximately 0.5 for random classifier."""
        torch.manual_seed(42)
        X = torch.randn(100, 10)
        y = torch.sign(torch.randn(100))
        y[y == 0] = 1

        model = LinearModel(input_dim=10, output_dim=1)
        # Very small random weights
        with torch.no_grad():
            model.linear.weight.data = torch.randn(1, 10) * 0.001

        acc = compute_accuracy(model, X, y)
        # Should be around 0.5 for random
        assert 0.3 < acc < 0.7


class TestMetricsTracker:
    """Tests for metrics tracker."""

    def test_log_and_retrieve(self):
        """Test logging and retrieving metrics."""
        tracker = MetricsTracker()

        for i in range(5):
            tracker.log(MetricsSnapshot(
                epoch=i,
                loss=1.0 / (i + 1),
                accuracy=0.5 + i * 0.1,
                weight_norm_l2=1.0,
                weight_norm_linf=0.5,
                margin=0.1 * i,
                normalized_margin=0.1 * i,
                gradient_norm=0.01
            ))

        losses = tracker.get_metric_series('loss')
        assert len(losses) == 5
        assert losses[0] == 1.0
        assert losses[-1] == 0.2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        tracker = MetricsTracker()
        tracker.log(MetricsSnapshot(
            epoch=0, loss=0.5, accuracy=0.8,
            weight_norm_l2=1.0, weight_norm_linf=0.5,
            margin=0.1, normalized_margin=0.1,
            gradient_norm=0.01
        ))

        d = tracker.to_dict()
        assert 'loss' in d
        assert 'accuracy' in d
        assert d['loss'][0] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
