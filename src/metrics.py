"""
Metrics computation and tracking for implicit bias experiments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class MetricsSnapshot:
    """Single snapshot of metrics at a given epoch."""
    epoch: int
    loss: float
    accuracy: float
    weight_norm_l2: float
    weight_norm_linf: float
    margin: float
    normalized_margin: float
    gradient_norm: float
    gradient_noise: Optional[float] = None


class MetricsTracker:
    """Track and log metrics throughout training."""

    def __init__(self):
        self.history: List[MetricsSnapshot] = []
        self.gradient_history: List[torch.Tensor] = []

    def log(self, snapshot: MetricsSnapshot):
        """Add a metrics snapshot."""
        self.history.append(snapshot)

    def get_metric_series(self, metric_name: str) -> List[float]:
        """Get time series of a specific metric."""
        return [getattr(s, metric_name) for s in self.history]

    def to_dict(self) -> Dict:
        """Convert history to dictionary."""
        return {
            'epochs': [s.epoch for s in self.history],
            'loss': [s.loss for s in self.history],
            'accuracy': [s.accuracy for s in self.history],
            'weight_norm_l2': [s.weight_norm_l2 for s in self.history],
            'weight_norm_linf': [s.weight_norm_linf for s in self.history],
            'margin': [s.margin for s in self.history],
            'normalized_margin': [s.normalized_margin for s in self.history],
            'gradient_norm': [s.gradient_norm for s in self.history],
            'gradient_noise': [s.gradient_noise for s in self.history],
        }

    def save(self, path: str):
        """Save metrics to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'MetricsTracker':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        tracker = cls()
        for i in range(len(data['epochs'])):
            tracker.log(MetricsSnapshot(
                epoch=data['epochs'][i],
                loss=data['loss'][i],
                accuracy=data['accuracy'][i],
                weight_norm_l2=data['weight_norm_l2'][i],
                weight_norm_linf=data['weight_norm_linf'][i],
                margin=data['margin'][i],
                normalized_margin=data['normalized_margin'][i],
                gradient_norm=data['gradient_norm'][i],
                gradient_noise=data['gradient_noise'][i] if data['gradient_noise'][i] else None,
            ))
        return tracker


def compute_weight_norms(model: nn.Module) -> Tuple[float, float]:
    """
    Compute ℓ2 and ℓ∞ norms of all model weights.

    Returns:
        (l2_norm, linf_norm): Tuple of weight norms
    """
    all_weights = []
    for param in model.parameters():
        all_weights.append(param.data.flatten())

    if not all_weights:
        return 0.0, 0.0

    weights = torch.cat(all_weights)
    l2_norm = weights.norm(p=2).item()
    linf_norm = weights.abs().max().item()

    return l2_norm, linf_norm


def compute_layer_weight_norms(model: nn.Module) -> Dict[str, Tuple[float, float]]:
    """Compute per-layer weight norms."""
    norms = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2 = param.data.norm(p=2).item()
            linf = param.data.abs().max().item()
            norms[name] = (l2, linf)
    return norms


def compute_margin(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    normalize: bool = False
) -> float:
    """
    Compute classification margin.

    For binary classification, margin = y * f(x) where y ∈ {-1, +1}

    Args:
        model: Neural network model
        X: Input features
        y: Labels (should be -1 or +1)
        normalize: If True, normalize by weight norm (ℓ2)

    Returns:
        Minimum margin across all samples
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X).squeeze()
        margins = y * outputs
        min_margin = margins.min().item()

        if normalize:
            l2_norm, _ = compute_weight_norms(model)
            if l2_norm > 1e-8:
                min_margin /= l2_norm

    model.train()
    return min_margin


def compute_all_margins(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor
) -> Tuple[float, float, float, float]:
    """
    Compute multiple margin statistics.

    Returns:
        (min_margin, mean_margin, normalized_min, normalized_mean)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X).squeeze()
        margins = y * outputs

        min_margin = margins.min().item()
        mean_margin = margins.mean().item()

        l2_norm, _ = compute_weight_norms(model)
        if l2_norm > 1e-8:
            normalized_min = min_margin / l2_norm
            normalized_mean = mean_margin / l2_norm
        else:
            normalized_min = normalized_mean = 0.0

    model.train()
    return min_margin, mean_margin, normalized_min, normalized_mean


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return np.sqrt(total_norm)


def compute_gradient_noise(
    model: nn.Module,
    loss_fn: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
    n_samples: int = 10
) -> float:
    """
    Estimate gradient noise by computing variance of mini-batch gradients.

    Args:
        model: Neural network model
        loss_fn: Loss function
        X: Full dataset features
        y: Full dataset labels
        batch_size: Size of mini-batches
        n_samples: Number of mini-batches to sample

    Returns:
        Gradient noise (variance of gradient norms)
    """
    n = X.size(0)
    gradient_norms = []

    model.train()
    for _ in range(n_samples):
        # Random mini-batch
        indices = torch.randperm(n)[:batch_size]
        X_batch = X[indices]
        y_batch = y[indices]

        model.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = loss_fn(outputs, y_batch)
        loss.backward()

        grad_norm = compute_gradient_norm(model)
        gradient_norms.append(grad_norm)

    return np.var(gradient_norms)


def compute_accuracy(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor
) -> float:
    """Compute classification accuracy."""
    model.eval()
    with torch.no_grad():
        outputs = model(X).squeeze()
        predictions = torch.sign(outputs)
        # Handle zero outputs
        predictions[predictions == 0] = 1
        accuracy = (predictions == y).float().mean().item()
    model.train()
    return accuracy


def compute_directional_convergence(
    weight_history: List[torch.Tensor]
) -> List[float]:
    """
    Compute how stable the weight direction becomes over time.

    Returns cosine similarity between consecutive weight directions.
    """
    similarities = []
    for i in range(1, len(weight_history)):
        w_prev = weight_history[i - 1]
        w_curr = weight_history[i]

        # Normalize
        w_prev_norm = w_prev / (w_prev.norm() + 1e-8)
        w_curr_norm = w_curr / (w_curr.norm() + 1e-8)

        sim = torch.dot(w_prev_norm.flatten(), w_curr_norm.flatten()).item()
        similarities.append(sim)

    return similarities


def compute_margin_ratios(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor
) -> Tuple[float, float]:
    """
    Compute ℓ2-margin and ℓ∞-margin ratios.

    These ratios help distinguish optimizer implicit biases:
    - GD maximizes ℓ2-margin
    - Adam/Lion maximize ℓ∞-margin

    Returns:
        (l2_margin_ratio, linf_margin_ratio)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X).squeeze()
        margins = y * outputs
        min_margin = margins.min().item()

        l2_norm, linf_norm = compute_weight_norms(model)

        l2_ratio = min_margin / (l2_norm + 1e-8)
        linf_ratio = min_margin / (linf_norm + 1e-8)

    model.train()
    return l2_ratio, linf_ratio
