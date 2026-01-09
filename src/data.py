"""
Data generation utilities for implicit bias experiments.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import make_moons, make_circles


def generate_linearly_separable(
    n_samples: int = 100,
    dim: int = 2,
    margin: float = 0.5,
    noise: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate linearly separable data with controlled margin.

    Args:
        n_samples: Total number of samples
        dim: Dimensionality of features
        margin: Minimum separation between classes
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        (X, y): Features and labels (±1)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_per_class = n_samples // 2

    # Generate points on opposite sides of a hyperplane
    X_pos = torch.randn(n_per_class, dim)
    X_pos[:, 0] += margin  # Shift positive class

    X_neg = torch.randn(n_per_class, dim)
    X_neg[:, 0] -= margin  # Shift negative class

    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([
        torch.ones(n_per_class),
        -torch.ones(n_per_class)
    ])

    if noise > 0:
        X += torch.randn_like(X) * noise

    # Shuffle
    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


def generate_xor_data(
    n_samples: int = 200,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate XOR pattern data (not linearly separable).

    Args:
        n_samples: Total number of samples
        noise: Standard deviation of Gaussian noise
        seed: Random seed

    Returns:
        (X, y): Features and labels (±1)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_per_quadrant = n_samples // 4

    # Four quadrants
    centers = [
        (1, 1),   # +
        (-1, -1), # +
        (1, -1),  # -
        (-1, 1),  # -
    ]
    labels = [1, 1, -1, -1]

    X_list = []
    y_list = []

    for (cx, cy), label in zip(centers, labels):
        X_q = torch.randn(n_per_quadrant, 2) * noise
        X_q[:, 0] += cx
        X_q[:, 1] += cy
        X_list.append(X_q)
        y_list.append(torch.full((n_per_quadrant,), label, dtype=torch.float))

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list)

    # Shuffle
    perm = torch.randperm(X.size(0))
    return X[perm], y[perm]


def generate_spiral_data(
    n_samples: int = 200,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two interleaved spirals (highly nonlinear).

    Args:
        n_samples: Total number of samples
        noise: Standard deviation of noise
        seed: Random seed

    Returns:
        (X, y): Features and labels (±1)
    """
    if seed is not None:
        np.random.seed(seed)

    n_per_class = n_samples // 2

    theta = np.sqrt(np.random.rand(n_per_class)) * 2 * np.pi

    # First spiral
    r1 = 2 * theta + np.pi
    x1 = r1 * np.cos(theta) + np.random.randn(n_per_class) * noise
    y1 = r1 * np.sin(theta) + np.random.randn(n_per_class) * noise

    # Second spiral (rotated by pi)
    r2 = 2 * theta + np.pi
    x2 = -r2 * np.cos(theta) + np.random.randn(n_per_class) * noise
    y2 = -r2 * np.sin(theta) + np.random.randn(n_per_class) * noise

    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])
    y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])

    # Shuffle
    perm = np.random.permutation(n_samples)

    return torch.tensor(X[perm], dtype=torch.float32), torch.tensor(y[perm], dtype=torch.float32)


def generate_moons_data(
    n_samples: int = 200,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two interleaving half circles.

    Args:
        n_samples: Total number of samples
        noise: Standard deviation of noise
        seed: Random seed

    Returns:
        (X, y): Features and labels (±1)
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = 2 * y - 1  # Convert to ±1

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def generate_circles_data(
    n_samples: int = 200,
    noise: float = 0.1,
    factor: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two concentric circles.

    Args:
        n_samples: Total number of samples
        noise: Standard deviation of noise
        factor: Scale factor between inner and outer circle
        seed: Random seed

    Returns:
        (X, y): Features and labels (±1)
    """
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=seed
    )
    y = 2 * y - 1  # Convert to ±1

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def generate_high_dim_separable(
    n_samples: int = 100,
    dim: int = 100,
    effective_dim: int = 5,
    margin: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate high-dimensional data that is separable in a low-dimensional subspace.

    Useful for studying implicit bias in overparameterized settings.

    Args:
        n_samples: Total number of samples
        dim: Total dimensionality
        effective_dim: Number of "relevant" dimensions
        margin: Separation margin
        seed: Random seed

    Returns:
        (X, y): Features and labels (±1)
    """
    if seed is not None:
        torch.manual_seed(seed)

    n_per_class = n_samples // 2

    # Generate in effective dimensions
    X_eff = torch.randn(n_samples, effective_dim)

    # Labels based on first dimension
    y = torch.sign(X_eff[:, 0])
    y[y == 0] = 1  # Handle exact zeros

    # Add margin
    X_eff[:, 0] += margin * y

    # Embed in high dimensions with random rotation
    rotation = torch.randn(effective_dim, dim)
    rotation = torch.linalg.qr(rotation.T)[0].T  # Orthogonalize

    X = X_eff @ rotation[:effective_dim, :]

    # Add noise in null space
    null_noise = torch.randn(n_samples, dim) * 0.1
    X += null_noise

    return X, y


def get_dataset(
    name: str,
    n_samples: int = 200,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factory function to get dataset by name."""
    datasets = {
        'linear': generate_linearly_separable,
        'xor': generate_xor_data,
        'spiral': generate_spiral_data,
        'moons': generate_moons_data,
        'circles': generate_circles_data,
        'high_dim': generate_high_dim_separable,
    }

    name = name.lower()
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")

    return datasets[name](n_samples=n_samples, **kwargs)
