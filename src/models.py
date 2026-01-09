"""
Neural network models for implicit bias experiments.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class LinearModel(nn.Module):
    """Single linear layer for baseline experiments on linearly separable data."""

    def __init__(self, input_dim: int, output_dim: int = 1, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_weights(self) -> torch.Tensor:
        """Return weight vector for margin computation."""
        return self.linear.weight.data.flatten()


class ShallowMLP(nn.Module):
    """2-layer MLP with configurable hidden size and activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        activation: str = "relu",
        bias: bool = True
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.activation = self._get_activation(activation)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        return self.fc2(x)


class DeepMLP(nn.Module):
    """Configurable depth MLP for deeper network experiments."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        activation: str = "relu",
        bias: bool = True
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))

        self.output_layer = nn.Linear(dims[-1], output_dim, bias=bias)
        self.activation = self._get_activation(activation)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

    @property
    def depth(self) -> int:
        """Return number of layers (including output)."""
        return len(self.layers) + 1


class OverparameterizedMLP(nn.Module):
    """Highly overparameterized network for studying extreme cases."""

    def __init__(
        self,
        input_dim: int,
        width: int = 1024,
        depth: int = 2,
        output_dim: int = 1,
        activation: str = "relu",
        bias: bool = True
    ):
        super().__init__()
        hidden_dims = [width] * depth
        self.network = DeepMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
