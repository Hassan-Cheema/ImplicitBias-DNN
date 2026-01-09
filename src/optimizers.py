"""
Custom optimizer implementations including Full-Batch GD and Lion.
"""

import torch
from torch.optim import Optimizer
from typing import Callable, Iterable, Tuple
import math


class FullBatchGD(Optimizer):
    """
    Full-batch Gradient Descent optimizer.

    Unlike SGD, this is meant to be used with the full dataset
    for studying the exact implicit bias of gradient descent.
    """

    def __init__(self, params: Iterable, lr: float = 1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-group['lr'])

        return loss


class Lion(Optimizer):
    """
    Lion optimizer (EvoLved Sign Momentum).

    Chen et al., 2023: "Symbolic Discovery of Optimization Algorithms"

    Key characteristics:
    - Uses sign of momentum for updates (uniform magnitude)
    - Memory efficient (no second moment like Adam)
    - Tends toward ℓ∞-like implicit bias
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Lion update: sign of interpolated momentum
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])

                # Update momentum for next step
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class SignGD(Optimizer):
    """
    Sign Gradient Descent (SignSGD).

    Updates weights using only the sign of gradients.
    Useful for studying ℓ∞-margin behavior in isolation.
    """

    def __init__(self, params: Iterable, lr: float = 1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.add_(torch.sign(p.grad), alpha=-group['lr'])

        return loss


class NormalizedGD(Optimizer):
    """
    Normalized Gradient Descent.

    Normalizes the gradient to have unit norm before updating.
    Useful for studying directional convergence.
    """

    def __init__(self, params: Iterable, lr: float = 1e-2, eps: float = 1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                grad_norm = grad.norm()
                if grad_norm > group['eps']:
                    normalized_grad = grad / grad_norm
                    p.add_(normalized_grad, alpha=-group['lr'])

        return loss


def get_optimizer(name: str, params: Iterable, lr: float, **kwargs) -> Optimizer:
    """Factory function to get optimizer by name."""
    optimizers = {
        'gd': FullBatchGD,
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'lion': Lion,
        'signgd': SignGD,
        'normalized_gd': NormalizedGD,
    }

    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")

    return optimizers[name](params, lr=lr, **kwargs)
