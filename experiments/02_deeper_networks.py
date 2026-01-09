"""
Experiment 2: Deeper Networks

Investigate how implicit bias of different optimizers changes with network depth.

Key questions:
- Does the ℓ2 vs ℓ∞ margin distinction persist in deeper networks?
- How does depth affect the convergence rate?
- Do deeper networks show qualitatively different behavior?

Run: python experiments/02_deeper_networks.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import json
import numpy as np

from src.models import DeepMLP
from src.optimizers import get_optimizer
from src.metrics import (
    MetricsTracker, MetricsSnapshot,
    compute_weight_norms, compute_margin, compute_accuracy,
    compute_gradient_norm, compute_layer_weight_norms
)
from src.data import generate_linearly_separable
from src.visualization import plot_training_curves, plot_depth_analysis


def train_model(
    model: nn.Module,
    optimizer_name: str,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    log_interval: int = 10
) -> MetricsTracker:
    """Train a model and return metrics tracker."""

    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    y_bce = (y + 1) / 2

    tracker = MetricsTracker()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X).squeeze()
        loss = loss_fn(outputs, y_bce)
        loss.backward()

        grad_norm = compute_gradient_norm(model)
        optimizer.step()

        if epoch % log_interval == 0 or epoch == epochs - 1:
            l2_norm, linf_norm = compute_weight_norms(model)
            margin = compute_margin(model, X, y, normalize=False)
            norm_margin = compute_margin(model, X, y, normalize=True)
            acc = compute_accuracy(model, X, y)

            tracker.log(MetricsSnapshot(
                epoch=epoch,
                loss=loss.item(),
                accuracy=acc,
                weight_norm_l2=l2_norm,
                weight_norm_linf=linf_norm,
                margin=margin,
                normalized_margin=norm_margin,
                gradient_norm=grad_norm,
                gradient_noise=None
            ))

    return tracker


def run_experiment(args):
    """Run the deeper networks experiment."""

    print("=" * 60)
    print("Experiment 2: Deeper Networks")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    print(f"\nGenerating data (n={args.n_samples}, dim={args.dim})...")
    X, y = generate_linearly_separable(
        n_samples=args.n_samples,
        dim=args.dim,
        margin=1.0,
        seed=args.seed
    )

    # Depths to test
    depths = [1, 2, 4, 6, 8]
    optimizers = ['gd', 'sgd', 'adam', 'lion']
    hidden_width = args.hidden_width

    # Store all results
    depth_results = {}

    for depth in depths:
        print(f"\n{'='*50}")
        print(f"Testing depth: {depth} layers (width: {hidden_width})")
        print(f"{'='*50}")

        depth_results[depth] = {}

        for opt_name in optimizers:
            print(f"\n  Training with {opt_name.upper()}...")

            # Create model with specified depth
            if depth == 1:
                hidden_dims = [hidden_width]
            else:
                hidden_dims = [hidden_width] * depth

            model = DeepMLP(
                input_dim=args.dim,
                hidden_dims=hidden_dims,
                output_dim=1,
                activation='relu'
            )

            # Same initialization seed
            torch.manual_seed(args.seed)
            for param in model.parameters():
                if param.dim() > 1:
                    nn.init.xavier_normal_(param)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Model parameters: {n_params:,}")

            tracker = train_model(
                model=model,
                optimizer_name=opt_name,
                X=X, y=y,
                epochs=args.epochs,
                lr=args.lr,
                log_interval=args.log_interval
            )

            depth_results[depth][opt_name.upper()] = tracker.to_dict()

            # Print summary
            final = tracker.history[-1]
            print(f"    Final - Loss: {final.loss:.4f}, Acc: {final.accuracy:.2%}, "
                  f"NormMargin: {final.normalized_margin:.4f}")

            # Layer-wise analysis
            layer_norms = compute_layer_weight_norms(model)
            tracker.save(str(output_dir / f"metrics_depth{depth}_{opt_name}.json"))

    # Save all results
    with open(output_dir / "depth_results.json", 'w') as f:
        json.dump(depth_results, f, indent=2)

    # Visualization
    print("\n" + "=" * 40)
    print("Generating Visualizations")
    print("=" * 40)

    plot_depth_analysis(
        depth_results,
        metric='normalized_margin',
        save_path=str(output_dir / "depth_margin_analysis.png")
    )

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Final Normalized Margin by Depth and Optimizer")
    print("=" * 60)

    header = "Depth    | " + " | ".join(f"{opt:>8}" for opt in optimizers)
    print(header)
    print("-" * len(header))

    for depth in depths:
        row = f"{depth:>5} layers | "
        row += " | ".join(
            f"{depth_results[depth][opt.upper()]['normalized_margin'][-1]:>8.4f}"
            for opt in optimizers
        )
        print(row)

    print(f"\nResults saved to: {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Deeper Networks Experiment")
    parser.add_argument('--n_samples', type=int, default=200, help='Number of samples')
    parser.add_argument('--dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--hidden_width', type=int, default=64, help='Hidden layer width')
    parser.add_argument('--epochs', type=int, default=3000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/deeper_networks',
                        help='Output directory')

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
