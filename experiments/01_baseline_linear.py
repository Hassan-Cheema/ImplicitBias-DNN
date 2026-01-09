"""
Baseline Experiment: Linear Logistic Regression on Separable Data

This experiment reproduces the theoretical results:
- GD converges to max ℓ2-margin solution
- Adam converges to max ℓ∞-margin solution

Run: python experiments/01_baseline_linear.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import json

from src.models import LinearModel
from src.optimizers import get_optimizer
from src.metrics import (
    MetricsTracker, MetricsSnapshot,
    compute_weight_norms, compute_margin, compute_accuracy,
    compute_gradient_norm, compute_gradient_noise, compute_margin_ratios
)
from src.data import generate_linearly_separable
from src.visualization import (
    plot_training_curves, plot_margin_comparison,
    plot_weight_norm_ratio, plot_decision_boundary,
    plot_multi_decision_boundaries
)


def train_single_optimizer(
    model: nn.Module,
    optimizer_name: str,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    log_interval: int = 10,
    compute_noise: bool = True
) -> MetricsTracker:
    """Train a model with a single optimizer and track metrics."""

    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Convert y from ±1 to 0/1 for BCE loss
    y_bce = (y + 1) / 2

    tracker = MetricsTracker()

    for epoch in tqdm(range(epochs), desc=f"Training {optimizer_name}"):
        model.train()
        optimizer.zero_grad()

        outputs = model(X).squeeze()
        loss = loss_fn(outputs, y_bce)
        loss.backward()

        grad_norm = compute_gradient_norm(model)
        optimizer.step()

        # Log metrics
        if epoch % log_interval == 0 or epoch == epochs - 1:
            l2_norm, linf_norm = compute_weight_norms(model)
            margin = compute_margin(model, X, y, normalize=False)
            norm_margin = compute_margin(model, X, y, normalize=True)
            acc = compute_accuracy(model, X, y)

            # Gradient noise (expensive, sample occasionally)
            grad_noise = None
            if compute_noise and epoch % (log_interval * 10) == 0:
                grad_noise = compute_gradient_noise(model, loss_fn, X, y_bce)

            tracker.log(MetricsSnapshot(
                epoch=epoch,
                loss=loss.item(),
                accuracy=acc,
                weight_norm_l2=l2_norm,
                weight_norm_linf=linf_norm,
                margin=margin,
                normalized_margin=norm_margin,
                gradient_norm=grad_norm,
                gradient_noise=grad_noise
            ))

    return tracker


def run_experiment(args):
    """Run the baseline linear experiment."""

    print("=" * 60)
    print("Baseline Experiment: Linear Model on Separable Data")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    print(f"\nGenerating linearly separable data (n={args.n_samples}, dim={args.dim})...")
    X, y = generate_linearly_separable(
        n_samples=args.n_samples,
        dim=args.dim,
        margin=args.data_margin,
        seed=args.seed
    )

    print(f"Data shape: {X.shape}, Labels: {y.unique().tolist()}")

    # Optimizers to compare
    optimizers = ['gd', 'sgd', 'adam', 'lion']

    # Store results
    all_metrics = {}
    all_models = {}

    for opt_name in optimizers:
        print(f"\n{'='*40}")
        print(f"Training with {opt_name.upper()}")
        print(f"{'='*40}")

        # Fresh model for each optimizer
        model = LinearModel(input_dim=args.dim, output_dim=1, bias=False)

        # Use same initialization
        torch.manual_seed(args.seed)
        nn.init.normal_(model.linear.weight, mean=0, std=0.01)

        tracker = train_single_optimizer(
            model=model,
            optimizer_name=opt_name,
            X=X, y=y,
            epochs=args.epochs,
            lr=args.lr,
            log_interval=args.log_interval,
            compute_noise=not args.quick
        )

        all_metrics[opt_name.upper()] = tracker.to_dict()
        all_models[opt_name.upper()] = model

        # Save individual tracker
        tracker.save(str(output_dir / f"metrics_{opt_name}.json"))

        # Print final results
        final = tracker.history[-1]
        l2_ratio, linf_ratio = compute_margin_ratios(model, X, y)

        print(f"\nFinal Results for {opt_name.upper()}:")
        print(f"  Loss: {final.loss:.6f}")
        print(f"  Accuracy: {final.accuracy:.2%}")
        print(f"  Weight norm (ℓ2): {final.weight_norm_l2:.4f}")
        print(f"  Weight norm (ℓ∞): {final.weight_norm_linf:.4f}")
        print(f"  Margin: {final.margin:.4f}")
        print(f"  Normalized margin: {final.normalized_margin:.4f}")
        print(f"  ℓ2-margin ratio: {l2_ratio:.4f}")
        print(f"  ℓ∞-margin ratio: {linf_ratio:.4f}")

    # Save combined results
    with open(output_dir / "all_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Generate visualizations
    print("\n" + "=" * 40)
    print("Generating Visualizations")
    print("=" * 40)

    plot_training_curves(all_metrics, save_path=str(output_dir / "training_curves.png"))
    plot_margin_comparison(all_metrics, save_path=str(output_dir / "margin_comparison.png"))
    plot_weight_norm_ratio(all_metrics, save_path=str(output_dir / "weight_norm_ratio.png"))

    if args.dim == 2:
        plot_multi_decision_boundaries(
            all_models, X, y,
            save_path=str(output_dir / "decision_boundaries.png")
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Implicit Bias Comparison")
    print("=" * 60)

    print("\nNormalized Margin (margin / ||w||₂):")
    for opt, metrics in all_metrics.items():
        print(f"  {opt}: {metrics['normalized_margin'][-1]:.4f}")

    print("\nInterpretation:")
    print("  - GD should show larger ℓ2-margin (larger normalized margin)")
    print("  - Adam/Lion should show ℓ∞ bias (check weight norm ratio plot)")

    print(f"\nResults saved to: {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Baseline Linear Experiment")
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--dim', type=int, default=2, help='Input dimension')
    parser.add_argument('--data_margin', type=float, default=1.0, help='Data separation margin')
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/baseline', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode (skip expensive computations)')

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
