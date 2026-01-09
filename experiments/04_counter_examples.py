"""
Experiment 4: Counter-Examples

Find cases where the theoretical predictions about implicit bias break down.

Scenarios to explore:
1. Non-separable data (finite margin impossible)
2. Extreme overparameterization
3. Near-singular data configurations
4. Very small learning rates (long-time dynamics)

Run: python experiments/04_counter_examples.py
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
import matplotlib.pyplot as plt

from src.models import LinearModel, ShallowMLP, OverparameterizedMLP
from src.optimizers import get_optimizer
from src.metrics import (
    MetricsTracker, MetricsSnapshot,
    compute_weight_norms, compute_margin, compute_accuracy,
    compute_gradient_norm, compute_margin_ratios
)
from src.data import (
    generate_linearly_separable, generate_high_dim_separable,
    generate_xor_data, generate_moons_data
)


def train_and_track(
    model: nn.Module,
    optimizer_name: str,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    log_interval: int = 10
) -> MetricsTracker:
    """Train and track metrics."""

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


def counter_example_1_non_separable(args, output_dir):
    """
    Counter-example 1: Non-separable data

    Theory predicts margin growth for separable data.
    What happens with overlapping classes?
    """
    print("\n" + "=" * 60)
    print("Counter-Example 1: Non-Separable Data")
    print("=" * 60)

    # Generate overlapping data (negative margin)
    print("\nGenerating non-separable data with overlapping classes...")
    X, y = generate_linearly_separable(
        n_samples=200,
        dim=2,
        margin=-0.5,  # Negative margin = overlap
        noise=0.5,
        seed=args.seed
    )

    results = {}
    optimizers = ['gd', 'adam', 'lion']

    for opt_name in optimizers:
        print(f"\n  Training with {opt_name.upper()}...")
        model = LinearModel(input_dim=2, output_dim=1, bias=False)

        torch.manual_seed(args.seed)
        nn.init.normal_(model.linear.weight, std=0.01)

        tracker = train_and_track(model, opt_name, X, y, args.epochs, args.lr)
        results[opt_name] = tracker.to_dict()

        final = tracker.history[-1]
        print(f"    Final - Loss: {final.loss:.4f}, Acc: {final.accuracy:.2%}")
        print(f"    Margin: {final.margin:.4f} (expected to be negative)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, metric in enumerate(['loss', 'accuracy', 'margin']):
        for opt, data in results.items():
            axes[i].plot(data['epochs'], data[metric], label=opt.upper())
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.title())
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[2].axhline(y=0, color='red', linestyle='--', label='Zero margin')

    plt.suptitle('Counter-Example 1: Non-Separable Data')
    plt.tight_layout()
    plt.savefig(output_dir / "counter_example_1_non_separable.png", dpi=150)
    plt.show()

    return results


def counter_example_2_extreme_overparameterization(args, output_dir):
    """
    Counter-example 2: Extreme overparameterization

    With 100x more parameters than samples, how do implicit biases change?
    """
    print("\n" + "=" * 60)
    print("Counter-Example 2: Extreme Overparameterization")
    print("=" * 60)

    n_samples = 50
    X, y = generate_linearly_separable(n_samples=n_samples, dim=2, seed=args.seed)

    # Create massively overparameterized network
    model_params = n_samples * 100  # 100x overparameterized
    width = int(np.sqrt(model_params / 3))  # Approximate width for 2 layers

    print(f"\nData: {n_samples} samples")

    results = {}
    widths_to_test = [16, 64, 256, 1024]

    for width in widths_to_test:
        model = OverparameterizedMLP(
            input_dim=2,
            width=width,
            depth=2,
            output_dim=1
        )
        n_params = model.count_parameters()
        ratio = n_params / n_samples

        print(f"\n  Width {width}: {n_params:,} params ({ratio:.1f}x overparameterized)")

        width_results = {}
        for opt_name in ['gd', 'adam', 'lion']:
            torch.manual_seed(args.seed)
            model = OverparameterizedMLP(input_dim=2, width=width, depth=2, output_dim=1)

            tracker = train_and_track(model, opt_name, X, y, args.epochs // 2, args.lr * 0.1)
            width_results[opt_name] = tracker.to_dict()

            final = tracker.history[-1]
            l2_ratio, linf_ratio = compute_margin_ratios(model, X, y)
            print(f"    {opt_name.upper()}: Acc={final.accuracy:.2%}, "
                  f"ℓ2-ratio={l2_ratio:.4f}, ℓ∞-ratio={linf_ratio:.4f}")

        results[f"width_{width}"] = width_results

    # Analyze: does the ℓ2/ℓ∞ distinction persist?
    print("\n  Key finding: Check if margin ratio distinction persists with overparameterization")

    return results


def counter_example_3_high_dimensional(args, output_dir):
    """
    Counter-example 3: High-dimensional data with low effective dimension

    Does implicit bias favor the right subspace?
    """
    print("\n" + "=" * 60)
    print("Counter-Example 3: High-Dimensional with Low Effective Dimension")
    print("=" * 60)

    X, y = generate_high_dim_separable(
        n_samples=100,
        dim=100,
        effective_dim=5,
        margin=1.0,
        seed=args.seed
    )

    print(f"\nData: {X.shape[0]} samples in {X.shape[1]}D, effective dim=5")

    results = {}

    for opt_name in ['gd', 'adam', 'lion']:
        print(f"\n  Training with {opt_name.upper()}...")

        model = LinearModel(input_dim=100, output_dim=1, bias=False)
        torch.manual_seed(args.seed)
        nn.init.normal_(model.linear.weight, std=0.01)

        tracker = train_and_track(model, opt_name, X, y, args.epochs, args.lr * 0.1)
        results[opt_name] = tracker.to_dict()

        # Analyze weight distribution
        weights = model.get_weights()
        weight_sparsity = (weights.abs() < 0.01 * weights.abs().max()).float().mean()

        final = tracker.history[-1]
        print(f"    Accuracy: {final.accuracy:.2%}")
        print(f"    Weight sparsity (≈0): {weight_sparsity:.2%}")
        print(f"    Weight ℓ∞/ℓ2 ratio: {final.weight_norm_linf/final.weight_norm_l2:.4f}")

    # Plot weight distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, opt_name in zip(axes, ['gd', 'adam', 'lion']):
        model = LinearModel(input_dim=100, output_dim=1, bias=False)
        torch.manual_seed(args.seed)
        nn.init.normal_(model.linear.weight, std=0.01)

        train_and_track(model, opt_name, X, y, args.epochs, args.lr * 0.1)

        weights = model.get_weights().numpy()
        ax.bar(range(len(weights)), np.abs(weights))
        ax.set_title(f'{opt_name.upper()} Weight Magnitudes')
        ax.set_xlabel('Weight Index')
        ax.set_ylabel('|w|')

    plt.suptitle('Counter-Example 3: Weight Distribution in High Dimensions')
    plt.tight_layout()
    plt.savefig(output_dir / "counter_example_3_high_dim_weights.png", dpi=150)
    plt.show()

    return results


def counter_example_4_learning_rate_extremes(args, output_dir):
    """
    Counter-example 4: Extreme learning rates

    Very small LR: do long-time dynamics differ from theory?
    Very large LR: does instability affect implicit bias?
    """
    print("\n" + "=" * 60)
    print("Counter-Example 4: Learning Rate Extremes")
    print("=" * 60)

    X, y = generate_linearly_separable(n_samples=100, dim=2, seed=args.seed)

    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    results = {}

    for lr in learning_rates:
        print(f"\n  Learning rate: {lr}")
        results[f"lr_{lr}"] = {}

        for opt_name in ['gd', 'adam']:
            model = LinearModel(input_dim=2, output_dim=1, bias=False)
            torch.manual_seed(args.seed)
            nn.init.normal_(model.linear.weight, std=0.01)

            # More epochs for small LR
            epochs = int(args.epochs * (0.01 / max(lr, 1e-4)))
            epochs = min(epochs, 20000)

            try:
                tracker = train_and_track(model, opt_name, X, y, epochs, lr, log_interval=max(1, epochs//100))
                final = tracker.history[-1]

                results[f"lr_{lr}"][opt_name] = {
                    'converged': final.accuracy > 0.99,
                    'final_loss': final.loss,
                    'final_margin': final.normalized_margin,
                    'weight_ratio': final.weight_norm_linf / (final.weight_norm_l2 + 1e-8)
                }

                print(f"    {opt_name.upper()}: Converged={final.accuracy > 0.99}, "
                      f"Margin={final.normalized_margin:.4f}")
            except Exception as e:
                print(f"    {opt_name.upper()}: Failed ({e})")
                results[f"lr_{lr}"][opt_name] = {'converged': False, 'error': str(e)}

    return results


def run_experiment(args):
    """Run all counter-example experiments."""

    print("=" * 60)
    print("Experiment 4: Counter-Examples Where Theory Breaks")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run each counter-example
    all_results['non_separable'] = counter_example_1_non_separable(args, output_dir)
    all_results['overparameterization'] = counter_example_2_extreme_overparameterization(args, output_dir)
    all_results['high_dimensional'] = counter_example_3_high_dimensional(args, output_dir)
    all_results['learning_rates'] = counter_example_4_learning_rate_extremes(args, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Counter-Examples Findings")
    print("=" * 60)

    print("""
Key Observations:

1. NON-SEPARABLE DATA:
   - Theory assumes separability; without it, margin can be negative
   - Optimizers still differ in their approach to the problem

2. EXTREME OVERPARAMETERIZATION:
   - With many solutions available, implicit bias still guides selection
   - But the distinction may become less pronounced

3. HIGH-DIMENSIONAL:
   - Implicit bias determines which subspace is used
   - Different optimizers may use different weight structures

4. LEARNING RATE EFFECTS:
   - Very small LR: true limit behavior
   - Large LR: discrete-time effects dominate
    """)

    # Save summary
    with open(output_dir / "counter_examples_summary.json", 'w') as f:
        # Serialize what we can
        json.dump({
            'experiments_run': list(all_results.keys()),
            'learning_rates_tested': all_results.get('learning_rates', {})
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Counter-Examples Experiment")
    parser.add_argument('--epochs', type=int, default=3000, help='Base training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Base learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/counter_examples',
                        help='Output directory')

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
