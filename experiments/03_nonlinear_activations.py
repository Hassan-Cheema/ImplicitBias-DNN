"""
Experiment 3: Nonlinear Activations

Investigate how different activation functions affect implicit bias.

Key questions:
- Does the implicit bias differ between ReLU, Tanh, GELU?
- How do activations affect margin growth dynamics?
- Are there activation-optimizer interactions?

Run: python experiments/03_nonlinear_activations.py
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

from src.models import ShallowMLP
from src.optimizers import get_optimizer
from src.metrics import (
    MetricsTracker, MetricsSnapshot,
    compute_weight_norms, compute_margin, compute_accuracy,
    compute_gradient_norm
)
from src.data import generate_xor_data, generate_moons_data, get_dataset
from src.visualization import plot_multi_decision_boundaries


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
    """Run the nonlinear activations experiment."""

    print("=" * 60)
    print("Experiment 3: Nonlinear Activations")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate XOR data (requires nonlinearity)
    print(f"\nGenerating {args.dataset} data (n={args.n_samples})...")
    X, y = get_dataset(args.dataset, n_samples=args.n_samples, seed=args.seed)

    print(f"Data shape: {X.shape}")

    # Activations and optimizers to test
    activations = ['relu', 'tanh', 'gelu', 'leaky_relu']
    optimizers = ['gd', 'sgd', 'adam', 'lion']

    # Store results: activation -> optimizer -> metrics
    all_results = {}

    for activation in activations:
        print(f"\n{'='*50}")
        print(f"Testing activation: {activation.upper()}")
        print(f"{'='*50}")

        all_results[activation] = {}

        for opt_name in optimizers:
            print(f"\n  Training with {opt_name.upper()}...")

            model = ShallowMLP(
                input_dim=X.shape[1],
                hidden_dim=args.hidden_dim,
                output_dim=1,
                activation=activation
            )

            # Same initialization
            torch.manual_seed(args.seed)
            for param in model.parameters():
                if param.dim() > 1:
                    nn.init.xavier_normal_(param)

            tracker = train_model(
                model=model,
                optimizer_name=opt_name,
                X=X, y=y,
                epochs=args.epochs,
                lr=args.lr,
                log_interval=args.log_interval
            )

            all_results[activation][opt_name.upper()] = {
                'metrics': tracker.to_dict(),
                'final_acc': tracker.history[-1].accuracy,
                'final_margin': tracker.history[-1].normalized_margin,
            }

            final = tracker.history[-1]
            print(f"    Final - Loss: {final.loss:.4f}, Acc: {final.accuracy:.2%}, "
                  f"NormMargin: {final.normalized_margin:.4f}")

            tracker.save(str(output_dir / f"metrics_{activation}_{opt_name}.json"))

    # Save combined results
    with open(output_dir / "activation_results.json", 'w') as f:
        # Can't directly serialize, create summary
        summary = {
            act: {
                opt: {
                    'final_acc': res['final_acc'],
                    'final_margin': res['final_margin']
                }
                for opt, res in opts.items()
            }
            for act, opts in all_results.items()
        }
        json.dump(summary, f, indent=2)

    # Create comparison visualization
    print("\n" + "=" * 40)
    print("Generating Visualizations")
    print("=" * 40)

    # Heatmap: Activation x Optimizer for final accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy heatmap
    acc_matrix = np.array([
        [all_results[act][opt]['final_acc'] for opt in optimizers]
        for act in activations
    ])

    ax = axes[0]
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(optimizers)))
    ax.set_xticklabels([o.upper() for o in optimizers])
    ax.set_yticks(range(len(activations)))
    ax.set_yticklabels([a.upper() for a in activations])
    ax.set_title('Final Accuracy')
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(len(activations)):
        for j in range(len(optimizers)):
            ax.text(j, i, f'{acc_matrix[i, j]:.2%}', ha='center', va='center')

    # Margin heatmap
    margin_matrix = np.array([
        [all_results[act][opt]['final_margin'] for opt in optimizers]
        for act in activations
    ])

    ax = axes[1]
    im = ax.imshow(margin_matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(optimizers)))
    ax.set_xticklabels([o.upper() for o in optimizers])
    ax.set_yticks(range(len(activations)))
    ax.set_yticklabels([a.upper() for a in activations])
    ax.set_title('Final Normalized Margin')
    plt.colorbar(im, ax=ax)

    for i in range(len(activations)):
        for j in range(len(optimizers)):
            ax.text(j, i, f'{margin_matrix[i, j]:.3f}', ha='center', va='center',
                   color='white' if margin_matrix[i, j] < margin_matrix.mean() else 'black')

    plt.tight_layout()
    plt.savefig(output_dir / "activation_optimizer_heatmap.png", dpi=150)
    plt.show()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Activation vs Optimizer Interactions")
    print("=" * 60)

    print("\nBest performing combinations (by accuracy):")
    flat_results = [
        (act, opt, all_results[act][opt]['final_acc'])
        for act in activations
        for opt in [o.upper() for o in optimizers]
    ]
    flat_results.sort(key=lambda x: x[2], reverse=True)

    for act, opt, acc in flat_results[:5]:
        print(f"  {act.upper()} + {opt}: {acc:.2%}")

    print(f"\nResults saved to: {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Nonlinear Activations Experiment")
    parser.add_argument('--dataset', type=str, default='xor',
                        choices=['xor', 'moons', 'circles', 'spiral'],
                        help='Dataset type')
    parser.add_argument('--n_samples', type=int, default=300, help='Number of samples')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/nonlinear_activations',
                        help='Output directory')

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
