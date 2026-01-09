"""
Visualization utilities for implicit bias experiments.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set style - use ggplot which is always available
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


def plot_training_curves(
    metrics_dict: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    show: bool = False
):
    """
    Plot training curves for multiple optimizers.

    Args:
        metrics_dict: Dict mapping optimizer name to metrics dict
        save_path: Path to save figure
        figsize: Figure size
        show: Whether to display the plot interactively
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    metrics_to_plot = [
        ('loss', 'Training Loss'),
        ('accuracy', 'Accuracy'),
        ('weight_norm_l2', 'Weight Norm (L2)'),
        ('weight_norm_linf', 'Weight Norm (Linf)'),
        ('margin', 'Margin'),
        ('normalized_margin', 'Normalized Margin'),
    ]

    for ax, (metric, title) in zip(axes.flatten(), metrics_to_plot):
        for opt_name, metrics in metrics_dict.items():
            if metric in metrics:
                ax.plot(metrics['epochs'], metrics[metric], label=opt_name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_margin_comparison(
    metrics_dict: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    show: bool = False
):
    """
    Plot L2 vs Linf margin evolution comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Normalized margin over time
    ax = axes[0]
    for opt_name, metrics in metrics_dict.items():
        ax.plot(metrics['epochs'], metrics['normalized_margin'],
                label=opt_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Margin')
    ax.set_title('Margin / ||w||_2 Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final margin comparison bar chart
    ax = axes[1]
    opt_names = list(metrics_dict.keys())
    final_margins = [metrics_dict[name]['normalized_margin'][-1] for name in opt_names]

    colors = plt.cm.tab10(np.linspace(0, 1, len(opt_names)))
    bars = ax.bar(opt_names, final_margins, color=colors)
    ax.set_ylabel('Final Normalized Margin')
    ax.set_title('Final Margin Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, final_margins):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_weight_norm_ratio(
    metrics_dict: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = False
):
    """
    Plot ratio of Linf/L2 weight norm over time.

    Higher ratio suggests Linf-margin bias (Adam, Lion).
    Lower ratio suggests L2-margin bias (GD).
    """
    fig, ax = plt.subplots(figsize=figsize)

    for opt_name, metrics in metrics_dict.items():
        l2_norms = np.array(metrics['weight_norm_l2'])
        linf_norms = np.array(metrics['weight_norm_linf'])
        ratio = linf_norms / (l2_norms + 1e-8)
        ax.plot(metrics['epochs'], ratio, label=opt_name, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('||w||_inf / ||w||_2')
    ax.set_title('Weight Norm Ratio (Higher = Linf bias)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_decision_boundary(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    title: str = "Decision Boundary",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    resolution: int = 200,
    show: bool = False
):
    """
    Plot 2D decision boundary.

    Args:
        model: Trained model
        X: Input features (n_samples, 2)
        y: Labels (+/-1)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        resolution: Grid resolution
        show: Whether to display interactively
    """
    if X.shape[1] != 2:
        raise ValueError("Can only plot 2D decision boundaries")

    model.eval()

    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()],
        dtype=torch.float32
    )

    with torch.no_grad():
        Z = model(grid).squeeze().numpy()
    Z = Z.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Contour fill
    contour = ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.7)
    plt.colorbar(contour, ax=ax, label='Model Output')

    # Decision boundary
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

    # Data points
    X_np = X.numpy()
    y_np = y.numpy()
    ax.scatter(X_np[y_np > 0, 0], X_np[y_np > 0, 1],
               c='blue', edgecolors='white', s=50, label='+1')
    ax.scatter(X_np[y_np < 0, 0], X_np[y_np < 0, 1],
               c='red', edgecolors='white', s=50, label='-1')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    model.train()


def plot_gradient_noise(
    metrics_dict: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = False
):
    """Plot gradient noise over training."""
    fig, ax = plt.subplots(figsize=figsize)

    for opt_name, metrics in metrics_dict.items():
        if 'gradient_noise' in metrics and metrics['gradient_noise'][0] is not None:
            ax.plot(metrics['epochs'], metrics['gradient_noise'],
                    label=opt_name, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Noise (Variance)')
    ax.set_title('Gradient Noise Over Training')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_multi_decision_boundaries(
    models: Dict[str, torch.nn.Module],
    X: torch.Tensor,
    y: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4),
    show: bool = False
):
    """
    Plot decision boundaries for multiple optimizers side by side.
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 150),
        np.linspace(y_min, y_max, 150)
    )

    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()],
        dtype=torch.float32
    )

    X_np = X.numpy()
    y_np = y.numpy()

    for ax, (name, model) in zip(axes, models.items()):
        model.eval()
        with torch.no_grad():
            Z = model(grid).squeeze().numpy()
        Z = Z.reshape(xx.shape)

        # Contour fill
        ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.7)
        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

        # Data points
        ax.scatter(X_np[y_np > 0, 0], X_np[y_np > 0, 1],
                   c='blue', edgecolors='white', s=30)
        ax.scatter(X_np[y_np < 0, 0], X_np[y_np < 0, 1],
                   c='red', edgecolors='white', s=30)

        ax.set_title(name)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        model.train()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_depth_analysis(
    depth_results: Dict[int, Dict[str, Dict]],
    metric: str = 'normalized_margin',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    show: bool = False
):
    """
    Plot how a metric changes with network depth for different optimizers.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    depths = sorted(depth_results.keys())
    optimizers = list(next(iter(depth_results.values())).keys())

    # Final metric vs depth
    ax = axes[0]
    for opt_name in optimizers:
        final_values = [depth_results[d][opt_name][metric][-1] for d in depths]
        ax.plot(depths, final_values, 'o-', label=opt_name, linewidth=2, markersize=8)

    ax.set_xlabel('Network Depth')
    ax.set_ylabel(f'Final {metric.replace("_", " ").title()}')
    ax.set_title(f'{metric.replace("_", " ").title()} vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Heatmap
    ax = axes[1]
    final_values_matrix = np.array([
        [depth_results[d][opt][metric][-1] for opt in optimizers]
        for d in depths
    ])

    im = ax.imshow(final_values_matrix, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(optimizers)))
    ax.set_xticklabels(optimizers)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f'{d} layers' for d in depths])
    ax.set_title(f'{metric.replace("_", " ").title()} Heatmap')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def create_summary_figure(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show: bool = False
):
    """
    Create a comprehensive summary figure with all key visualizations.
    """
    fig = plt.figure(figsize=figsize)

    # This is a template - customize based on actual results structure
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Add subplots based on what data is available
    # ... (implementation depends on result structure)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
