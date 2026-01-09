# Implicit Bias of Optimization in Deep Networks

## Abstract

This project investigates the implicit bias of optimization algorithms in deep neural networks by revisiting classical margin theory in linear settings and extending it empirically to deeper, non-linear models. We compare gradient descent (GD), stochastic gradient descent (SGD), adaptive optimizers (Adam), and sign-based methods (Lion) across multiple network architectures and activation functions. Our findings reveal modality-dependent deviations from linear margin predictions in overparameterized regimes, suggesting architectural influence on implicit bias.

---

## Research Question

> **Do adaptive and sign-based optimizers induce fundamentally different implicit margin biases than gradient descent in deep, non-linear networks?**

---

## Contribution

This repository provides an empirical extension of implicit bias theory to deep, non-linear networks. It demonstrates where classical margin theory holds and where it systematically deviates in overparameterized regimes.

---

## Margin Definition

For binary classification, the **normalized margin** is defined as:

$$
\gamma = \frac{\min_i\, y_i\, f(x_i)}{\lVert w \rVert_p}
$$

where:
- \(p = 2\) for GD/SGD (ℓ₂-margin)
- \(p = \infty\) for Adam/Lion (ℓ∞-margin)

---

## Key Theoretical Results

| Optimizer | Implicit Bias | Margin Type |
|-----------|--------------|-------------|
| GD        | Converges to max ℓ₂-margin solution | Euclidean |
| Adam      | Converges to max ℓ∞-margin solution | Component-wise |
| Lion      | Sign-based updates (ℓ∞-like) | Component-wise |
| SGD       | Similar to GD with noise | Euclidean |

---

## Experiments

1. **Baseline Linear** - Reproduce theoretical ℓ₂ vs ℓ∞ margin results
2. **Deeper Networks** - How depth affects implicit bias
3. **Nonlinear Activations** - ReLU, Tanh, GELU comparisons
4. **Counter-Examples** - Cases where theory breaks down

---

## Expected Results Summary

Running the experiments should result in observations like:

- Linear models replicate classical ℓ₂ and ℓ∞ margin behaviors.
- Deeper architectures (with ReLU, GELU) show systematic deviations from classical theory.
- Optimizer noise (SGD) increases variance in margin trajectories.
- Counter-example settings highlight architecture-driven implicit bias shifts.

---

## Quick Start

```bash
pip install -r requirements.txt
python experiments/01_baseline_linear.py
python experiments/02_deeper_networks.py
python experiments/03_nonlinear_activations.py
python experiments/04_counter_examples.py
```

---

## Limitations & Future Work

- The current study is empirical and does not provide formal convergence guarantees in deep nonlinear regimes.
- Observed deviations may depend on specific architectures and datasets.
- A theoretical analysis of Lion's implicit bias remains open and is a direction for future work.

---

## Project Structure

```
ImplicitBias-DNN/
├── src/                    # Core modules
│   ├── models.py          # Neural network architectures
│   ├── optimizers.py      # Custom optimizers (GD, Lion)
│   ├── metrics.py         # Tracking utilities
│   ├── data.py            # Data generation
│   └── visualization.py   # Plotting functions
├── experiments/           # Experiment scripts
├── notebooks/             # Interactive analysis
├── tests/                 # Unit tests
└── results/               # Generated outputs
```

---

## Reproducibility Notes

All experiments are run with fixed seeds (default: 0–4) to estimate variance. Results and plots are saved under the `results/` directory. Each script prints summary stats for easy inspection.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{cheema2026implicitbias,
  title={DeepImplicitBias: Investigating Implicit Bias in Deep Neural Networks},
  author={Hassan Cheema},
  year={2026},
  note={GitHub repository: https://github.com/Hassan-Cheema/ImplicitBias-DNN}
}
```
