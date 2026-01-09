# Implicit Bias of Optimization in Deep Networks

## Abstract

This project investigates the implicit bias of optimization algorithms in deep neural networks by revisiting classical margin theory in linear settings and extending it empirically to deeper, non-linear models. We compare gradient descent (GD), stochastic gradient descent (SGD), adaptive optimizers (Adam), and sign-based methods (Lion) across multiple network architectures and activation functions. Our findings reveal modality-dependent deviations from linear margin predictions in overparameterized regimes, suggesting architectural influence on implicit bias.

---

## Research Question

> **Do adaptive and sign-based optimizers induce fundamentally different implicit margin biases than gradient descent in deep, non-linear networks?**

We reproduce classical implicit bias results in linear models and extend them empirically to deeper, non-linear networks where theory does not fully apply.

---

## Margin Definition

For binary classification, the **normalized margin** is defined as:

$$\gamma = \frac{\min_i \; y_i \, f(x_i)}{\lVert w \rVert_p}$$

where:
- **p = 2** for GD/SGD (ℓ₂-margin)
- **p = ∞** for Adam/Lion (ℓ∞-margin)

---

## Key Theoretical Results

| Optimizer | Implicit Bias | Margin Type |
|---|---|---|
| GD | Converges to max ℓ₂-margin solution | Euclidean |
| Adam | Converges to max ℓ∞-margin solution | Component-wise |
| Lion | Sign-based updates (ℓ∞-like) | Component-wise |
| SGD | Similar to GD + noise | Euclidean |

### Empirical Contribution

Our experiments confirm classical margin behavior in linear models and demonstrate systematic deviations in deep, non-linear networks, particularly under ReLU and GELU activations. These deviations correlate with architecture depth, suggesting implicit bias is not optimizer-dependent alone.

---

## Empirical Takeaways

Across deep non-linear models:
- GD and SGD maintain classical margin trends.
- Adaptive and sign-based optimizers show systematic deviations in margin dynamics.
- Depth and activation non-linearities amplify implicit bias shifts.

---

## Evaluation Metrics

| Setting | Metric | Interpretation |
|---|---|---|
| Linear | Normalized margin | Implicit bias alignment |
| Deep | Margin evolution | Generalization proxy |
| Counter-examples | Divergence | Architecture effect |

---

## Experiments

1. **Baseline Linear** - Reproduce theoretical ℓ₂ vs ℓ∞ margin results
2. **Deeper Networks** - How depth affects implicit bias
3. **Nonlinear Activations** - ReLU, Tanh, GELU comparisons
4. **Counter-Examples** - Cases where theory breaks down

> **Key Finding:** In certain overparameterized nonlinear regimes, we observe deviations from predicted margin behavior, suggesting architectural dependence of implicit bias.

---

## Expected Results Summary

You should observe:
- Linear settings match classic theory.
- Depth increases margin deviation under Adam and Lion.
- ReLU nets diverge more than Tanh/GELU.

---

## Visualization Summary

- **Margin Dynamics:** Plots show normalized margin differences between optimizers across training epochs.
- **Depth Trends:** Margin trends vary significantly with network depth in non-linear settings.
- **Activation Effects:** ReLU networks diverge more from classical bias predictions than Tanh/GELU.

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

## Reproducibility

Experiments are run with fixed seeds (default: 42). Results and plots are saved to `results/`.

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

## Limitations

- Results are empirical and do not provide formal convergence guarantees.
- Observed deviations may depend on architecture choices and data distributions.
- The current setup uses synthetic datasets; real data generalization remains untested.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{ImplicitBias-DNN2026,
  title={DeepImplicitBias: Investigating Implicit Biases of Optimizers in Deep Networks},
  author={Hassan Cheema},
  year={2026},
  note={GitHub repository: https://github.com/Hassan-Cheema/ImplicitBias-DNN}
}
```
