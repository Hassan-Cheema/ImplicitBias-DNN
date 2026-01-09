# Implicit Bias of Optimization in Deep Networks

Research project investigating why different optimizers converge to different solutions.

## Research Question

> **Do adaptive and sign-based optimizers induce fundamentally different implicit margin biases than gradient descent in deep, non-linear networks?**

We reproduce classical implicit bias results in linear models and extend them empirically to deeper, non-linear networks where theory does not fully apply.

---

## Margin Definition

For binary classification, the **normalized margin** is defined as:

$$\gamma = \frac{\min_i \, y_i \cdot f(x_i)}{\|w\|_p}$$

where:
- **p = 2** for GD/SGD (ℓ₂-margin)
- **p = ∞** for Adam/Lion (ℓ∞-margin)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline experiment (reproduces ℓ2 vs ℓ∞ margin theory)
python experiments/01_baseline_linear.py

# Run all experiments
python experiments/01_baseline_linear.py
python experiments/02_deeper_networks.py
python experiments/03_nonlinear_activations.py
python experiments/04_counter_examples.py
```

---

## Key Theoretical Results

| Optimizer | Implicit Bias | Margin Type |
|-----------|--------------|-------------|
| GD        | Converges to max ℓ₂-margin solution | Euclidean |
| Adam      | Converges to max ℓ∞-margin solution | Component-wise |
| Lion      | Sign-based updates (ℓ∞-like) | Component-wise |
| SGD       | Similar to GD + noise | Euclidean |

---

## Experiments

1. **Baseline Linear** - Reproduce theoretical ℓ₂ vs ℓ∞ margin results
2. **Deeper Networks** - How depth affects implicit bias
3. **Nonlinear Activations** - ReLU, Tanh, GELU comparisons
4. **Counter-Examples** - Cases where theory breaks down

> **Key Finding:** In certain overparameterized nonlinear regimes, we observe deviations from predicted margin behavior, suggesting architectural dependence of implicit bias.

---

## Project Structure

```
IBODN/
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

## Limitations and Open Questions

- Results are empirical and do not provide formal guarantees in deep nonlinear settings.
- Margin behavior appears architecture-dependent; identifying sufficient conditions remains open.
- Interaction between optimizer noise and implicit bias requires further theoretical analysis.
