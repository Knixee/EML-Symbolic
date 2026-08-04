<div align="center">

# EML-Symbolic

**Differentiable symbolic regression built on a single binary operator.**

*One gate. Every elementary function. A neural network that learns equations, not just numbers.*

[![PyPI](https://img.shields.io/pypi/v/eml-symbolic?case=preserve&label=PyPI&color=blue)](https://pypi.org/project/eml-symbolic/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn API](https://img.shields.io/badge/scikit--learn-compatible-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

</div>

---

## What is this?

`emlsymbolic` is a small, sklearn-compatible library for **symbolic regression** — fitting data to an explicit, human-readable mathematical formula instead of an opaque set of weights.

Instead of hand-picking a library of primitive functions (`sin`, `cos`, `log`, `sqrt`, `+`, `×`, ...) and searching over combinations of them, this project builds every formula out of **one** operator:

```
eml(x, y) = exp(x) − ln(|y|)
```

This is the **EML (Exponential‑Minus‑Logarithm) gate**, discovered in 2026 by Andrzej Odrzywołek (Institute of Theoretical Physics, Jagiellonian University) and described in *[All Elementary Functions from a Single Binary Operator](https://arxiv.org/abs/2603.21852)*. The paper shows that `eml`, paired with the constant `1`, is functionally complete for the standard "scientific calculator" basis: addition, subtraction, multiplication, division, powers, roots, and the transcendental functions can all be written as a binary tree of nested `eml` calls (e.g. `exp(x) = eml(x, 1)`, `ln(x) = eml(1, eml(eml(1, x), 1))`).

The analogy the paper draws is to the **NAND gate** in digital logic: just as any Boolean circuit can be built from NAND alone, any elementary-function expression can be built from `eml` alone. That uniformity is exactly what makes it a good fit for gradient-based architecture search — every node in the network is the *same* operator, so choosing a formula reduces to choosing *which inputs feed which node*, a structural search problem that can be relaxed into something differentiable.

This project exists because I read the preprint on alphaXiv and wanted a working, installable implementation of the idea rather than just a mental note. For the primary source itself:

- 📄 Odrzywołek, A. — [*All Elementary Functions from a Single Binary Operator*](https://arxiv.org/abs/2603.21852) (arXiv:2603.21852) — also readable with inline discussion on [alphaXiv](https://www.alphaxiv.org/abs/2603.21852v2)
- 📚 [EML Operator — Wolfram MathWorld](https://mathworld.wolfram.com/EMLOperator.html) — a concise, citable definition and reference list
- 💻 [VA00/SymbolicRegressionPackage](https://github.com/VA00/SymbolicRegressionPackage) — the author's own reference implementation

`emlsymbolic` is **not** affiliated with Odrzywołek or the paper above — it's an independent, unofficial reimplementation of the core idea (differentiable search over nested `eml` gates), structured as a proper installable package rather than a research script. Treat it as an enthusiast's implementation, not a canonical or peer-reviewed one.

---

## How it works

The core idea is a **two-phase differentiable architecture search**:

### 1. The building block: `ParametricSafeEMLNode`

Every computational node in the network computes:

```
f(x, y) = exp(clamp(scale_x · x)) − ln(|scale_y · y| + ε)
```

where `x` and `y` are not fixed inputs — they are *selected* from a growing pool of available values (the original input features, a constant `1.0`, and the outputs of every earlier node) using a **Straight-Through Gumbel-Softmax** estimator. This lets the network pick, in a differentiable way, *which two quantities* each `eml` gate should combine, rather than requiring that choice to be fixed in advance.

### 2. The network: layered `eml` gates over a growing feature pool

`_EMLInternalNetwork` stacks several layers of these nodes. After each layer, its outputs are concatenated back into the shared pool, so later layers can route through earlier ones — building up nested expressions like `eml(eml(x, 1), eml(1, x))` organically, purely from gradient descent.

A final sparse linear layer combines whichever pool columns matter into the prediction, with an L1 penalty encouraging most of them to drop to zero.

### 3. Two training phases

| Phase | What happens |
|---|---|
| **Phase 1 — Relaxation** | Gumbel-Softmax temperature anneals from `2.0 → 0.2`. Routing logits (`gate_x`, `gate_y`) and node scales are all trained jointly with an entropy + L1 penalty pushing the soft routing toward a hard, sparse choice. |
| **Phase 2 — Fine-tuning** | The discrete structure is frozen (`argmax` of each gate), the output layer is pruned below `pruning_threshold`, and only the surviving numeric parameters (scales, final linear weights) are fine-tuned — now optimizing pure numbers, not architecture. |

The result is a *fixed, discrete computational graph* of `eml` gates with tuned numeric coefficients — which is just a formula.

### 4. Reading out the equation

Because the structure is frozen and every node has a known formula, the fitted model can be decoded directly into a string:

- **`to_symbolic()`** walks the frozen graph and emits the raw nested `exp`/`log` expression.
- **`to_standard_equation()`** pipes that through SymPy (`simplify` + `trigsimp`) to fold the `eml` nesting back down into familiar closed forms where possible (e.g. recovering something that reads like `sin(x)` instead of a tower of `exp`/`log` calls).

This is the whole point of the project: **you get an equation, not just a predictor.**

---

## Installation

```bash
git clone https://github.com/Knixee/EML-Symbolic.git
cd EML-Symbolic
pip install -e .
```

Requires Python 3.9+, PyTorch, scikit-learn, NumPy, and SymPy (for equation simplification).

---

## Quick start

```python
import numpy as np
from sklearn.model_selection import train_test_split
from emlsymbolic import EMLSymbolicRegressor

# A deliberately non-linear target: sin(x)*exp(-0.1x) + 0.5x^2
def target_function(x):
    return np.sin(x) * np.exp(-0.1 * x) + 0.5 * (x ** 2)

X = np.linspace(0.0, 5.0, 1000).reshape(-1, 1)
y = target_function(X[:, 0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = EMLSymbolicRegressor(
    n_layers=3,
    nodes_per_layer=3,
    epochs=1000,
    l1_lambda=0.05,
    pruning_threshold=0.03,
    random_state=42,
)

model.fit(X_train, y_train)

print("R² (train):", model.score(X_train, y_train))
print("R² (test): ", model.score(X_test, y_test))

print("Raw EML formula: ", model.to_symbolic(feature_names=["x"]))
print("Simplified form:  ", model.to_standard_equation())
```

Because `EMLSymbolicRegressor` subclasses scikit-learn's `BaseEstimator`/`RegressorMixin`, it drops straight into `Pipeline`, `GridSearchCV`, cross-validation, and every other tool in the sklearn ecosystem.

### Useful knobs

| Parameter | Role |
|---|---|
| `n_layers`, `nodes_per_layer` | Depth/width of the search space — bigger means richer formulas but slower, harder-to-simplify output. |
| `l1_lambda`, `gate_l1_lambda`, `gate_entropy_lambda` | Sparsity pressure on the output layer and on the routing gates during Phase 1. |
| `pruning_threshold` | Cutoff for dropping a term from the final formula. |
| `clamp_min` / `clamp_max` | Numerical safety rails on the `exp` argument — prevents Phase‑1 exploration from overflowing to `inf`/`NaN`. |
| `lr_phase1` / `lr_phase2` | Separate learning rates for architecture search vs. numeric fine-tuning. |
| `eval_set`, `max_nonfinite_streak` | Runtime diagnostics passed to `fit()`: validation tracking and an abort guard against training divergence. |

---

## Project layout

```
.
├── src/
│   └── emlsymbolic/
│       ├── __init__.py
│       ├── network.py       # _EMLInternalNetwork
│       ├── regressor.py     # EMLSymbolicRegressor (sklearn-compatible estimator)
│       └── nodes.py         # ParametricSafeEMLNode
├── tests/
│   └── test_regressor.py
├── benchmarks/               # (planned) standalone benchmark scripts
│   └── docs/                 # (planned) result plots, comparison tables, write-ups
├── pyproject.toml
└── README.md
```

> **Benchmarks:** the `benchmarks/` directory is planned to hold self-contained scripts comparing `emlsymbolic` against standard baselines (classic symbolic regression, plain MLPs, ground-truth recovery on synthetic functions), with generated plots collected under `benchmarks/docs/` for a visual before/after story.

---

## Why a single operator?

Traditional symbolic regression tools search over a *hand-designed* library of primitives, which means:

- the search space is combinatorial across primitive *types*, not just structure,
- adding a new function family means touching the search algorithm itself,
- gradient-based training over a heterogeneous primitive set is awkward — different primitives have different domains, gradients, and failure modes.

Because every node here is *literally the same operator*, the entire search space becomes homogeneous: choosing a formula is nothing more than choosing **which two pool entries feed each identical gate**. That is exactly the kind of discrete-choice problem Gumbel-Softmax relaxation was designed for, which is what makes an architecture like this trainable end-to-end with plain backprop.

---

## Status & disclaimers

This is a research-grade implementation exploring an idea from a 2026 preprint. Expression blow-up (formulas nesting `exp`/`log` many layers deep) is a known characteristic of the EML representation itself, not a bug specific to this code — `to_standard_equation()` includes a length guard (`max_sympify_length`) to avoid hanging SymPy on pathologically large expressions rather than trying to simplify everything unconditionally.

Contributions, issues, and benchmark results are welcome.

---

## Contributing
 
This started as a one-person itch-scratching project, so there's plenty of room to make it better — and plenty of ways to help that don't require touching the core architecture:
 
- 🐛 **Found a bug, a weird edge case, or a formula that doesn't simplify right?** Open an [issue](../../issues) — even a small reproducible example (a dataset shape + parameters that break things) is genuinely useful.
- 💡 **Have an idea** (a new benchmark, a cleaner API, a smarter pruning rule, better SymPy simplification)? Open an issue to discuss it before diving into a big PR, so effort doesn't go to waste on something that doesn't fit.
- 🔀 **Fork it.** If you want to experiment with a different node type, a different relaxation schedule, or just poke at the internals — forking and hacking is exactly what this is for. If it turns into something interesting, a PR back is always welcome.
No formal process beyond that — open an issue or a PR and it'll get a look.

---

## Citation

If you use **this implementation** in academic work and it was helpful, you can cite it:
 
```bibtex
@software{emlsymbolic2026,
  title   = {emlsymbolic: Differentiable symbolic regression via the EML operator},
  author  = {Knixee},
  url     = {https://github.com/Knixee/EML-Symbolic},
  year    = {2026}
}
```
 
However, the core mathematical contribution is Odrzywołek's. If you're using the EML operator concept, **please cite the original paper**:
 
```bibtex
@article{odrzywolek2026eml,
  title   = {All Elementary Functions from a Single Binary Operator},
  author  = {Odrzywołek, Andrzej},
  journal = {arXiv preprint arXiv:2603.21852},
  year    = {2026}
}
```
 
If both the idea and this specific implementation mattered to your work, citing both is perfectly fine — and honestly, more transparent about what you actually used.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) for details.