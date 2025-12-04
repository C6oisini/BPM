# BPM Privacy Toolkit

An open-source implementation of the **Bounded Perturbation Mechanism (BPM)** for privacy-preserving clustering.  
The goal of this repository is to provide a production-ready Python package plus a pair of reusable scripts that make it easy to reproduce the results from *“K-means clustering with local dχ-privacy”* and adapt them to new datasets.

---

## Highlights

- **Complete BPM primitives** – `bpm_privacy.BPM` exposes λ<sub>L</sub>, p<sub>L</sub>, λ<sub>2,r</sub>, density evaluation, and the exact sampling routines described in the paper.
- **Drop-in private clustering** – `PrivateKMeans`, `PrivateGMM`, and `PrivateTMM` encapsulate the “user-side perturbation + server-side clustering” workflow so you can adopt BPM with one import.
- **Single entry point for experiments** – `scripts/run_experiment.py` sweeps ε/L grids, runs multiple trials, and exports SSE / Silhouette / ARI / NMI summaries to stdout or CSV.
- **Mechanism inspection utility** – `scripts/inspect_mechanism.py` prints numerical constants and optionally visualizes 2-D sampling behavior for sanity checks.
- **Tested building blocks** – the lightweight `pytest` suite covers both the mechanism math and the K-means integration to catch regressions early.

---

## Installation

### Recommended: uv workflow

[uv](https://github.com/astral-sh/uv) provides deterministic lockfiles (`uv.lock`) and fast isolated environments.

```bash
uv venv                               # creates .venv using settings from pyproject
source .venv/bin/activate             # PowerShell: .\.venv\Scripts\Activate.ps1
uv pip install -e .                   # editable install with uv’s resolver
# optional extras
uv pip install -r requirements.txt    # installs pinned runtime dependencies
```

You can then execute any script via `uv run`, e.g. `uv run scripts/run_experiment.py ...` or `uv run pytest`.

### Plain pip (alternative)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt  # optional pinned spec
```

---

## Repository layout

```
.
├── bpm_privacy/              # reusable Python package
│   ├── mechanism.py          # λL, pL, μL, λ2r, density helpers
│   ├── sampling.py           # Algorithms 2/3/4 from the paper
│   ├── private_kmeans.py     # Algorithm 1 implementation
│   ├── private_gmm.py        # BPM-augmented Gaussian Mixture Model
│   └── private_tmm.py        # Student's t Mixture + BPM
├── scripts/
│   ├── run_experiment.py     # CLI for batch experiments
│   └── inspect_mechanism.py  # CLI for λL / pL diagnostics
├── tests/                    # pytest-based regression checks
├── figures/                  # archived example plots (optional)
├── BPM.pdf                   # original paper for reference
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

---

## Quick start

### 1. Run a full experiment sweep

```bash
python scripts/run_experiment.py \
  --dataset iris \
  --algorithms kmeans gmm \
  --epsilons 0.5 1 2 4 \
  --Ls 0.3 0.5 \
  --trials 5 \
  --csv results.csv
```

Key CLI options:

| Flag | Description |
|------|-------------|
| `--dataset {iris,blobs}` | Use the Iris benchmark or generate synthetic blobs. |
| `--algorithms {kmeans,gmm,tmm}` | Evaluate one or more algorithms in a single run. |
| `--epsilons`, `--Ls` | Provide one or many ε/L values to map the privacy–utility trade-off. |
| `--trials N` | Repeat each configuration `N` times and average the metrics. |
| `--skip-baseline` | Skip the non-private baseline if you only need private runs. |
| `--csv path` | Persist the summary table for downstream analysis. |
| `--plot path` | Save multi-metric line charts (SSE/Silhouette/ARI/NMI vs ε) for every algorithm/L pair. |

**Enumerated CLI options from `scripts/run_experiment.py`:**

- `--dataset {iris, blobs}`
- `--algorithms {kmeans, gmm, tmm}`
- `--Ls` accepts any float(s); use multiple values to explore different bounds.
- `--epsilons` accepts any float(s); defaults to `1.0 2.0 4.0`.

For completeness, the remaining parameters are continuous/integers: `--samples`, `--features`, `--clusters`, `--trials`, `--seed`, `--tmm-nu`, `--tmm-alpha`, `--tmm-max-iter`, plus the flags `--skip-baseline` and `--csv`.

The script automatically normalizes every dataset to `[0,1]^d` so the BPM assumptions hold.

### 2. Inspect mechanism constants and sampling quality

```bash
python scripts/inspect_mechanism.py \
  --epsilon 2.0 \
  --L 0.3 \
  --dimension 2 \
  --samples 2000 \
  --plot figures/bpm_scatter.png
```

This command prints λ<sub>L</sub>, p<sub>L</sub>, λ<sub>2,r</sub>, and validates that the empirical fraction of samples inside the L-ball matches the theoretical probability.  
In 2-D, it also saves a scatter plot that highlights the ball boundary and report space.

### 3. Rebuild legacy figures

Need the original PNGs from the paper replication? Run:

```bash
python scripts/legacy_figures.py           # regenerates every historical figure
python scripts/legacy_plots.py             # focused Iris metric curves & scatter grid
python scripts/legacy_figures.py --figures bpm_sampling_distribution iris_evaluation
```

All figures are saved under `figures/` using their historical filenames (e.g., `iris_evaluation.png`, `all_methods_comparison.png`), so downstream docs continue to work.

### 4. Integrate directly in your own pipeline

```python
import numpy as np
from sklearn.datasets import make_blobs
from bpm_privacy import PrivateKMeans

X, _ = make_blobs(n_samples=300, n_features=2, centers=3, random_state=42)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # map to [0,1]^d

model = PrivateKMeans(n_clusters=3, epsilon=2.0, L=0.4, random_state=42)
model.fit(X)
labels = model.predict(X)
print("Private SSE:", model.compute_sse(X))
```

---

## Testing

```bash
pytest
```

- `tests/test_mechanism.py` confirms that densities decrease with distance, all samples stay inside the report cube, and the inside-ball probability matches p<sub>L</sub>.
- `tests/test_private_kmeans.py` checks that perturbations respect bounds and that increasing ε improves SSE (within a tolerance), giving you confidence in the privacy/utility trade-off.

---

## Roadmap

- [ ] Add Laplace / Gaussian baselines for direct comparisons.
|- [ ] Provide ready-to-run Jupyter notebooks for teaching/demo purposes.
|- [ ] Publish pre-generated CSV benchmarks for typical ε/L grids.

Contributions are welcome! Please open an issue for feature proposals or submit a PR directly.

---

## Citation

If you build upon this implementation in academic work, cite the original paper and optionally link back to this repository:

> Mengmeng Yang, Ivan Tjuawinata, Kwok-Yan Lam. *K-means clustering with local dχ-privacy for privacy-preserving data analysis.* Journal of LaTeX Class Files, Vol. 14, No. 8, 2021.

The PDF is included at `BPM.pdf` for convenience.

---

## License & attribution

This toolkit is released for research and practical experimentation. Please ensure that any downstream use complies with your jurisdiction’s privacy regulations and that the original authors of BPM are credited appropriately. If you have questions or would like to showcase a project built on top of this work, feel free to open an issue. 
