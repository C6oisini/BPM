#!/usr/bin/env python3
"""
Consolidated experiment runner for BPM-based clustering.

Usage examples:
    python scripts/run_experiment.py --dataset iris --algorithms kmeans gmm --epsilons 1 2 4
    python scripts/run_experiment.py --dataset blobs --clusters 4 --algorithms kmeans \
        --epsilons 0.5 1 --Ls 0.2 0.4 --trials 5 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from bpm_privacy import BPGT, PrivateGMM, PrivateKMeans, PrivateTMM, TMM


def normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalize data to the [0, 1]^d domain required by BPM."""
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    denom = np.clip(X_max - X_min, 1e-12, None)
    return (X - X_min) / denom


def compute_sse(X: np.ndarray, labels: np.ndarray, centers: Optional[np.ndarray]) -> float:
    """Compute the sum of squared errors for a clustering assignment."""
    if centers is None or labels is None:
        return float("nan")
    sse = 0.0
    for idx, center in enumerate(centers):
        cluster = X[labels == idx]
        if len(cluster) > 0:
            sse += np.sum((cluster - center) ** 2)
    return float(sse)


def safe_silhouette(X: np.ndarray, labels: Optional[np.ndarray]) -> float:
    """Return silhouette score or NaN if undefined."""
    if labels is None:
        return float("nan")
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def safe_ari(y_true: Optional[np.ndarray], labels: Optional[np.ndarray]) -> float:
    """Return ARI if ground truth is available; NaN otherwise."""
    if y_true is None or labels is None:
        return float("nan")
    if len(np.unique(labels)) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(adjusted_rand_score(y_true, labels))


def safe_nmi(y_true: Optional[np.ndarray], labels: Optional[np.ndarray]) -> float:
    """Return NMI if ground truth is available; NaN otherwise."""
    if y_true is None or labels is None:
        return float("nan")
    if len(np.unique(labels)) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(normalized_mutual_info_score(y_true, labels))


def dataset_loader(name: str, samples: int, features: int, clusters: int, seed: int) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Load or generate a dataset."""
    if name == "iris":
        iris = load_iris()
        return iris.data, iris.target, clusters or len(np.unique(iris.target))
    if name == "blobs":
        X, y = make_blobs(
            n_samples=samples,
            n_features=features,
            centers=clusters,
            cluster_std=0.5,
            random_state=seed,
        )
        return X, y, clusters
    raise ValueError(f"Unsupported dataset '{name}'.")


@dataclass
class TrialMetrics:
    labels: Optional[np.ndarray]
    centers: Optional[np.ndarray]


def summarize_metrics(X: np.ndarray, y_true: Optional[np.ndarray], trial_results: Iterable[TrialMetrics]) -> Dict[str, float]:
    """Aggregate metrics across multiple trials."""
    sses, silhouettes, aris, nmis = [], [], [], []
    for trial in trial_results:
        sses.append(compute_sse(X, trial.labels, trial.centers))
        silhouettes.append(safe_silhouette(X, trial.labels))
        aris.append(safe_ari(y_true, trial.labels))
        nmis.append(safe_nmi(y_true, trial.labels))

    def nanmean(values: List[float]) -> float:
        arr = np.array(values, dtype=float)
        if np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    return {
        "sse": nanmean(sses),
        "silhouette": nanmean(silhouettes),
        "ari": nanmean(aris),
        "nmi": nanmean(nmis),
    }


def run_baseline(
    algorithm: str,
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    n_clusters: int,
    seed: int,
    nu: float,
    alpha: float,
    max_iter: int,
) -> Dict[str, float]:
    """Execute the non-private baseline for an algorithm."""
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        labels = model.fit_predict(X)
        centers = model.cluster_centers_
    elif algorithm == "gmm":
        model = GaussianMixture(n_components=n_clusters, n_init=10, random_state=seed)
        model.fit(X)
        labels = model.predict(X)
        centers = model.means_
    elif algorithm in {"tmm", "bpgt"}:
        model = TMM(
            n_components=n_clusters,
            nu=nu,
            alpha=alpha,
            max_iter=max_iter,
            random_state=seed,
            fixed_nu=True,
        )
        model.fit(X)
        labels = model.labels_
        centers = model.means_
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'.")

    metrics = summarize_metrics(X, y_true, [TrialMetrics(labels=labels, centers=centers)])
    return metrics


def run_private_trials(
    algorithm: str,
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    n_clusters: int,
    epsilon: float,
    L: float,
    trials: int,
    seed: int,
    nu: float,
    alpha: float,
    max_iter: int,
    bpgt_params: Optional[dict] = None,
) -> Dict[str, float]:
    """Execute multiple private trials and aggregate metrics."""
    trial_results: List[TrialMetrics] = []
    for t in range(trials):
        trial_seed = seed + t
        if algorithm == "kmeans":
            model = PrivateKMeans(
                n_clusters=n_clusters,
                epsilon=epsilon,
                L=L,
                random_state=trial_seed,
            )
        elif algorithm == "gmm":
            model = PrivateGMM(
                n_components=n_clusters,
                epsilon=epsilon,
                L=L,
                random_state=trial_seed,
            )
        elif algorithm == "tmm":
            model = PrivateTMM(
                n_components=n_clusters,
                epsilon=epsilon,
                L=L,
                nu=nu,
                alpha=alpha,
                max_iter=max_iter,
                random_state=trial_seed,
                fixed_nu=True,
            )
        elif algorithm == "bpgt":
            cfg = bpgt_params or {}
            model = BPGT(
                n_clusters=n_clusters,
                epsilon=epsilon,
                L=L,
                random_state=trial_seed,
                gd_lr=cfg.get("gd_lr", 0.05),
                gd_tol=cfg.get("gd_tol", 1e-3),
                gd_max_iter=cfg.get("gd_max_iter", 200),
                tmm_max_iter=cfg.get("tmm_max_iter", 100),
                tmm_tol=cfg.get("tmm_tol", 1e-3),
            )
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}'.")

        model.fit(X)
        if hasattr(model, "predict"):
            labels = model.predict(X)
        else:
            labels = getattr(model, "labels_", None)
        centers = getattr(model, "cluster_centers_", None)
        if centers is None:
            centers = getattr(model, "means_", None)
        trial_results.append(TrialMetrics(labels=labels, centers=centers))

    return summarize_metrics(X, y_true, trial_results)


def format_float(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:0.4f}"


def print_table(rows: List[Dict[str, object]]) -> None:
    """Pretty-print experiment results."""
    header = f"{'Mode':<10} {'Algorithm':<10} {'ε':<6} {'L':<6} {'SSE':<12} {'Silhouette':<12} {'ARI':<12} {'NMI':<12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['mode']:<10} {row['algorithm']:<10} "
            f"{(row['epsilon'] if row['epsilon'] is not None else '-'):>6} "
            f"{(row['L'] if row['L'] is not None else '-'):>6} "
            f"{format_float(row['sse']):<12} "
            f"{format_float(row['silhouette']):<12} "
            f"{format_float(row['ari']):<12} "
            f"{format_float(row['nmi']):<12}"
        )


def maybe_write_csv(path: Optional[Path], rows: List[Dict[str, object]]) -> None:
    """Persist results to CSV for reproducibility."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mode", "algorithm", "epsilon", "L", "sse", "silhouette", "ari", "nmi"]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_curves(rows: List[Dict[str, object]], output: Optional[Path]) -> None:
    """Plot metric curves vs epsilon for each (algorithm, L) pair."""
    if output is None:
        return

    series_keys = sorted(
        {(row["algorithm"], row["L"]) for row in rows if row["mode"] == "private"}
    )
    epsilons = sorted({row["epsilon"] for row in rows if row["epsilon"] is not None})
    metrics = ["sse", "silhouette", "ari", "nmi"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for metric, ax in zip(metrics, axes.flatten()):
        for algorithm, L in series_keys:
            ys = []
            for epsilon in epsilons:
                for row in rows:
                    if (
                        row["mode"] == "private"
                        and row["algorithm"] == algorithm
                        and row["L"] == L
                        and row["epsilon"] == epsilon
                    ):
                        ys.append(row[metric])
                        break
                else:
                    ys.append(np.nan)
            ax.plot(
                epsilons,
                ys,
                marker="o",
                label=f"{algorithm} (L={L})",
            )
        ax.set_xlabel("ε")
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BPM privacy-preserving clustering experiments.")
    parser.add_argument("--dataset", choices=["iris", "blobs"], default="iris")
    parser.add_argument("--samples", type=int, default=300, help="Number of samples for synthetic datasets.")
    parser.add_argument("--features", type=int, default=2, help="Number of features for synthetic datasets.")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters/components.")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["kmeans"],
        choices=["kmeans", "gmm", "tmm", "bpgt"],
    )
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--Ls", nargs="+", type=float, default=[0.3])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, help="Optional path to save results as CSV.")
    parser.add_argument("--plot", type=Path, help="Optional path to save metric curves figure (line plots).")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip non-private baselines.")
    parser.add_argument("--tmm-nu", type=float, default=15.0)
    parser.add_argument("--tmm-alpha", type=float, default=0.01)
    parser.add_argument("--tmm-max-iter", type=int, default=200)
    parser.add_argument("--bpgt-gd-lr", type=float, default=0.05)
    parser.add_argument("--bpgt-gd-tol", type=float, default=1e-3)
    parser.add_argument("--bpgt-gd-max-iter", type=int, default=200)
    parser.add_argument("--bpgt-tmm-max-iter", type=int, default=100)
    parser.add_argument("--bpgt-tmm-tol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_X, y_true, inferred_clusters = dataset_loader(
        args.dataset, args.samples, args.features, args.clusters, args.seed
    )
    n_clusters = args.clusters or inferred_clusters
    X = normalize_data(raw_X)

    results: List[Dict[str, object]] = []

    if not args.skip_baseline:
        for algorithm in args.algorithms:
            baseline_metrics = run_baseline(
                algorithm,
                X,
                y_true,
                n_clusters,
                args.seed,
                args.tmm_nu,
                args.tmm_alpha,
                args.tmm_max_iter,
            )
            results.append(
                {
                    "mode": "baseline",
                    "algorithm": algorithm,
                    "epsilon": None,
                    "L": None,
                    **baseline_metrics,
                }
            )

    for algorithm in args.algorithms:
        for epsilon in args.epsilons:
            for L in args.Ls:
                metrics = run_private_trials(
                    algorithm,
                    X,
                    y_true,
                    n_clusters,
                    epsilon,
                    L,
                    args.trials,
                    args.seed,
                    args.tmm_nu,
                    args.tmm_alpha,
                    args.tmm_max_iter,
                    bpgt_params=dict(
                        gd_lr=args.bpgt_gd_lr,
                        gd_tol=args.bpgt_gd_tol,
                        gd_max_iter=args.bpgt_gd_max_iter,
                        tmm_max_iter=args.bpgt_tmm_max_iter,
                        tmm_tol=args.bpgt_tmm_tol,
                    ),
                )
                results.append(
                    {
                        "mode": "private",
                        "algorithm": algorithm,
                        "epsilon": epsilon,
                        "L": L,
                        **metrics,
                    }
                )

    print_table(results)
    maybe_write_csv(args.csv, results)
    plot_curves(results, args.plot)


if __name__ == "__main__":
    main()
