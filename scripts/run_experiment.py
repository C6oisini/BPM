#!/usr/bin/env python3
"""
Consolidated experiment runner for BPM-based clustering.

Usage examples:
    python scripts/run_experiment.py --dataset iris --mechanisms bpm bpgm --servers kmeans gmm --epsilons 1 2 4
    python scripts/run_experiment.py --dataset blobs --clusters 4 --mechanisms bpgm --servers tmm \
        --epsilons 0.5 1 --Ls 0.2 0.4 --trials 5 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from d_privacy import (
    ClientMechanism,
    BPMMechanism,
    BPGMMechanism,
    BLMMechanism,
    CIMMechanism,
    GMMServer,
    ServerAlgorithm,
    KMeansServer,
    PrivacyClusteringPipeline,
    TMM,
    TMMServer,
)


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


def summarize_metrics(
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    trial_results: Iterable[TrialMetrics],
    true_centers: Optional[np.ndarray],
) -> Dict[str, float]:
    """Aggregate metrics across multiple trials."""
    sses, silhouettes, aris, nmis, res = [], [], [], [], []
    for trial in trial_results:
        sses.append(compute_sse(X, trial.labels, trial.centers))
        silhouettes.append(safe_silhouette(X, trial.labels))
        aris.append(safe_ari(y_true, trial.labels))
        nmis.append(safe_nmi(y_true, trial.labels))
        res.append(compute_re(trial.centers, true_centers))

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
        "re": nanmean(res),
    }


def compute_true_centers(
    X: np.ndarray, y_true: Optional[np.ndarray], n_clusters: int
) -> Optional[np.ndarray]:
    if y_true is None:
        return None
    unique_labels = np.unique(y_true)
    if len(unique_labels) != n_clusters:
        return None
    centers = []
    for label in sorted(unique_labels):
        mask = y_true == label
        if not np.any(mask):
            return None
        centers.append(np.mean(X[mask], axis=0))
    return np.asarray(centers)


def compute_re(
    centers: Optional[np.ndarray], true_centers: Optional[np.ndarray]
) -> float:
    if centers is None or true_centers is None:
        return float("nan")
    if centers.shape[0] != true_centers.shape[0]:
        return float("nan")

    cost_matrix = np.linalg.norm(
        centers[:, np.newaxis, :] - true_centers[np.newaxis, :, :], axis=2
    )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    errors = []
    for pred_idx, true_idx in zip(row_ind, col_ind):
        denom = np.linalg.norm(true_centers[true_idx])
        denom = denom if denom > 1e-12 else 1e-12
        errors.append(
            np.linalg.norm(centers[pred_idx] - true_centers[true_idx]) / denom
        )
    if not errors:
        return float("nan")
    return float(np.mean(errors))


def create_mechanism(
    name: str,
    epsilon: float,
    L: Optional[float],
    seed: int,
    bpgt_args: Dict[str, float],
    distance_args: Dict[str, float],
) -> ClientMechanism:
    if name == "bpm":
        if L is None:
            raise ValueError("BPM requires L; ensure it is provided in --Ls.")
        return BPMMechanism(epsilon=epsilon, L=L, random_state=seed)
    if name == "bpgm":
        if L is None:
            raise ValueError("BPGM requires L; ensure it is provided in --Ls.")
        return BPGMMechanism(
            epsilon=epsilon,
            L=L,
            random_state=seed,
            lr=bpgt_args["gd_lr"],
            tol=bpgt_args["gd_tol"],
            max_iter=bpgt_args["gd_max_iter"],
        )
    if name == "blm":
        return BLMMechanism(
            epsilon=epsilon,
            random_state=seed,
            synth_lr=distance_args["lr"],
            synth_tol=distance_args["tol"],
            synth_max_iter=distance_args["max_iter"],
        )
    if name == "cim":
        return CIMMechanism(
            epsilon=epsilon,
            random_state=seed,
            synth_lr=distance_args["lr"],
            synth_tol=distance_args["tol"],
            synth_max_iter=distance_args["max_iter"],
        )
    raise ValueError(f"Unknown mechanism '{name}'")


def create_server(
    name: str,
    n_clusters: int,
    seed: int,
    tmm_params: Dict[str, float],
) -> ServerAlgorithm:
    if name == "kmeans":
        return KMeansServer(n_clusters=n_clusters, random_state=seed)
    if name == "gmm":
        return GMMServer(n_components=n_clusters, random_state=seed)
    if name == "tmm":
        return TMMServer(
            n_components=n_clusters,
            nu=tmm_params["nu"],
            alpha=tmm_params["alpha"],
            max_iter=tmm_params["max_iter"],
            tol=tmm_params["tol"],
            random_state=seed,
        )
    raise ValueError(f"Unknown server '{name}'")


def run_baseline(
    server: str,
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    n_clusters: int,
    seed: int,
    tmm_params: Dict[str, float],
    true_centers: Optional[np.ndarray],
) -> Dict[str, float]:
    """Non-private baseline using the specified server-side algorithm only."""
    if server == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        labels = model.fit_predict(X)
        centers = model.cluster_centers_
    elif server == "gmm":
        model = GaussianMixture(n_components=n_clusters, n_init=10, random_state=seed)
        model.fit(X)
        labels = model.predict(X)
        centers = model.means_
    elif server == "tmm":
        model = TMM(
            n_components=n_clusters,
            nu=tmm_params["nu"],
            alpha=tmm_params["alpha"],
            max_iter=tmm_params["max_iter"],
            tol=tmm_params["tol"],
            random_state=seed,
            fixed_nu=True,
        )
        model.fit(X)
        labels = model.labels_
        centers = model.means_
    else:
        raise ValueError(f"Unsupported server '{server}'.")

    return summarize_metrics(
        X, y_true, [TrialMetrics(labels=labels, centers=centers)], true_centers
    )


def run_private_trials(
    mechanism_name: str,
    server_name: str,
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    n_clusters: int,
    epsilon: float,
    L: Optional[float],
    trials: int,
    seed: int,
    tmm_params: Dict[str, float],
    bpgt_params: Dict[str, float],
    distance_params: Dict[str, float],
    true_centers: Optional[np.ndarray],
) -> Dict[str, float]:
    """Execute multiple private trials and aggregate metrics."""
    trial_results: List[TrialMetrics] = []
    for t in range(trials):
        trial_seed = seed + t
        mechanism = create_mechanism(
            mechanism_name,
            epsilon,
            L,
            trial_seed,
            bpgt_params,
            distance_params,
        )
        server = create_server(server_name, n_clusters, trial_seed, tmm_params)
        pipeline = PrivacyClusteringPipeline(mechanism, server, n_clusters=n_clusters)
        pipeline.fit(X)
        trial_results.append(
            TrialMetrics(labels=pipeline.predict(X), centers=pipeline.cluster_centers_)
        )

    return summarize_metrics(X, y_true, trial_results, true_centers)


def format_float(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:0.4f}"


def print_table(rows: List[Dict[str, object]]) -> None:
    """Pretty-print experiment results."""
    header = (
        f"{'Mode':<10} {'Mechanism':<12} {'Server':<10} {'ε':<6} {'L':<6} "
        f"{'SSE':<12} {'Silhouette':<12} {'ARI':<12} {'NMI':<12} {'RE':<12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['mode']:<10} {str(row['mechanism']):<12} {row['server']:<10} "
            f"{(row['epsilon'] if row['epsilon'] is not None else '-'):>6} "
            f"{(row['L'] if row['L'] is not None else '-'):>6} "
            f"{format_float(row['sse']):<12} "
            f"{format_float(row['silhouette']):<12} "
            f"{format_float(row['ari']):<12} "
            f"{format_float(row['nmi']):<12} "
            f"{format_float(row.get('re', float('nan'))):<12}"
        )


def maybe_write_csv(path: Optional[Path], rows: List[Dict[str, object]]) -> None:
    """Persist results to CSV for reproducibility."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "mechanism",
        "server",
        "epsilon",
        "L",
        "sse",
        "silhouette",
        "ari",
        "nmi",
        "re",
    ]
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
        {
            (row["mechanism"], row["server"], row["L"])
            for row in rows
            if row["mode"] == "private"
        }
    )
    epsilons = sorted({row["epsilon"] for row in rows if row["epsilon"] is not None})
    metrics = ["sse", "silhouette", "ari", "nmi", "re"]

    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for metric, ax in zip(metrics, axes):
        for mechanism, server, L in series_keys:
            ys = []
            for epsilon in epsilons:
                for row in rows:
                    if (
                        row["mode"] == "private"
                        and row["mechanism"] == mechanism
                        and row["server"] == server
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
                label=f"{mechanism}+{server} (L={L})",
            )
        ax.set_xlabel("ε")
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)
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
        "--mechanisms",
        nargs="+",
        default=["bpm"],
        choices=["bpm", "bpgm", "blm", "cim"],
    )
    parser.add_argument(
        "--servers",
        nargs="+",
        default=["kmeans"],
        choices=["kmeans", "gmm", "tmm"],
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
    parser.add_argument("--tmm-tol", type=float, default=1e-3)
    parser.add_argument("--bpgt-gd-lr", type=float, default=0.05)
    parser.add_argument("--bpgt-gd-tol", type=float, default=1e-3)
    parser.add_argument("--bpgt-gd-max-iter", type=int, default=200)
    parser.add_argument("--bpgt-tmm-max-iter", type=int, default=100)
    parser.add_argument("--bpgt-tmm-tol", type=float, default=1e-3)
    parser.add_argument("--distance-lr", type=float, default=0.1, help="Learning rate for BLM/CIM synthetic reconstruction.")
    parser.add_argument("--distance-tol", type=float, default=1e-4, help="Tolerance for the synthetic distance matching objective.")
    parser.add_argument("--distance-max-iter", type=int, default=200, help="Maximum GD iterations for BLM/CIM synthetic reconstruction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_X, y_true, inferred_clusters = dataset_loader(
        args.dataset, args.samples, args.features, args.clusters, args.seed
    )
    n_clusters = args.clusters or inferred_clusters
    X = normalize_data(raw_X)

    results: List[Dict[str, object]] = []

    tmm_params = {
        "nu": args.tmm_nu,
        "alpha": args.tmm_alpha,
        "max_iter": args.tmm_max_iter,
        "tol": args.tmm_tol,
    }
    bpgt_params = {
        "gd_lr": args.bpgt_gd_lr,
        "gd_tol": args.bpgt_gd_tol,
        "gd_max_iter": args.bpgt_gd_max_iter,
    }
    distance_params = {
        "lr": args.distance_lr,
        "tol": args.distance_tol,
        "max_iter": args.distance_max_iter,
    }
    true_centers = compute_true_centers(X, y_true, n_clusters)

    if not args.skip_baseline:
        for server in args.servers:
            baseline_metrics = run_baseline(
                server,
                X,
                y_true,
                n_clusters,
                args.seed,
                tmm_params,
                true_centers,
            )
            results.append(
                {
                    "mode": "baseline",
                    "mechanism": "none",
                    "server": server,
                    "epsilon": None,
                    "L": None,
                    **baseline_metrics,
                }
            )

    mechanisms = args.mechanisms
    for mechanism in mechanisms:
        L_values = args.Ls if mechanism in {"bpm", "bpgm"} else [None]
        for server in args.servers:
            for epsilon in args.epsilons:
                for L in L_values:
                    metrics = run_private_trials(
                        mechanism,
                        server,
                        X,
                        y_true,
                        n_clusters,
                        epsilon,
                        L,
                        args.trials,
                        args.seed,
                        tmm_params,
                        bpgt_params,
                        distance_params,
                        true_centers,
                    )
                    results.append(
                        {
                            "mode": "private",
                            "mechanism": mechanism,
                            "server": server,
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
