#!/usr/bin/env python3
"""
Recreates the legacy visualization flow for BPM-based private K-means.

This script mirrors the plotting logic that previously lived in evaluate_iris.py
so existing documentation and figures can still be reproduced.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from bpm_privacy import PrivateKMeans


def normalize_data(X: np.ndarray) -> np.ndarray:
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    denom = np.clip(X_max - X_min, 1e-12, None)
    return (X - X_min) / denom


def compute_sse(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    total = 0.0
    for k in range(len(centers)):
        cluster_points = X[labels == k]
        if cluster_points.size:
            total += np.sum(np.linalg.norm(cluster_points - centers[k], axis=1) ** 2)
    return float(total)


def run_private_kmeans(
    X: np.ndarray,
    y_true: np.ndarray,
    n_clusters: int,
    configs: Sequence[Tuple[float, float]],
    seeds: Sequence[int],
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    baseline_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_std = baseline_kmeans.fit_predict(X)
    centers_std = baseline_kmeans.cluster_centers_
    sse_std = compute_sse(X, labels_std, centers_std)
    silhouette_std = silhouette_score(X, labels_std)
    ari_std = adjusted_rand_score(y_true, labels_std)
    nmi_std = normalized_mutual_info_score(y_true, labels_std)

    print("\n=== Baseline (non-private K-means) ===")
    print(f"SSE: {sse_std:.4f}")
    print(f"Silhouette: {silhouette_std:.4f}")
    print(f"ARI: {ari_std:.4f}")
    print(f"NMI: {nmi_std:.4f}")

    print("\n=== Private configurations ===")
    header = f"{'ε':<8}{'L':<8}{'SSE':<12}{'Silhouette':<12}{'ARI':<12}{'NMI':<12}{'ΔSSE%':<10}"
    print(header)
    print("-" * len(header))

    for epsilon, L in configs:
        sses: List[float] = []
        silhouettes: List[float] = []
        aris: List[float] = []
        nmis: List[float] = []

        for seed in seeds:
            model = PrivateKMeans(
                n_clusters=n_clusters,
                epsilon=epsilon,
                L=L,
                random_state=seed,
            )
            model.fit(X)
            labels_priv = model.predict(X)

            sses.append(model.compute_sse(X))
            silhouettes.append(silhouette_score(X, labels_priv))
            aris.append(adjusted_rand_score(y_true, labels_priv))
            nmis.append(normalized_mutual_info_score(y_true, labels_priv))

        sse_mean = float(np.mean(sses))
        sil_mean = float(np.mean(silhouettes))
        ari_mean = float(np.mean(aris))
        nmi_mean = float(np.mean(nmis))
        sse_change = ((sse_mean - sse_std) / sse_std) * 100

        print(
            f"{epsilon:<8.1f}{L:<8.2f}"
            f"{sse_mean:<12.4f}{sil_mean:<12.4f}"
            f"{ari_mean:<12.4f}{nmi_mean:<12.4f}"
            f"{sse_change:>+8.2f}%"
        )

        results.append(
            {
                "epsilon": epsilon,
                "L": L,
                "sse": sse_mean,
                "silhouette": sil_mean,
                "ari": ari_mean,
                "nmi": nmi_mean,
                "sse_delta_pct": sse_change,
                "labels": labels_priv,
                "model": model,
            }
        )

    return results


def plot_metric_curves(
    configs: Sequence[Tuple[float, float]],
    results: List[Dict[str, float]],
    output: Path,
) -> None:
    epsilons = [cfg[0] for cfg in configs]

    metrics = {
        "SSE": [r["sse"] for r in results],
        "Silhouette": [r["silhouette"] for r in results],
        "ARI": [r["ari"] for r in results],
        "NMI": [r["nmi"] for r in results],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_list = axes.flatten()

    for ax, (name, values) in zip(axes_list, metrics.items()):
        ax.plot(epsilons, values, marker="o", linewidth=2)
        ax.set_xlabel("ε (privacy budget)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs ε")
        ax.grid(alpha=0.3)

    fig.suptitle("Private K-means Metrics (Iris dataset)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved metric curves to {output}")


def plot_cluster_assignments(
    X: np.ndarray,
    results: List[Dict[str, float]],
    configs: Iterable[Tuple[float, float]],
    output: Path,
) -> None:
    selected: List[Dict[str, float]] = []
    for epsilon, L in configs:
        for result in results:
            if result["epsilon"] == epsilon and result["L"] == L:
                selected.append(result)
                break

    if not selected:
        print("No matching configurations found for scatter plot; skipping.")
        return

    fig, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 5))
    if len(selected) == 1:
        axes = [axes]

    for ax, result in zip(axes, selected):
        labels = result["labels"]
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=40, alpha=0.8)
        centers = result["model"].cluster_centers_
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            c="red",
            marker="X",
            s=200,
            edgecolors="black",
            linewidths=1.5,
        )
        ax.set_title(f"ε={result['epsilon']}, L={result['L']}")
        ax.set_xlabel("Feature 1 (normalized)")
        ax.set_ylabel("Feature 2 (normalized)")
        ax.grid(alpha=0.2)

    fig.suptitle("Private K-means Cluster Assignments", fontsize=16, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved cluster assignment grids to {output}")


def parse_configs(config_str: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for token in config_str.split(","):
        if not token:
            continue
        epsilon_str, L_str = token.split(":")
        pairs.append((float(epsilon_str), float(L_str)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy plotting workflow for BPM private K-means.")
    parser.add_argument(
        "--configs",
        type=str,
        default="0.5:0.3,1.0:0.3,2.0:0.3,4.0:0.3,8.0:0.5",
        help="Comma-separated ε:L pairs used for metrics.",
    )
    parser.add_argument(
        "--scatter-configs",
        type=str,
        default="0.5:0.3,2.0:0.3,8.0:0.5",
        help="Comma-separated ε:L pairs to visualize as cluster scatter plots.",
    )
    parser.add_argument(
        "--metric-figure",
        type=Path,
        default=Path("figures/legacy_metrics.png"),
        help="Where to save the metric curve figure.",
    )
    parser.add_argument(
        "--scatter-figure",
        type=Path,
        default=Path("figures/legacy_clusters.png"),
        help="Where to save the cluster assignment grids.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of random seeds to average for each configuration.",
    )
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    scatter_configs = parse_configs(args.scatter_configs)
    seeds = [42 + i for i in range(args.trials)]

    iris = load_iris()
    X = normalize_data(iris.data)
    y = iris.target
    results = run_private_kmeans(X, y, n_clusters=3, configs=configs, seeds=seeds)
    plot_metric_curves(configs, results, args.metric_figure)
    plot_cluster_assignments(X, results, scatter_configs, args.scatter_figure)


if __name__ == "__main__":
    main()
