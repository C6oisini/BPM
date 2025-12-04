#!/usr/bin/env python3
"""
Rebuilds the legacy figure suite for BPM-based clustering experiments.

Generates the figures that previously lived in standalone scripts such as:
  - bpm_sampling_distribution.png
  - iris_evaluation.png
  - iris_comprehensive.png
  - silhouette_evaluation.png
  - nonlinear_analysis.png
  - kmeans_vs_gmm.png
  - all_methods_comparison.png
  - comprehensive_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from bpm_privacy import BPM, PrivateGMM, PrivateKMeans, PrivateTMM, TMM, bpm_sampling


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def normalize_data(X: np.ndarray) -> np.ndarray:
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    denom = np.clip(X_max - X_min, 1e-12, None)
    return (X - X_min) / denom


def compute_sse(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    total = 0.0
    for idx, center in enumerate(centers):
        pts = X[labels == idx]
        if pts.size:
            total += np.sum(np.linalg.norm(pts - center, axis=1) ** 2)
    return float(total)


def safe_metrics(X: np.ndarray, y_true: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if labels is None or len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "ari": np.nan, "nmi": np.nan}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
    }


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Figure builders
# --------------------------------------------------------------------------- #


def figure_bpm_sampling_distribution(path: Path) -> None:
    bpm = BPM(epsilon=2.0, L=0.3, dimension=2)
    anchor = np.array([0.5, 0.5])
    samples = np.array(
        [
            bpm_sampling(anchor, bpm.k, bpm.L, bpm.p_L, bpm.lambda_2r)
            for _ in range(1500)
        ]
    )
    distances = np.linalg.norm(samples - anchor, axis=1)
    inside = np.mean(distances <= bpm.L)
    ensure_dir(path)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(samples[:, 0], samples[:, 1], s=12, alpha=0.35, label="Samples")
    ax.scatter(anchor[0], anchor[1], c="crimson", marker="*", s=120, label="True point")
    circle = plt.Circle(anchor, bpm.L, fill=False, color="steelblue", linestyle="--", label=f"r = {bpm.L}")
    ax.add_patch(circle)
    ax.set_xlim(-bpm.L - 0.1, 1 + bpm.L + 0.1)
    ax.set_ylim(-bpm.L - 0.1, 1 + bpm.L + 0.1)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("BPM Sampling Distribution (ε=2.0, L=0.3)")
    ax.legend()
    ax.grid(alpha=0.3)
    text = f"λ_L={bpm.lambda_L:.4f}\np_L={bpm.p_L:.3f}\nInside rate={inside:.3f}"
    ax.text(1 + bpm.L - 0.2, -bpm.L + 0.1, text, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[done] {path}")


def figure_iris_evaluation(path: Path) -> None:
    iris = load_iris()
    X = normalize_data(iris.data)
    configs = [(0.5, 0.3), (2.0, 0.3), (8.0, 0.5)]
    ensure_dir(path)
    fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5))
    baseline = KMeans(n_clusters=3, random_state=42, n_init=10)
    baseline_labels = baseline.fit_predict(X)
    baseline_metrics = safe_metrics(X, iris.target, baseline_labels)

    for ax, (epsilon, L) in zip(axes, configs):
        model = PrivateKMeans(n_clusters=3, epsilon=epsilon, L=L, random_state=42)
        model.fit(X)
        labels = model.predict(X)
        metrics = safe_metrics(X, iris.target, labels)
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=40, alpha=0.8)
        centers = model.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200,
                   edgecolors="black", linewidths=1.5, label="Centroids")
        ax.set_title(f"ε={epsilon}, L={L}\nSSE={model.compute_sse(X):.2f}, Sil={metrics['silhouette']:.2f}")
        ax.set_xlabel("Feature 1 (normalized)")
        ax.set_ylabel("Feature 2 (normalized)")
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"Iris Private K-means (Baseline Silhouette={baseline_metrics['silhouette']:.2f})",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[done] {path}")


def figure_iris_comprehensive(path: Path) -> None:
    iris = load_iris()
    X = normalize_data(iris.data)
    y = iris.target
    epsilons = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
    L_values = [0.3, 0.5]
    baseline = KMeans(n_clusters=3, random_state=42, n_init=10)
    baseline_labels = baseline.fit_predict(X)
    baseline_sse = compute_sse(X, baseline_labels, baseline.cluster_centers_)

    ensure_dir(path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metric_axes = {
        "SSE": axes[0, 0],
        "Silhouette": axes[0, 1],
        "ARI": axes[1, 0],
        "NMI": axes[1, 1],
    }

    for L in L_values:
        sse_vals, sil_vals, ari_vals, nmi_vals = [], [], [], []
        for epsilon in epsilons:
            model = PrivateKMeans(n_clusters=3, epsilon=epsilon, L=L, random_state=42)
            model.fit(X)
            labels = model.predict(X)
            metrics = safe_metrics(X, y, labels)
            sse_vals.append(model.compute_sse(X))
            sil_vals.append(metrics["silhouette"])
            ari_vals.append(metrics["ari"])
            nmi_vals.append(metrics["nmi"])

        metric_axes["SSE"].plot(epsilons, sse_vals, marker="o", label=f"L={L}")
        metric_axes["Silhouette"].plot(epsilons, sil_vals, marker="o", label=f"L={L}")
        metric_axes["ARI"].plot(epsilons, ari_vals, marker="o", label=f"L={L}")
        metric_axes["NMI"].plot(epsilons, nmi_vals, marker="o", label=f"L={L}")

    for name, ax in metric_axes.items():
        ax.set_xlabel("ε")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(f"{name} vs ε")

    axes[0, 0].axhline(baseline_sse, color="gray", linestyle="--", label="Baseline SSE")
    axes[0, 0].legend()
    fig.suptitle("Iris Comprehensive Sweep", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[done] {path}")


def figure_silhouette_evaluation(path: Path) -> None:
    np.random.seed(42)
    X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5, random_state=42)
    X = normalize_data(X)
    configs = [(1.0, 0.3), (2.0, 0.3), (4.0, 0.3), (4.0, 0.5), (8.0, 0.3)]
    baseline = KMeans(n_clusters=3, random_state=42, n_init=10)
    baseline_labels = baseline.fit_predict(X)
    baseline_sse = compute_sse(X, baseline_labels, baseline.cluster_centers_)
    baseline_sil = silhouette_score(X, baseline_labels)

    sse_vals, sil_vals = [], []
    for epsilon, L in configs:
        model = PrivateKMeans(n_clusters=3, epsilon=epsilon, L=L, random_state=42)
        model.fit(X)
        labels = model.predict(X)
        sse_vals.append(model.compute_sse(X))
        sil_vals.append(silhouette_score(X, labels))

    ensure_dir(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(range(len(configs)), sse_vals, color="#4c72b0")
    axes[0].axhline(baseline_sse, color="gray", linestyle="--", label="Baseline")
    axes[0].set_xticks(range(len(configs)))
    axes[0].set_xticklabels([f"ε={e}\nL={L}" for e, L in configs])
    axes[0].set_ylabel("SSE")
    axes[0].set_title("SSE across configs")
    axes[0].legend()

    axes[1].bar(range(len(configs)), sil_vals, color="#55a868")
    axes[1].axhline(baseline_sil, color="gray", linestyle="--", label="Baseline")
    axes[1].set_xticks(range(len(configs)))
    axes[1].set_xticklabels([f"ε={e}\nL={L}" for e, L in configs])
    axes[1].set_ylabel("Silhouette")
    axes[1].set_title("Silhouette across configs")
    axes[1].legend()

    fig.suptitle("Silhouette Evaluation (Synthetic Blobs)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[done] {path}")


def figure_nonlinear_analysis(path: Path) -> None:
    iris = load_iris()
    X = normalize_data(iris.data)
    y = iris.target
    epsilons = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    baseline = KMeans(n_clusters=3, random_state=42, n_init=10)
    baseline_labels = baseline.fit_predict(X)
    baseline_centers = baseline.cluster_centers_

    results = {"epsilon": [], "sse": [], "silhouette": [], "ari": [], "nmi": [],
               "centroid_distance": [], "reassignment_rate": []}

    for epsilon in epsilons:
        model = PrivateKMeans(n_clusters=3, epsilon=epsilon, L=0.5, random_state=42)
        model.fit(X)
        labels = model.predict(X)
        centers = model.cluster_centers_
        metrics = safe_metrics(X, y, labels)
        results["epsilon"].append(epsilon)
        results["sse"].append(model.compute_sse(X))
        results["silhouette"].append(metrics["silhouette"])
        results["ari"].append(metrics["ari"])
        results["nmi"].append(metrics["nmi"])
        from scipy.spatial.distance import cdist
        distances = cdist(centers, baseline_centers).min(axis=1)
        results["centroid_distance"].append(float(np.mean(distances)))
        reassignment = np.mean(labels != baseline_labels)
        results["reassignment_rate"].append(float(reassignment))

    ensure_dir(path)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metric_names = ["sse", "silhouette", "ari", "nmi", "centroid_distance", "reassignment_rate"]
    titles = [
        "SSE vs ε", "Silhouette vs ε",
        "ARI vs ε", "NMI vs ε",
        "Avg. centroid drift vs ε", "Reassignment rate vs ε",
    ]
    for ax, metric, title in zip(axes.flatten(), metric_names, titles):
        ax.plot(results["epsilon"], results[metric], marker="o")
        ax.set_xlabel("ε")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("Why the metrics change non-linearly", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[done] {path}")


def _collect_algorithm_metrics(X: np.ndarray, y: np.ndarray, epsilon: float, L: float
                               ) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    # Standard
    kmeans_std = KMeans(n_clusters=3, random_state=42, n_init=10)
    gmm_std = GaussianMixture(n_components=3, random_state=42, n_init=10)
    metrics["K-means"] = {"mode": "standard"}
    metrics["GMM"] = {"mode": "standard"}
    metrics["TMM"] = {"mode": "standard"}
    kmeans_std_labels = kmeans_std.fit_predict(X)
    gmm_std.fit(X)
    gmm_std_labels = gmm_std.predict(X)
    metrics["K-means"].update(
        sse=compute_sse(X, kmeans_std_labels, kmeans_std.cluster_centers_),
        **safe_metrics(X, y, kmeans_std_labels),
    )
    metrics["GMM"].update(
        sse=compute_sse(X, gmm_std_labels, gmm_std.means_),
        **safe_metrics(X, y, gmm_std_labels),
    )
    tmm_std = TMM(
        n_components=3,
        nu=15.0,
        alpha=0.01,
        max_iter=80,
        tol=0.1,
        random_state=42,
        fixed_nu=True,
        verbose=False,
    )
    tmm_std.fit(X)
    tmm_std_labels = tmm_std.labels_
    metrics["TMM"].update(
        sse=compute_sse(X, tmm_std_labels, tmm_std.means_),
        **safe_metrics(X, y, tmm_std_labels),
    )

    # Private
    kmeans_priv = PrivateKMeans(n_clusters=3, epsilon=epsilon, L=L, random_state=42)
    kmeans_priv.fit(X)
    labels_km_priv = kmeans_priv.predict(X)
    metrics["Private K-means"] = {
        "mode": "private",
        "sse": kmeans_priv.compute_sse(X),
        **safe_metrics(X, y, labels_km_priv),
    }
    gmm_priv = PrivateGMM(n_components=3, epsilon=epsilon, L=L, random_state=42)
    gmm_priv.fit(X)
    labels_gmm_priv = gmm_priv.predict(X)
    metrics["Private GMM"] = {
        "mode": "private",
        "sse": compute_sse(X, labels_gmm_priv, gmm_priv.means_),
        **safe_metrics(X, y, labels_gmm_priv),
    }
    tmm_priv = PrivateTMM(n_components=3, epsilon=epsilon, L=L, nu=15.0,
                          alpha=0.01, max_iter=80, tol=0.1, random_state=42,
                          fixed_nu=True, verbose=False)
    tmm_priv.fit(X)
    labels_tmm_priv = tmm_priv.labels_
    metrics["Private TMM"] = {
        "mode": "private",
        "sse": compute_sse(X, labels_tmm_priv, tmm_priv.means_),
        **safe_metrics(X, y, labels_tmm_priv),
    }
    return metrics


def figure_kmeans_vs_gmm(path: Path) -> None:
    iris = load_iris()
    X = normalize_data(iris.data)
    y = iris.target
    metrics = _collect_algorithm_metrics(X, y, epsilon=8.0, L=2.0)
    ensure_dir(path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    categories = ["K-means", "Private K-means", "GMM", "Private GMM"]
    values = {metric: [metrics[name][metric] for name in categories] for metric in ["sse", "silhouette", "ari", "nmi"]}
    titles = ["SSE", "Silhouette", "ARI", "NMI"]
    for ax, title, metric_name in zip(axes.flatten(), titles, ["sse", "silhouette", "ari", "nmi"]):
        ax.bar(range(len(categories)), values[metric_name], color="#4c72b0")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=20)
        ax.set_ylabel(title)
        ax.set_title(f"{title} comparison")
    fig.suptitle("K-means vs GMM (Standard vs Private)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[done] {path}")


def figure_all_methods(path: Path) -> None:
    iris = load_iris()
    X = normalize_data(iris.data)
    y = iris.target
    metrics = _collect_algorithm_metrics(X, y, epsilon=8.0, L=2.0)
    ensure_dir(path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    labels = list(metrics.keys())
    x = np.arange(len(labels))
    width = 0.2

    for ax, metric_name in zip(axes.flatten(), ["sse", "silhouette", "ari", "nmi"]):
        values = [metrics[label][metric_name] for label in labels]
        colors = ["#4c72b0" if metrics[label]["mode"] == "standard" else "#dd8452" for label in labels]
        ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25)
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{metric_name.upper()} comparison")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("K-means vs GMM vs TMM (Standard vs Private)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[done] {path}")


def figure_comprehensive_comparison(path: Path) -> None:
    iris = load_iris()
    blobs_X, blobs_y = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=0)
    datasets = [
        ("Iris", normalize_data(iris.data), iris.target),
        ("Blobs", normalize_data(blobs_X), blobs_y),
    ]
    algorithms = ["K-means", "Private K-means", "GMM", "Private GMM", "Private TMM"]
    epsilon, L = 6.0, 1.5
    data = []

    for dataset_name, X, y in datasets:
        kg_metrics = _collect_algorithm_metrics(X, y, epsilon=epsilon, L=L)
        for algo in algorithms:
            if algo not in kg_metrics:
                continue
            data.append((dataset_name, algo, kg_metrics[algo]["sse"], kg_metrics[algo]["silhouette"]))

    ensure_dir(path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric_idx, title in zip(axes, [2, 3], ["SSE", "Silhouette"]):
        for dataset in set(d for d, _, _, _ in data):
            xs, vals = [], []
            for algo in algorithms:
                for d, a, sse, sil in data:
                    if d == dataset and a == algo:
                        xs.append(algo)
                        vals.append(sse if metric_idx == 2 else sil)
                        break
            ax.plot(range(len(xs)), vals, marker="o", label=dataset)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs, rotation=25)
        ax.set_title(f"{title} across datasets")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Comprehensive Comparison Across Datasets", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[done] {path}")


# Mapping of figure names to builders and default output paths
FIGURES = {
    "bpm_sampling_distribution": (figure_bpm_sampling_distribution, Path("figures/bpm_sampling_distribution.png")),
    "iris_evaluation": (figure_iris_evaluation, Path("figures/iris_evaluation.png")),
    "iris_comprehensive": (figure_iris_comprehensive, Path("figures/iris_comprehensive.png")),
    "silhouette_evaluation": (figure_silhouette_evaluation, Path("figures/silhouette_evaluation.png")),
    "nonlinear_analysis": (figure_nonlinear_analysis, Path("figures/nonlinear_analysis.png")),
    "kmeans_vs_gmm": (figure_kmeans_vs_gmm, Path("figures/kmeans_vs_gmm.png")),
    "all_methods_comparison": (figure_all_methods, Path("figures/all_methods_comparison.png")),
    "comprehensive_comparison": (figure_comprehensive_comparison, Path("figures/comprehensive_comparison.png")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate legacy BPM figures.")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        choices=["all"] + list(FIGURES.keys()),
        help="Which figures to regenerate. Defaults to all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = FIGURES.keys() if "all" in args.figures else args.figures
    for name in targets:
        builder, output = FIGURES[name]
        builder(output)


if __name__ == "__main__":
    main()
