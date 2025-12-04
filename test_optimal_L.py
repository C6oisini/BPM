"""
Test to find optimal L value for BPM.

According to Remark 2 in the paper, L should be chosen to minimize
the expected squared distance E[||X - v||_2^2].
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from clustering import PrivateKMeans


def normalize_data(X):
    """Normalize data to [0,1]^d."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-10)


def compute_sse(X, labels, centers):
    """Compute Sum of Squared Errors."""
    sse = 0.0
    for k in range(len(centers)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            sse += np.sum(np.linalg.norm(cluster_points - centers[k], axis=1) ** 2)
    return sse


def main():
    print("=" * 80)
    print("Finding Optimal L for BPM")
    print("=" * 80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 300
    n_features = 2
    n_clusters = 3

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=42
    )
    X_normalized = normalize_data(X)

    print(f"\nDataset: {n_samples} samples, {n_features} features, {n_clusters} clusters")

    # Standard K-means
    kmeans_standard = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_standard = kmeans_standard.fit_predict(X_normalized)
    sse_standard = compute_sse(X_normalized, labels_standard, kmeans_standard.cluster_centers_)

    print(f"Standard K-means SSE: {sse_standard:.4f}")

    # Test different L values
    epsilon_values = [1.0, 2.0, 4.0]
    L_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    print("\n" + "=" * 80)
    print("Testing different (ε, L) configurations:")
    print("=" * 80)

    best_config = None
    best_sse = float('inf')

    for epsilon in epsilon_values:
        print(f"\nε = {epsilon}:")
        print(f"{'L':<8} {'SSE':<12} {'SSE Increase %':<15} {'Status':<10}")
        print("-" * 50)

        for L in L_values:
            try:
                kmeans_private = PrivateKMeans(
                    n_clusters=n_clusters,
                    epsilon=epsilon,
                    L=L,
                    random_state=42
                )
                kmeans_private.fit(X_normalized)
                sse_private = kmeans_private.compute_sse(X_normalized)
                sse_increase_pct = ((sse_private - sse_standard) / sse_standard) * 100

                status = "✓"
                if sse_private < best_sse:
                    best_sse = sse_private
                    best_config = (epsilon, L)
                    status = "✓ BEST"

                print(f"{L:<8.2f} {sse_private:<12.4f} {sse_increase_pct:>12.2f}% {status:<10}")

            except Exception as e:
                print(f"{L:<8.2f} {'ERROR':<12} {str(e)[:30]:<15}")

    print("\n" + "=" * 80)
    print("Best Configuration:")
    print("=" * 80)
    if best_config:
        print(f"ε = {best_config[0]}, L = {best_config[1]}")
        print(f"SSE = {best_sse:.4f}")
        print(f"SSE increase = {((best_sse - sse_standard) / sse_standard) * 100:.2f}%")


if __name__ == "__main__":
    main()
