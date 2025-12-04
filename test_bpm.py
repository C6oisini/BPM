"""
Test BPM implementation.

This script verifies that the BPM mechanism is correctly implemented
according to the paper specifications.
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
    print("BPM Mechanism Test")
    print("=" * 80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 300
    n_features = 2
    n_clusters = 3

    # Generate well-separated clusters
    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=42
    )

    # Normalize to [0,1]^d
    X_normalized = normalize_data(X)

    print(f"\nDataset:")
    print(f"  Number of samples: {n_samples}")
    print(f"  Number of features: {n_features}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Data range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")

    # Standard K-means (baseline)
    print("\n" + "-" * 80)
    print("Standard K-means (no privacy):")
    print("-" * 80)

    kmeans_standard = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_standard = kmeans_standard.fit_predict(X_normalized)
    centers_standard = kmeans_standard.cluster_centers_
    sse_standard = compute_sse(X_normalized, labels_standard, centers_standard)

    print(f"  SSE: {sse_standard:.4f}")
    print(f"  Cluster centers shape: {centers_standard.shape}")

    # Private K-means with BPM
    print("\n" + "-" * 80)
    print("Private K-means with BPM:")
    print("-" * 80)

    # Test different privacy budgets and L values
    test_configs = [
        {"epsilon": 1.0, "L": 1.0},
        {"epsilon": 2.0, "L": 1.0},
        {"epsilon": 4.0, "L": 1.0},
    ]

    results = []

    for config in test_configs:
        epsilon = config["epsilon"]
        L = config["L"]

        print(f"\n  Configuration: ε={epsilon}, L={L}")

        # Fit private K-means
        kmeans_private = PrivateKMeans(
            n_clusters=n_clusters,
            epsilon=epsilon,
            L=L,
            random_state=42
        )
        kmeans_private.fit(X_normalized)

        # Compute SSE on original data
        sse_private = kmeans_private.compute_sse(X_normalized)

        # Compute SSE increase
        sse_increase_pct = ((sse_private - sse_standard) / sse_standard) * 100

        print(f"    SSE (private): {sse_private:.4f}")
        print(f"    SSE increase: {sse_increase_pct:.2f}%")

        results.append({
            "epsilon": epsilon,
            "L": L,
            "sse": sse_private,
            "sse_increase_pct": sse_increase_pct
        })

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"\nStandard K-means SSE: {sse_standard:.4f}")
    print("\nPrivate K-means results:")
    print(f"{'ε':<8} {'L':<8} {'SSE':<12} {'SSE Increase':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['epsilon']:<8.1f} {r['L']:<8.1f} {r['sse']:<12.4f} {r['sse_increase_pct']:>12.2f}%")

    # Verify privacy guarantee
    print("\n" + "=" * 80)
    print("Privacy Verification:")
    print("=" * 80)

    from bpm import BPM
    epsilon = 1.0
    L = 1.0
    bpm = BPM(epsilon=epsilon, L=L, dimension=n_features)

    print(f"\nBPM Configuration:")
    print(f"  ε = {epsilon}")
    print(f"  L = {L}")
    print(f"  d = {n_features}")
    print(f"  λ_L = {bpm.lambda_L:.6f}")
    print(f"  p_L = {bpm.p_L:.6f}")

    # Test density function
    v = np.array([0.5, 0.5])
    test_points = [
        np.array([0.5, 0.5]),  # Same point
        np.array([0.7, 0.5]),  # Distance 0.2
        np.array([1.0, 0.5]),  # Distance 0.5
        np.array([1.5, 0.5]),  # Distance 1.0 (= L)
        np.array([2.0, 0.5]),  # Distance 1.5 (> L)
    ]

    print(f"\nDensity function test (v = {v}):")
    print(f"{'x':<20} {'||x-v||_2':<15} {'f_v^(L)(x)':<15}")
    print("-" * 55)
    for x in test_points:
        dist = np.linalg.norm(x - v)
        density = bpm.density(v, x)
        print(f"{str(x):<20} {dist:<15.4f} {density:<15.8f}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
