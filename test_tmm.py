"""
Test the TMM (Student's t Mixture Model) implementation on Iris dataset.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from clustering.tmm import TMM, PrivateTMM


def compute_sse(X, labels, centers):
    """Compute sum of squared errors."""
    sse = 0
    for i, center in enumerate(centers):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - center) ** 2)
    return sse


def main():
    print("=" * 80)
    print("Testing TMM (Student's t Mixture Model) on Iris Dataset")
    print("=" * 80)

    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    n_samples, n_features = X.shape

    print(f"\nDataset: Iris")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {len(np.unique(y_true))}")

    # Test 1: Standard TMM (no privacy)
    print("\n" + "-" * 80)
    print("Test 1: Standard TMM (no privacy)")
    print("-" * 80)

    tmm = TMM(n_components=3, nu=15.0, alpha=0.01, max_iter=100,
              random_state=42, fixed_nu=True)
    tmm.fit(X)

    labels_std = tmm.labels_
    sse_std = compute_sse(X, labels_std, tmm.means_)
    sil_std = silhouette_score(X, labels_std)
    ari_std = adjusted_rand_score(y_true, labels_std)
    nmi_std = normalized_mutual_info_score(y_true, labels_std)

    print(f"\n[Standard TMM Results]")
    print(f"  SSE:         {sse_std:.4f}")
    print(f"  Silhouette:  {sil_std:.4f}")
    print(f"  ARI:         {ari_std:.4f}")
    print(f"  NMI:         {nmi_std:.4f}")
    print(f"  Converged:   {tmm.converged_}")
    print(f"  Iterations:  {tmm.n_iter_}")

    # Test 2: Private TMM with BPM
    print("\n" + "-" * 80)
    print("Test 2: Private TMM with BPM (epsilon=8.0, L=2.0)")
    print("-" * 80)

    private_tmm = PrivateTMM(
        n_components=3,
        epsilon=8.0,
        L=2.0,
        nu=15.0,
        alpha=0.01,
        max_iter=100,
        random_state=42,
        fixed_nu=True
    )
    private_tmm.fit(X)

    labels_priv = private_tmm.labels_
    sse_priv = compute_sse(X, labels_priv, private_tmm.means_)
    sil_priv = silhouette_score(X, labels_priv)
    ari_priv = adjusted_rand_score(y_true, labels_priv)
    nmi_priv = normalized_mutual_info_score(y_true, labels_priv)

    print(f"\n[Private TMM Results]")
    print(f"  SSE:         {sse_priv:.4f} (+{(sse_priv/sse_std - 1)*100:.1f}%)")
    print(f"  Silhouette:  {sil_priv:.4f} ({sil_priv/sil_std*100:.1f}%)")
    print(f"  ARI:         {ari_priv:.4f} ({ari_priv/ari_std*100:.1f}%)")
    print(f"  NMI:         {nmi_priv:.4f} ({nmi_priv/nmi_std*100:.1f}%)")

    # Test 3: Different epsilon values
    print("\n" + "-" * 80)
    print("Test 3: Testing different epsilon values (L=2.0)")
    print("-" * 80)

    epsilons = [0.5, 1.0, 2.0, 4.0, 8.0]
    print(f"\n{'Epsilon':<10} {'SSE':<10} {'Silhouette':<12} {'ARI':<10} {'NMI':<10}")
    print("-" * 52)

    for eps in epsilons:
        private_tmm = PrivateTMM(
            n_components=3,
            epsilon=eps,
            L=2.0,
            nu=15.0,
            alpha=0.01,
            max_iter=100,
            random_state=42,
            fixed_nu=True
        )
        private_tmm.fit(X)

        labels = private_tmm.labels_
        sse = compute_sse(X, labels, private_tmm.means_)
        sil = silhouette_score(X, labels)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)

        print(f"{eps:<10.1f} {sse:<10.2f} {sil:<12.4f} {ari:<10.4f} {nmi:<10.4f}")

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
