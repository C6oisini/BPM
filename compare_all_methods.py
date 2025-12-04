"""
Comprehensive comparison of K-means, GMM, and TMM with BPM privacy protection.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from clustering.kmeans import PrivateKMeans
from clustering.gmm import PrivateGMM
from clustering.tmm import TMM, PrivateTMM
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.mixture import GaussianMixture


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
    print("BPM机制：K-means vs GMM vs TMM 全面对比分析")
    print("=" * 80)

    # Load Iris dataset
    print("\n" + "-" * 80)
    print("1. 准备数据集")
    print("-" * 80)

    iris = load_iris()
    X = iris.data
    y_true = iris.target
    n_samples, n_features = X.shape

    # Normalize data to [0,1]^d for BPM mechanism
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_normalized = (X - X_min) / (X_max - X_min)

    print(f"  数据集: Iris")
    print(f"  样本数: {n_samples}")
    print(f"  特征数: {n_features}")
    print(f"  簇数: {len(np.unique(y_true))}")

    # Standard algorithms (no privacy)
    print("\n" + "-" * 80)
    print("2. 标准算法基准（无隐私保护）")
    print("-" * 80)

    # K-means
    print("\n  [K-means]")
    kmeans_std = SKLearnKMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_std.fit(X)
    labels_km = kmeans_std.labels_
    sse_km = compute_sse(X, labels_km, kmeans_std.cluster_centers_)
    sil_km = silhouette_score(X, labels_km)
    ari_km = adjusted_rand_score(y_true, labels_km)
    nmi_km = normalized_mutual_info_score(y_true, labels_km)

    print(f"    SSE:          {sse_km:.4f}")
    print(f"    轮廓系数:      {sil_km:.4f}")
    print(f"    ARI:          {ari_km:.4f}")
    print(f"    NMI:          {nmi_km:.4f}")

    # GMM
    print("\n  [GMM]")
    gmm_std = GaussianMixture(n_components=3, random_state=42, n_init=10)
    gmm_std.fit(X)
    labels_gmm = gmm_std.predict(X)
    sse_gmm = compute_sse(X, labels_gmm, gmm_std.means_)
    sil_gmm = silhouette_score(X, labels_gmm)
    ari_gmm = adjusted_rand_score(y_true, labels_gmm)
    nmi_gmm = normalized_mutual_info_score(y_true, labels_gmm)
    ll_gmm = gmm_std.score(X) * len(X)

    print(f"    SSE:          {sse_gmm:.4f}")
    print(f"    轮廓系数:      {sil_gmm:.4f}")
    print(f"    ARI:          {ari_gmm:.4f}")
    print(f"    NMI:          {nmi_gmm:.4f}")
    print(f"    对数似然:      {ll_gmm:.4f}")

    # TMM
    print("\n  [TMM]")
    tmm_std = TMM(n_components=3, nu=15.0, alpha=0.01, max_iter=100,
                  random_state=42, fixed_nu=True)
    tmm_std.fit(X)
    labels_tmm = tmm_std.labels_
    sse_tmm = compute_sse(X, labels_tmm, tmm_std.means_)
    sil_tmm = silhouette_score(X, labels_tmm)
    ari_tmm = adjusted_rand_score(y_true, labels_tmm)
    nmi_tmm = normalized_mutual_info_score(y_true, labels_tmm)

    print(f"    SSE:          {sse_tmm:.4f}")
    print(f"    轮廓系数:      {sil_tmm:.4f}")
    print(f"    ARI:          {ari_tmm:.4f}")
    print(f"    NMI:          {nmi_tmm:.4f}")

    # Private algorithms with BPM
    print("\n" + "-" * 80)
    print("3. 私有算法对比（BPM机制）")
    print("-" * 80)
    print("\n  配置: ε=8.0, L=2.0")

    # Private K-means
    print("\n  [私有K-means]")
    private_kmeans = PrivateKMeans(n_clusters=3, epsilon=8.0, L=2.0, random_state=42)
    private_kmeans.fit(X_normalized)
    labels_km_priv = private_kmeans.labels_
    sse_km_priv = compute_sse(X, labels_km_priv, private_kmeans.cluster_centers_ * (X_max - X_min) + X_min)
    sil_km_priv = silhouette_score(X, labels_km_priv)
    ari_km_priv = adjusted_rand_score(y_true, labels_km_priv)
    nmi_km_priv = normalized_mutual_info_score(y_true, labels_km_priv)

    print(f"    SSE:          {sse_km_priv:.4f} (+{(sse_km_priv/sse_km - 1)*100:.1f}%)")
    print(f"    轮廓系数:      {sil_km_priv:.4f} ({sil_km_priv/sil_km*100:.1f}%)")
    print(f"    ARI:          {ari_km_priv:.4f} ({ari_km_priv/ari_km*100:.1f}%)")
    print(f"    NMI:          {nmi_km_priv:.4f} ({nmi_km_priv/nmi_km*100:.1f}%)")

    # Private GMM
    print("\n  [私有GMM]")
    private_gmm = PrivateGMM(n_components=3, epsilon=8.0, L=2.0, random_state=42)
    private_gmm.fit(X_normalized)
    labels_gmm_priv = private_gmm.predict(X)
    n_unique_labels_gmm = len(np.unique(labels_gmm_priv))
    if n_unique_labels_gmm < 2:
        print(f"    警告: 所有点被分配到同一个簇 (unique labels: {n_unique_labels_gmm})")
        sse_gmm_priv = float('inf')
        sil_gmm_priv = 0.0
        ari_gmm_priv = 0.0
        nmi_gmm_priv = 0.0
    else:
        sse_gmm_priv = compute_sse(X, labels_gmm_priv, private_gmm.means_ * (X_max - X_min) + X_min)
        sil_gmm_priv = silhouette_score(X, labels_gmm_priv)
        ari_gmm_priv = adjusted_rand_score(y_true, labels_gmm_priv)
        nmi_gmm_priv = normalized_mutual_info_score(y_true, labels_gmm_priv)

    print(f"    SSE:          {sse_gmm_priv:.4f} (+{(sse_gmm_priv/sse_gmm - 1)*100:.1f}%)")
    print(f"    轮廓系数:      {sil_gmm_priv:.4f} ({sil_gmm_priv/sil_gmm*100:.1f}%)")
    print(f"    ARI:          {ari_gmm_priv:.4f} ({ari_gmm_priv/ari_gmm*100:.1f}%)")
    print(f"    NMI:          {nmi_gmm_priv:.4f} ({nmi_gmm_priv/nmi_gmm*100:.1f}%)")

    # Private TMM
    print("\n  [私有TMM]")
    private_tmm = PrivateTMM(n_components=3, epsilon=8.0, L=2.0, nu=15.0,
                             alpha=0.01, max_iter=100, random_state=42, fixed_nu=True)
    private_tmm.fit(X_normalized)
    labels_tmm_priv = private_tmm.labels_
    n_unique_labels_tmm = len(np.unique(labels_tmm_priv))
    if n_unique_labels_tmm < 2:
        print(f"    警告: 所有点被分配到同一个簇 (unique labels: {n_unique_labels_tmm})")
        sse_tmm_priv = float('inf')
        sil_tmm_priv = 0.0
        ari_tmm_priv = 0.0
        nmi_tmm_priv = 0.0
    else:
        sse_tmm_priv = compute_sse(X, labels_tmm_priv, private_tmm.means_ * (X_max - X_min) + X_min)
        sil_tmm_priv = silhouette_score(X, labels_tmm_priv)
        ari_tmm_priv = adjusted_rand_score(y_true, labels_tmm_priv)
        nmi_tmm_priv = normalized_mutual_info_score(y_true, labels_tmm_priv)

    print(f"    SSE:          {sse_tmm_priv:.4f} (+{(sse_tmm_priv/sse_tmm - 1)*100:.1f}%)")
    print(f"    轮廓系数:      {sil_tmm_priv:.4f} ({sil_tmm_priv/sil_tmm*100:.1f}%)")
    print(f"    ARI:          {ari_tmm_priv:.4f} ({ari_tmm_priv/ari_tmm*100:.1f}%)")
    print(f"    NMI:          {nmi_tmm_priv:.4f} ({nmi_tmm_priv/nmi_tmm*100:.1f}%)")

    # Visualization
    print("\n" + "-" * 80)
    print("4. 生成可视化对比图")
    print("-" * 80)

    # Use PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('K-means vs GMM vs TMM Comparison (with BPM Privacy)',
                 fontsize=16, fontweight='bold')

    # Row 1: Standard algorithms
    methods_std = [
        ('Standard K-means', labels_km, kmeans_std.cluster_centers_, sse_km, sil_km),
        ('Standard GMM', labels_gmm, gmm_std.means_, sse_gmm, sil_gmm),
        ('Standard TMM', labels_tmm, tmm_std.means_, sse_tmm, sil_tmm)
    ]

    for idx, (title, labels, centers, sse, sil) in enumerate(methods_std):
        ax = axes[0, idx]
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                           cmap='viridis', alpha=0.6, s=50)
        centers_pca = pca.transform(centers)
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
                  c='red', marker='X', s=200, edgecolors='black', linewidths=2)
        ax.set_title(f'{title}\nSSE={sse:.2f}, Sil={sil:.3f}', fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)

    # Row 2: Private algorithms (denormalize centers for visualization)
    methods_priv = [
        ('Private K-means (BPM)', labels_km_priv,
         private_kmeans.cluster_centers_ * (X_max - X_min) + X_min,
         sse_km_priv, sil_km_priv),
        ('Private GMM (BPM)', labels_gmm_priv,
         private_gmm.means_ * (X_max - X_min) + X_min,
         sse_gmm_priv, sil_gmm_priv),
        ('Private TMM (BPM)', labels_tmm_priv,
         private_tmm.means_ * (X_max - X_min) + X_min,
         sse_tmm_priv, sil_tmm_priv)
    ]

    for idx, (title, labels, centers, sse, sil) in enumerate(methods_priv):
        ax = axes[1, idx]
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                           cmap='viridis', alpha=0.6, s=50)
        centers_pca = pca.transform(centers)
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
                  c='red', marker='X', s=200, edgecolors='black', linewidths=2)
        ax.set_title(f'{title}\nε=8.0, L=2.0\nSSE={sse:.2f}, Sil={sil:.3f}',
                    fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('all_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("\n  可视化已保存到: all_methods_comparison.png")

    # Detailed analysis
    print("\n" + "=" * 80)
    print("5. 详细对比分析")
    print("=" * 80)

    print("\n【标准算法对比（无隐私保护）】")
    print(f"  K-means: SSE={sse_km:.2f}, 轮廓系数={sil_km:.3f}, ARI={ari_km:.3f}")
    print(f"  GMM:     SSE={sse_gmm:.2f}, 轮廓系数={sil_gmm:.3f}, ARI={ari_gmm:.3f}")
    print(f"  TMM:     SSE={sse_tmm:.2f}, 轮廓系数={sil_tmm:.3f}, ARI={ari_tmm:.3f}")

    print("\n【私有算法对比（ε=8.0, L=2.0）】")
    print(f"  私有K-means: SSE={sse_km_priv:.2f}, 轮廓系数={sil_km_priv:.3f}, ARI={ari_km_priv:.3f}")
    print(f"  私有GMM:     SSE={sse_gmm_priv:.2f}, 轮廓系数={sil_gmm_priv:.3f}, ARI={ari_gmm_priv:.3f}")
    print(f"  私有TMM:     SSE={sse_tmm_priv:.2f}, 轮廓系数={sil_tmm_priv:.3f}, ARI={ari_tmm_priv:.3f}")

    print("\n【隐私代价分析】")
    print(f"  K-means: SSE增加 {(sse_km_priv/sse_km - 1)*100:.1f}%, "
          f"轮廓系数保留 {sil_km_priv/sil_km*100:.1f}%, ARI保留 {ari_km_priv/ari_km*100:.1f}%")
    print(f"  GMM:     SSE增加 {(sse_gmm_priv/sse_gmm - 1)*100:.1f}%, "
          f"轮廓系数保留 {sil_gmm_priv/sil_gmm*100:.1f}%, ARI保留 {ari_gmm_priv/ari_gmm*100:.1f}%")
    print(f"  TMM:     SSE增加 {(sse_tmm_priv/sse_tmm - 1)*100:.1f}%, "
          f"轮廓系数保留 {sil_tmm_priv/sil_tmm*100:.1f}%, ARI保留 {ari_tmm_priv/ari_tmm*100:.1f}%")

    print("\n【各方法特点总结】")
    print("\n  ✓ K-means:")
    print("    - 简单快速，计算效率高")
    print("    - 硬聚类（每个点仅属于一个簇）")
    print("    - 对噪声和异常值敏感")
    print("    - BPM噪声下仍然表现稳定")

    print("\n  ✓ GMM (高斯混合模型):")
    print("    - 软聚类（每个点有概率分布）")
    print("    - 可捕捉椭圆形簇和不同大小的簇")
    print("    - 提供概率模型（对数似然、AIC、BIC）")
    print("    - 对BPM噪声较敏感（需估计协方差矩阵）")

    print("\n  ✓ TMM (T混合模型):")
    print("    - 软聚类（基于Student's t分布）")
    print("    - 比GMM对异常值更鲁棒（厚尾分布）")
    print("    - 适合有异常值的数据")
    print("    - BPM噪声下表现最稳定")

    print("\n【推荐使用场景】")
    print("  → 数据干净、追求速度：使用 K-means")
    print("  → 需要软聚类、概率建模：使用 GMM")
    print("  → 数据有异常值、追求鲁棒性：使用 TMM")
    print("  → 隐私保护 + 鲁棒性：优先考虑 TMM")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
