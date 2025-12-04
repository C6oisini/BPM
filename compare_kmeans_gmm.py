"""
比较BPM机制在K-means和GMM上的表现。

展示软聚类（GMM）相比硬聚类（K-means）的优势。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from clustering import PrivateKMeans, PrivateGMM


def normalize_data(X):
    """归一化数据到[0,1]^d"""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-10)


def compute_sse(X, labels, centers):
    """计算SSE"""
    sse = 0.0
    for k in range(len(centers)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            sse += np.sum(np.linalg.norm(cluster_points - centers[k], axis=1) ** 2)
    return sse


def main():
    print("=" * 80)
    print("BPM机制：K-means vs GMM 对比分析")
    print("=" * 80)

    # ========================================================================
    # 1. 加载和准备数据
    # ========================================================================
    print("\n" + "-" * 80)
    print("1. 准备数据集")
    print("-" * 80)

    # 使用Iris数据集
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    X_normalized = normalize_data(X)

    n_samples, n_features = X_normalized.shape
    n_components = 3

    print(f"  数据集: Iris")
    print(f"  样本数: {n_samples}")
    print(f"  特征数: {n_features}")
    print(f"  簇数: {n_components}")

    # ========================================================================
    # 2. 标准算法基准（无隐私）
    # ========================================================================
    print("\n" + "-" * 80)
    print("2. 标准算法基准（无隐私保护）")
    print("-" * 80)

    # 标准K-means
    print("\n  [K-means]")
    kmeans_std = KMeans(n_clusters=n_components, random_state=42, n_init=10)
    labels_km = kmeans_std.fit_predict(X_normalized)

    sse_km = compute_sse(X_normalized, labels_km, kmeans_std.cluster_centers_)
    sil_km = silhouette_score(X_normalized, labels_km)
    ari_km = adjusted_rand_score(y_true, labels_km)
    nmi_km = normalized_mutual_info_score(y_true, labels_km)

    print(f"    SSE:          {sse_km:.4f}")
    print(f"    轮廓系数:      {sil_km:.4f}")
    print(f"    ARI:          {ari_km:.4f}")
    print(f"    NMI:          {nmi_km:.4f}")

    # 标准GMM
    print("\n  [GMM]")
    gmm_std = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm_std.fit(X_normalized)
    labels_gmm = gmm_std.predict(X_normalized)
    proba_gmm = gmm_std.predict_proba(X_normalized)

    sse_gmm = compute_sse(X_normalized, labels_gmm, gmm_std.means_)
    sil_gmm = silhouette_score(X_normalized, labels_gmm)
    ari_gmm = adjusted_rand_score(y_true, labels_gmm)
    nmi_gmm = normalized_mutual_info_score(y_true, labels_gmm)
    log_likelihood = gmm_std.score(X_normalized) * n_samples
    aic = gmm_std.aic(X_normalized)
    bic = gmm_std.bic(X_normalized)

    print(f"    SSE:          {sse_gmm:.4f}")
    print(f"    轮廓系数:      {sil_gmm:.4f}")
    print(f"    ARI:          {ari_gmm:.4f}")
    print(f"    NMI:          {nmi_gmm:.4f}")
    print(f"    对数似然:      {log_likelihood:.4f}")
    print(f"    AIC:          {aic:.4f}")
    print(f"    BIC:          {bic:.4f}")

    # ========================================================================
    # 3. 私有算法对比（BPM）
    # ========================================================================
    print("\n" + "-" * 80)
    print("3. 私有算法对比（BPM机制）")
    print("-" * 80)

    epsilon = 8.0
    L = 2.0

    print(f"\n  配置: ε={epsilon}, L={L}")

    # 私有K-means
    print("\n  [私有K-means]")
    kmeans_priv = PrivateKMeans(
        n_clusters=n_components,
        epsilon=epsilon,
        L=L,
        random_state=42
    )
    kmeans_priv.fit(X_normalized)
    labels_km_priv = kmeans_priv.predict(X_normalized)

    sse_km_priv = kmeans_priv.compute_sse(X_normalized)
    sil_km_priv = silhouette_score(X_normalized, labels_km_priv)
    ari_km_priv = adjusted_rand_score(y_true, labels_km_priv)
    nmi_km_priv = normalized_mutual_info_score(y_true, labels_km_priv)

    print(f"    SSE:          {sse_km_priv:.4f} (+{(sse_km_priv-sse_km)/sse_km*100:.1f}%)")
    print(f"    轮廓系数:      {sil_km_priv:.4f} ({sil_km_priv/sil_km*100:.1f}%)")
    print(f"    ARI:          {ari_km_priv:.4f} ({ari_km_priv/ari_km*100:.1f}%)")
    print(f"    NMI:          {nmi_km_priv:.4f} ({nmi_km_priv/nmi_km*100:.1f}%)")

    # 私有GMM
    print("\n  [私有GMM]")
    gmm_priv = PrivateGMM(
        n_components=n_components,
        epsilon=epsilon,
        L=L,
        random_state=42
    )
    gmm_priv.fit(X_normalized)
    labels_gmm_priv = gmm_priv.predict(X_normalized)
    proba_gmm_priv = gmm_priv.predict_proba(X_normalized)

    sse_gmm_priv = compute_sse(X_normalized, labels_gmm_priv, gmm_priv.means_)
    sil_gmm_priv = silhouette_score(X_normalized, labels_gmm_priv)
    ari_gmm_priv = adjusted_rand_score(y_true, labels_gmm_priv)
    nmi_gmm_priv = normalized_mutual_info_score(y_true, labels_gmm_priv)
    log_likelihood_priv = gmm_priv.score(X_normalized) * n_samples
    aic_priv = gmm_priv.compute_aic(X_normalized)
    bic_priv = gmm_priv.compute_bic(X_normalized)

    print(f"    SSE:          {sse_gmm_priv:.4f} (+{(sse_gmm_priv-sse_gmm)/sse_gmm*100:.1f}%)")
    print(f"    轮廓系数:      {sil_gmm_priv:.4f} ({sil_gmm_priv/sil_gmm*100:.1f}%)")
    print(f"    ARI:          {ari_gmm_priv:.4f} ({ari_gmm_priv/ari_gmm*100:.1f}%)")
    print(f"    NMI:          {nmi_gmm_priv:.4f} ({nmi_gmm_priv/nmi_gmm*100:.1f}%)")
    print(f"    对数似然:      {log_likelihood_priv:.4f}")
    print(f"    AIC:          {aic_priv:.4f}")
    print(f"    BIC:          {bic_priv:.4f}")

    # ========================================================================
    # 4. 可视化对比
    # ========================================================================
    print("\n" + "-" * 80)
    print("4. 生成可视化对比图")
    print("-" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 使用前两个主成分可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_normalized)

    # 图1: 标准K-means
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_km,
                          cmap='viridis', alpha=0.6, s=30)
    centers_2d_km = pca.transform(kmeans_std.cluster_centers_)
    ax1.scatter(centers_2d_km[:, 0], centers_2d_km[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    ax1.set_title(f'Standard K-means\nSSE={sse_km:.2f}, Sil={sil_km:.3f}',
                 fontsize=11, fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.grid(True, alpha=0.3)

    # 图2: 标准GMM
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_gmm,
                          cmap='viridis', alpha=0.6, s=30)
    centers_2d_gmm = pca.transform(gmm_std.means_)
    ax2.scatter(centers_2d_gmm[:, 0], centers_2d_gmm[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    ax2.set_title(f'Standard GMM\nSSE={sse_gmm:.2f}, Sil={sil_gmm:.3f}',
                 fontsize=11, fontweight='bold')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid(True, alpha=0.3)

    # 图3: GMM概率分布（标准）
    ax3 = axes[0, 2]
    # 显示每个点属于主要簇的概率
    max_proba = proba_gmm.max(axis=1)
    scatter3 = ax3.scatter(X_2d[:, 0], X_2d[:, 1], c=max_proba,
                          cmap='RdYlGn', alpha=0.6, s=30, vmin=0.33, vmax=1.0)
    plt.colorbar(scatter3, ax=ax3, label='Max Probability')
    ax3.set_title('Standard GMM\nSoft Assignment Confidence',
                 fontsize=11, fontweight='bold')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.grid(True, alpha=0.3)

    # 图4: 私有K-means
    ax4 = axes[1, 0]
    scatter4 = ax4.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_km_priv,
                          cmap='viridis', alpha=0.6, s=30)
    centers_2d_km_priv = pca.transform(kmeans_priv.cluster_centers_)
    ax4.scatter(centers_2d_km_priv[:, 0], centers_2d_km_priv[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    ax4.set_title(f'Private K-means (BPM)\nε={epsilon}, L={L}\n'
                 f'SSE={sse_km_priv:.2f}, Sil={sil_km_priv:.3f}',
                 fontsize=11, fontweight='bold')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.grid(True, alpha=0.3)

    # 图5: 私有GMM
    ax5 = axes[1, 1]
    scatter5 = ax5.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_gmm_priv,
                          cmap='viridis', alpha=0.6, s=30)
    centers_2d_gmm_priv = pca.transform(gmm_priv.means_)
    ax5.scatter(centers_2d_gmm_priv[:, 0], centers_2d_gmm_priv[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    ax5.set_title(f'Private GMM (BPM)\nε={epsilon}, L={L}\n'
                 f'SSE={sse_gmm_priv:.2f}, Sil={sil_gmm_priv:.3f}',
                 fontsize=11, fontweight='bold')
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.grid(True, alpha=0.3)

    # 图6: GMM概率分布（私有）
    ax6 = axes[1, 2]
    max_proba_priv = proba_gmm_priv.max(axis=1)
    scatter6 = ax6.scatter(X_2d[:, 0], X_2d[:, 1], c=max_proba_priv,
                          cmap='RdYlGn', alpha=0.6, s=30, vmin=0.33, vmax=1.0)
    plt.colorbar(scatter6, ax=ax6, label='Max Probability')
    ax6.set_title('Private GMM (BPM)\nSoft Assignment Confidence',
                 fontsize=11, fontweight='bold')
    ax6.set_xlabel('PC1')
    ax6.set_ylabel('PC2')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kmeans_vs_gmm.png', dpi=150, bbox_inches='tight')
    print("\n  可视化已保存到: kmeans_vs_gmm.png")

    # ========================================================================
    # 5. 详细对比分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. 详细对比分析")
    print("=" * 80)

    print(f"\n【K-means vs GMM（标准算法）】")
    print(f"  K-means: SSE={sse_km:.2f}, 轮廓系数={sil_km:.3f}, ARI={ari_km:.3f}")
    print(f"  GMM:     SSE={sse_gmm:.2f}, 轮廓系数={sil_gmm:.3f}, ARI={ari_gmm:.3f}")
    print(f"  → GMM在ARI上{'优于' if ari_gmm > ari_km else '不如'}K-means")

    print(f"\n【K-means vs GMM（BPM私有算法）】")
    print(f"  私有K-means: SSE={sse_km_priv:.2f}, 轮廓系数={sil_km_priv:.3f}, ARI={ari_km_priv:.3f}")
    print(f"  私有GMM:     SSE={sse_gmm_priv:.2f}, 轮廓系数={sil_gmm_priv:.3f}, ARI={ari_gmm_priv:.3f}")
    print(f"  → 私有GMM在ARI上{'优于' if ari_gmm_priv > ari_km_priv else '不如'}私有K-means")

    print(f"\n【隐私代价（K-means）】")
    print(f"  SSE增加:       {(sse_km_priv-sse_km)/sse_km*100:>6.1f}%")
    print(f"  轮廓系数保留:   {sil_km_priv/sil_km*100:>6.1f}%")
    print(f"  ARI保留:       {ari_km_priv/ari_km*100:>6.1f}%")

    print(f"\n【隐私代价（GMM）】")
    print(f"  SSE增加:       {(sse_gmm_priv-sse_gmm)/sse_gmm*100:>6.1f}%")
    print(f"  轮廓系数保留:   {sil_gmm_priv/sil_gmm*100:>6.1f}%")
    print(f"  ARI保留:       {ari_gmm_priv/ari_gmm*100:>6.1f}%")

    print(f"\n【GMM特有优势】")
    print(f"  1. 软聚类：提供每个点属于各簇的概率分布")
    print(f"     - 标准GMM平均最大概率: {max_proba.mean():.3f}")
    print(f"     - 私有GMM平均最大概率: {max_proba_priv.mean():.3f}")
    print(f"  2. 概率模型：可计算对数似然、AIC、BIC")
    print(f"     - 标准GMM对数似然: {log_likelihood:.2f}")
    print(f"     - 私有GMM对数似然: {log_likelihood_priv:.2f}")
    print(f"  3. 更灵活：可捕捉椭圆形簇和不同大小的簇")

    print(f"\n【推荐】")
    if ari_gmm_priv >= ari_km_priv:
        print(f"  ✓ 对于Iris数据集，推荐使用私有GMM（BPM）")
        print(f"    - ARI更高: {ari_gmm_priv:.3f} vs {ari_km_priv:.3f}")
        print(f"    - 提供软聚类概率")
    else:
        print(f"  ✓ 对于Iris数据集，两种方法性能接近")
        print(f"    - 需要软聚类时使用GMM")
        print(f"    - 需要简单快速时使用K-means")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
