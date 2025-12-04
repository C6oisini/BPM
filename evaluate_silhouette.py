"""
使用轮廓系数评估BPM聚类质量。

轮廓系数（Silhouette Coefficient）衡量聚类的紧密度和分离度：
- 取值范围：[-1, 1]
- 接近 +1：样本与自己的簇很匹配，与其他簇分离良好
- 接近  0：样本在两个簇的边界上
- 接近 -1：样本可能被分配到了错误的簇
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from clustering import PrivateKMeans
import matplotlib.pyplot as plt


def normalize_data(X):
    """归一化数据到[0,1]^d"""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-10)


def compute_sse(X, labels, centers):
    """计算SSE（Sum of Squared Errors）"""
    sse = 0.0
    for k in range(len(centers)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            sse += np.sum(np.linalg.norm(cluster_points - centers[k], axis=1) ** 2)
    return sse


def main():
    print("=" * 80)
    print("使用轮廓系数评估BPM聚类质量")
    print("=" * 80)

    # 生成数据
    np.random.seed(42)
    n_samples = 300
    n_features = 2
    n_clusters = 3

    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=42
    )
    X_normalized = normalize_data(X)

    print(f"\n数据集信息:")
    print(f"  样本数: {n_samples}")
    print(f"  特征数: {n_features}")
    print(f"  簇数: {n_clusters}")

    # ========================================================================
    # 1. 标准K-means（无隐私保护）
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. 标准K-means（无隐私保护）")
    print("=" * 80)

    kmeans_std = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_std = kmeans_std.fit_predict(X_normalized)
    centers_std = kmeans_std.cluster_centers_

    # 计算指标
    sse_std = compute_sse(X_normalized, labels_std, centers_std)
    silhouette_std = silhouette_score(X_normalized, labels_std)

    print(f"  SSE: {sse_std:.4f}")
    print(f"  轮廓系数: {silhouette_std:.4f}")

    # ========================================================================
    # 2. 不同配置的私有K-means
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. 私有K-means（BPM）- 不同配置")
    print("=" * 80)

    configs = [
        {"epsilon": 1.0, "L": 0.3},
        {"epsilon": 2.0, "L": 0.3},
        {"epsilon": 4.0, "L": 0.3},
        {"epsilon": 4.0, "L": 0.5},
        {"epsilon": 8.0, "L": 0.3},
    ]

    results = []

    print(f"\n{'ε':<8} {'L':<8} {'SSE':<12} {'轮廓系数':<12} {'SSE增加':<12} {'轮廓系数降低':<15}")
    print("-" * 75)

    for config in configs:
        epsilon = config["epsilon"]
        L = config["L"]

        # 运行私有K-means
        kmeans_priv = PrivateKMeans(
            n_clusters=n_clusters,
            epsilon=epsilon,
            L=L,
            random_state=42
        )
        kmeans_priv.fit(X_normalized)

        # 预测标签
        labels_priv = kmeans_priv.predict(X_normalized)

        # 计算指标
        sse_priv = kmeans_priv.compute_sse(X_normalized)
        silhouette_priv = silhouette_score(X_normalized, labels_priv)

        # 计算变化
        sse_increase_pct = ((sse_priv - sse_std) / sse_std) * 100
        silhouette_decrease = silhouette_std - silhouette_priv
        silhouette_decrease_pct = (silhouette_decrease / silhouette_std) * 100

        print(f"{epsilon:<8.1f} {L:<8.2f} {sse_priv:<12.4f} {silhouette_priv:<12.4f} "
              f"{sse_increase_pct:>10.2f}% {silhouette_decrease_pct:>13.2f}%")

        results.append({
            "epsilon": epsilon,
            "L": L,
            "sse": sse_priv,
            "silhouette": silhouette_priv,
            "sse_increase_pct": sse_increase_pct,
            "silhouette_decrease_pct": silhouette_decrease_pct
        })

    # ========================================================================
    # 3. 可视化对比
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. 生成可视化对比图")
    print("=" * 80)

    # 选择最佳配置（轮廓系数最高的）
    best_config = max(results, key=lambda x: x["silhouette"])
    epsilon_best = best_config["epsilon"]
    L_best = best_config["L"]

    print(f"\n最佳配置（基于轮廓系数）:")
    print(f"  ε = {epsilon_best}, L = {L_best}")
    print(f"  轮廓系数: {best_config['silhouette']:.4f}")
    print(f"  SSE: {best_config['sse']:.4f}")

    # 重新运行最佳配置以获取详细结果
    kmeans_best = PrivateKMeans(
        n_clusters=n_clusters,
        epsilon=epsilon_best,
        L=L_best,
        random_state=42
    )
    kmeans_best.fit(X_normalized)
    labels_best = kmeans_best.predict(X_normalized)

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 图1: 标准K-means聚类结果
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(X_normalized[:, 0], X_normalized[:, 1],
                           c=labels_std, cmap='viridis', alpha=0.6, s=30)
    ax1.scatter(centers_std[:, 0], centers_std[:, 1],
                c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                label='质心')
    ax1.set_xlabel('特征 1')
    ax1.set_ylabel('特征 2')
    ax1.set_title(f'标准K-means\nSSE={sse_std:.4f}, 轮廓系数={silhouette_std:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 私有K-means聚类结果
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(X_normalized[:, 0], X_normalized[:, 1],
                           c=labels_best, cmap='viridis', alpha=0.6, s=30)
    ax2.scatter(kmeans_best.cluster_centers_[:, 0],
                kmeans_best.cluster_centers_[:, 1],
                c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                label='质心')
    ax2.set_xlabel('特征 1')
    ax2.set_ylabel('特征 2')
    ax2.set_title(f'私有K-means (ε={epsilon_best}, L={L_best})\n'
                  f'SSE={best_config["sse"]:.4f}, 轮廓系数={best_config["silhouette"]:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 轮廓系数对比
    ax3 = axes[1, 0]
    epsilons = [r["epsilon"] for r in results if r["L"] == 0.3]
    silhouettes = [r["silhouette"] for r in results if r["L"] == 0.3]
    ax3.plot([0] + epsilons, [silhouette_std] + silhouettes, 'o-', linewidth=2, markersize=8)
    ax3.axhline(y=silhouette_std, color='red', linestyle='--', alpha=0.5, label='标准K-means')
    ax3.set_xlabel('隐私预算 ε')
    ax3.set_ylabel('轮廓系数')
    ax3.set_title('轮廓系数 vs 隐私预算 (L=0.3)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 图4: SSE对比
    ax4 = axes[1, 1]
    sses = [r["sse"] for r in results if r["L"] == 0.3]
    ax4.plot([0] + epsilons, [sse_std] + sses, 's-', linewidth=2, markersize=8, color='orange')
    ax4.axhline(y=sse_std, color='red', linestyle='--', alpha=0.5, label='标准K-means')
    ax4.set_xlabel('隐私预算 ε')
    ax4.set_ylabel('SSE')
    ax4.set_title('SSE vs 隐私预算 (L=0.3)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('silhouette_evaluation.png', dpi=150, bbox_inches='tight')
    print("\n  可视化已保存到: silhouette_evaluation.png")

    # ========================================================================
    # 4. 详细分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. 详细分析")
    print("=" * 80)

    print(f"\n【隐私-效用权衡】")
    print(f"  标准K-means基准:")
    print(f"    - SSE: {sse_std:.4f}")
    print(f"    - 轮廓系数: {silhouette_std:.4f}")

    print(f"\n  最佳私有K-means (ε={epsilon_best}, L={L_best}):")
    print(f"    - SSE: {best_config['sse']:.4f} (+{best_config['sse_increase_pct']:.2f}%)")
    print(f"    - 轮廓系数: {best_config['silhouette']:.4f} (-{best_config['silhouette_decrease_pct']:.2f}%)")

    print(f"\n【关键发现】")
    print(f"  1. 增大ε（隐私预算）通常会提高聚类质量")
    print(f"  2. 轮廓系数比SSE更能反映聚类的内在质量")
    print(f"  3. 即使SSE增加较多，轮廓系数的下降可能较小")
    print(f"  4. 需要在隐私保护和聚类质量之间做出权衡")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
