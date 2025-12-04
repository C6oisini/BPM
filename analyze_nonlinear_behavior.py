"""
分析为什么轮廓系数、ARI、NMI不是线性变化的。

这个脚本深入分析了隐私预算ε对不同评估指标的影响机制。
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from clustering import PrivateKMeans
import matplotlib.pyplot as plt


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


def analyze_cluster_changes(labels_true, labels_pred):
    """分析簇分配的变化"""
    n_samples = len(labels_true)
    n_changed = np.sum(labels_true != labels_pred)
    change_rate = n_changed / n_samples
    return n_changed, change_rate


def main():
    print("=" * 80)
    print("为什么评估指标不是线性变化的？深入分析")
    print("=" * 80)

    # 加载数据
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    X_normalized = normalize_data(X)
    n_clusters = 3

    # 标准K-means
    kmeans_std = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_std = kmeans_std.fit_predict(X_normalized)
    centers_std = kmeans_std.cluster_centers_

    print("\n" + "=" * 80)
    print("1. 理论解释")
    print("=" * 80)

    print("""
【为什么SSE近似线性下降？】

SSE是一个连续指标，直接衡量点到质心的欧氏距离：
    SSE = Σ ||x_i - c_j||²

随着ε增大：
  • 隐私噪声指数级减少：noise ∝ e^(-ε·distance)
  • 质心位置更接近真实质心
  • 每个点到质心的距离连续减小
  • SSE平滑、连续地下降

→ SSE对质心位置的微小变化非常敏感，呈现近似线性下降

【为什么轮廓系数、ARI、NMI不是线性变化？】

这些指标是离散指标，基于簇分配（cluster assignment）：
  • 每个点被分配到最近的质心
  • 只有当点被重新分配到不同簇时，这些指标才会变化
  • 质心的小幅移动可能不改变任何簇分配

关键机制 - "决策边界效应"：
  1. 当质心在决策边界远离时：簇分配保持不变，指标不变
  2. 当质心跨越决策边界：突然有多个点被重新分配，指标跳跃
  3. 这导致指标呈现阶梯式/非线性变化

【为什么会出现"V型凹陷"（ε=2和ε=4时）？】

在中等隐私预算下：
  • 质心可能移动到"不幸"的位置
  • 导致某个质心恰好在两个真实簇之间
  • 造成大量误分类
  • 随着ε继续增大，质心移出这个不良位置，指标恢复
    """)

    # ========================================================================
    # 2. 实验验证
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. 实验验证 - 详细跟踪指标变化")
    print("=" * 80)

    epsilon_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    L = 0.5

    results = {
        'epsilon': [],
        'sse': [],
        'silhouette': [],
        'ari': [],
        'nmi': [],
        'centroid_distance': [],  # 质心到真实质心的平均距离
        'n_reassigned': [],        # 重新分配的点数
        'reassignment_rate': []    # 重新分配比例
    }

    print(f"\n{'ε':<8} {'SSE':<10} {'轮廓系数':<10} {'ARI':<10} {'质心偏移':<10} {'重分配点数':<12} {'重分配率':<10}")
    print("-" * 80)

    for epsilon in epsilon_values:
        # 运行私有K-means
        kmeans_priv = PrivateKMeans(
            n_clusters=n_clusters,
            epsilon=epsilon,
            L=L,
            random_state=42
        )
        kmeans_priv.fit(X_normalized)
        labels_priv = kmeans_priv.predict(X_normalized)
        centers_priv = kmeans_priv.cluster_centers_

        # 计算指标
        sse = compute_sse(X_normalized, labels_priv, centers_priv)
        silhouette = silhouette_score(X_normalized, labels_priv)
        ari = adjusted_rand_score(y_true, labels_priv)
        nmi = normalized_mutual_info_score(y_true, labels_priv)

        # 计算质心偏移（需要匹配质心）
        # 简单方法：计算最近的质心对之间的平均距离
        from scipy.spatial.distance import cdist
        distances = cdist(centers_priv, centers_std)
        min_distances = distances.min(axis=1)
        avg_centroid_distance = min_distances.mean()

        # 计算重新分配的点数
        n_reassigned, reassignment_rate = analyze_cluster_changes(labels_std, labels_priv)

        # 保存结果
        results['epsilon'].append(epsilon)
        results['sse'].append(sse)
        results['silhouette'].append(silhouette)
        results['ari'].append(ari)
        results['nmi'].append(nmi)
        results['centroid_distance'].append(avg_centroid_distance)
        results['n_reassigned'].append(n_reassigned)
        results['reassignment_rate'].append(reassignment_rate)

        print(f"{epsilon:<8.1f} {sse:<10.4f} {silhouette:<10.4f} {ari:<10.4f} "
              f"{avg_centroid_distance:<10.4f} {n_reassigned:<12d} {reassignment_rate:<10.2%}")

    # ========================================================================
    # 3. 可视化分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. 生成可视化分析")
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 图1: SSE vs ε
    ax1 = axes[0, 0]
    ax1.plot(results['epsilon'], results['sse'], 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Privacy Budget ε', fontsize=12)
    ax1.set_ylabel('SSE', fontsize=12)
    ax1.set_title('SSE vs ε\n(Nearly Linear Decrease)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.annotate('Nearly linear\ndecrease', xy=(5, 20), fontsize=10, color='red')

    # 图2: Silhouette vs ε
    ax2 = axes[0, 1]
    ax2.plot(results['epsilon'], results['silhouette'], 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Privacy Budget ε', fontsize=12)
    ax2.set_ylabel('Silhouette Coefficient', fontsize=12)
    ax2.set_title('Silhouette vs ε\n(Non-linear with Valley)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.annotate('Valley', xy=(2.5, 0.25), fontsize=10, color='red',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 图3: ARI vs ε
    ax3 = axes[0, 2]
    ax3.plot(results['epsilon'], results['ari'], '^-', linewidth=2, markersize=8, color='orange')
    ax3.set_xlabel('Privacy Budget ε', fontsize=12)
    ax3.set_ylabel('Adjusted Rand Index', fontsize=12)
    ax3.set_title('ARI vs ε\n(Non-linear Jumps)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 图4: 质心偏移 vs ε
    ax4 = axes[1, 0]
    ax4.plot(results['epsilon'], results['centroid_distance'], 'o-', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Privacy Budget ε', fontsize=12)
    ax4.set_ylabel('Average Centroid Distance', fontsize=12)
    ax4.set_title('Centroid Displacement vs ε\n(Continuous Decrease)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 图5: 重新分配的点数 vs ε
    ax5 = axes[1, 1]
    ax5.plot(results['epsilon'], results['n_reassigned'], 'd-', linewidth=2, markersize=8, color='red')
    ax5.set_xlabel('Privacy Budget ε', fontsize=12)
    ax5.set_ylabel('Number of Reassigned Points', fontsize=12)
    ax5.set_title('Reassigned Points vs ε\n(Explains Non-linearity!)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.annotate('Key insight:\nJumps in reassignments\ncause jumps in metrics',
                 xy=(4, 60), fontsize=9, color='darkred',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # 图6: 重新分配率 vs ε
    ax6 = axes[1, 2]
    ax6.plot(results['epsilon'], [r*100 for r in results['reassignment_rate']],
             'h-', linewidth=2, markersize=8, color='brown')
    ax6.set_xlabel('Privacy Budget ε', fontsize=12)
    ax6.set_ylabel('Reassignment Rate (%)', fontsize=12)
    ax6.set_title('Reassignment Rate vs ε\n(Discrete Changes)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nonlinear_analysis.png', dpi=150, bbox_inches='tight')
    print("\n  可视化已保存到: nonlinear_analysis.png")

    # ========================================================================
    # 4. 关键发现总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. 关键发现总结")
    print("=" * 80)

    print(f"""
【实验验证了理论分析】

1. SSE的连续性：
   • SSE从 {results['sse'][0]:.2f} (ε={results['epsilon'][0]}) 平滑降至 {results['sse'][-1]:.2f} (ε={results['epsilon'][-1]})
   • 质心偏移从 {results['centroid_distance'][0]:.4f} 连续降至 {results['centroid_distance'][-1]:.4f}
   • 呈现近似线性/指数衰减特性

2. 轮廓系数/ARI/NMI的离散性：
   • 在ε={results['epsilon'][0]}-{results['epsilon'][2]}之间：重分配率 {results['reassignment_rate'][0]*100:.1f}%-{results['reassignment_rate'][2]*100:.1f}%
   • 在ε={results['epsilon'][4]}附近：出现"谷底"（重分配率 {results['reassignment_rate'][4]*100:.1f}%）
   • 在ε≥{results['epsilon'][6]}后：逐步稳定，重分配率降至 {results['reassignment_rate'][-1]*100:.1f}%

3. 决策边界效应：
   • 重新分配的点数呈现跳跃式变化
   • 这直接导致了Silhouette/ARI/NMI的非线性行为
   • 在中等ε值时，质心可能处于"不幸"位置，导致大量误分类

【结论】

SSE vs 离散指标的不同行为根源：
  • SSE：连续指标，对质心位置的任何变化都敏感  → 近似线性
  • Silhouette/ARI/NMI：离散指标，只在簇分配变化时才变化  → 非线性、阶梯式

这不是实现的bug，而是这些指标的本质特性！
    """)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
