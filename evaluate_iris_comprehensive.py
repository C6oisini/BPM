"""
使用Iris数据集进行全面评估。

测试配置：
- L值: 2, 6, 10
- ε值: 0.1, 0.5, 1, 2, 4, 8
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from clustering import PrivateKMeans
import matplotlib.pyplot as plt
import pandas as pd


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
    print("Iris数据集全面评估 - 不同L和ε配置")
    print("=" * 80)

    # 加载数据
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    X_normalized = normalize_data(X)
    n_clusters = 3

    print(f"\n数据集信息:")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  类别数: {n_clusters}")

    # ========================================================================
    # 1. 标准K-means基准
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. 标准K-means（基准）")
    print("=" * 80)

    kmeans_std = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_std = kmeans_std.fit_predict(X_normalized)
    centers_std = kmeans_std.cluster_centers_

    sse_std = compute_sse(X_normalized, labels_std, centers_std)
    silhouette_std = silhouette_score(X_normalized, labels_std)
    ari_std = adjusted_rand_score(y_true, labels_std)
    nmi_std = normalized_mutual_info_score(y_true, labels_std)

    print(f"\n  SSE:          {sse_std:.4f}")
    print(f"  轮廓系数:      {silhouette_std:.4f}")
    print(f"  ARI:          {ari_std:.4f}")
    print(f"  NMI:          {nmi_std:.4f}")

    # ========================================================================
    # 2. 全面测试不同配置
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. 测试不同(ε, L)配置")
    print("=" * 80)

    epsilon_values = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
    L_values = [2, 6, 10]

    all_results = []

    for L in L_values:
        print(f"\n{'─'*80}")
        print(f"L = {L}")
        print(f"{'─'*80}")
        print(f"{'ε':<8} {'SSE':<12} {'SSE增加%':<12} {'轮廓系数':<12} {'ARI':<12} {'NMI':<12}")
        print("-" * 80)

        for epsilon in epsilon_values:
            # 运行多次取平均（减少随机性影响）
            n_runs = 5
            sses, silhouettes, aris, nmis = [], [], [], []

            for seed in range(42, 42 + n_runs):
                try:
                    kmeans_priv = PrivateKMeans(
                        n_clusters=n_clusters,
                        epsilon=epsilon,
                        L=L,
                        random_state=seed
                    )
                    kmeans_priv.fit(X_normalized)
                    labels_priv = kmeans_priv.predict(X_normalized)

                    # 检查是否所有点都在同一个簇（会导致silhouette失败）
                    n_unique_labels = len(np.unique(labels_priv))
                    if n_unique_labels < 2:
                        # 所有点在同一簇，使用最差值
                        sses.append(float('inf'))
                        silhouettes.append(0.0)
                        aris.append(0.0)
                        nmis.append(0.0)
                        continue

                    sse = kmeans_priv.compute_sse(X_normalized)
                    silhouette = silhouette_score(X_normalized, labels_priv)
                    ari = adjusted_rand_score(y_true, labels_priv)
                    nmi = normalized_mutual_info_score(y_true, labels_priv)

                    sses.append(sse)
                    silhouettes.append(silhouette)
                    aris.append(ari)
                    nmis.append(nmi)
                except Exception as e:
                    # 发生错误时使用最差值
                    sses.append(float('inf'))
                    silhouettes.append(0.0)
                    aris.append(0.0)
                    nmis.append(0.0)

            # 计算平均值（过滤掉inf值）
            valid_sses = [s for s in sses if s != float('inf')]
            if valid_sses:
                sse_avg = np.mean(valid_sses)
            else:
                sse_avg = float('inf')

            silhouette_avg = np.mean(silhouettes)
            ari_avg = np.mean(aris)
            nmi_avg = np.mean(nmis)

            if sse_avg != float('inf'):
                sse_increase = ((sse_avg - sse_std) / sse_std) * 100
            else:
                sse_increase = float('inf')

            print(f"{epsilon:<8.1f} {sse_avg:<12.4f} {sse_increase:>10.2f}% "
                  f"{silhouette_avg:<12.4f} {ari_avg:<12.4f} {nmi_avg:<12.4f}")

            all_results.append({
                'L': L,
                'epsilon': epsilon,
                'sse': sse_avg,
                'sse_increase': sse_increase,
                'silhouette': silhouette_avg,
                'ari': ari_avg,
                'nmi': nmi_avg
            })

    # ========================================================================
    # 3. 数据分析和可视化
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. 生成综合分析图表")
    print("=" * 80)

    df = pd.DataFrame(all_results)

    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    colors = {2: 'blue', 6: 'green', 10: 'red'}
    markers = {2: 'o', 6: 's', 10: '^'}

    # 图1: SSE vs ε (不同L)
    ax1 = fig.add_subplot(gs[0, 0])
    for L in L_values:
        data = df[df['L'] == L]
        ax1.plot(data['epsilon'], data['sse'],
                marker=markers[L], linewidth=2, markersize=8,
                color=colors[L], label=f'L={L}')
    ax1.axhline(y=sse_std, color='black', linestyle='--', alpha=0.5, label='Standard')
    ax1.set_xlabel('Privacy Budget ε', fontsize=11)
    ax1.set_ylabel('SSE', fontsize=11)
    ax1.set_title('SSE vs ε (Different L)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 图2: 轮廓系数 vs ε (不同L)
    ax2 = fig.add_subplot(gs[0, 1])
    for L in L_values:
        data = df[df['L'] == L]
        ax2.plot(data['epsilon'], data['silhouette'],
                marker=markers[L], linewidth=2, markersize=8,
                color=colors[L], label=f'L={L}')
    ax2.axhline(y=silhouette_std, color='black', linestyle='--', alpha=0.5, label='Standard')
    ax2.set_xlabel('Privacy Budget ε', fontsize=11)
    ax2.set_ylabel('Silhouette Coefficient', fontsize=11)
    ax2.set_title('Silhouette vs ε (Different L)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: ARI vs ε (不同L)
    ax3 = fig.add_subplot(gs[0, 2])
    for L in L_values:
        data = df[df['L'] == L]
        ax3.plot(data['epsilon'], data['ari'],
                marker=markers[L], linewidth=2, markersize=8,
                color=colors[L], label=f'L={L}')
    ax3.axhline(y=ari_std, color='black', linestyle='--', alpha=0.5, label='Standard')
    ax3.set_xlabel('Privacy Budget ε', fontsize=11)
    ax3.set_ylabel('Adjusted Rand Index', fontsize=11)
    ax3.set_title('ARI vs ε (Different L)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4: NMI vs ε (不同L)
    ax4 = fig.add_subplot(gs[1, 0])
    for L in L_values:
        data = df[df['L'] == L]
        ax4.plot(data['epsilon'], data['nmi'],
                marker=markers[L], linewidth=2, markersize=8,
                color=colors[L], label=f'L={L}')
    ax4.axhline(y=nmi_std, color='black', linestyle='--', alpha=0.5, label='Standard')
    ax4.set_xlabel('Privacy Budget ε', fontsize=11)
    ax4.set_ylabel('Normalized Mutual Info', fontsize=11)
    ax4.set_title('NMI vs ε (Different L)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 图5: 热图 - 轮廓系数
    ax5 = fig.add_subplot(gs[1, 1])
    pivot_silhouette = df.pivot(index='L', columns='epsilon', values='silhouette')
    im1 = ax5.imshow(pivot_silhouette, cmap='RdYlGn', aspect='auto', vmin=0, vmax=silhouette_std)
    ax5.set_xticks(range(len(epsilon_values)))
    ax5.set_xticklabels(epsilon_values)
    ax5.set_yticks(range(len(L_values)))
    ax5.set_yticklabels(L_values)
    ax5.set_xlabel('ε', fontsize=11)
    ax5.set_ylabel('L', fontsize=11)
    ax5.set_title('Silhouette Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax5)
    # 添加数值标注
    for i in range(len(L_values)):
        for j in range(len(epsilon_values)):
            text = ax5.text(j, i, f'{pivot_silhouette.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

    # 图6: 热图 - ARI
    ax6 = fig.add_subplot(gs[1, 2])
    pivot_ari = df.pivot(index='L', columns='epsilon', values='ari')
    im2 = ax6.imshow(pivot_ari, cmap='RdYlGn', aspect='auto', vmin=0, vmax=ari_std)
    ax6.set_xticks(range(len(epsilon_values)))
    ax6.set_xticklabels(epsilon_values)
    ax6.set_yticks(range(len(L_values)))
    ax6.set_yticklabels(L_values)
    ax6.set_xlabel('ε', fontsize=11)
    ax6.set_ylabel('L', fontsize=11)
    ax6.set_title('ARI Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax6)
    # 添加数值标注
    for i in range(len(L_values)):
        for j in range(len(epsilon_values)):
            text = ax6.text(j, i, f'{pivot_ari.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

    # 图7: SSE增加百分比对比
    ax7 = fig.add_subplot(gs[2, 0])
    x = np.arange(len(epsilon_values))
    width = 0.25
    for idx, L in enumerate(L_values):
        data = df[df['L'] == L]
        ax7.bar(x + idx*width, data['sse_increase'], width,
               label=f'L={L}', color=colors[L], alpha=0.7)
    ax7.set_xlabel('ε', fontsize=11)
    ax7.set_ylabel('SSE Increase (%)', fontsize=11)
    ax7.set_title('SSE Increase Percentage', fontsize=12, fontweight='bold')
    ax7.set_xticks(x + width)
    ax7.set_xticklabels(epsilon_values)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.set_yscale('log')

    # 图8: 最佳配置识别
    ax8 = fig.add_subplot(gs[2, 1:])
    # 综合得分：考虑轮廓系数和ARI的平均值
    df['综合得分'] = (df['silhouette'] / silhouette_std + df['ari'] / ari_std) / 2

    # 为每个L找最佳ε
    best_configs = []
    for L in L_values:
        data = df[df['L'] == L]
        best_idx = data['综合得分'].idxmax()
        best = data.loc[best_idx]
        best_configs.append(best)

    config_labels = [f"L={int(b['L'])}, ε={b['epsilon']}" for b in best_configs]
    x_pos = np.arange(len(best_configs))

    # 绘制综合得分
    bars = ax8.bar(x_pos, [b['综合得分'] for b in best_configs],
                  color=[colors[int(b['L'])] for b in best_configs], alpha=0.7)
    ax8.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Standard = 1.0')
    ax8.set_xlabel('Configuration', fontsize=11)
    ax8.set_ylabel('Composite Score\n(Silhouette + ARI) / 2', fontsize=11)
    ax8.set_title('Best Configuration for Each L', fontsize=12, fontweight='bold')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(config_labels)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # 在柱子上添加具体数值
    for i, (bar, config) in enumerate(zip(bars, best_configs)):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\nSSE+{config["sse_increase"]:.0f}%',
                ha='center', va='bottom', fontsize=9)

    plt.savefig('iris_comprehensive.png', dpi=150, bbox_inches='tight')
    print("\n  可视化已保存到: iris_comprehensive.png")

    # ========================================================================
    # 4. 详细分析报告
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. 详细分析报告")
    print("=" * 80)

    print(f"\n【最佳配置总结】")
    for config in best_configs:
        print(f"\n  L = {int(config['L'])} 的最佳ε: {config['epsilon']}")
        print(f"    综合得分:      {config['综合得分']:.4f} (标准=1.0)")
        print(f"    SSE增加:       {config['sse_increase']:.2f}%")
        print(f"    轮廓系数:      {config['silhouette']:.4f} (标准={silhouette_std:.4f})")
        print(f"    ARI:           {config['ari']:.4f} (标准={ari_std:.4f})")
        print(f"    NMI:           {config['nmi']:.4f} (标准={nmi_std:.4f})")

    # 找全局最佳
    global_best_idx = df['综合得分'].idxmax()
    global_best = df.loc[global_best_idx]

    print(f"\n【全局最佳配置】")
    print(f"  L = {int(global_best['L'])}, ε = {global_best['epsilon']}")
    print(f"  综合得分:      {global_best['综合得分']:.4f}")
    print(f"  SSE增加:       {global_best['sse_increase']:.2f}%")
    print(f"  轮廓系数:      {global_best['silhouette']:.4f} (保留 {global_best['silhouette']/silhouette_std*100:.1f}%)")
    print(f"  ARI:           {global_best['ari']:.4f} (保留 {global_best['ari']/ari_std*100:.1f}%)")
    print(f"  NMI:           {global_best['nmi']:.4f} (保留 {global_best['nmi']/nmi_std*100:.1f}%)")

    print(f"\n【关键发现】")
    print(f"  1. L值的影响：")
    for L in L_values:
        data = df[df['L'] == L]
        print(f"     L={L}: 平均轮廓系数 = {data['silhouette'].mean():.4f}, "
              f"平均ARI = {data['ari'].mean():.4f}")

    print(f"\n  2. ε值的影响：")
    for eps in epsilon_values:
        data = df[df['epsilon'] == eps]
        print(f"     ε={eps}: 平均轮廓系数 = {data['silhouette'].mean():.4f}, "
              f"平均ARI = {data['ari'].mean():.4f}")

    print(f"\n  3. 推荐配置（按使用场景）：")
    print(f"     高质量需求:     L={int(global_best['L'])}, ε={global_best['epsilon']}")

    # 找SSE增加最小的配置
    min_sse_idx = df['sse_increase'].idxmin()
    min_sse_config = df.loc[min_sse_idx]
    print(f"     低SSE需求:      L={int(min_sse_config['L'])}, ε={min_sse_config['epsilon']} "
          f"(SSE仅增加{min_sse_config['sse_increase']:.1f}%)")

    # 找中等隐私预算的最佳配置
    mid_privacy = df[df['epsilon'].isin([1.0, 2.0])]
    mid_best_idx = mid_privacy['综合得分'].idxmax()
    mid_best = mid_privacy.loc[mid_best_idx]
    print(f"     中等隐私预算:   L={int(mid_best['L'])}, ε={mid_best['epsilon']} "
          f"(综合得分{mid_best['综合得分']:.3f})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
