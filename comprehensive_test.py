"""
Comprehensive comparison of K-means, GMM, and TMM on multiple datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import (load_iris, load_wine, load_breast_cancer,
                               load_digits, make_blobs, make_moons, make_circles)
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from clustering.kmeans import PrivateKMeans
from clustering.gmm import PrivateGMM
from clustering.tmm import TMM, PrivateTMM


def compute_sse(X, labels, centers):
    """Compute sum of squared errors."""
    sse = 0
    for i, center in enumerate(centers):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - center) ** 2)
    return sse


def prepare_datasets():
    """Prepare multiple datasets for testing."""
    datasets = {}

    # 1. Iris dataset
    iris = load_iris()
    datasets['Iris'] = {
        'X': iris.data,
        'y': iris.target,
        'n_clusters': 3,
        'description': 'Iris flowers (4D, 150 samples)'
    }

    # 2. Wine dataset
    wine = load_wine()
    datasets['Wine'] = {
        'X': wine.data,
        'y': wine.target,
        'n_clusters': 3,
        'description': 'Wine quality (13D, 178 samples)'
    }

    # 3. Breast Cancer dataset (binary, so use 2 clusters)
    cancer = load_breast_cancer()
    datasets['Breast Cancer'] = {
        'X': cancer.data,
        'y': cancer.target,
        'n_clusters': 2,
        'description': 'Breast cancer (30D, 569 samples)'
    }

    # 4. Digits subset (only first 3 digits)
    digits = load_digits()
    mask = np.isin(digits.target, [0, 1, 2])
    datasets['Digits (0-2)'] = {
        'X': digits.data[mask],
        'y': digits.target[mask],
        'n_clusters': 3,
        'description': 'Handwritten digits 0-2 (64D, ~180 samples)'
    }

    # 5. Synthetic: Blobs (well-separated)
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, n_features=4,
                                   cluster_std=0.5, random_state=42)
    datasets['Synthetic Blobs'] = {
        'X': X_blobs,
        'y': y_blobs,
        'n_clusters': 3,
        'description': 'Well-separated Gaussian blobs (4D, 300 samples)'
    }

    # 6. Synthetic: Overlapping blobs
    X_overlap, y_overlap = make_blobs(n_samples=300, centers=3, n_features=4,
                                       cluster_std=2.0, random_state=42)
    datasets['Synthetic Overlap'] = {
        'X': X_overlap,
        'y': y_overlap,
        'n_clusters': 3,
        'description': 'Overlapping Gaussian blobs (4D, 300 samples)'
    }

    return datasets


def test_standard_algorithms(X, y_true, n_clusters):
    """Test standard algorithms without privacy."""
    results = {}

    # K-means
    kmeans = SKLearnKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels_km = kmeans.labels_
    results['K-means'] = {
        'SSE': compute_sse(X, labels_km, kmeans.cluster_centers_),
        'Silhouette': silhouette_score(X, labels_km) if len(np.unique(labels_km)) > 1 else 0.0,
        'ARI': adjusted_rand_score(y_true, labels_km),
        'NMI': normalized_mutual_info_score(y_true, labels_km),
        'n_clusters': len(np.unique(labels_km))
    }

    # GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
    gmm.fit(X)
    labels_gmm = gmm.predict(X)
    results['GMM'] = {
        'SSE': compute_sse(X, labels_gmm, gmm.means_),
        'Silhouette': silhouette_score(X, labels_gmm) if len(np.unique(labels_gmm)) > 1 else 0.0,
        'ARI': adjusted_rand_score(y_true, labels_gmm),
        'NMI': normalized_mutual_info_score(y_true, labels_gmm),
        'n_clusters': len(np.unique(labels_gmm))
    }

    # TMM
    try:
        tmm = TMM(n_components=n_clusters, nu=15.0, alpha=0.01, max_iter=50,
                  random_state=42, fixed_nu=True)
        tmm.fit(X)
        labels_tmm = tmm.labels_
        results['TMM'] = {
            'SSE': compute_sse(X, labels_tmm, tmm.means_),
            'Silhouette': silhouette_score(X, labels_tmm) if len(np.unique(labels_tmm)) > 1 else 0.0,
            'ARI': adjusted_rand_score(y_true, labels_tmm),
            'NMI': normalized_mutual_info_score(y_true, labels_tmm),
            'n_clusters': len(np.unique(labels_tmm))
        }
    except Exception as e:
        print(f"    TMM failed: {e}")
        results['TMM'] = {
            'SSE': float('inf'),
            'Silhouette': 0.0,
            'ARI': 0.0,
            'NMI': 0.0,
            'n_clusters': 0
        }

    return results


def test_private_algorithms(X, y_true, n_clusters, epsilon=8.0, L=0.5):
    """Test private algorithms with BPM."""
    # Normalize data
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0  # Avoid division by zero
    X_normalized = (X - X_min) / X_range

    results = {}

    # Private K-means
    try:
        private_kmeans = PrivateKMeans(n_clusters=n_clusters, epsilon=epsilon, L=L, random_state=42)
        private_kmeans.fit(X_normalized)
        labels_km_priv = private_kmeans.labels_
        centers_km_priv = private_kmeans.cluster_centers_ * X_range + X_min

        n_unique = len(np.unique(labels_km_priv))
        results['Private K-means'] = {
            'SSE': compute_sse(X, labels_km_priv, centers_km_priv) if n_unique > 1 else float('inf'),
            'Silhouette': silhouette_score(X, labels_km_priv) if n_unique > 1 else 0.0,
            'ARI': adjusted_rand_score(y_true, labels_km_priv),
            'NMI': normalized_mutual_info_score(y_true, labels_km_priv),
            'n_clusters': n_unique
        }
    except Exception as e:
        print(f"    Private K-means failed: {e}")
        results['Private K-means'] = {
            'SSE': float('inf'),
            'Silhouette': 0.0,
            'ARI': 0.0,
            'NMI': 0.0,
            'n_clusters': 0
        }

    # Private GMM
    try:
        private_gmm = PrivateGMM(n_components=n_clusters, epsilon=epsilon, L=L, random_state=42)
        private_gmm.fit(X_normalized)
        labels_gmm_priv = private_gmm.predict(X)
        centers_gmm_priv = private_gmm.means_ * X_range + X_min

        n_unique = len(np.unique(labels_gmm_priv))
        results['Private GMM'] = {
            'SSE': compute_sse(X, labels_gmm_priv, centers_gmm_priv) if n_unique > 1 else float('inf'),
            'Silhouette': silhouette_score(X, labels_gmm_priv) if n_unique > 1 else 0.0,
            'ARI': adjusted_rand_score(y_true, labels_gmm_priv),
            'NMI': normalized_mutual_info_score(y_true, labels_gmm_priv),
            'n_clusters': n_unique
        }
    except Exception as e:
        print(f"    Private GMM failed: {e}")
        results['Private GMM'] = {
            'SSE': float('inf'),
            'Silhouette': 0.0,
            'ARI': 0.0,
            'NMI': 0.0,
            'n_clusters': 0
        }

    # Private TMM
    try:
        private_tmm = PrivateTMM(n_components=n_clusters, epsilon=epsilon, L=L,
                                 nu=15.0, alpha=0.01, max_iter=50,
                                 random_state=42, fixed_nu=True)
        private_tmm.fit(X_normalized)
        labels_tmm_priv = private_tmm.labels_
        centers_tmm_priv = private_tmm.means_ * X_range + X_min

        n_unique = len(np.unique(labels_tmm_priv))
        results['Private TMM'] = {
            'SSE': compute_sse(X, labels_tmm_priv, centers_tmm_priv) if n_unique > 1 else float('inf'),
            'Silhouette': silhouette_score(X, labels_tmm_priv) if n_unique > 1 else 0.0,
            'ARI': adjusted_rand_score(y_true, labels_tmm_priv),
            'NMI': normalized_mutual_info_score(y_true, labels_tmm_priv),
            'n_clusters': n_unique
        }
    except Exception as e:
        print(f"    Private TMM failed: {e}")
        results['Private TMM'] = {
            'SSE': float('inf'),
            'Silhouette': 0.0,
            'ARI': 0.0,
            'NMI': 0.0,
            'n_clusters': 0
        }

    return results


def main():
    print("=" * 80)
    print("K-means vs GMM vs TMM: 多数据集全面对比")
    print("=" * 80)

    # Prepare datasets
    datasets = prepare_datasets()

    # Test configuration
    epsilon = 4.0
    L = 0.5

    # Store all results
    all_results = []

    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"数据集: {dataset_name}")
        print(f"描述: {dataset_info['description']}")
        print(f"簇数: {dataset_info['n_clusters']}")
        print("=" * 80)

        X = dataset_info['X']
        y_true = dataset_info['y']
        n_clusters = dataset_info['n_clusters']

        # Test standard algorithms
        print("\n[标准算法 - 无隐私保护]")
        std_results = test_standard_algorithms(X, y_true, n_clusters)
        for method, metrics in std_results.items():
            print(f"  {method:15s}: ARI={metrics['ARI']:.4f}, Sil={metrics['Silhouette']:.4f}, "
                  f"Clusters={metrics['n_clusters']}")
            all_results.append({
                'Dataset': dataset_name,
                'Algorithm': method,
                'Privacy': 'No',
                'Epsilon': None,
                'L': None,
                'SSE': metrics['SSE'],
                'Silhouette': metrics['Silhouette'],
                'ARI': metrics['ARI'],
                'NMI': metrics['NMI'],
                'N_Clusters': metrics['n_clusters']
            })

        # Test private algorithms
        print(f"\n[私有算法 - BPM机制 (ε={epsilon}, L={L})]")
        priv_results = test_private_algorithms(X, y_true, n_clusters, epsilon=epsilon, L=L)
        for method, metrics in priv_results.items():
            print(f"  {method:15s}: ARI={metrics['ARI']:.4f}, Sil={metrics['Silhouette']:.4f}, "
                  f"Clusters={metrics['n_clusters']}")
            all_results.append({
                'Dataset': dataset_name,
                'Algorithm': method,
                'Privacy': 'Yes',
                'Epsilon': epsilon,
                'L': L,
                'SSE': metrics['SSE'] if metrics['SSE'] != float('inf') else np.nan,
                'Silhouette': metrics['Silhouette'],
                'ARI': metrics['ARI'],
                'NMI': metrics['NMI'],
                'N_Clusters': metrics['n_clusters']
            })

    # Create DataFrame for analysis
    df = pd.DataFrame(all_results)

    # Save to CSV
    df.to_csv('comprehensive_results.csv', index=False)
    print(f"\n{'=' * 80}")
    print("结果已保存到: comprehensive_results.csv")

    # Generate visualizations
    print("\n生成可视化对比图...")
    generate_visualizations(df)

    # Generate summary report
    print("\n生成汇总报告...")
    generate_summary_report(df)

    print("\n" + "=" * 80)
    print("全面测试完成！")
    print("=" * 80)


def generate_visualizations(df):
    """Generate comprehensive visualization charts."""

    # Separate standard and private algorithms
    df_std = df[df['Privacy'] == 'No'].copy()
    df_priv = df[df['Privacy'] == 'Yes'].copy()

    # Map algorithm names for grouping
    df_std['Method'] = df_std['Algorithm']
    df_priv['Method'] = df_priv['Algorithm'].str.replace('Private ', '')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('K-means vs GMM vs TMM: Comprehensive Comparison Across Datasets',
                 fontsize=16, fontweight='bold')

    metrics = ['ARI', 'NMI', 'Silhouette']
    titles = ['Adjusted Rand Index (Higher is Better)',
              'Normalized Mutual Information (Higher is Better)',
              'Silhouette Coefficient (Higher is Better)']

    # Plot standard algorithms
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[0, idx]
        pivot_std = df_std.pivot(index='Dataset', columns='Method', values=metric)
        pivot_std.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'Standard Algorithms\n{title}', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xlabel('')
        ax.legend(title='Algorithm', loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Plot private algorithms
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[1, idx]
        pivot_priv = df_priv.pivot(index='Dataset', columns='Method', values=metric)
        pivot_priv.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'Private Algorithms (ε=4.0, L=0.5)\n{title}', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xlabel('')
        ax.legend(title='Algorithm', loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("  可视化已保存到: comprehensive_comparison.png")


def generate_summary_report(df):
    """Generate a summary report."""
    print("\n" + "=" * 80)
    print("汇总报告")
    print("=" * 80)

    # Separate standard and private
    df_std = df[df['Privacy'] == 'No']
    df_priv = df[df['Privacy'] == 'Yes']

    # Average performance by algorithm
    print("\n【标准算法平均性能】")
    std_avg = df_std.groupby('Algorithm')[['ARI', 'NMI', 'Silhouette']].mean()
    print(std_avg.to_string())

    print("\n【私有算法平均性能】")
    priv_avg = df_priv.groupby('Algorithm')[['ARI', 'NMI', 'Silhouette']].mean()
    print(priv_avg.to_string())

    # Best algorithm per dataset (standard)
    print("\n【各数据集最佳算法（标准版本）】")
    for dataset in df_std['Dataset'].unique():
        df_dataset = df_std[df_std['Dataset'] == dataset]
        best_ari = df_dataset.loc[df_dataset['ARI'].idxmax()]
        print(f"  {dataset:20s}: {best_ari['Algorithm']:15s} (ARI={best_ari['ARI']:.4f})")

    # Best algorithm per dataset (private)
    print("\n【各数据集最佳算法（私有版本）】")
    for dataset in df_priv['Dataset'].unique():
        df_dataset = df_priv[df_priv['Dataset'] == dataset]
        best_ari = df_dataset.loc[df_dataset['ARI'].idxmax()]
        print(f"  {dataset:20s}: {best_ari['Algorithm']:15s} (ARI={best_ari['ARI']:.4f})")

    # Privacy cost analysis
    print("\n【隐私代价分析】")
    for algo in ['K-means', 'GMM', 'TMM']:
        std_data = df_std[df_std['Algorithm'] == algo][['ARI', 'NMI', 'Silhouette']].mean()
        priv_data = df_priv[df_priv['Algorithm'] == f'Private {algo}'][['ARI', 'NMI', 'Silhouette']].mean()

        if not std_data.empty and not priv_data.empty:
            retention = (priv_data / std_data * 100).fillna(0)
            print(f"\n  {algo}:")
            print(f"    ARI保留: {retention['ARI']:.1f}%")
            print(f"    NMI保留: {retention['NMI']:.1f}%")
            print(f"    轮廓系数保留: {retention['Silhouette']:.1f}%")


if __name__ == "__main__":
    main()
