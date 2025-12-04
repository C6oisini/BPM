"""
基于BPM的私有高斯混合模型（Private GMM with BPM）。

将BPM机制与GMM结合，实现隐私保护的软聚类。
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from bpm import BPM
from bpm.sampling import bpm_sampling


class PrivateGMM:
    """
    使用BPM机制的隐私保护高斯混合模型。

    与K-means的主要区别：
    - 软聚类：每个点有属于各个簇的概率分布
    - 估计协方差矩阵：捕捉簇的形状和方向
    - 混合权重：估计每个簇的先验概率
    """

    def __init__(self, n_components, epsilon, L,
                 covariance_type='full', max_iter=100, random_state=None):
        """
        初始化私有GMM。

        Args:
            n_components: 混合成分数量（簇数）
            epsilon: 隐私预算 (ε)
            L: BPM的阈值距离
            covariance_type: 协方差类型 ('full', 'tied', 'diag', 'spherical')
            max_iter: EM算法最大迭代次数
            random_state: 随机种子
        """
        self.n_components = n_components
        self.epsilon = epsilon
        self.L = L
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state

        # GMM参数（拟合后填充）
        self.means_ = None           # 均值
        self.covariances_ = None     # 协方差矩阵
        self.weights_ = None         # 混合权重
        self.converged_ = None       # 是否收敛
        self.n_iter_ = None          # 实际迭代次数
        self.lower_bound_ = None     # 对数似然下界

        # 扰动后的数据
        self.X_perturbed_ = None

    def fit(self, X):
        """
        拟合私有GMM。

        算法流程：
        1. [用户侧] 每个用户使用BPM扰动自己的数据
        2. [服务器侧] 对扰动后的数据运行GMM的EM算法

        Args:
            X: 数据矩阵 (n_samples, n_features)，应在[0,1]^d范围内

        Returns:
            self
        """
        n_samples, d = X.shape

        # 检查数据范围
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("数据必须归一化到[0,1]^d范围")

        # 初始化BPM机制
        bpm = BPM(epsilon=self.epsilon, L=self.L, dimension=d)

        # 步骤1：用户侧 - 每个数据点使用BPM扰动
        X_perturbed = np.zeros_like(X)
        for i in range(n_samples):
            v = X[i]
            x = bpm_sampling(
                v=v,
                k=bpm.k,
                L=bpm.L,
                p_L=bpm.p_L,
                lambda_2r=bpm.lambda_2r
            )
            X_perturbed[i] = x

        self.X_perturbed_ = X_perturbed

        # 步骤2：服务器侧 - 对扰动数据运行GMM
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=10
        )
        gmm.fit(X_perturbed)

        # 保存GMM参数
        self.means_ = gmm.means_
        self.covariances_ = gmm.covariances_
        self.weights_ = gmm.weights_
        self.converged_ = gmm.converged_
        self.n_iter_ = gmm.n_iter_
        self.lower_bound_ = gmm.lower_bound_

        return self

    def predict(self, X):
        """
        预测簇标签（硬分配）。

        Args:
            X: 数据矩阵 (n_samples, n_features)

        Returns:
            簇标签数组 (n_samples,)
        """
        if self.means_ is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        # 使用学习到的GMM参数进行预测
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type
        )
        gmm.means_ = self.means_
        gmm.covariances_ = self.covariances_
        gmm.weights_ = self.weights_
        gmm.precisions_cholesky_ = self._compute_precisions_cholesky()

        return gmm.predict(X)

    def predict_proba(self, X):
        """
        预测每个点属于各簇的概率（软分配）。

        这是GMM相比K-means的主要优势。

        Args:
            X: 数据矩阵 (n_samples, n_features)

        Returns:
            概率矩阵 (n_samples, n_components)
            每行和为1，表示该点属于各簇的概率
        """
        if self.means_ is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type
        )
        gmm.means_ = self.means_
        gmm.covariances_ = self.covariances_
        gmm.weights_ = self.weights_
        gmm.precisions_cholesky_ = self._compute_precisions_cholesky()

        return gmm.predict_proba(X)

    def score(self, X):
        """
        计算给定数据的对数似然。

        Args:
            X: 数据矩阵 (n_samples, n_features)

        Returns:
            平均对数似然
        """
        if self.means_ is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type
        )
        gmm.means_ = self.means_
        gmm.covariances_ = self.covariances_
        gmm.weights_ = self.weights_
        gmm.precisions_cholesky_ = self._compute_precisions_cholesky()

        return gmm.score(X)

    def sample(self, n_samples=1):
        """
        从学习到的GMM中采样。

        Args:
            n_samples: 要生成的样本数

        Returns:
            samples: 生成的样本 (n_samples, n_features)
            labels: 样本的簇标签 (n_samples,)
        """
        if self.means_ is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type
        )
        gmm.means_ = self.means_
        gmm.covariances_ = self.covariances_
        gmm.weights_ = self.weights_
        gmm.precisions_cholesky_ = self._compute_precisions_cholesky()

        return gmm.sample(n_samples)

    def _compute_precisions_cholesky(self):
        """
        从协方差矩阵计算精度矩阵的Cholesky分解。

        这是sklearn内部需要的。
        """
        from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

        if self.covariance_type == 'full':
            # 对于full协方差，需要为每个成分计算Cholesky分解
            n_components, n_features, _ = self.covariances_.shape
            precisions_chol = np.empty((n_components, n_features, n_features))
            for k in range(n_components):
                cov = self.covariances_[k]
                try:
                    precisions_chol[k] = np.linalg.cholesky(np.linalg.inv(cov))
                except np.linalg.LinAlgError:
                    # 如果矩阵不可逆，添加小的正则化
                    cov_reg = cov + np.eye(n_features) * 1e-6
                    precisions_chol[k] = np.linalg.cholesky(np.linalg.inv(cov_reg))
        elif self.covariance_type == 'tied':
            try:
                precisions_chol = np.linalg.cholesky(np.linalg.inv(self.covariances_))
            except np.linalg.LinAlgError:
                n_features = self.covariances_.shape[0]
                cov_reg = self.covariances_ + np.eye(n_features) * 1e-6
                precisions_chol = np.linalg.cholesky(np.linalg.inv(cov_reg))
        elif self.covariance_type == 'diag':
            precisions_chol = 1.0 / np.sqrt(self.covariances_)
        else:  # spherical
            precisions_chol = 1.0 / np.sqrt(self.covariances_)

        return precisions_chol

    def compute_aic(self, X):
        """
        计算AIC (Akaike Information Criterion)。

        AIC用于模型选择，越小越好。

        Args:
            X: 数据矩阵

        Returns:
            AIC值
        """
        if self.means_ is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type
        )
        gmm.means_ = self.means_
        gmm.covariances_ = self.covariances_
        gmm.weights_ = self.weights_
        gmm.precisions_cholesky_ = self._compute_precisions_cholesky()

        return gmm.aic(X)

    def compute_bic(self, X):
        """
        计算BIC (Bayesian Information Criterion)。

        BIC用于模型选择，越小越好。

        Args:
            X: 数据矩阵

        Returns:
            BIC值
        """
        if self.means_ is None:
            raise ValueError("模型尚未拟合，请先调用fit()")

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type
        )
        gmm.means_ = self.means_
        gmm.covariances_ = self.covariances_
        gmm.weights_ = self.weights_
        gmm.precisions_cholesky_ = self._compute_precisions_cholesky()

        return gmm.bic(X)
