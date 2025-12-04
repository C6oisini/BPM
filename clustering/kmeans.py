"""
Private K-means clustering with BPM.

This module implements Algorithm 1 from the paper:
K-means clustering with ε-d_E privacy.
"""

import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from bpm import BPM
from bpm.sampling import bpm_sampling


class PrivateKMeans:
    """
    Privacy-preserving K-means clustering using BPM.

    This implements Algorithm 1 from the paper.
    """

    def __init__(self, n_clusters, epsilon, L, max_iter=300, random_state=None):
        """
        Initialize private K-means.

        Args:
            n_clusters: Number of clusters (K)
            epsilon: Privacy budget (ε)
            L: Threshold distance for BPM
            max_iter: Maximum number of iterations for K-means
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.L = L
        self.max_iter = max_iter
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        """
        Fit private K-means clustering.

        Algorithm 1:
        1. [User side] Each user perturbs their data using BPM
        2. [Server side] Perform K-means on perturbed data

        Args:
            X: Data matrix (n_samples, n_features), should be in [0,1]^d

        Returns:
            self
        """
        n_samples, d = X.shape

        # Check that data is in [0,1]^d
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("Data must be normalized to [0,1]^d")

        # Initialize BPM mechanism
        bpm = BPM(epsilon=self.epsilon, L=self.L, dimension=d)

        # Step 1: User side - Perturb each data point
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

        # Step 2: Server side - Perform K-means on perturbed data
        kmeans = SKLearnKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=10
        )
        kmeans.fit(X_perturbed)

        # Store results
        self.cluster_centers_ = kmeans.cluster_centers_
        self.labels_ = kmeans.labels_
        self.inertia_ = kmeans.inertia_
        self.X_perturbed_ = X_perturbed

        return self

    def predict(self, X):
        """
        Predict cluster labels for data X.

        Note: This uses the true (unperturbed) data to assign to nearest centroid.

        Args:
            X: Data matrix (n_samples, n_features)

        Returns:
            Cluster labels
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Assign each point to nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_[np.newaxis, :], axis=2)
        return np.argmin(distances, axis=1)

    def score(self, X):
        """
        Compute the Sum of Squared Errors (SSE) for the original data.

        SSE = Σ_{i=1}^K Σ_{v_j ∈ C_i} ||v_j - c_i||^2

        where c_i is the centroid of cluster C_i.

        Args:
            X: Original (unperturbed) data matrix

        Returns:
            Negative SSE (following scikit-learn convention)
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        labels = self.predict(X)

        sse = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroid = self.cluster_centers_[k]
                sse += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)

        return -sse  # Negative for scikit-learn compatibility

    def compute_sse(self, X):
        """
        Compute SSE directly (positive value).

        Args:
            X: Original (unperturbed) data matrix

        Returns:
            SSE value
        """
        return -self.score(X)
