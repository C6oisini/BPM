"""
Composable privacy-preserving clustering pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .mechanisms import ClientMechanism
from .server_algorithms import ServerAlgorithm


@dataclass
class PipelineResult:
    centers: np.ndarray
    labels: np.ndarray
    perturbed: np.ndarray


class PrivacyClusteringPipeline:
    """Glue code between a client mechanism and a server clustering algorithm."""

    def __init__(
        self,
        mechanism: ClientMechanism,
        server: ServerAlgorithm,
        *,
        n_clusters: int,
    ) -> None:
        self.mechanism = mechanism
        self.server = server
        self.n_clusters = n_clusters
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.X_perturbed_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PrivacyClusteringPipeline":
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("Input data must be normalized to [0,1]^d.")

        perturbed = self.mechanism.perturb(X)
        centers, _ = self.server.fit(perturbed)
        labels = self.predict_with_centers(X, centers)

        self.X_perturbed_ = perturbed
        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict_with_centers(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, np.newaxis] - centers[np.newaxis, :], axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("Pipeline not fitted.")
        return self.predict_with_centers(X, self.cluster_centers_)

    def compute_sse(self, X: np.ndarray) -> float:
        if self.cluster_centers_ is None or self.labels_ is None:
            raise ValueError("Pipeline not fitted.")
        sse = 0.0
        for idx in range(self.n_clusters):
            points = X[self.labels_ == idx]
            if points.size == 0:
                continue
            sse += np.sum(np.linalg.norm(points - self.cluster_centers_[idx], axis=1) ** 2)
        return float(sse)
