"""Composable client/server clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .client import ClientMechanism, BPGMMechanism
from .server import ServerAlgorithm, TMMServer


@dataclass
class PipelineResult:
    centers: np.ndarray
    labels: np.ndarray
    perturbed: np.ndarray


class PrivacyClusteringPipeline:
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
        centers, labels = self.server.fit(perturbed)
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


class BPGT(PrivacyClusteringPipeline):
    def __init__(
        self,
        n_clusters: int,
        epsilon: float,
        L: float,
        random_state: Optional[int] = None,
        *,
        gd_lr: float = 0.05,
        gd_tol: float = 1e-3,
        gd_max_iter: int = 200,
        tmm_max_iter: int = 100,
        tmm_tol: float = 1e-3,
    ) -> None:
        mechanism = BPGMMechanism(
            epsilon=epsilon,
            L=L,
            random_state=random_state,
            lr=gd_lr,
            tol=gd_tol,
            max_iter=gd_max_iter,
        )
        server = TMMServer(
            n_components=n_clusters,
            max_iter=tmm_max_iter,
            tol=tmm_tol,
            random_state=random_state,
        )
        super().__init__(mechanism, server, n_clusters=n_clusters)
