"""
Server-side clustering algorithms that consume perturbed data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans as SKKMeans
from sklearn.mixture import GaussianMixture

from .private_tmm import TMM


class ServerAlgorithm(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cluster_centers, labels) for the perturbed data."""


class KMeansServer(ServerAlgorithm):
    def __init__(self, n_clusters: int, max_iter: int = 300, random_state: Optional[int] = None) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        model = SKKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=10,
        )
        labels = model.fit_predict(X)
        return model.cluster_centers_, labels


class GMMServer(ServerAlgorithm):
    def __init__(
        self,
        n_components: int,
        covariance_type: str = "full",
        max_iter: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=10,
        )
        model.fit(X)
        labels = model.predict(X)
        return model.means_, labels


class TMMServer(ServerAlgorithm):
    def __init__(
        self,
        n_components: int,
        nu: float = 15.0,
        alpha: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
        fixed_nu: bool = True,
    ) -> None:
        self.n_components = n_components
        self.nu = nu
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.fixed_nu = fixed_nu

    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        model = TMM(
            n_components=self.n_components,
            nu=self.nu,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            fixed_nu=self.fixed_nu,
        )
        model.fit(X)
        labels = model.labels_
        return model.means_, labels
