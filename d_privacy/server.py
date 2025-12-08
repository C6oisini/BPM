"""Server-side clustering algorithms for distance-private data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from scipy.special import digamma
from sklearn.cluster import KMeans as SKKMeans
from sklearn.mixture import GaussianMixture


class TMM:
    """Student's t Mixture Model (robust alternative to Gaussian mixtures)."""

    def __init__(
        self,
        n_components: int = 3,
        nu: float = 15.0,
        alpha: float = 0.01,
        max_iter: int = 500,
        tol: float = 0.1,
        convergence_count: int = 3,
        random_state: Optional[int] = None,
        fixed_nu: bool = False,
        verbose: bool = False,
    ) -> None:
        self.n_components = n_components
        self.nu = nu
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.convergence_count = convergence_count
        self.random_state = random_state
        self.fixed_nu = fixed_nu
        self.verbose = verbose
        self.means_: Optional[np.ndarray] = None
        self.tau_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.converged_: bool = False

    def fit(self, X: np.ndarray) -> "TMM":
        if self.random_state is not None:
            np.random.seed(self.random_state)
        x = X.T
        p, n = x.shape
        g = self.n_components
        mu = np.zeros((p, g))
        low = x.min(axis=1, keepdims=True)
        up = x.max(axis=1, keepdims=True)
        for i in range(g):
            mu[:, i] = np.random.uniform(low.squeeze(), up.squeeze())
        nu = self.nu
        alpha = self.alpha
        tau = np.zeros((g, n))
        rho = np.zeros((g, n))
        D = np.zeros((g, n))
        c = (p + nu) / 2
        Q = 0
        dQ = 0
        count = 0
        for itr in range(1, self.max_iter + 1):
            for i in range(g):
                for j in range(n):
                    diff = x[:, j] - mu[:, i]
                    D[i, j] = np.dot(diff, diff)
            temp = 1.0 / np.power(1 + D / (nu * alpha), c)
            for i in range(g):
                denom = np.sum(temp[:, :], axis=0)
                tau[i, :] = temp[i, :] / denom
                rho[i, :] = (nu + p) / (nu + D[i, :] / alpha)
            Q_old = Q
            Q = 0
            for i in range(g):
                val = tau[i, :] * np.log(temp[i, :])
                val[np.isnan(val)] = 0
                Q += np.sum(val)
            for i in range(g):
                frac_m0 = np.zeros(p)
                frac_m1 = 0
                for j in range(n):
                    frac_m0 += tau[i, j] * x[:, j] / rho[i, j]
                    frac_m1 += tau[i, j] / rho[i, j]
                mu[:, i] = frac_m0 / frac_m1 if frac_m1 != 0 else frac_m0
            mu[np.isnan(mu)] = 0
            frac_a0 = 0
            frac_a1 = 0
            for i in range(g):
                frac_a0 += np.sum(tau[i, :] * rho[i, :] * D[i, :])
                frac_a1 += np.sum(tau[i, :])
            alpha = frac_a0 / frac_a1 / p
            if not self.fixed_nu:
                star = 0
                for i in range(g):
                    numerator = nu + p
                    denom = nu + D[i, :] / alpha
                    star += np.sum(tau[i, :] * (np.log(numerator / denom) - numerator / denom)) / np.sum(tau[i, :])
                star = star / g
                nu_old = nu
                nu = 1.0 / (-1 - star + np.log((nu + p) / 2) - digamma((nu + p) / 2))
                delta = 10
                while delta > 0:
                    nu = nu + 0.01
                    delta = (
                        -digamma(nu)
                        + np.log(nu)
                        + 1
                        + star
                        - np.log((nu_old + p) / 2)
                        + digamma((nu_old + p) / 2)
                    )
                c = (p + nu) / 2
            if self.verbose:
                print(f"Iteration {itr}, Q={Q:.4f}, nu={nu:.4f}, alpha={alpha:.6f}")
            if np.isnan(Q):
                break
            if abs(Q_old - Q) < self.tol or abs(dQ - abs(Q_old - Q)) < self.tol:
                count += 1
            else:
                count = 0
            dQ = abs(Q_old - Q)
            if count > self.convergence_count:
                self.converged_ = True
                break
        labels = np.argmax(tau, axis=0)
        self.means_ = mu.T
        self.tau_ = tau
        self.labels_ = labels
        self.n_iter_ = itr
        self.nu = nu
        self.alpha = alpha
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = X.T
        p, n = x.shape
        g = self.n_components
        mu = self.means_.T
        D = np.zeros((g, n))
        for i in range(g):
            for j in range(n):
                diff = x[:, j] - mu[:, i]
                D[i, j] = np.dot(diff, diff)
        c = (p + self.nu) / 2
        temp = 1.0 / np.power(1 + D / (self.nu * self.alpha), c)
        tau = np.zeros((g, n))
        for i in range(g):
            denom = np.sum(temp[:, :], axis=0)
            tau[i, :] = temp[i, :] / denom
        labels = np.argmax(tau, axis=0)
        return labels


class ServerAlgorithm(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...


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
        return model.means_, model.labels_
