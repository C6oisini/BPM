"""
Implementation of the BPGT framework from:
Fan Chen et al. "BPGT: A Novel Privacy-Preserving K-Means Clustering Framework
to Guarantee Local dχ-privacy" (2024).

This module provides:
1. BPGM — the Bounded Perturbation Generation Mechanism that samples a
   truncated exponential noise distance and synthesizes a new record with
   constrained gradient descent (Algorithm 2 in the paper).
2. BPGT — a clustering wrapper that perturbs every record with BPGM on the
   user side and runs TK-means (implemented via the TMM class) on the server
   side (Algorithm 1 in the paper).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .private_tmm import TMM


def _ensure_rng(random_state: Optional[int]) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


@dataclass
class BPGMConfig:
    epsilon: float
    L: float
    lr: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    tol: float = 1e-3
    max_iter: int = 200
    clamp: bool = True


class BPGM:
    r"""
    Bounded Perturbation Generation Mechanism (user-side).

    - Samples a noise distance \hat{d} from the truncated exponential
      distribution defined in Eq. (5) of the paper.
    - Generates a synthetic record \hat{r} by minimizing the loss
      L = (||r - \hat{r}|| - \hat{d})^2 via Adam (Algorithm 2).
    """

    def __init__(self, config: BPGMConfig, random_state: Optional[int] = None) -> None:
        self.config = config
        self.rng = _ensure_rng(random_state)

    # ------------------------------------------------------------------ #
    # Noise-distance sampling
    # ------------------------------------------------------------------ #
    def sample_distance(self) -> float:
        eps = self.config.epsilon
        L = self.config.L
        # Inverse CDF sampling for truncated exponential over [0, L].
        u = self.rng.random()
        normalizer = 1.0 - math.exp(-eps * L)
        inner = 1.0 - normalizer * u
        inner = max(inner, 1e-12)
        return -math.log(inner) / eps

    # ------------------------------------------------------------------ #
    # Synthetic data generation
    # ------------------------------------------------------------------ #
    def synthesize(self, point: np.ndarray) -> np.ndarray:
        target_distance = self.sample_distance()
        x = self.rng.uniform(-self.config.L, 1.0 + self.config.L, size=point.shape)
        m = np.zeros_like(point)
        v = np.zeros_like(point)

        for t in range(1, self.config.max_iter + 1):
            diff = x - point
            dist = np.linalg.norm(diff) + 1e-12
            loss = (dist - target_distance) ** 2
            # Gradient of loss wrt x
            grad = (
                np.zeros_like(point)
                if dist == 0.0
                else 2.0 * (dist - target_distance) * (diff / dist)
            )

            m = self.config.beta1 * m + (1 - self.config.beta1) * grad
            v = self.config.beta2 * v + (1 - self.config.beta2) * (grad ** 2)
            m_hat = m / (1 - self.config.beta1 ** t)
            v_hat = v / (1 - self.config.beta2 ** t)
            step = self.config.lr * m_hat / (np.sqrt(v_hat) + self.config.adam_eps)
            x = x - step

            if self.config.clamp:
                x = np.clip(x, -self.config.L, 1.0 + self.config.L)

            if math.sqrt(loss) <= self.config.tol:
                break

        return x

    def perturb_dataset(self, X: np.ndarray) -> np.ndarray:
        synthetic = np.empty_like(X)
        for idx, point in enumerate(X):
            synthetic[idx] = self.synthesize(point)
        return synthetic


class BPGT:
    """
    Full BPGT framework: user-side BPGM + server-side TK-means (TMM).
    """

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
        tmm_fixed_nu: bool = True,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.bpgm = BPGM(
            BPGMConfig(
                epsilon=epsilon,
                L=L,
                lr=gd_lr,
                tol=gd_tol,
                max_iter=gd_max_iter,
            ),
            random_state=random_state,
        )
        self.tmm_params = dict(
            n_components=n_clusters,
            nu=15.0,
            alpha=0.01,
            max_iter=tmm_max_iter,
            tol=tmm_tol,
            random_state=random_state,
            fixed_nu=tmm_fixed_nu,
            verbose=False,
        )
        self.synthetic_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.model_: Optional[TMM] = None
        self.L = L

    def fit(self, X: np.ndarray) -> "BPGT":
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("BPGT expects data normalized to [0, 1]^d.")

        synthetic = self.bpgm.perturb_dataset(X)
        tmm = TMM(**self.tmm_params)
        tmm.fit(synthetic)

        self.synthetic_ = synthetic
        self.model_ = tmm
        self.cluster_centers_ = tmm.means_
        self.labels_ = tmm.labels_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet.")
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.cluster_centers_[np.newaxis, :], axis=2
        )
        return np.argmin(distances, axis=1)

    def compute_sse(self, X: np.ndarray) -> float:
        if self.cluster_centers_ is None or self.labels_ is None:
            raise ValueError("Model not fitted yet.")
        sse = 0.0
        for idx in range(self.n_clusters):
            cluster_points = X[self.labels_ == idx]
            if cluster_points.size == 0:
                continue
            sse += np.sum(
                np.linalg.norm(cluster_points - self.cluster_centers_[idx], axis=1) ** 2
            )
        return float(sse)
