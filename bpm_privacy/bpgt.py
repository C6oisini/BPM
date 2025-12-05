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

from .mechanisms import BPGMMechanism
from .server_algorithms import TMMServer
from .pipeline import PrivacyClusteringPipeline


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


class BPGT(PrivacyClusteringPipeline):
    """
    Convenience wrapper: BPGM mechanism + TMM server (i.e., original BPGT).
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
