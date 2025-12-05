"""
Client-side privacy mechanisms.

This module separates the perturbation logic (BPM vs. BPGM) from the
server-side clustering algorithms so different combinations can be composed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .mechanism import BPM as BPMCore
from .sampling import bpm_sampling
from .bpgt import BPGMConfig, BPGM as BPGMCore


class ClientMechanism(ABC):
    """Base interface for client-side perturbation mechanisms."""

    @abstractmethod
    def perturb(self, X: np.ndarray) -> np.ndarray:
        """Return perturbed/synthetic data with the same shape as ``X``."""


class BPMMechanism(ClientMechanism):
    """Classic BPM perturbation as described in the original paper."""

    def __init__(self, epsilon: float, L: float, random_state: Optional[int] = None) -> None:
        self.epsilon = epsilon
        self.L = L
        self.random_state = random_state

    def perturb(self, X: np.ndarray) -> np.ndarray:
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("BPM requires the data to be normalized to [0,1]^d.")
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, d = X.shape
        bpm = BPMCore(epsilon=self.epsilon, L=self.L, dimension=d)
        perturbed = np.zeros_like(X)
        for i in range(n_samples):
            perturbed[i] = bpm_sampling(
                v=X[i],
                k=bpm.k,
                L=bpm.L,
                p_L=bpm.p_L,
                lambda_2r=bpm.lambda_2r,
            )
        return perturbed


class BPGMMechanism(ClientMechanism):
    """Bounded Perturbation Generation Mechanism (BPGM) using Adam."""

    def __init__(
        self,
        epsilon: float,
        L: float,
        random_state: Optional[int] = None,
        *,
        lr: float = 0.05,
        tol: float = 1e-3,
        max_iter: int = 200,
    ) -> None:
        self.config = BPGMConfig(
            epsilon=epsilon,
            L=L,
            lr=lr,
            tol=tol,
            max_iter=max_iter,
        )
        self.random_state = random_state

    def perturb(self, X: np.ndarray) -> np.ndarray:
        mechanism = BPGMCore(self.config, random_state=self.random_state)
        return mechanism.perturb_dataset(X)
