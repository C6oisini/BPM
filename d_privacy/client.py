"""Client-side mechanisms for distance-private clustering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import integrate
from scipy.special import factorial


# ---------------------------------------------------------------------------
# BPM core mathematics
# ---------------------------------------------------------------------------


def compute_ball_volume(d: int, R: float) -> float:
    if d % 2 == 0:
        return (np.pi ** (d / 2) * R ** d) / factorial(d / 2)
    product_term = 1.0
    for ell in range(1, (d - 1) // 2):
        product_term *= (2 * ell) / (2 * ell + 1)
    return (4 * np.pi ** ((d - 1) / 2) * R ** d * product_term) / (d * factorial((d - 3) / 2))


def compute_S3_integral(j: int) -> float:
    if j == 0:
        return np.pi
    if j == 1:
        return 2.0
    result, _ = integrate.quad(lambda x: np.sin(x) ** j, 0, np.pi)
    return result


def compute_B_L_integral(d: int, L: float, k: float) -> float:
    exp_series_sum = 0.0
    for i in range(d):
        exp_series_sum += (k * L) ** i / factorial(i)
    product_S3 = 1.0
    for j in range(1, d - 1):
        product_S3 *= compute_S3_integral(d - 1 - j)
    return 2 * np.pi * (factorial(d - 1) / (k ** d)) * (np.exp(k * L) - exp_series_sum) * product_S3


def compute_mu_L(d: int, L: float, k: float) -> float:
    B_L = compute_B_L_integral(d, L, k)
    V_L = compute_ball_volume(d, L)
    return B_L + np.exp(-k * L) * ((1 + 2 * L) ** d - V_L)


def compute_lambda_L(d: int, L: float, k: float) -> float:
    return 1.0 / compute_mu_L(d, L, k)


def compute_p_L(d: int, L: float, k: float) -> float:
    B_L = compute_B_L_integral(d, L, k)
    V_L = compute_ball_volume(d, L)
    return B_L / (B_L + np.exp(-k * L) * ((1 + 2 * L) ** d - V_L))


def compute_lambda_2r(d: int, L: float, k: float) -> float:
    exp_series_sum = 0.0
    for i in range(d):
        exp_series_sum += (k * L) ** i / factorial(i)
    return (k ** d * np.exp(k * L)) / (factorial(d - 1) * (np.exp(k * L) - exp_series_sum))


@dataclass
class BPM:
    epsilon: float
    L: float
    dimension: int

    def __post_init__(self) -> None:
        self.k = self.epsilon
        self.lambda_L = compute_lambda_L(self.dimension, self.L, self.k)
        self.p_L = compute_p_L(self.dimension, self.L, self.k)
        self.lambda_2r = compute_lambda_2r(self.dimension, self.L, self.k)

    def density(self, v: np.ndarray, x: np.ndarray) -> float:
        distance = np.linalg.norm(x - v)
        return self.lambda_L * np.exp(-self.k * min(distance, self.L))


# ---------------------------------------------------------------------------
# Sampling helpers used by BPM
# ---------------------------------------------------------------------------


def sample_f1(v: np.ndarray, L: float) -> np.ndarray:
    d = len(v)
    x = v.copy()
    while np.linalg.norm(x - v) <= L:
        x = np.random.uniform(-L, 1 + L, size=d)
    return x


def sample_radius(d: int, L: float, k: float, lambda_2r: float) -> float:
    r_max = (d - 1) / k
    if r_max > L:
        r_max = L
    if r_max <= 0:
        r_max = L
    max_density = lambda_2r * (r_max ** max(d - 1, 0)) * np.exp(-k * r_max)
    while True:
        r = np.random.uniform(0, L)
        density = lambda_2r * (r ** (d - 1)) * np.exp(-k * r)
        if np.random.uniform(0, max_density) <= density:
            return r


def sample_f2(v: np.ndarray, L: float, k: float, lambda_2r: float) -> np.ndarray:
    d = len(v)
    r = sample_radius(d, L, k, lambda_2r)
    direction = np.random.randn(d)
    direction = direction / np.linalg.norm(direction)
    return v + r * direction


def bpm_sampling(v: np.ndarray, k: float, L: float, p_L: float, lambda_2r: float) -> np.ndarray:
    if np.random.uniform(0, 1) > p_L:
        return sample_f1(v, L)
    return sample_f2(v, L, k, lambda_2r)


# ---------------------------------------------------------------------------
# BPGM core
# ---------------------------------------------------------------------------


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
    def __init__(self, config: BPGMConfig, random_state: Optional[int] = None) -> None:
        self.config = config
        self.rng = _ensure_rng(random_state)

    def sample_distance(self) -> float:
        eps = self.config.epsilon
        L = self.config.L
        u = self.rng.random()
        normalizer = 1.0 - math.exp(-eps * L)
        inner = 1.0 - normalizer * u
        inner = max(inner, 1e-12)
        return -math.log(inner) / eps

    def synthesize(self, point: np.ndarray) -> np.ndarray:
        target_distance = self.sample_distance()
        x = self.rng.uniform(-self.config.L, 1.0 + self.config.L, size=point.shape)
        m = np.zeros_like(point)
        v = np.zeros_like(point)
        for t in range(1, self.config.max_iter + 1):
            diff = x - point
            dist = np.linalg.norm(diff) + 1e-12
            loss = (dist - target_distance) ** 2
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


# ---------------------------------------------------------------------------
# Distance-based synthetic helpers (BLM/CIM)
# ---------------------------------------------------------------------------


def truncated_exponential_sample(
    rng: np.random.Generator, rate: float, max_value: float
) -> float:
    if max_value <= 0:
        return 0.0
    if rate <= 0:
        return float(rng.uniform(0.0, max_value))
    exp_term = np.exp(-rate * max_value)
    u = rng.uniform(0.0, 1.0)
    return float(-np.log(1.0 - u * (1.0 - exp_term)) / rate)


def estimate_domain_bounds(L: float) -> Tuple[float, float]:
    return (-L, 1.0 + L)


def compute_max_pairwise_distance(X: np.ndarray) -> float:
    if len(X) <= 1:
        return 0.0
    diff = X[:, None, :] - X[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return float(np.max(dist))


@dataclass
class SyntheticGeneratorConfig:
    lr: float = 0.1
    tol: float = 1e-4
    max_iter: int = 200
    clip_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None


def synthesise_record(
    rng: np.random.Generator,
    real_point: np.ndarray,
    target_distance: float,
    config: SyntheticGeneratorConfig,
) -> np.ndarray:
    if target_distance <= 0:
        return real_point.copy()
    direction = rng.normal(size=real_point.shape)
    norm = np.linalg.norm(direction)
    if norm == 0:
        direction = np.ones_like(real_point)
        norm = np.linalg.norm(direction)
    direction /= norm
    synthetic = real_point + target_distance * direction
    if config.clip_bounds is not None:
        low, high = config.clip_bounds
        synthetic = np.clip(synthetic, low, high)
    for _ in range(max(config.max_iter, 1)):
        diff_vector = synthetic - real_point
        current_distance = np.linalg.norm(diff_vector)
        error = current_distance - target_distance
        if abs(error) <= config.tol:
            break
        if current_distance == 0:
            direction = rng.normal(size=real_point.shape)
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction /= norm
            synthetic = real_point + target_distance * direction
        else:
            grad = 2.0 * error * diff_vector / current_distance
            synthetic = synthetic - config.lr * grad
            if config.clip_bounds is not None:
                low, high = config.clip_bounds
                synthetic = np.clip(synthetic, low, high)
    return synthetic


# ---------------------------------------------------------------------------
# Mechanism interfaces
# ---------------------------------------------------------------------------


class ClientMechanism:
    def perturb(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BPMMechanism(ClientMechanism):
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
        bpm = BPM(epsilon=self.epsilon, L=self.L, dimension=d)
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
        mechanism = BPGM(self.config, random_state=self.random_state)
        return mechanism.perturb_dataset(X)


class _DistancePerturbationMechanism(ClientMechanism):
    def __init__(
        self,
        epsilon: float,
        random_state: Optional[int] = None,
        *,
        synth_lr: float = 0.1,
        synth_tol: float = 1e-4,
        synth_max_iter: int = 200,
    ) -> None:
        self.epsilon = epsilon
        self.random_state = random_state
        self.synthetic_config = SyntheticGeneratorConfig(
            lr=synth_lr,
            tol=synth_tol,
            max_iter=synth_max_iter,
            clip_bounds=None,
        )

    def _clip_bounds(self, X: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return None

    def _support(self, X: np.ndarray) -> float:
        raise NotImplementedError

    def perturb(self, X: np.ndarray) -> np.ndarray:
        if np.any(X < 0) or np.any(X > 1):
            raise ValueError("Distance mechanisms require input normalized to [0,1]^d.")
        clip = self._clip_bounds(X)
        self.synthetic_config.clip_bounds = clip
        rng = np.random.default_rng(self.random_state)
        support = self._support(X)
        rate = self.epsilon
        perturbed = np.zeros_like(X)
        for idx, point in enumerate(X):
            noisy_distance = truncated_exponential_sample(rng, rate, support)
            perturbed[idx] = synthesise_record(
                rng,
                point,
                noisy_distance,
                self.synthetic_config,
            )
        return perturbed


class BLMMechanism(_DistancePerturbationMechanism):
    def __init__(
        self,
        epsilon: float,
        random_state: Optional[int] = None,
        *,
        synth_lr: float = 0.1,
        synth_tol: float = 1e-4,
        synth_max_iter: int = 200,
    ) -> None:
        super().__init__(
            epsilon=epsilon,
            random_state=random_state,
            synth_lr=synth_lr,
            synth_tol=synth_tol,
            synth_max_iter=synth_max_iter,
        )

    def _clip_bounds(self, X: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return (X.min(axis=0), X.max(axis=0))

    def _support(self, X: np.ndarray) -> float:
        return max(compute_max_pairwise_distance(X), 1e-9)


class CIMMechanism(_DistancePerturbationMechanism):
    def __init__(
        self,
        epsilon: float,
        random_state: Optional[int] = None,
        *,
        synth_lr: float = 0.1,
        synth_tol: float = 1e-4,
        synth_max_iter: int = 200,
        max_distance: float = 1.0,
    ) -> None:
        self.max_distance = max_distance
        super().__init__(
            epsilon=epsilon,
            random_state=random_state,
            synth_lr=synth_lr,
            synth_tol=synth_tol,
            synth_max_iter=synth_max_iter,
        )

    def _clip_bounds(self, X: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return (X.min(axis=0), X.max(axis=0))

    def _support(self, X: np.ndarray) -> float:
        return self.max_distance
