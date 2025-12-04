"""
Bounded Perturbation Mechanism (BPM) implementation.
This module implements the BPM mechanism as described in the paper:
"K-means clustering with local d_χ-privacy for privacy-preserving data analysis"
"""

import numpy as np
from scipy import integrate
from scipy.special import factorial
import math


def compute_ball_volume(d, R):
    """
    Compute volume of d-dimensional ball with radius R.

    From Lemma 3 in the paper:
    - If d is even: V_R^(d) = (π^(d/2) * R^d) / (d/2)!
    - If d is odd: V_R^(d) = (4 * π^((d-1)/2) * R^d) / (d * ((d-3)/2)! * ∏_{ℓ=1}^{(d-3)/2} (2ℓ)/(2ℓ+1))

    Args:
        d: Dimension
        R: Radius

    Returns:
        Volume of d-dimensional ball
    """
    if d % 2 == 0:  # d is even
        return (np.pi ** (d / 2) * R ** d) / factorial(d / 2)
    else:  # d is odd
        product_term = 1.0
        for ell in range(1, (d - 1) // 2):
            product_term *= (2 * ell) / (2 * ell + 1)
        return (4 * np.pi ** ((d - 1) / 2) * R ** d * product_term) / (d * factorial((d - 3) / 2))


def compute_S3_integral(j):
    """
    Compute S_{3,j} = ∫_0^π sin^j(x) dx.

    This uses the recursive formula or direct computation.

    Args:
        j: Power of sine

    Returns:
        Value of the integral
    """
    if j == 0:
        return np.pi
    elif j == 1:
        return 2.0

    # Use the reduction formula: S_{3,j} = ((j-1)/j) * S_{3,j-2}
    # But for numerical stability, we use scipy's integrate
    result, _ = integrate.quad(lambda x: np.sin(x) ** j, 0, np.pi)
    return result


def compute_B_L_integral(d, L, k):
    """
    Compute B_L^(d) = ∫_{B_L^(d)} e^{-k||x||_2} dx.

    From Lemma 3, for d-dimensional ball:
    B_L^(d) = (2π) * (d-1)!/k^d * (e^{kL} - Σ_{i=0}^{d-1} (kL)^i/i!) * ∏_{j=1}^{d-2} S_{3,d-1-j}

    But there's also a formula: there exists c_R ∈ (0, R) such that
    B_R^(d) = (volume formula) * e^{k(c_R - R)}

    For simplicity and accuracy, we use the series formula from Lemma 4.

    Args:
        d: Dimension
        L: Threshold distance
        k: Privacy parameter (ε)

    Returns:
        Value of B_L^(d)
    """
    # Compute the exponential series sum
    exp_series_sum = 0.0
    for i in range(d):
        exp_series_sum += (k * L) ** i / factorial(i)

    # Compute the product of S_{3,j} integrals
    product_S3 = 1.0
    for j in range(1, d - 1):
        product_S3 *= compute_S3_integral(d - 1 - j)

    # From Lemma 4's formula for μ_L^(1)
    result = 2 * np.pi * (factorial(d - 1) / (k ** d)) * (np.exp(k * L) - exp_series_sum) * product_S3

    return result


def compute_mu_L(d, L, k):
    """
    Compute μ_L from Lemma 4.

    μ_L = B_L^(d) + e^{-kL} * [(1 + 2L)^d - V_L^(d)]

    where:
    - B_L^(d) is the integral over the ball
    - V_L^(d) is the volume of the ball

    Args:
        d: Dimension
        L: Threshold distance
        k: Privacy parameter (ε)

    Returns:
        Value of μ_L
    """
    B_L = compute_B_L_integral(d, L, k)
    V_L = compute_ball_volume(d, L)

    mu_L = B_L + np.exp(-k * L) * ((1 + 2 * L) ** d - V_L)

    return mu_L


def compute_lambda_L(d, L, k):
    """
    Compute λ_L = μ_L^{-1}.

    From Lemma 4, λ_{v,L} is independent of v, so we denote it as λ_L.

    Args:
        d: Dimension
        L: Threshold distance
        k: Privacy parameter (ε)

    Returns:
        Value of λ_L
    """
    mu_L = compute_mu_L(d, L, k)
    return 1.0 / mu_L


def compute_p_L(d, L, k):
    """
    Compute p_L, the probability of sampling inside the ball B_{v,L}.

    From page 5 of the paper:
    p_L = λ_L * B_L^(d) = B_L^(d) / (B_L^(d) + e^{-kL} * [(1 + 2L)^d - V_L^(d)])

    Args:
        d: Dimension
        L: Threshold distance
        k: Privacy parameter (ε)

    Returns:
        Probability p_L
    """
    B_L = compute_B_L_integral(d, L, k)
    V_L = compute_ball_volume(d, L)

    p_L = B_L / (B_L + np.exp(-k * L) * ((1 + 2 * L) ** d - V_L))

    return p_L


def compute_lambda_2r(d, L, k):
    """
    Compute λ_{2,r} for sampling the radius in Algorithm 4.

    From page 6:
    λ_{2,r} = (k^d * e^{kL}) / ((d-1)! * [e^{kL} - Σ_{i=0}^{d-1} (kL)^i/i!])

    Args:
        d: Dimension
        L: Threshold distance
        k: Privacy parameter (ε)

    Returns:
        Value of λ_{2,r}
    """
    # Compute the exponential series sum
    exp_series_sum = 0.0
    for i in range(d):
        exp_series_sum += (k * L) ** i / factorial(i)

    lambda_2r = (k ** d * np.exp(k * L)) / (factorial(d - 1) * (np.exp(k * L) - exp_series_sum))

    return lambda_2r


class BPM:
    """
    Bounded Perturbation Mechanism (BPM).

    This class implements the BPM mechanism that satisfies ε-d_E privacy.
    """

    def __init__(self, epsilon, L, dimension):
        """
        Initialize BPM mechanism.

        Args:
            epsilon: Privacy budget (ε)
            L: Threshold distance (determines report space [-L, 1+L]^d)
            dimension: Data dimension (d)
        """
        self.epsilon = epsilon
        self.L = L
        self.d = dimension
        self.k = epsilon  # k is set to ε for ε-d_E privacy (Theorem 8)

        # Precompute constants
        self.lambda_L = compute_lambda_L(self.d, self.L, self.k)
        self.p_L = compute_p_L(self.d, self.L, self.k)
        self.lambda_2r = compute_lambda_2r(self.d, self.L, self.k)

    def density(self, v, x):
        """
        Compute the probability density function f_v^(L)(x).

        From Equation (2):
        f_v^(L)(x) = λ_L * e^{-k * min{||x-v||_2, L}}

        Args:
            v: True data point (shape: (d,))
            x: Report point (shape: (d,))

        Returns:
            Density value
        """
        distance = np.linalg.norm(x - v)
        return self.lambda_L * np.exp(-self.k * min(distance, self.L))

    def __repr__(self):
        return f"BPM(ε={self.epsilon}, L={self.L}, d={self.d})"
