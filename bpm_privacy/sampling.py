"""
Sampling algorithms for BPM mechanism.

This module implements Algorithms 2, 3, and 4 from the paper.
"""

import numpy as np
from scipy.stats import norm


def sample_f1(v, L):
    """
    Algorithm 3: Sampling f_1^{(v,L)}.

    Sample uniformly from [-L, 1+L]^d \ B_{v,L}^(d).
    This is done by repeatedly sampling from [-L, 1+L]^d until ||x - v||_2 > L.

    Args:
        v: True data point (shape: (d,))
        L: Threshold distance

    Returns:
        Sample x from f_1^{(v,L)}
    """
    d = len(v)
    x = v.copy()

    # Keep sampling until ||x - v||_2 > L
    while np.linalg.norm(x - v) <= L:
        # Sample each coordinate uniformly from [-L, 1+L]
        x = np.random.uniform(-L, 1 + L, size=d)

    return x


def sample_radius(d, L, k, lambda_2r):
    """
    Sample radius r from the distribution f_2^{(v,L,k),(s,r)}(r).

    From page 6:
    f_2^{(v,L,k),(s,r)}(r) = λ_{2,r} * r^{d-1} * e^{-kr}

    for r ∈ [0, L].

    This uses inverse sampling method as mentioned in the paper.

    Args:
        d: Dimension
        L: Threshold distance
        k: Privacy parameter (ε)
        lambda_2r: Normalization constant λ_{2,r}

    Returns:
        Sampled radius r
    """
    # We use rejection sampling for simplicity
    # The maximum of r^{d-1} * e^{-kr} occurs at r = (d-1)/k
    r_max = (d - 1) / k
    if r_max > L:
        r_max = L

    # Maximum value of the unnormalized density
    if r_max > 0:
        max_density = lambda_2r * (r_max ** (d - 1)) * np.exp(-k * r_max)
    else:
        max_density = lambda_2r * np.exp(0)  # r=0 case

    # Rejection sampling
    while True:
        r = np.random.uniform(0, L)
        density = lambda_2r * (r ** (d - 1)) * np.exp(-k * r)
        u = np.random.uniform(0, max_density)
        if u <= density:
            return r


def sample_f2(v, L, k, lambda_2r):
    """
    Algorithm 4: Sampling f_2^{(v,L,k)}.

    Sample from the ball B_{v,L}^(d) with density proportional to e^{-k||x-v||_2}.

    Steps:
    1. Sample radius r from f_2^{(v,L,k),(s,r)}(r)
    2. Sample direction uniformly on sphere S_{v,r}^(d) using Muller's method

    Args:
        v: True data point (shape: (d,))
        L: Threshold distance
        k: Privacy parameter (ε)
        lambda_2r: Normalization constant λ_{2,r}

    Returns:
        Sample x from f_2^{(v,L,k)}
    """
    d = len(v)

    # Step 1: Sample radius r
    r = sample_radius(d, L, k, lambda_2r)

    # Step 2: Sample direction uniformly on sphere using Muller's method
    # Generate d independent standard normal random variables
    t = np.random.randn(d)

    # Normalize to get uniform direction on unit sphere
    t = t / np.linalg.norm(t)

    # Scale by radius and translate by v
    x = v + r * t

    return x


def bpm_sampling(v, k, L, p_L, lambda_2r):
    """
    Algorithm 2: BPM Sampling.

    Sample from the BPM distribution with density f_v^(L).

    Two-stage sampling:
    1. With probability p_L: sample from f_2 (inside ball)
    2. With probability 1-p_L: sample from f_1 (outside ball)

    Args:
        v: True data point (shape: (d,))
        k: Privacy parameter (ε)
        L: Threshold distance
        p_L: Probability of sampling inside ball
        lambda_2r: Normalization constant for radius sampling

    Returns:
        Perturbed report x
    """
    # Sample from Bernoulli with probability p_L
    p = np.random.uniform(0, 1)

    if p > p_L:
        # Sample from f_1 (outside ball, uniform)
        x = sample_f1(v, L)
    else:
        # Sample from f_2 (inside ball, exponential decay)
        x = sample_f2(v, L, k, lambda_2r)

    return x
