"""
Unified BPM (Bounded Perturbation Mechanism) toolkit.

This package bundles the core BPM mechanism plus privacy-preserving clustering
algorithms so callers can simply import ``bpm_privacy`` instead of juggling
separate ``bpm`` and ``clustering`` modules.
"""

from .mechanism import (
    BPM,
    compute_ball_volume,
    compute_B_L_integral,
    compute_lambda_L,
    compute_lambda_2r,
    compute_mu_L,
    compute_p_L,
)
from .sampling import bpm_sampling, sample_f1, sample_f2, sample_radius
from .private_kmeans import PrivateKMeans
from .private_gmm import PrivateGMM
from .private_tmm import PrivateTMM, TMM

__all__ = [
    "BPM",
    "PrivateGMM",
    "PrivateKMeans",
    "PrivateTMM",
    "TMM",
    "bpm_sampling",
    "compute_B_L_integral",
    "compute_ball_volume",
    "compute_lambda_L",
    "compute_lambda_2r",
    "compute_mu_L",
    "compute_p_L",
    "sample_f1",
    "sample_f2",
    "sample_radius",
]
