"""
Unified BPM/BPGM toolkit.

Mechanisms (client-side perturbation) and server-side clustering algorithms
can be composed via :class:`PrivacyClusteringPipeline`.
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
from .mechanisms import ClientMechanism, BPMMechanism, BPGMMechanism
from .server_algorithms import ServerAlgorithm, KMeansServer, GMMServer, TMMServer
from .pipeline import PrivacyClusteringPipeline
from .bpgt import BPGM, BPGT
from .private_tmm import TMM

__all__ = [
    "BPM",
    "BPGM",
    "BPGMMechanism",
    "BPMMechanism",
    "BPGT",
    "ClientMechanism",
    "ServerAlgorithm",
    "PrivacyClusteringPipeline",
    "KMeansServer",
    "GMMServer",
    "TMMServer",
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
