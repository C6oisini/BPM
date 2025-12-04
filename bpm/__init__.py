"""
BPM (Bounded Perturbation Mechanism) package.

This package implements the BPM mechanism for ε-d_E privacy as described in:
"K-means clustering with local d_χ-privacy for privacy-preserving data analysis"
"""

from .mechanism import BPM, compute_lambda_L, compute_p_L, compute_mu_L
from .sampling import bpm_sampling, sample_f1, sample_f2

__all__ = [
    'BPM',
    'bpm_sampling',
    'sample_f1',
    'sample_f2',
    'compute_lambda_L',
    'compute_p_L',
    'compute_mu_L'
]
