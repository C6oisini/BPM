import numpy as np
from sklearn.datasets import make_blobs

from bpm_privacy.bpgt import BPGM, BPGMConfig, BPGT


def _normalize(X: np.ndarray) -> np.ndarray:
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    denom = np.clip(X_max - X_min, 1e-12, None)
    return (X - X_min) / denom


def test_bpgm_distance_and_bounds():
    cfg = BPGMConfig(epsilon=2.0, L=0.4, max_iter=50, tol=1e-2)
    bpgm = BPGM(cfg, random_state=0)
    distances = [bpgm.sample_distance() for _ in range(200)]
    assert max(distances) <= cfg.L + 1e-6
    base = np.array([0.5, 0.5])
    synthetic = bpgm.synthesize(base)
    assert np.all(synthetic <= 1 + cfg.L + 1e-6)
    assert np.all(synthetic >= -cfg.L - 1e-6)


def test_bpgt_end_to_end():
    X, _ = make_blobs(n_samples=200, n_features=2, centers=3, random_state=42)
    X = _normalize(X)
    bpgt = BPGT(
        n_clusters=3,
        epsilon=2.0,
        L=0.4,
        random_state=0,
        gd_max_iter=50,
    )
    bpgt.fit(X)
    labels = bpgt.predict(X)
    assert len(np.unique(labels)) >= 2
    sse = bpgt.compute_sse(X)
    assert np.isfinite(sse)
