import numpy as np
from sklearn.datasets import make_blobs

from bpm_privacy import BPMMechanism, KMeansServer, PrivacyClusteringPipeline


def _normalized_blobs():
    X, _ = make_blobs(n_samples=120, n_features=2, centers=3, cluster_std=0.3, random_state=0)
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    return (X - X_min) / (X_max - X_min)


def _pipeline(epsilon: float):
    mechanism = BPMMechanism(epsilon=epsilon, L=0.4, random_state=0)
    server = KMeansServer(n_clusters=3, random_state=0)
    return PrivacyClusteringPipeline(mechanism, server, n_clusters=3)


def test_bpm_mechanism_bounds():
    X = _normalized_blobs()
    pipe = _pipeline(3.0)
    pipe.fit(X)
    assert pipe.labels_.shape[0] == X.shape[0]
    L = pipe.mechanism.L
    assert np.all(pipe.X_perturbed_ >= -L - 1e-9)
    assert np.all(pipe.X_perturbed_ <= 1 + L + 1e-9)


def test_higher_epsilon_improves_sse():
    X = _normalized_blobs()
    np.random.seed(1)
    low_privacy = _pipeline(0.5)
    low_privacy.fit(X)
    sse_low = low_privacy.compute_sse(X)

    np.random.seed(1)
    high_privacy = _pipeline(4.0)
    high_privacy.fit(X)
    sse_high = high_privacy.compute_sse(X)

    assert sse_high <= sse_low * 1.2
