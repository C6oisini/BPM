import numpy as np
from sklearn.datasets import make_blobs

from bpm_privacy import PrivateKMeans


def _normalized_blobs():
    X, _ = make_blobs(n_samples=120, n_features=2, centers=3, cluster_std=0.3, random_state=0)
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    return (X - X_min) / (X_max - X_min)


def test_private_kmeans_outputs_valid_labels():
    X = _normalized_blobs()
    np.random.seed(0)
    model = PrivateKMeans(n_clusters=3, epsilon=3.0, L=0.4, random_state=0)
    model.fit(X)
    labels = model.predict(X)
    assert labels.shape[0] == X.shape[0]
    assert np.all(model.X_perturbed_ >= -model.L - 1e-9)
    assert np.all(model.X_perturbed_ <= 1 + model.L + 1e-9)


def test_higher_epsilon_improves_sse():
    X = _normalized_blobs()
    np.random.seed(1)
    low_privacy = PrivateKMeans(n_clusters=3, epsilon=0.5, L=0.4, random_state=0)
    low_privacy.fit(X)
    sse_low = low_privacy.compute_sse(X)

    np.random.seed(1)
    high_privacy = PrivateKMeans(n_clusters=3, epsilon=4.0, L=0.4, random_state=0)
    high_privacy.fit(X)
    sse_high = high_privacy.compute_sse(X)

    assert sse_high <= sse_low * 1.2
