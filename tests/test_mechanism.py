import numpy as np

from bpm_privacy import BPM, bpm_sampling


def test_density_monotonic_decrease():
    bpm = BPM(epsilon=2.0, L=0.4, dimension=2)
    anchor = np.array([0.5, 0.5])
    near = np.array([0.6, 0.5])
    far = np.array([1.0, 0.5])

    density_at_anchor = bpm.density(anchor, anchor)
    density_near = bpm.density(anchor, near)
    density_far = bpm.density(anchor, far)

    assert density_at_anchor > density_near > density_far


def test_sampling_respects_bounds_and_probability():
    bpm = BPM(epsilon=1.5, L=0.3, dimension=2)
    anchor = np.array([0.5, 0.5])
    np.random.seed(123)
    samples = np.array([bpm_sampling(anchor, bpm.k, bpm.L, bpm.p_L, bpm.lambda_2r) for _ in range(1000)])

    assert np.all(samples >= -bpm.L - 1e-9)
    assert np.all(samples <= 1 + bpm.L + 1e-9)

    distances = np.linalg.norm(samples - anchor, axis=1)
    inside_rate = np.mean(distances <= bpm.L)
    assert abs(inside_rate - bpm.p_L) < 0.1
