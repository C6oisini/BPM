"""
Debug BPM implementation by examining the perturbed data.
"""

import numpy as np
import matplotlib.pyplot as plt
from bpm import BPM
from bpm.sampling import bpm_sampling


def test_sampling_distribution():
    """Test that sampling follows the correct distribution."""
    print("=" * 80)
    print("Testing BPM Sampling Distribution")
    print("=" * 80)

    # Configuration
    epsilon = 2.0
    L = 0.3
    d = 2

    # True data point at center of [0,1]^2
    v = np.array([0.5, 0.5])

    print(f"\nConfiguration:")
    print(f"  ε = {epsilon}")
    print(f"  L = {L}")
    print(f"  d = {d}")
    print(f"  v = {v}")

    # Initialize BPM
    bpm = BPM(epsilon=epsilon, L=L, dimension=d)

    print(f"\nBPM parameters:")
    print(f"  λ_L = {bpm.lambda_L:.6f}")
    print(f"  p_L = {bpm.p_L:.6f}")
    print(f"  k = {bpm.k:.6f}")

    # Generate samples
    n_samples = 1000
    samples = np.zeros((n_samples, d))

    print(f"\nGenerating {n_samples} samples...")
    for i in range(n_samples):
        samples[i] = bpm_sampling(v, bpm.k, bpm.L, bpm.p_L, bpm.lambda_2r)

    # Analyze samples
    distances = np.linalg.norm(samples - v, axis=1)

    print(f"\nSample statistics:")
    print(f"  Mean distance: {distances.mean():.4f}")
    print(f"  Median distance: {np.median(distances):.4f}")
    print(f"  Max distance: {distances.max():.4f}")
    print(f"  Min distance: {distances.min():.4f}")

    # Count samples inside/outside ball
    inside_ball = np.sum(distances <= L)
    outside_ball = np.sum(distances > L)

    print(f"\n  Samples inside ball (≤ L={L}): {inside_ball} ({100*inside_ball/n_samples:.1f}%)")
    print(f"  Samples outside ball (> L): {outside_ball} ({100*outside_ball/n_samples:.1f}%)")
    print(f"  Expected p_L = {bpm.p_L:.3f} ({100*bpm.p_L:.1f}%)")

    # Check if samples are in report space [-L, 1+L]^d
    in_report_space = np.all((samples >= -L) & (samples <= 1 + L), axis=1)
    print(f"\n  Samples in report space [-L, 1+L]^d: {np.sum(in_report_space)} ({100*np.sum(in_report_space)/n_samples:.1f}%)")

    # Check samples in data domain [0,1]^d
    in_data_domain = np.all((samples >= 0) & (samples <= 1), axis=1)
    print(f"  Samples in data domain [0,1]^d: {np.sum(in_data_domain)} ({100*np.sum(in_data_domain)/n_samples:.1f}%)")

    # Expected squared distance
    squared_distances = distances ** 2
    print(f"\nExpected squared distance E[||X-v||^2]:")
    print(f"  Empirical: {squared_distances.mean():.4f}")

    # Visualize (only for 2D)
    if d == 2:
        plt.figure(figsize=(10, 8))

        # Plot samples
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, label='Samples')

        # Plot true point
        plt.scatter([v[0]], [v[1]], c='red', s=100, marker='*', label=f'True point v={v}', zorder=5)

        # Plot ball boundary
        circle = plt.Circle(v, L, fill=False, color='blue', linestyle='--', label=f'Ball boundary (r={L})')
        plt.gca().add_patch(circle)

        # Plot data domain
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.axhline(y=1, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=1, color='k', linestyle='-', linewidth=0.5)

        # Plot report space boundary
        plt.axhline(y=-L, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.axhline(y=1+L, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.axvline(x=-L, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.axvline(x=1+L, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'BPM Sampling Distribution (ε={epsilon}, L={L})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        # Set limits to show report space
        margin = 0.1
        plt.xlim(-L - margin, 1 + L + margin)
        plt.ylim(-L - margin, 1 + L + margin)

        plt.tight_layout()
        plt.savefig('bpm_sampling_distribution.png', dpi=150)
        print(f"\nVisualization saved to: bpm_sampling_distribution.png")

    return samples, distances


if __name__ == "__main__":
    samples, distances = test_sampling_distribution()
