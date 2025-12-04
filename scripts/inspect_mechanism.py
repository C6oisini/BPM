#!/usr/bin/env python3
"""
Small utility to inspect BPM parameters and sampling behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from bpm_privacy import BPM, bpm_sampling

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib is an optional dependency here
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect BPM constants and sampling behavior.")
    parser.add_argument("--epsilon", type=float, default=2.0, help="Privacy budget ε.")
    parser.add_argument("--L", type=float, default=0.3, help="Bound L defining the report space.")
    parser.add_argument("--dimension", type=int, default=2, help="Data dimensionality.")
    parser.add_argument("--samples", type=int, default=0, help="Number of samples to draw for diagnostics.")
    parser.add_argument("--plot", type=Path, help="Optional path to save a 2D scatter of sampled points.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before sampling to keep runs reproducible.",
    )
    return parser.parse_args()


def describe_bpm(bpm: BPM) -> None:
    print("=" * 72)
    print("BPM Configuration")
    print("=" * 72)
    print(f"ε (privacy budget): {bpm.epsilon}")
    print(f"L (bound):          {bpm.L}")
    print(f"Dimension (d):      {bpm.d}")
    print(f"λ_L:                {bpm.lambda_L:.8f}")
    print(f"p_L:                {bpm.p_L:.8f}")
    print(f"λ_2,r:              {bpm.lambda_2r:.8f}")


def sample_points(bpm: BPM, n_samples: int, seed: int) -> np.ndarray:
    rng_state = np.random.get_state()
    np.random.seed(seed)
    samples = np.zeros((n_samples, bpm.d))
    anchor = np.full(bpm.d, 0.5)
    for idx in range(n_samples):
        samples[idx] = bpm_sampling(anchor, bpm.k, bpm.L, bpm.p_L, bpm.lambda_2r)
    np.random.set_state(rng_state)
    return samples


def summarize_samples(samples: np.ndarray, bpm: BPM) -> None:
    center = np.full(bpm.d, 0.5)
    distances = np.linalg.norm(samples - center, axis=1)
    inside = distances <= bpm.L
    print("\nSampling diagnostics")
    print("-" * 72)
    print(f"Samples generated:           {len(samples)}")
    print(f"Mean distance to anchor:     {distances.mean():.4f}")
    print(f"Median distance to anchor:   {np.median(distances):.4f}")
    print(f"Fraction inside ball:        {inside.mean():.4f} (expected {bpm.p_L:.4f})")
    in_bounds = np.all((samples >= -bpm.L) & (samples <= 1 + bpm.L), axis=1).mean()
    print(f"Fraction within report cube: {in_bounds:.4f}")


def maybe_plot(samples: np.ndarray, output: Optional[Path], bpm: BPM) -> None:
    if output is None or plt is None:
        return
    if samples.shape[1] != 2:
        print("Skipping plot: visualization only supported for 2D data.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    center = np.full(2, 0.5)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(samples[:, 0], samples[:, 1], s=15, alpha=0.35, label="Samples")
    ax.scatter(center[0], center[1], s=120, c="crimson", marker="*", label="True point")
    circle = plt.Circle(center, bpm.L, fill=False, color="steelblue", linestyle="--", label=f"r = {bpm.L}")
    ax.add_patch(circle)
    ax.set_xlim(-bpm.L - 0.1, 1 + bpm.L + 0.1)
    ax.set_ylim(-bpm.L - 0.1, 1 + bpm.L + 0.1)
    ax.set_title(f"BPM Samples (ε={bpm.epsilon}, L={bpm.L})")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved scatter plot to {output}")


def main() -> None:
    args = parse_args()
    bpm = BPM(epsilon=args.epsilon, L=args.L, dimension=args.dimension)
    describe_bpm(bpm)

    if args.samples > 0:
        samples = sample_points(bpm, args.samples, args.seed)
        summarize_samples(samples, bpm)
        maybe_plot(samples, args.plot, bpm)


if __name__ == "__main__":
    main()
