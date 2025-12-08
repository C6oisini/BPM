#!/usr/bin/env python3
"""Inspect BPGM sampling behavior and optionally plot the synthetic region."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from d_privacy.client import BPGM, BPGMConfig

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect BPGM sampling behavior.")
    parser.add_argument("--epsilon", type=float, default=2.0, help="Privacy budget ε.")
    parser.add_argument("--L", type=float, default=0.3, help="Bound L defining the report space.")
    parser.add_argument("--dimension", type=int, default=2, help="Data dimensionality.")
    parser.add_argument("--samples", type=int, default=0, help="Number of synthetic points to draw.")
    parser.add_argument("--plot", type=Path, help="Path to save the scatter plot (2D only).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed controlling reproducibility.")
    parser.add_argument("--lr", type=float, default=0.05, help="BPGM Adam learning rate.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Stopping tolerance for Adam.")
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum Adam iterations.")
    return parser.parse_args()


def describe(config: BPGMConfig, dimension: int) -> None:
    print("=" * 72)
    print("BPGM Configuration")
    print("=" * 72)
    print(f"ε (privacy budget): {config.epsilon}")
    print(f"L (bound):          {config.L}")
    print(f"Dimension (d):      {dimension}")
    print(f"Adam lr:            {config.lr}")
    print(f"Tolerance:          {config.tol}")
    print(f"Max iterations:     {config.max_iter}")


def sample_points(config: BPGMConfig, dimension: int, n_samples: int, seed: int) -> np.ndarray:
    rng_state = np.random.get_state()
    np.random.seed(seed)
    sampler = BPGM(config, random_state=seed)
    anchor = np.full(dimension, 0.5)
    samples = np.zeros((n_samples, dimension))
    for idx in range(n_samples):
        samples[idx] = sampler.synthesize(anchor)
    np.random.set_state(rng_state)
    return samples


def summarize(samples: np.ndarray, L: float) -> None:
    center = np.full(samples.shape[1], 0.5)
    distances = np.linalg.norm(samples - center, axis=1)
    inside = distances <= L
    print("\nSampling diagnostics")
    print("-" * 72)
    print(f"Samples generated:           {len(samples)}")
    print(f"Mean distance to anchor:     {distances.mean():.4f}")
    print(f"Median distance to anchor:   {np.median(distances):.4f}")
    print(f"Fraction inside ball (r=L):  {inside.mean():.4f}")
    cube_fraction = np.all((samples >= -L) & (samples <= 1 + L), axis=1).mean()
    print(f"Fraction within report cube: {cube_fraction:.4f}")


def maybe_plot(samples: np.ndarray, L: float, output: Optional[Path]) -> None:
    if output is None:
        return
    if plt is None:
        print("matplotlib not available; skipping plot.")
        return
    if samples.shape[1] != 2:
        print("Skipping plot: visualization only supported for 2D data.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(samples[:, 0], samples[:, 1], s=15, alpha=0.35, label="BPGM samples")
    center = np.full(2, 0.5)
    ax.scatter(center[0], center[1], s=120, c="crimson", marker="*", label="Anchor")
    circle = plt.Circle(center, L, color="steelblue", fill=False, linestyle="--", label=f"r = {L}")
    ax.add_patch(circle)
    square = np.array(
        [
            [-L, -L],
            [1 + L, -L],
            [1 + L, 1 + L],
            [-L, 1 + L],
            [-L, -L],
        ]
    )
    ax.plot(square[:, 0], square[:, 1], "k-", linewidth=0.5, label="Report cube")
    ax.set_xlim(-L - 0.1, 1 + L + 0.1)
    ax.set_ylim(-L - 0.1, 1 + L + 0.1)
    ax.set_title("BPGM Samples")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved scatter plot to {output}")


def main() -> None:
    args = parse_args()
    config = BPGMConfig(
        epsilon=args.epsilon,
        L=args.L,
        lr=args.lr,
        tol=args.tol,
        max_iter=args.max_iter,
    )
    describe(config, args.dimension)
    if args.samples > 0:
        samples = sample_points(config, args.dimension, args.samples, args.seed)
        summarize(samples, args.L)
        maybe_plot(samples, args.L, args.plot)
    else:
        print("No samples requested (use --samples N). Only printed configuration.")


if __name__ == "__main__":
    main()
