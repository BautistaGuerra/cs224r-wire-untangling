"""
Seed control for StickReorderEnv.

The placement sampler captures a *reference* to the env's RNG at construction
time, so naively reassigning ``env.rng`` would leave the sampler holding the
old generator. Both attributes are rebound here so the next ``reset()``
produces deterministic placements.

Per-demo seed convention (matches the stashed N=3 plan): ``top_seed`` sets the
generator for the whole collection, and the i-th demo uses

    demo_seed = top_seed * 1_000_003 + i

so seeds are stable across re-runs and don't collide with neighbouring runs.
"""

from __future__ import annotations

import numpy as np


PRIME_OFFSET = 1_000_003


def seed_env(raw_env, seed: int) -> None:
    """Reseed both ``raw_env.rng`` and any placement initializer in lock-step.

    Call this immediately before ``env.reset()`` to make the next reset's
    stick placement (and any other RNG-dependent randomization) deterministic.
    """
    rng = np.random.default_rng(int(seed))
    raw_env.rng = rng
    initializer = getattr(raw_env, "placement_initializer", None)
    if initializer is not None:
        initializer.rng = rng


def demo_seed(top_seed: int, demo_index: int) -> int:
    """Stable per-demo seed: top_seed * PRIME_OFFSET + i."""
    return int(top_seed) * PRIME_OFFSET + int(demo_index)
