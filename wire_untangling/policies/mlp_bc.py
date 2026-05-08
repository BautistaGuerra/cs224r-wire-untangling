"""MLP behavior-cloning policy for the StickReorderEnv.

A plain feedforward network: state → action. Trained with MSE on (obs, action)
pairs from expert demos. Designed as the frozen base policy for the residual
SAC experiments in proposal §7 — all three residual variants (phase-agnostic,
phase-conditional, phase-gated) sit on top of this MLP-BC.

Why MLP-BC and not a generative policy:
  - Single-modal expert (deterministic scripted policy → no demo multimodality
    for diffusion / flow matching to exploit).
  - Deterministic inference: same state always produces same action, no
    sampling variance to add covariate shift on top of the BC drift problem.
  - ~3 minutes to train on MPS, vs ~25 minutes for flow matching, with no
    inference-time integration loop.

The 60-D obs vector mixes joint positions in radians, joint velocities, EEF
positions in metres, quaternions, and goal positions — magnitudes differ
~100x across dims. State z-score normalisation is therefore mandatory; the
training script computes per-dim mean/std over the demo corpus and saves
them in the checkpoint so inference applies the same transform.

Output activation is ``tanh`` so actions lie in [-1, 1] — matches the
robosuite OSC_POSE action space (signed pose deltas + signed gripper
command, where -1 = open, +1 = closed). Without the tanh the network's
unbounded output would let the OSC controller clamp at the boundary
silently and lose precision on saturated commands.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBCPolicy(nn.Module):
    """Plain MLP mapping state → action, with tanh output.

    Args:
        state_dim: dimensionality of the (normalised) state vector.
        action_dim: dimensionality of the action vector (7 for OSC_POSE).
        hidden_dims: tuple of hidden layer widths. Default (256, 256, 256)
            mirrors the SB3 default for SAC actor networks and is plenty
            of capacity for a unimodal expert at N=1.
        dropout: dropout probability applied between hidden layers. 0.0
            disables. Small values (0.05-0.1) help when training data is
            limited; we default off because expert demos are noise-free.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)

        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, state_dim) → action: (B, action_dim) in [-1, 1]."""
        return self.net(state)


def mse_loss(policy: MLPBCPolicy, s_batch: torch.Tensor,
             a_batch: torch.Tensor) -> torch.Tensor:
    """Standard BC objective: MSE between predicted and expert actions.

    Args:
        policy: MLPBCPolicy.
        s_batch: normalised states, shape (B, state_dim).
        a_batch: expert actions, shape (B, action_dim) — already in [-1, 1].

    Returns:
        Scalar MSE loss (mean over batch and action dims).
    """
    pred = policy(s_batch)
    return torch.mean((pred - a_batch) ** 2)
