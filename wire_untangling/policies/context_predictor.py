"""Supervised phase / active-stick predictor for learned BC context.

The model is intentionally stateless: it maps the current raw observation to
two categorical predictions used to rebuild the same hard one-hot context that
phase-active MLP-BC was trained with.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextPredictor(nn.Module):
    """Shared MLP trunk with phase and active-stick classification heads."""

    def __init__(
        self,
        obs_dim: int,
        num_phases: int = 8,
        num_sticks: int = 1,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_phases = int(num_phases)
        self.num_sticks = int(num_sticks)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout = float(dropout)

        layers: list[nn.Module] = []
        prev = self.obs_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            prev = h

        self.trunk = nn.Sequential(*layers)
        self.phase_head = nn.Linear(prev, self.num_phases)
        self.active_head = nn.Linear(prev, self.num_sticks)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (phase_logits, active_stick_logits)."""
        features = self.trunk(obs)
        return self.phase_head(features), self.active_head(features)
