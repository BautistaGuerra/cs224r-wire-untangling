"""Per-dimension z-score normalizer shared by training and inference.

Provides a single Normalizer class that replaces the ad-hoc normalize_array /
denormalize_array helpers in train_flow_matching.py and the inline normalization
inside DPFMModelPolicy and MLPBCModelPolicy.  Both numpy and torch code paths
are supported so the same object works in the data-loading pipeline (numpy) and
inside GPU-side policy forward passes (torch).

Typical usage
-------------
Training::

    obs_norm  = Normalizer.from_data(flat_obs)
    act_norm  = Normalizer.from_data(flat_actions, clip_low=action_low, clip_high=action_high)
    normed_obs = obs_norm.normalize(obs)
    torch.save({"obs_norm": obs_norm.state_dict(), ...}, path)

Inference::

    ckpt = torch.load(path)
    obs_norm = Normalizer.from_state_dict(ckpt["obs_norm"])
    normed  = obs_norm.normalize(raw_obs)
    raw_act = act_norm.denormalize(predicted_action)
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from torch import distributions as pyd

logger = logging.getLogger(__name__)

DEFAULT_SCALE_OBSERVATIONS = 1e-6 #1e-3
DEFAULT_SCALE_ACTIONS = 1e-6 #1e-2

class Normalizer:
    """Per-dimension z-score normalizer that works with both numpy and torch.

    Forward:  (x - loc) / scale
    Inverse:  x * scale + loc

    ``scale`` is clamped to ``eps`` so constant-variance dims produce zero
    rather than +/-inf.
    """

    # Floor for scale values.  Any dimension whose standard deviation is below
    # this threshold is treated as constant; its normalized output will be ~0.
    EPS = 1e-6 # 1e-1

    def __init__(
        self,
        loc: np.ndarray | torch.Tensor,
        scale: np.ndarray | torch.Tensor,
        clip_low: np.ndarray | torch.Tensor | None = None,
        clip_high: np.ndarray | torch.Tensor | None = None,
        warn_on_clip: bool = False,
        default_scale: float = EPS
    ):
        """Construct a Normalizer from pre-computed statistics.

        Args:
            loc:       Per-dimension mean (or any shift), shape (D,).
            scale:     Per-dimension std  (or any scale), shape (D,).
                       Values below ``EPS`` are clamped up to ``EPS``.
            clip_low:  Optional per-dimension lower bound applied after
                       denormalization.  Typically the env's action_low.
            clip_high: Optional per-dimension upper bound applied after
                       denormalization.  Typically the env's action_high.
            warn_on_clip: If True, log a warning when denormalized values
                       exceed clip bounds.
            default_scale: the default gard value on the stnadard deviation. If the
                        std. dev of the dimension is < default_scale, std. dev is
                        set to default scale to prevent blow of OOD values.
        """
        self.loc = np.asarray(loc, dtype=np.float32)
        self.scale = np.maximum(np.asarray(scale, dtype=np.float32), default_scale)
        self.clip_low = np.asarray(clip_low, dtype=np.float32) if clip_low is not None else None
        self.clip_high = np.asarray(clip_high, dtype=np.float32) if clip_high is not None else None
        self.warn_on_clip = warn_on_clip

    # ── numpy paths ──────────────────────────────────────────────────────

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalize: (x - loc) / scale.  Works on single rows or batches."""
        return (x - self.loc) / self.scale

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Invert z-score: x * scale + loc, then clip if bounds were provided.

        Logs a warning when any denormalized value falls outside the clip
        bounds, reporting the worst per-dimension overshoot.
        """
        raw = x * self.scale + self.loc
        if self.clip_low is not None or self.clip_high is not None:
            if self.warn_on_clip:
                self._warn_clip_violation(raw)
            raw = np.clip(raw, self.clip_low, self.clip_high)
        return raw

    # ── torch paths ──────────────────────────────────────────────────────

    def normalize_torch(self, x: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Z-score normalize a torch tensor.  Loc/scale are moved to ``device``."""
        loc = torch.as_tensor(self.loc, dtype=torch.float32, device=device or x.device)
        scale = torch.as_tensor(self.scale, dtype=torch.float32, device=device or x.device)
        return (x - loc) / scale

    def denormalize_torch(self, x: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Invert z-score on a torch tensor, with optional clipping.

        Logs a warning when any denormalized value falls outside the clip
        bounds, reporting the worst per-dimension overshoot.
        """
        loc = torch.as_tensor(self.loc, dtype=torch.float32, device=device or x.device)
        scale = torch.as_tensor(self.scale, dtype=torch.float32, device=device or x.device)
        raw = x * scale + loc
        if self.clip_low is not None or self.clip_high is not None:
            if self.warn_on_clip:
                self._warn_clip_violation(raw.detach().cpu().numpy())
            low = torch.as_tensor(self.clip_low, device=x.device) if self.clip_low is not None else None
            high = torch.as_tensor(self.clip_high, device=x.device) if self.clip_high is not None else None
            raw = torch.clamp(raw, min=low, max=high)
        return raw

    # ── clip violation guard ────────────────────────────────────────────

    def _warn_clip_violation(self, raw: np.ndarray) -> None:
        """Log a warning if any values in ``raw`` exceed the clip bounds.

        Reports the worst overshoot below clip_low and above clip_high on a
        per-dimension basis so the caller can tell which action dims are
        drifting and by how much.
        """
        violations = []
        if self.clip_low is not None:
            under = self.clip_low - raw  # positive where raw < clip_low
            max_under = np.max(under, axis=0) if raw.ndim > 1 else under
            bad = max_under > 0.1
            if np.any(bad):
                dims = np.where(bad)[0]
                for d in dims:
                    violations.append(f"dim {d}: {max_under[d]:.4f} below min ({self.clip_low[d]:.4f})")
        if self.clip_high is not None:
            over = raw - self.clip_high  # positive where raw > clip_high
            max_over = np.max(over, axis=0) if raw.ndim > 1 else over
            bad = max_over > 0.1
            if np.any(bad):
                dims = np.where(bad)[0]
                for d in dims:
                    violations.append(f"dim {d}: {max_over[d]:.4f} above max ({self.clip_high[d]:.4f})")
        if violations:
            logger.warning("Denormalized actions outside clip bounds:\n  " + "\n  ".join(violations))

    # ── serialization ────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Export loc, scale, and clip bounds as torch tensors for checkpoint storage."""
        d = {
            "loc": torch.from_numpy(self.loc),
            "scale": torch.from_numpy(self.scale),
        }
        if self.clip_low is not None:
            d["clip_low"] = torch.from_numpy(self.clip_low)
        if self.clip_high is not None:
            d["clip_high"] = torch.from_numpy(self.clip_high)
        return d

    @classmethod
    def from_state_dict(cls, d: dict) -> Normalizer:
        """Reconstruct a Normalizer from a checkpoint state dict.

        Uses a small default_scale (1e-6) to preserve the exact scale values
        that were stored at training time, avoiding re-clamping with the
        current class-level EPS which may differ.
        """
        return cls(
            loc=d["loc"].cpu().numpy(),
            scale=d["scale"].cpu().numpy(),
            clip_low=(d.get("clip_low").cpu().numpy() if "clip_low" in d else None),
            clip_high=(d.get("clip_high").cpu().numpy() if "clip_high" in d else None),
        )

    # ── factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        clip_low: np.ndarray | None = None,
        clip_high: np.ndarray | None = None,
        default_scale: float = EPS
    ) -> Normalizer:
        """Compute per-dimension mean/std from a (N, D) dataset and build a Normalizer."""
        return cls(
            loc=data.mean(axis=0).astype(np.float32),
            scale=data.std(axis=0).astype(np.float32),
            clip_low=clip_low,
            clip_high=clip_high,
            default_scale=default_scale
        )


# ── Tensor batch-dimension helpers ──────────────────────────────────────────
# Adapted from residual-offpolicy-rl (Amazon, CC-BY-NC-4.0)

def maybe_unsqueeze(t: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Add a leading batch dim if tensor is 1-D. Returns (tensor, was_unsqueezed)."""
    if t.dim() == 1:
        return t.unsqueeze(0), True
    return t, False


def maybe_squeeze(t: torch.Tensor, was_unsqueezed: bool) -> torch.Tensor:
    """Remove the leading batch dim if it was added by maybe_unsqueeze."""
    if was_unsqueezed:
        return t.squeeze(0)
    return t


# ── Action norm clipping ────────────────────────────────────────────────────
# Taken from the residual-offpolicy-rl Amazon codebase.
from torch.distributions.utils import _standard_normal
def clip_action_norm(action, max_norm):
    assert max_norm > 0
    assert action.dim() == 2 and action.size(1) == 7

    ee_action = action[:, :6]
    gripper_action = action[:, 6:]

    ee_action_norm = ee_action.norm(dim=1).unsqueeze(1)
    ee_action = ee_action / ee_action_norm
    assert (ee_action.norm(dim=1).min() - 1).abs() <= 1e-5
    scale = ee_action_norm.clamp(max=max_norm)
    ee_action = ee_action * scale
    action = torch.cat([ee_action, gripper_action], dim=1)
    return action  # noqa: RET504


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6, max_action_norm: float = -1):
        if isinstance(scale, float):
            scale = torch.ones_like(loc) * scale

        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.max_action_norm = max_action_norm

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x
    def sample(self, clip=None, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size()
        shape = self._extended_shape(sample_shape)
        # Sample the "additional noise" from the unit normal
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        # Scale it down to the residual action scale
        eps *= self.scale
        if clip is not None:
            # If we request, clip the predicted action into the [-clip; clip] range.
            eps = torch.clamp(eps, -clip, clip)
        # Add residual noise to the predicted mean.
        x = self.loc + eps
        x = self._clamp(x)
        if self.max_action_norm > 0:
            # Not used currently
            x = clip_action_norm(x, self.max_action_norm)
        return x
