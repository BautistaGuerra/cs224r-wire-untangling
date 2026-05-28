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
        default_scale: float = EPS,
        normalize_dims: list[int] | None = None,
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
            normalize_dims: Optional list of dimension indices to normalize.
                        Excluded dims get identity transform (loc=0, scale=1).
                        If None, all dims are normalized.
        """
        self.loc = np.asarray(loc, dtype=np.float32)
        self.scale = np.maximum(np.asarray(scale, dtype=np.float32), default_scale)
        self.clip_low = np.asarray(clip_low, dtype=np.float32) if clip_low is not None else None
        self.clip_high = np.asarray(clip_high, dtype=np.float32) if clip_high is not None else None
        self.warn_on_clip = warn_on_clip
        # Filter out any indices that exceed the data dimension
        if normalize_dims is not None:
            normalize_dims = [d for d in normalize_dims if d < len(self.loc)]
            if not normalize_dims:
                normalize_dims = None
        self.normalize_dims = normalize_dims

        # Zero out loc and set scale=1 for excluded dims so the formula
        # (x - loc) / scale works uniformly without branching.
        if self.normalize_dims is not None:
            mask = np.zeros(len(self.loc), dtype=bool)
            mask[self.normalize_dims] = True
            self.loc[~mask] = 0.0
            self.scale[~mask] = 1.0

        # Print out stats
        ndims = len(self.loc)
        normed = len(self.normalize_dims) if self.normalize_dims is not None else ndims
        print(
            f"Normalizer created: dims={ndims}, normalized={normed}, "
            f"loc=[{self.loc.min():.4f}, {self.loc.max():.4f}], "
            f"scale=[{self.scale.min():.6f}, {self.scale.max():.4f}]"
            + (f", clip=[{self.clip_low.min():.2f}, {self.clip_high.max():.2f}]"
               if self.clip_low is not None else "")
        )

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
            "type": "zscore",
            "loc": torch.from_numpy(self.loc),
            "scale": torch.from_numpy(self.scale),
        }
        if self.clip_low is not None:
            d["clip_low"] = torch.from_numpy(self.clip_low)
        if self.clip_high is not None:
            d["clip_high"] = torch.from_numpy(self.clip_high)
        if self.normalize_dims is not None:
            d["normalize_dims"] = torch.tensor(self.normalize_dims, dtype=torch.int64)
        return d

    @classmethod
    def from_state_dict(cls, d: dict) -> Normalizer:
        """Reconstruct a Normalizer from a checkpoint state dict.

        Uses a small default_scale (1e-6) to preserve the exact scale values
        that were stored at training time, avoiding re-clamping with the
        current class-level EPS which may differ.
        """
        normalize_dims = None
        if "normalize_dims" in d:
            normalize_dims = d["normalize_dims"].cpu().tolist()
        return cls(
            loc=d["loc"].cpu().numpy(),
            scale=d["scale"].cpu().numpy(),
            clip_low=(d.get("clip_low").cpu().numpy() if "clip_low" in d else None),
            clip_high=(d.get("clip_high").cpu().numpy() if "clip_high" in d else None),
            normalize_dims=normalize_dims,
        )

    # ── factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        clip_low: np.ndarray | None = None,
        clip_high: np.ndarray | None = None,
        default_scale: float = EPS,
        normalize_dims: list[int] | None = None,
    ) -> Normalizer:
        """Compute per-dimension mean/std from a (N, D) dataset and build a Normalizer."""
        return cls(
            loc=data.mean(axis=0).astype(np.float32),
            scale=data.std(axis=0).astype(np.float32),
            clip_low=clip_low,
            clip_high=clip_high,
            default_scale=default_scale,
            normalize_dims=normalize_dims,
        )


# ── Identity (no-op) normalizer ──────────────────────────────────────────────

class IdentityNormalizer:
    """No-op normalizer that passes data through unchanged. Same API as Normalizer."""

    def __init__(self, ndims: int = 0):
        self.ndims = ndims
        self.normalize_dims = None

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return x

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x

    def normalize_torch(self, x: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        return x

    def denormalize_torch(self, x: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        return x

    def state_dict(self) -> dict:
        return {"type": "identity", "ndims": torch.tensor(self.ndims)}

    @classmethod
    def from_state_dict(cls, d: dict) -> IdentityNormalizer:
        return cls(ndims=int(d.get("ndims", 0)))

    @classmethod
    def from_data(cls, data: np.ndarray, **kwargs) -> IdentityNormalizer:
        return cls(ndims=data.shape[1] if data.ndim > 1 else data.shape[0])


# ── Min-max normalizer ──────────────────────────────────────────────────────
# Adapted from ActionScaler in residual-offpolicy-rl (Amazon, CC-BY-NC-4.0)
# Source: resfit/rl_finetuning/utils/normalization.py, class ActionScaler

class MinMaxNormalizer:
    """Per-dimension min-max normalizer that scales data to [-1, 1].

    Forward:  2 * (x - min) / range - 1   (only on masked dims)
    Inverse:  (x + 1) * range / 2 + min   (only on masked dims)

    Excluded dims (not in normalize_dims) pass through unchanged.
    Works with both numpy and torch. Provides the same API as Normalizer
    (normalize/denormalize, state_dict/from_state_dict, from_data).
    """

    def __init__(
        self,
        data_min: np.ndarray,
        data_max: np.ndarray,
        min_range: float = 1e-1,
        normalize_dims: list[int] | None = None,
    ):
        """Construct a MinMaxNormalizer from per-dimension min/max.

        Args:
            data_min:  Per-dimension minimum values, shape (D,).
            data_max:  Per-dimension maximum values, shape (D,).
            min_range: Minimum range per dimension to prevent blow-up on
                       constant dims. Matches reference ActionScaler default.
            normalize_dims: Optional list of dimension indices to normalize.
                        Excluded dims pass through unchanged.
                        If None, all dims are normalized.
        """
        data_min = np.asarray(data_min, dtype=np.float32)
        data_max = np.asarray(data_max, dtype=np.float32)
        ndims = len(data_min)

        mid = (data_min + data_max) / 2
        half_range = (data_max - data_min) / 2
        half_range = np.maximum(half_range, min_range / 2)

        self.data_min = mid - half_range
        self.data_max = mid + half_range
        self.data_range = self.data_max - self.data_min

        # Build boolean mask: True for dims that get normalized
        if normalize_dims is not None:
            normalize_dims = [d for d in normalize_dims if d < ndims]
            if not normalize_dims:
                normalize_dims = None
        self.normalize_dims = normalize_dims
        self._mask = np.ones(ndims, dtype=bool)
        if self.normalize_dims is not None:
            self._mask[:] = False
            self._mask[self.normalize_dims] = True

        normed = int(self._mask.sum())
        print(
            f"MinMaxNormalizer created: dims={ndims}, normalized={normed}, "
            f"range=[{self.data_range[self._mask].min():.4f}, {self.data_range[self._mask].max():.4f}]"
        )

    # ── numpy paths ──────────────────────────────────────────────────────

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Scale masked dims to [-1, 1]. Excluded dims pass through."""
        out = np.array(x, dtype=np.float32, copy=True)
        m = self._mask
        clamped = np.clip(out[..., m], self.data_min[m], self.data_max[m])
        out[..., m] = 2.0 * (clamped - self.data_min[m]) / self.data_range[m] - 1.0
        return out

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Unscale masked dims from [-1, 1]. Excluded dims pass through."""
        out = np.array(x, dtype=np.float32, copy=True)
        m = self._mask
        clamped = np.clip(out[..., m], -1.0, 1.0)
        out[..., m] = self.data_min[m] + (clamped + 1.0) * self.data_range[m] / 2.0
        return out

    # ── torch paths ──────────────────────────────────────────────────────

    def normalize_torch(self, x: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Scale masked dims to [-1, 1] (torch). Excluded dims pass through."""
        d = device or x.device
        out = x.clone()
        m = torch.as_tensor(self._mask, device=d)
        dmin = torch.as_tensor(self.data_min, dtype=torch.float32, device=d)[m]
        dmax = torch.as_tensor(self.data_max, dtype=torch.float32, device=d)[m]
        drange = torch.as_tensor(self.data_range, dtype=torch.float32, device=d)[m]
        clamped = torch.clamp(out[..., m], dmin, dmax)
        out[..., m] = 2.0 * (clamped - dmin) / drange - 1.0
        return out

    def denormalize_torch(self, x: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Unscale masked dims from [-1, 1] (torch). Excluded dims pass through."""
        d = device or x.device
        out = x.clone()
        m = torch.as_tensor(self._mask, device=d)
        dmin = torch.as_tensor(self.data_min, dtype=torch.float32, device=d)[m]
        drange = torch.as_tensor(self.data_range, dtype=torch.float32, device=d)[m]
        clamped = torch.clamp(out[..., m], -1.0, 1.0)
        out[..., m] = dmin + (clamped + 1.0) * drange / 2.0
        return out

    # ── serialization ────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        d = {
            "type": "minmax",
            "data_min": torch.from_numpy(self.data_min),
            "data_max": torch.from_numpy(self.data_max),
            "data_range": torch.from_numpy(self.data_range),
        }
        if self.normalize_dims is not None:
            d["normalize_dims"] = torch.tensor(self.normalize_dims, dtype=torch.int64)
        return d

    @classmethod
    def from_state_dict(cls, d: dict) -> MinMaxNormalizer:
        normalize_dims = None
        if "normalize_dims" in d:
            normalize_dims = d["normalize_dims"].cpu().tolist()
        obj = cls.__new__(cls)
        obj.data_min = d["data_min"].cpu().numpy()
        obj.data_max = d["data_max"].cpu().numpy()
        obj.data_range = d["data_range"].cpu().numpy()
        obj.normalize_dims = normalize_dims
        ndims = len(obj.data_min)
        obj._mask = np.ones(ndims, dtype=bool)
        if normalize_dims is not None:
            obj._mask[:] = False
            obj._mask[normalize_dims] = True
        return obj

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        min_range: float = 1e-1,
        normalize_dims: list[int] | None = None,
    ) -> MinMaxNormalizer:
        """Compute per-dimension min/max from a (N, D) dataset."""
        return cls(
            data_min=data.min(axis=0).astype(np.float32),
            data_max=data.max(axis=0).astype(np.float32),
            min_range=min_range,
            normalize_dims=normalize_dims,
        )


# Factory for loading any normalizer from a checkpoint

NORM_ZSCORE = "zscore"
NORM_MINMAX = "minmax"
NORM_IDENTITY = "identity"

NORMALIZER_TYPES = {
    NORM_ZSCORE: Normalizer,
    # TODO(alexta): MinMaxNormalizer results in a zero success rate in DPFM. Hihghly suspect there's a bug here.
    # Do not use for now.
    NORM_MINMAX: MinMaxNormalizer,
    NORM_IDENTITY: IdentityNormalizer,
}


def load_normalizer(d: dict):
    """Load any normalizer from a state_dict, dispatching on the 'type' key.

    For backward compatibility, if no 'type' key is present, defaults to zscore.
    """
    norm_type = d.get("type", NORM_ZSCORE)
    if isinstance(norm_type, torch.Tensor):
        norm_type = str(norm_type)
    cls = NORMALIZER_TYPES.get(norm_type)
    if cls is None:
        raise ValueError(f"Unknown normalizer type: {norm_type!r}. "
                         f"Valid types: {list(NORMALIZER_TYPES.keys())}")
    return cls.from_state_dict(d)


def create_normalizer_from_data(
    normalizer_type: str,
    data: np.ndarray,
    normalize_dims: list[int] | None = None,
    default_scale: float = 1e-6,
    clip_low: np.ndarray | None = None,
    clip_high: np.ndarray | None = None,
    min_range: float = 1e-1,
):
    """Create a normalizer of the specified type from a dataset.

    Args:
        normalizer_type: One of NORM_ZSCORE, NORM_MINMAX, NORM_IDENTITY.
        data: (N, D) array to compute statistics from.
        normalize_dims: Indices of dims to normalize (None = all).
        default_scale: Floor for z-score std (only used for zscore).
        clip_low: Per-dim lower clip bound after denorm (only used for zscore).
        clip_high: Per-dim upper clip bound after denorm (only used for zscore).
        min_range: Minimum per-dim range (only used for minmax).
    """
    if normalizer_type == NORM_ZSCORE:
        return Normalizer.from_data(data, default_scale=default_scale,
                                    clip_low=clip_low, clip_high=clip_high,
                                    normalize_dims=normalize_dims)
    elif normalizer_type == NORM_MINMAX:
        return MinMaxNormalizer.from_data(data, min_range=min_range,
                                          normalize_dims=normalize_dims)
    elif normalizer_type == NORM_IDENTITY:
        return IdentityNormalizer.from_data(data)
    else:
        raise ValueError(f"Unknown normalizer_type: {normalizer_type!r}. "
                         f"Valid types: {list(NORMALIZER_TYPES.keys())}")


# ── Observation/action dimension helpers ────────────────────────────────────

# Dimensionality of the current environment.

# TODO(alexta): currently we explicitly rely that the environment will form the dimensions as following:
# - proprioception
# - stick info (location, goal)
# - phase and stick one-hot encodings.
# I feel this is a weak assumption that is inherently fragile. We need a better way to create a predictable state vector.

# Num. dimensions for each stick (must be normalized)
DIMS_PER_STICK = 10  # pos(3) + quat(4) + goal(3)
# Robot proprioception dimensions  (must be normalized)
PROPRIOCEPTION_DIM = 50
# Num. action dimensions
ACTION_DIM = 7
ACTION_GRIPPER_DIM = 6  # first 6 dims are joint forces, dim 6 is gripper - do not normalize it.


def obs_normalize_dims(num_sticks: int) -> list[int] | None:
    """Return the list of observation dims to normalize for a given stick count.

    Obs layout: [stick0_pos(3), stick0_quat(4), goal0_pos(3), ..., proprioception(50)]
    All dims are continuous and should be normalized.
    Returns None (normalize all) if num_sticks is invalid.
    """
    assert num_sticks > 0
    total = num_sticks * DIMS_PER_STICK + PROPRIOCEPTION_DIM
    return list(range(total))


def action_normalize_dims(action_dim: int = ACTION_DIM) -> list[int] | None:
    """Return the list of action dims to normalize (excludes binary gripper).

    For the standard 7-dim OSC_POSE action space, dims 0-5 are joint forces
    and dim 6 is the binary gripper. For non-standard action dims, returns None
    (normalize all).
    """
    if action_dim == ACTION_DIM:
        return list(range(ACTION_GRIPPER_DIM))
    return None



# ── Tensor batch-dimension helpers ──────────────────────────────��───────────
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
    # We should not be use it for Residual RL
    raise NotImplementedError
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
