"""Train an MLP behavior-cloning policy on expert demos.

Canonical demo recipe (matches README):
    python scripts/collect_demos.py --num-demos 200 --output data/demos.hdf5 --seed 42

The script prints HDF5 provenance attrs at startup (env_config_hash, top_seed,
oracle_version, robosuite_version, num_demos). Use --require-config-hash and
--require-top-seed to hard-fail if the demos were generated with different
parameters than expected — useful for CI and shared-checkpoint workflows.

Loss is MSE on (state, action) pairs; states are z-score normalised per-dim
over the training corpus and the stats are saved with the checkpoint so
inference applies the same transform. ~3 minutes on MPS for 100 epochs.
"""

import argparse
import json
import os
import random

import h5py
import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader, TensorDataset

from wire_untangling.policies.mlp_bc import MLPBCPolicy, mse_loss


CONDITIONING_OBS = "obs"
CONDITIONING_PHASE_ACTIVE = "phase-active"
NUM_PHASES = 8


def read_demo_metadata(demo_path: str) -> dict:
    """Pull provenance attrs from the HDF5 root."""
    keys = ("env_config_hash", "top_seed", "oracle_version",
            "robosuite_version", "num_demos", "obs_dim", "total_samples")
    with h5py.File(demo_path, "r") as f:
        meta = {}
        for k in keys:
            if k in f.attrs:
                v = f.attrs[k]
                if isinstance(v, bytes):
                    v = v.decode()
                meta[k] = v
            else:
                meta[k] = None
    return meta


def print_demo_metadata(demo_path: str, meta: dict) -> None:
    print(f"Loaded demos from {demo_path}")
    for k in ("num_demos", "total_samples", "obs_dim",
              "top_seed", "env_config_hash", "oracle_version",
              "robosuite_version"):
        v = meta.get(k)
        print(f"  {k}: {v}")


def validate_demo_metadata(meta: dict, require_hash: str | None,
                           require_seed: int | None) -> None:
    if require_hash is not None and meta.get("env_config_hash") != require_hash:
        raise SystemExit(
            f"env_config_hash mismatch: HDF5 has {meta.get('env_config_hash')}, "
            f"required {require_hash}"
        )
    if require_seed is not None and int(meta.get("top_seed") or -1) != require_seed:
        raise SystemExit(
            f"top_seed mismatch: HDF5 has {meta.get('top_seed')}, "
            f"required {require_seed}"
        )


def _decode_json_attr(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(value)


def _infer_num_sticks(f: h5py.File) -> int:
    env_cfg = _decode_json_attr(f.attrs.get("env_config"))
    if env_cfg and "num_sticks" in env_cfg:
        return int(env_cfg["num_sticks"])

    max_stick = -1
    for key in f["data"].keys():
        demo = f["data"][key]
        if "active_stick" in demo:
            active = demo["active_stick"][:]
            if len(active):
                max_stick = max(max_stick, int(active.max()))
    return max_stick + 1 if max_stick >= 0 else 1


def make_phase_active_features(
    phases: np.ndarray,
    active_sticks: np.ndarray,
    num_sticks: int,
    num_phases: int = NUM_PHASES,
) -> np.ndarray:
    """One-hot encode per-step high-level labels for phase-conditioned BC."""
    phases = np.asarray(phases, dtype=np.int64)
    active_sticks = np.asarray(active_sticks, dtype=np.int64)
    if phases.shape != active_sticks.shape:
        raise ValueError(
            f"phase and active_stick shapes differ: {phases.shape} vs {active_sticks.shape}"
        )
    if np.any((phases < 0) | (phases >= num_phases)):
        raise ValueError(f"phase values must be in [0, {num_phases})")
    if np.any((active_sticks < 0) | (active_sticks >= num_sticks)):
        raise ValueError(f"active_stick values must be in [0, {num_sticks})")

    out = np.zeros((len(phases), num_phases + num_sticks), dtype=np.float32)
    out[np.arange(len(phases)), phases] = 1.0
    out[np.arange(len(active_sticks)), num_phases + active_sticks] = 1.0
    return out


def load_data(
    demo_path: str,
    batch_size: int = 256,
    shuffle: bool = True,
    conditioning: str = CONDITIONING_OBS,
) -> tuple[DataLoader, int, int, np.ndarray, np.ndarray, dict]:
    """Load (state, action) pairs from HDF5, normalise states.

    No action chunking — MLP-BC predicts one action per state, so each demo
    timestep is an independent training example.

    Returns (dataloader, state_dim, action_dim, state_mean, state_std, meta).
    """
    all_obs = []
    all_actions = []
    all_features = []

    with h5py.File(demo_path, "r") as f:
        num_sticks = _infer_num_sticks(f)
        for key in sorted(f["data"].keys()):
            demo = f["data"][key]
            all_obs.append(demo["obs"][:])
            all_actions.append(demo["actions"][:])
            if conditioning == CONDITIONING_PHASE_ACTIVE:
                if "phase" not in demo or "active_stick" not in demo:
                    raise ValueError(
                        "--conditioning phase-active requires phase and active_stick "
                        f"datasets; missing in data/{key}"
                    )
                all_features.append(
                    make_phase_active_features(
                        demo["phase"][:],
                        demo["active_stick"][:],
                        num_sticks=num_sticks,
                    )
                )

    flat_obs = np.concatenate(all_obs, axis=0)
    flat_actions = np.concatenate(all_actions, axis=0)
    raw_obs_dim = flat_obs.shape[1]
    action_dim = flat_actions.shape[1]
    state = flat_obs
    if conditioning == CONDITIONING_PHASE_ACTIVE:
        state = np.concatenate([flat_obs, np.concatenate(all_features, axis=0)], axis=1)
    elif conditioning != CONDITIONING_OBS:
        raise ValueError(
            f"Unsupported conditioning={conditioning!r}; "
            f"expected {CONDITIONING_OBS!r} or {CONDITIONING_PHASE_ACTIVE!r}"
        )
    state_dim = state.shape[1]

    # Per-dim z-score stats. eps avoids /0 on dims that happen to be
    # constant in the demos (e.g. goal_pos at N=1 is fixed every episode).
    state_mean = state.mean(axis=0).astype(np.float32)
    state_std = state.std(axis=0).astype(np.float32)
    state_std = np.maximum(state_std, 1e-6)
    state_normed = ((state - state_mean) / state_std).astype(np.float32)

    s = torch.tensor(state_normed, dtype=torch.float32)
    a = torch.tensor(flat_actions, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(s, a), batch_size=batch_size, shuffle=shuffle,
    )
    meta = {
        "conditioning": conditioning,
        "raw_obs_dim": raw_obs_dim,
        "num_phases": NUM_PHASES,
        "num_sticks": num_sticks,
    }
    return loader, state_dim, action_dim, state_mean, state_std, meta


def train(
    demos_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dims: tuple[int, ...],
    dropout: float,
    seed: int,
    use_wandb: bool,
    checkpoint_dir: str,
    conditioning: str = CONDITIONING_OBS,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training device: {device}")

    loader, state_dim, action_dim, state_mean, state_std, conditioning_meta = load_data(
        demos_path,
        batch_size=batch_size,
        conditioning=conditioning,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    if use_wandb:
        run = wandb.init(
            project="cs224r-wire-untangling",
            config={
                "policy": "mlp_bc",
                "seed": seed,
                "state_dim": state_dim,
                "raw_obs_dim": conditioning_meta["raw_obs_dim"],
                "action_dim": action_dim,
                "conditioning": conditioning,
                "hidden_dims": list(hidden_dims),
                "dropout": dropout,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
            },
            tags=["mlp-bc", "bc"],
        )
    else:
        run = None

    policy = MLPBCPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            loss = mse_loss(policy, s_batch, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s_batch.size(0)
            n += s_batch.size(0)
        avg_loss = total_loss / n
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        if run is not None:
            wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    save_path = os.path.join(checkpoint_dir, "mlp_bc_policy.pt")
    torch.save({
        "model_state_dict": policy.state_dict(),
        "state_dim": state_dim,
        "raw_obs_dim": conditioning_meta["raw_obs_dim"],
        "action_dim": action_dim,
        "conditioning": conditioning,
        "num_phases": conditioning_meta["num_phases"],
        "num_sticks": conditioning_meta["num_sticks"],
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        # Stored as torch tensors so torch.load(weights_only=True) accepts
        # them without an allowlist (PyTorch 2.6+ rejects numpy by default).
        "state_mean": torch.from_numpy(state_mean),
        "state_std": torch.from_numpy(state_std),
    }, save_path)
    print(f"Model saved to {save_path}")

    if run is not None:
        artifact = wandb.Artifact("mlp-bc-policy", type="model")
        artifact.add_file(save_path)
        run.log_artifact(artifact)
        run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos-path", default="data/demos.hdf5")
    parser.add_argument("--checkpoint-dir", default="checkpoints/mlp_bc")
    # 100 epochs left the policy ~half-converged (per-dim error ~0.05, 57%
    # success). 500 epochs drives error to ~0.025 and success to 92% on 1000
    # demos. Loss curve plateaus around epoch 400-500.
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--conditioning", choices=[CONDITIONING_OBS, CONDITIONING_PHASE_ACTIVE],
                        default=CONDITIONING_OBS,
                        help="Input features for BC. 'obs' uses only env observations; "
                             "'phase-active' appends one-hot phase and active_stick labels.")
    parser.add_argument("--require-config-hash", default=None,
                        help="Hard-fail if the demo HDF5's env_config_hash "
                             "doesn't match. Use to enforce a canonical dataset.")
    parser.add_argument("--require-top-seed", type=int, default=None,
                        help="Hard-fail if the demo HDF5's top_seed doesn't match.")
    args = parser.parse_args()

    meta = read_demo_metadata(args.demos_path)
    print_demo_metadata(args.demos_path, meta)
    validate_demo_metadata(meta, args.require_config_hash, args.require_top_seed)

    train(
        demos_path=args.demos_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        checkpoint_dir=args.checkpoint_dir,
        conditioning=args.conditioning,
    )


if __name__ == "__main__":
    main()
