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
import os
import random

import h5py
import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader, TensorDataset

from wire_untangling.policies.mlp_bc import MLPBCPolicy, mse_loss


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


def load_data(
    demo_path: str,
    batch_size: int = 256,
    shuffle: bool = True,
) -> tuple[DataLoader, int, int, np.ndarray, np.ndarray]:
    """Load (state, action) pairs from HDF5, normalise states.

    No action chunking — MLP-BC predicts one action per state, so each demo
    timestep is an independent training example.

    Returns (dataloader, state_dim, action_dim, state_mean, state_std).
    """
    all_obs = []
    all_actions = []
    with h5py.File(demo_path, "r") as f:
        for key in sorted(f["data"].keys()):
            demo = f["data"][key]
            all_obs.append(demo["obs"][:])
            all_actions.append(demo["actions"][:])

    flat_obs = np.concatenate(all_obs, axis=0)
    flat_actions = np.concatenate(all_actions, axis=0)
    state_dim = flat_obs.shape[1]
    action_dim = flat_actions.shape[1]

    # Per-dim z-score stats. eps avoids /0 on dims that happen to be
    # constant in the demos (e.g. goal_pos at N=1 is fixed every episode).
    state_mean = flat_obs.mean(axis=0).astype(np.float32)
    state_std = flat_obs.std(axis=0).astype(np.float32)
    state_std = np.maximum(state_std, 1e-6)
    flat_obs_normed = ((flat_obs - state_mean) / state_std).astype(np.float32)

    s = torch.tensor(flat_obs_normed, dtype=torch.float32)
    a = torch.tensor(flat_actions, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(s, a), batch_size=batch_size, shuffle=shuffle,
    )
    return loader, state_dim, action_dim, state_mean, state_std


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

    loader, state_dim, action_dim, state_mean, state_std = load_data(
        demos_path, batch_size=batch_size,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    if use_wandb:
        run = wandb.init(
            project="cs224r-wire-untangling",
            config={
                "policy": "mlp_bc",
                "seed": seed,
                "state_dim": state_dim,
                "action_dim": action_dim,
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
        "action_dim": action_dim,
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
    )


if __name__ == "__main__":
    main()
