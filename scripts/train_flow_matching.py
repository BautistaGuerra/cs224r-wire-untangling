import argparse
import os
import random

import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
import torch

from wire_untangling.policies.flow_matching_policy import FlowMatchingPolicy, flow_matching_loss
from wire_untangling.utils.normalizer import Normalizer, DEFAULT_SCALE_OBSERVATIONS, DEFAULT_SCALE_ACTIONS
import scripts.rrl_env_creation as rrl_env

def load_config(
    env_config: str = "configs/stick_reorder.yaml",
    policy_config: str = "configs/flow_matching.yaml",
) -> dict:
    cfg = {}
    for path in [env_config, policy_config]:
        with open(path) as f:
            cfg.update(yaml.safe_load(f))
    return cfg


def make_action_bounds(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Create the current one-stick env and return raw action bounds."""

    env_cfg = dict(config.get("env", {}))
    env_cfg["num_sticks"] = 1
    raw_env = rrl_env.make_rrl_gym_env_1stick(env_cfg)
    try:
        low, high = raw_env.action_spec
        print(f'Environment created with action bounds. low: {low}, high: {high}')
        return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
    finally:
        raw_env.close()


def train(config: dict, demos_path: str, seed: int = None, use_wandb: bool = True, checkpoint_dir: str = "checkpoints"):
    dpfm_cfg = config.get("dpfm", {})
    train_cfg = config.get("dpfm_train", {})

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(train_cfg.get("batch_size", 2048))
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("lr", 1e-4))

    chunk_size = int(dpfm_cfg.get("action_chunk_horizon", 20))
    execute_steps = int(dpfm_cfg.get("execute_steps", max(1, chunk_size // 2)))
    action_low, action_high = make_action_bounds(config)
    loader, state_dim, action_dim, obs_norm, action_norm = load_data(
        demos_path,
        chunk_size=chunk_size,
        batch_size=batch_size,
        action_low=action_low,
        action_high=action_high,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    if use_wandb:
        import wandb
        run = wandb.init(
            project="cs224r-wire-untangling",
            config={
                **config,
                "seed": seed,
                "state_dim": state_dim,
                "action_dim": action_dim,
            },
            tags=["flow-matching", "bc"],
        )
    else:
        run = None

    policy = FlowMatchingPolicy(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        pred_horizon=int(dpfm_cfg.get("action_chunk_horizon", 20)),
        num_integration_steps=int(dpfm_cfg.get("integration_steps", 10)),
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            loss = flow_matching_loss(policy, s_batch, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s_batch.size(0)
            n += s_batch.size(0)

        avg_loss = total_loss / n
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        if run is not None:
            wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    save_path = os.path.join(checkpoint_dir, "flow_matching_policy.pt")
    torch.save({
        "model_state_dict": policy.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "pred_horizon": chunk_size,
        "execute_steps": execute_steps,
        # TODO(alexta): rename to num_integration_steps
        "num_integration_steps": int(dpfm_cfg.get("integration_steps", 10)),
        "obs_norm": obs_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")

    if run is not None:
        artifact = wandb.Artifact("flow-matching-policy", type="model")
        artifact.add_file(save_path)
        run.log_artifact(artifact)
        run.finish()


def create_chunk_indices(
    episode_ends: np.ndarray,
    chunk_size: int,
    pad_after: int,
) -> np.ndarray:
    """Precompute valid (buffer_start, buffer_end, sample_start, sample_end) tuples.

    Adapted from diffusion_policy's create_indices (sampler.py). Each tuple
    describes how to slice the flat buffer and where to place the slice within
    a fixed-length chunk, enabling last-element padding at episode boundaries.

    Args:
        episode_ends: Cumulative episode end indices into the flat buffer.
        chunk_size: Action chunk length C.
        pad_after: Max padding past episode end (typically chunk_size - 1).
    """
    pad_after = min(max(pad_after, 0), chunk_size - 1)
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        max_start = episode_length - chunk_size + pad_after
        for idx in range(0, max_start + 1):
            buffer_start = min(idx, episode_length - 1) + start_idx
            buffer_end = min(idx + chunk_size, episode_length) + start_idx
            start_offset = buffer_start - (idx + start_idx)
            end_offset = (idx + chunk_size + start_idx) - buffer_end
            sample_start = start_offset
            sample_end = chunk_size - end_offset
            indices.append([buffer_start, buffer_end, sample_start, sample_end])
    return np.array(indices, dtype=np.int64)


def load_data(
    demo_path: str,
    chunk_size: int = 20,
    batch_size: int = 256,
    shuffle: bool = True,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> tuple[DataLoader, int, int, Normalizer, Normalizer]:
    """Load demos from HDF5, window actions into chunks.

    For each timestep t, produces (s_t, [a_t, ..., a_{t+C-1}]). When the
    chunk extends past the episode end, the last action is repeated to fill
    the chunk (same padding strategy as diffusion_policy's SequenceSampler).

    Observations and action chunks are z-score normalized using dataset stats.

    Returns (dataloader, state_dim, action_dim, obs_normalizer, action_normalizer).
    """
    import h5py

    all_obs = []
    all_actions = []
    episode_lengths = []
    with h5py.File(demo_path, "r") as f:
        data_grp = f["data"]
        for key in sorted(data_grp.keys()):
            demo = data_grp[key]
            obs = demo["obs"][:]
            actions = demo["actions"][:]
            all_obs.append(obs)
            all_actions.append(actions)
            episode_lengths.append(len(obs))

    state_dim = all_obs[0].shape[1]
    action_dim = all_actions[0].shape[1]
    for i, (obs, act) in enumerate(zip(all_obs, all_actions)):
        assert obs.shape[1] == state_dim, f"Demo {i}: obs_dim={obs.shape[1]} != {state_dim}"
        assert act.shape[1] == action_dim, f"Demo {i}: action_dim={act.shape[1]} != {action_dim}"

    flat_obs = np.concatenate(all_obs, axis=0)
    flat_actions = np.concatenate(all_actions, axis=0)

    # Initialize the normalizer from data
    obs_norm = Normalizer.from_data(flat_obs, default_scale=DEFAULT_SCALE_OBSERVATIONS)
    action_norm = Normalizer.from_data(flat_actions, clip_low=action_low, clip_high=action_high,
                                       default_scale=DEFAULT_SCALE_ACTIONS)

    episode_ends = np.cumsum(episode_lengths)
    indices = create_chunk_indices(episode_ends, chunk_size, pad_after=chunk_size - 1)

    chunked_states = []
    chunked_actions = []
    for buf_start, buf_end, samp_start, samp_end in indices:
        state = obs_norm.normalize(flat_obs[buf_start])

        action_chunk = np.zeros((chunk_size, action_dim), dtype=np.float32)
        action_slice = flat_actions[buf_start:buf_end]
        action_chunk[samp_start:samp_end] = action_slice
        if samp_end < chunk_size:
            action_chunk[samp_end:] = action_slice[-1]
        action_chunk = action_norm.normalize(action_chunk)

        chunked_states.append(state)
        chunked_actions.append(action_chunk.flatten())

    s_tensor = torch.tensor(np.array(chunked_states), dtype=torch.float32)
    a_tensor = torch.tensor(np.array(chunked_actions), dtype=torch.float32)
    loader = DataLoader(TensorDataset(s_tensor, a_tensor), batch_size=batch_size, shuffle=shuffle)
    return loader, state_dim, action_dim, obs_norm, action_norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/stick_reorder.yaml")
    parser.add_argument("--dpfm-config", default="configs/flow_matching.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/flow_matching")
    parser.add_argument("--demos-path", default="data/demos.hdf5")
    args = parser.parse_args()

    cfg = load_config(args.env_config, args.dpfm_config)

    # seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42)
    train(cfg, demos_path=args.demos_path, seed=args.seed, use_wandb=not args.no_wandb, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
