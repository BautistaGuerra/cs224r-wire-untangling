import argparse
import os
import random

import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
import torch

from wire_untangling.policies.flow_matching_policy import FlowMatchingPolicy, flow_matching_loss
from wire_untangling.utils.normalizer import (
    Normalizer, DEFAULT_SCALE_OBSERVATIONS, DEFAULT_SCALE_ACTIONS,
    action_normalize_dims,
    create_normalizer_from_data, NORM_ZSCORE, NORM_MINMAX, NORM_IDENTITY,
)


CONDITIONING_OBS = "obs"
CONDITIONING_PHASE_ACTIVE = "phase-active"
NUM_PHASES = 8

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
    """Create the configured env and return raw action bounds."""
    from wire_untangling.envs import StickReorderEnv

    env_cfg = dict(config.get("env", {}))
    kwargs = dict(
        robots=env_cfg.get("robot", "Panda"),
        num_sticks=env_cfg.get("num_sticks", 1),
        stick_length=env_cfg.get("stick_length", 0.20),
        stick_radius=env_cfg.get("stick_radius", 0.0075),
        goal_spacing=env_cfg.get("goal_spacing", 0.06),
        success_threshold=env_cfg.get("success_threshold", 0.03),
        orientation_threshold=env_cfg.get("orientation_threshold", np.deg2rad(10.0)),
        lambda_rot=env_cfg.get("lambda_rot", 0.1),
        goal_yaw=env_cfg.get("goal_yaw", 0.0),
        reward_shaping=env_cfg.get("reward_shaping", True),
        terminate_on_success=env_cfg.get("terminate_on_success", True),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=env_cfg.get("horizon", 500),
    )
    optional_env_keys = (
        "placement_mode",
        "init_x_range",
        "init_y_range",
        "side_init_x_range",
        "side_init_y_ranges",
        "side_init_yaw_range",
        "side_goal_x",
        "side_goal_y_ranges",
        "stick_color_indices",
    )
    for key in optional_env_keys:
        if key in env_cfg:
            kwargs[key] = env_cfg[key]

    raw_env = StickReorderEnv(**kwargs)
    try:
        low, high = raw_env.action_spec
        print(f'Environment created with action bounds. low: {low}, high: {high}')
        return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
    finally:
        raw_env.close()


def train(
    config: dict,
    demos_path: str,
    seed: int = None,
    use_wandb: bool = True,
    checkpoint_dir: str = "checkpoints",
    action_normalizer_type: str = "zscore",
    conditioning: str = CONDITIONING_OBS,
    wandb_run_id: str | None = None,
    wandb_name: str | None = None,
):
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
    execute_steps = int(dpfm_cfg.get("execute_steps", FlowMatchingPolicy.default_execute_steps(chunk_size)))
    val_fraction = float(train_cfg.get("val_fraction", 0.1))
    action_low, action_high = make_action_bounds(config)
    train_loader, val_loader, state_dim, action_dim, obs_norm, action_norm, data_meta = load_data(
        demos_path,
        chunk_size=chunk_size,
        batch_size=batch_size,
        action_low=action_low,
        action_high=action_high,
        conditioning=conditioning,
        val_fraction=val_fraction,
        action_normalizer_type=action_normalizer_type,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    if use_wandb:
        import wandb
        wandb_kwargs = {
            "project": "cs224r-wire-untangling",
            "config": {
                **config,
                "seed": seed,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "conditioning": conditioning,
                "action_normalizer": action_normalizer_type,
                "raw_obs_dim": data_meta["raw_obs_dim"],
                "num_phases": data_meta["num_phases"],
                "num_sticks": data_meta["num_sticks"],
            },
            "tags": ["flow-matching", "bc"],
        }
        if wandb_run_id:
            wandb_kwargs.update({"id": wandb_run_id, "resume": "allow"})
        if wandb_name:
            wandb_kwargs["name"] = wandb_name
        run = wandb.init(**wandb_kwargs)
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
        # Train epoch
        for s_batch, a_batch in train_loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            loss = flow_matching_loss(policy, s_batch, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s_batch.size(0)
            n += s_batch.size(0)

        avg_loss = total_loss / n
        log_msg = f"  Epoch {epoch+1}/{epochs}, train_loss: {avg_loss:.6f}"
        log_dict = {"epoch": epoch + 1, "train_loss": avg_loss}

        # Validation loss. Inference is pretty quick, so do evaluation every epoch
        if val_loader is not None:
            policy.eval()
            val_total = 0.0
            val_n = 0
            with torch.no_grad():
                for s_batch, a_batch in val_loader:
                    s_batch = s_batch.to(device)
                    a_batch = a_batch.to(device)
                    val_loss = flow_matching_loss(policy, s_batch, a_batch)
                    val_total += val_loss.item() * s_batch.size(0)
                    val_n += s_batch.size(0)
            avg_val_loss = val_total / val_n
            log_msg += f", val_loss: {avg_val_loss:.6f}"
            log_dict["val_loss"] = avg_val_loss
            policy.train()

        print(log_msg)
        if run is not None:
            wandb.log(log_dict)

    save_path = os.path.join(checkpoint_dir, "flow_matching_policy.pt")
    torch.save({
        "model_state_dict": policy.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "pred_horizon": chunk_size,
        "execute_steps": execute_steps,
        "num_integration_steps": int(dpfm_cfg.get("integration_steps", 10)),
        "conditioning": conditioning,
        "raw_obs_dim": data_meta["raw_obs_dim"],
        "num_phases": data_meta["num_phases"],
        "num_sticks": data_meta["num_sticks"],
        "obs_norm": obs_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")

    if run is not None:
        # For now, disable saving the network in the WanDB. It takes a huge amount of space there.
        # artifact = wandb.Artifact("flow-matching-policy", type="model")
        # artifact.add_file(save_path)
        # run.log_artifact(artifact)
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


def _decode_json_attr(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    import json
    return json.loads(value)


def _infer_num_sticks(f, raw_obs_dim: int | None = None) -> int:
    env_cfg = _decode_json_attr(f.attrs.get("env_config"))
    if env_cfg and "num_sticks" in env_cfg:
        return int(env_cfg["num_sticks"])
    if raw_obs_dim is not None and raw_obs_dim >= 60:
        inferred = (raw_obs_dim - 50) // 10
        if 50 + 10 * inferred == raw_obs_dim and inferred > 0:
            return int(inferred)

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
    chunk_size: int = 20,
    batch_size: int = 256,
    shuffle: bool = True,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    conditioning: str = CONDITIONING_OBS,
    val_fraction: float = 0.0,
    action_normalizer_type: str = NORM_ZSCORE,
):
    """Load demos from HDF5, window actions into chunks.

    For each timestep t, produces (s_t, [a_t, ..., a_{t+C-1}]). When the
    chunk extends past the episode end, the last action is repeated to fill
    the chunk (same padding strategy as diffusion_policy's SequenceSampler).

    Observations and action chunks are z-score normalized using dataset stats.
    Normalizers are always fit on the TRAINING split only.

    Args:
        val_fraction: Fraction of episodes to hold out for validation (0.0 = no val set).
            Split is done at the episode level, not timestep level.

    Returns (train_loader, val_loader, state_dim, action_dim, obs_normalizer, action_normalizer).
    val_loader is None if val_fraction == 0.
    """
    import h5py

    all_obs = []
    all_actions = []
    all_features = []
    episode_lengths = []
    with h5py.File(demo_path, "r") as f:
        data_grp = f["data"]
        keys = sorted(data_grp.keys())
        first_obs = data_grp[keys[0]]["obs"][:]
        raw_obs_dim = first_obs.shape[1]
        num_sticks = _infer_num_sticks(f, raw_obs_dim=raw_obs_dim)
        for key in sorted(data_grp.keys()):
            demo = data_grp[key]
            obs = demo["obs"][:]
            actions = demo["actions"][:]
            all_obs.append(obs)
            all_actions.append(actions)
            episode_lengths.append(len(obs))
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

    num_episodes = len(all_obs)
    action_dim = all_actions[0].shape[1]
    for i, (obs, act) in enumerate(zip(all_obs, all_actions)):
        assert obs.shape[1] == raw_obs_dim, f"Demo {i}: obs_dim={obs.shape[1]} != {raw_obs_dim}"
        assert act.shape[1] == action_dim, f"Demo {i}: action_dim={act.shape[1]} != {action_dim}"

    if conditioning == CONDITIONING_OBS:
        all_states = all_obs
    elif conditioning == CONDITIONING_PHASE_ACTIVE:
        all_states = [
            np.concatenate([obs, features], axis=1)
            for obs, features in zip(all_obs, all_features)
        ]
    else:
        raise ValueError(
            f"Unsupported conditioning={conditioning!r}; "
            f"expected {CONDITIONING_OBS!r} or {CONDITIONING_PHASE_ACTIVE!r}"
        )
    state_dim = all_states[0].shape[1]

    # Split episodes into train/val
    num_val = int(num_episodes * val_fraction)
    num_train = num_episodes - num_val
    # Shuffle episode indices for random split
    ep_indices = np.random.permutation(num_episodes)
    train_indices = sorted(ep_indices[:num_train])
    val_indices = sorted(ep_indices[num_train:]) if num_val > 0 else []

    # Fit normalizers on training episodes only
    train_states = [all_states[i] for i in train_indices]
    train_actions = [all_actions[i] for i in train_indices]
    flat_train_states = np.concatenate(train_states, axis=0)
    flat_train_actions = np.concatenate(train_actions, axis=0)

    obs_norm = Normalizer.from_data(
        flat_train_states,
        default_scale=DEFAULT_SCALE_OBSERVATIONS,
        normalize_dims=list(range(raw_obs_dim)),
    )

    action_ndims = action_normalize_dims(action_dim)
    action_norm = create_normalizer_from_data(
        action_normalizer_type, flat_train_actions,
        normalize_dims=action_ndims,
        default_scale=DEFAULT_SCALE_ACTIONS,
        clip_low=action_low, clip_high=action_high,
    )

    def _build_loader(ep_idxs: np.ndarray, all_states: list[np.ndarray], all_actions: list[np.ndarray],
                      episode_lengths, shuffle:bool=True):
        """Build a DataLoader from a subset of episodes.

        Args:
            ep_idxs: indices of the episodes that constitute this data loader (separate for train and eval)
            all_states, all_actions: numpy arrays of states / actions
        """
        subset_states = [all_states[i] for i in ep_idxs]
        subset_actions = [all_actions[i] for i in ep_idxs]
        subset_lengths = [episode_lengths[i] for i in ep_idxs]

        flat_states = np.concatenate(subset_states, axis=0)
        flat_actions = np.concatenate(subset_actions, axis=0)
        episode_ends = np.cumsum(subset_lengths)
        indices = create_chunk_indices(episode_ends, chunk_size, pad_after=chunk_size - 1)

        chunked_states = []
        chunked_actions = []
        for buf_start, buf_end, samp_start, samp_end in indices:
            state = obs_norm.normalize(flat_states[buf_start])

            action_chunk = np.zeros((chunk_size, action_dim), dtype=np.float32)
            action_slice = flat_actions[buf_start:buf_end]
            action_chunk[samp_start:samp_end] = action_slice
            if samp_end < chunk_size:
                action_chunk[samp_end:] = action_slice[-1]
            naction_chunk = action_norm.normalize(action_chunk)

            chunked_states.append(state)
            chunked_actions.append(naction_chunk.flatten())

        s_tensor = torch.tensor(np.array(chunked_states), dtype=torch.float32)
        a_tensor = torch.tensor(np.array(chunked_actions), dtype=torch.float32)
        return DataLoader(TensorDataset(s_tensor, a_tensor), batch_size=batch_size, shuffle=shuffle)

    train_loader = _build_loader(train_indices, all_states, all_actions, episode_lengths, shuffle=shuffle)
    val_loader = None
    if val_indices:
        val_loader = _build_loader(val_indices, all_states, all_actions, episode_lengths, shuffle=False)
        print(f"Train/val split: {num_train} train, {num_val} val episodes")

    meta = {
        "conditioning": conditioning,
        "raw_obs_dim": raw_obs_dim,
        "num_phases": NUM_PHASES,
        "num_sticks": num_sticks,
    }
    return train_loader, val_loader, state_dim, action_dim, obs_norm, action_norm, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/stick_reorder.yaml")
    parser.add_argument("--dpfm-config", default="configs/flow_matching.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--action-normalizer", choices=[NORM_ZSCORE, NORM_MINMAX, NORM_IDENTITY],
                        default=NORM_ZSCORE, help="Type of action normalizer (default: zscore)")
    parser.add_argument("--checkpoint-dir", default="checkpoints/flow_matching")
    parser.add_argument("--demos-path", default="data/demos.hdf5")
    parser.add_argument("--conditioning", choices=[CONDITIONING_OBS, CONDITIONING_PHASE_ACTIVE],
                        default=CONDITIONING_OBS)
    parser.add_argument("--action-chunk-horizon", type=int, default=None,
                        help="Override action chunk horizon from config")
    parser.add_argument("--execute-steps", type=int, default=None,
                        help="Override execute steps from config")
    parser.add_argument("--integration-steps", type=int, default=None,
                        help="Override number of flow matching integration steps from config")
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.env_config, args.dpfm_config)

    # CLI overrides for DPFM hyperparams
    if args.action_chunk_horizon is not None:
        cfg.setdefault("dpfm", {})["action_chunk_horizon"] = args.action_chunk_horizon
    if args.execute_steps is not None:
        cfg.setdefault("dpfm", {})["execute_steps"] = args.execute_steps
    if args.integration_steps is not None:
        cfg.setdefault("dpfm", {})["integration_steps"] = args.integration_steps

    train(cfg, demos_path=args.demos_path, seed=args.seed, use_wandb=not args.no_wandb,
          checkpoint_dir=args.checkpoint_dir, action_normalizer_type=args.action_normalizer,
          conditioning=args.conditioning, wandb_run_id=args.wandb_run_id,
          wandb_name=args.wandb_name)


if __name__ == "__main__":
    main()
