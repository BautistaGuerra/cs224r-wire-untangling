import argparse
import os
import random

import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
import torch

from wire_untangling.policies.flow_matching_policy import FlowMatchingPolicy, flow_matching_loss
from wire_untangling.utils.normalizer import Normalizer


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
        return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
    finally:
        raw_env.close()


def train(
    config: dict,
    demos_path: str,
    seed: int = 42,
    use_wandb: bool = True,
    checkpoint_dir: str = "checkpoints",
    conditioning: str = CONDITIONING_OBS,
    obs_std_floor: float = Normalizer.EPS,
    action_std_floor: float = Normalizer.EPS,
    wandb_run_id: str | None = None,
    wandb_name: str | None = None,
):
    dpfm_cfg = config.get("dpfm", {})
    train_cfg = config.get("dpfm_train", config.get("train", {}))

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
    loader, state_dim, action_dim, obs_norm, action_norm, data_meta = load_data(
        demos_path,
        chunk_size=chunk_size,
        batch_size=batch_size,
        action_low=action_low,
        action_high=action_high,
        conditioning=conditioning,
        obs_std_floor=obs_std_floor,
        action_std_floor=action_std_floor,
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
                "raw_obs_dim": data_meta["raw_obs_dim"],
                "num_phases": data_meta["num_phases"],
                "num_sticks": data_meta["num_sticks"],
                "obs_std_floor": obs_std_floor,
                "action_std_floor": action_std_floor,
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
        num_steps=int(dpfm_cfg.get("integration_steps", 10)),
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
        "num_steps": int(dpfm_cfg.get("integration_steps", 10)),
        "conditioning": conditioning,
        "raw_obs_dim": data_meta["raw_obs_dim"],
        "num_phases": data_meta["num_phases"],
        "num_sticks": data_meta["num_sticks"],
        "obs_std_floor": obs_std_floor,
        "action_std_floor": action_std_floor,
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


def _decode_json_attr(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    import json
    return json.loads(value)


def _infer_num_sticks(f) -> int:
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
    obs_std_floor: float = Normalizer.EPS,
    action_std_floor: float = Normalizer.EPS,
) -> tuple[DataLoader, int, int, Normalizer, Normalizer, dict]:
    """Load demos from HDF5, window actions into chunks.

    For each timestep t, produces (s_t, [a_t, ..., a_{t+C-1}]). When the
    chunk extends past the episode end, the last action is repeated to fill
    the chunk (same padding strategy as diffusion_policy's SequenceSampler).

    Observations and action chunks are z-score normalized using dataset stats.

    Returns (dataloader, state_dim, action_dim, obs_normalizer, action_normalizer, meta).
    """
    import h5py

    all_obs = []
    all_actions = []
    all_features = []
    episode_lengths = []
    with h5py.File(demo_path, "r") as f:
        data_grp = f["data"]
        num_sticks = _infer_num_sticks(f)
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

    raw_obs_dim = all_obs[0].shape[1]
    action_dim = all_actions[0].shape[1]
    for i, (obs, act) in enumerate(zip(all_obs, all_actions)):
        assert obs.shape[1] == raw_obs_dim, f"Demo {i}: obs_dim={obs.shape[1]} != {raw_obs_dim}"
        assert act.shape[1] == action_dim, f"Demo {i}: action_dim={act.shape[1]} != {action_dim}"

    flat_obs = np.concatenate(all_obs, axis=0)
    flat_actions = np.concatenate(all_actions, axis=0)
    state = flat_obs
    if conditioning == CONDITIONING_PHASE_ACTIVE:
        state = np.concatenate([flat_obs, np.concatenate(all_features, axis=0)], axis=1)
    elif conditioning != CONDITIONING_OBS:
        raise ValueError(
            f"Unsupported conditioning={conditioning!r}; "
            f"expected {CONDITIONING_OBS!r} or {CONDITIONING_PHASE_ACTIVE!r}"
        )
    state_dim = state.shape[1]

    # Initialize the normalizer from data
    obs_norm = Normalizer.from_data(state, default_scale=obs_std_floor)
    action_norm = Normalizer.from_data(
        flat_actions,
        clip_low=action_low,
        clip_high=action_high,
        default_scale=action_std_floor,
    )

    episode_ends = np.cumsum(episode_lengths)
    indices = create_chunk_indices(episode_ends, chunk_size, pad_after=chunk_size - 1)

    chunked_states = []
    chunked_actions = []
    for buf_start, buf_end, samp_start, samp_end in indices:
        state_row = obs_norm.normalize(state[buf_start])

        action_chunk = np.zeros((chunk_size, action_dim), dtype=np.float32)
        action_slice = flat_actions[buf_start:buf_end]
        action_chunk[samp_start:samp_end] = action_slice
        if samp_end < chunk_size:
            action_chunk[samp_end:] = action_slice[-1]
        action_chunk = action_norm.normalize(action_chunk)

        chunked_states.append(state_row)
        chunked_actions.append(action_chunk.flatten())

    s_tensor = torch.tensor(np.array(chunked_states), dtype=torch.float32)
    a_tensor = torch.tensor(np.array(chunked_actions), dtype=torch.float32)
    loader = DataLoader(TensorDataset(s_tensor, a_tensor), batch_size=batch_size, shuffle=shuffle)
    meta = {
        "conditioning": conditioning,
        "raw_obs_dim": raw_obs_dim,
        "num_phases": NUM_PHASES,
        "num_sticks": num_sticks,
    }
    return loader, state_dim, action_dim, obs_norm, action_norm, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/stick_reorder.yaml")
    parser.add_argument("--dpfm-config", default="configs/flow_matching.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/flow_matching")
    parser.add_argument("--demos-path", default="data/demos.hdf5")
    parser.add_argument("--conditioning", choices=[CONDITIONING_OBS, CONDITIONING_PHASE_ACTIVE],
                        default=CONDITIONING_OBS)
    parser.add_argument("--obs-std-floor", type=float, default=Normalizer.EPS,
                        help="DPFM-local observation normalizer std floor.")
    parser.add_argument("--action-std-floor", type=float, default=Normalizer.EPS,
                        help="DPFM-local action normalizer std floor.")
    args = parser.parse_args()

    cfg = load_config(args.env_config, args.dpfm_config)

    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42)
    train(
        cfg,
        demos_path=args.demos_path,
        seed=seed,
        use_wandb=not args.no_wandb,
        checkpoint_dir=args.checkpoint_dir,
        conditioning=args.conditioning,
        obs_std_floor=args.obs_std_floor,
        action_std_floor=args.action_std_floor,
    )


if __name__ == "__main__":
    main()
