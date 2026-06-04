"""Residual TD3 training script for the stick-reorder environment.

Collects online experience with a base DPFM policy + residual TD3 actor,
stores transitions in a replay buffer, and trains the residual policy
to improve upon the base policy's actions.

All observations and actions in the replay buffer are z-score normalized
using the same normalizers from the DPFM checkpoint.
"""
import argparse
import os
import random
import re
import time
from collections import deque
from typing import Callable

import numpy as np
import yaml
import torch

from robosuite.wrappers import GymWrapper

from wire_untangling.envs import StickReorderEnv
from wire_untangling.policies.rl.agent import TD3Agent
from wire_untangling.policies.policy_inference_wrappers import DPFMModelPolicy, ResidualRLPolicy
from wire_untangling.utils.normalizer import Normalizer, DEFAULT_SCALE_OBSERVATIONS, DEFAULT_SCALE_ACTIONS
from wire_untangling.utils.stick_order import StickOrderScheduler
import scripts.rrl_env_creation as rrl_env

# ── Config loading ───────────────────────────────────────────────────────────

class DictConfig:
    """Minimal recursive namespace for dot-access on nested dicts."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, DictConfig(v) if isinstance(v, dict) else v)
    def __repr__(self):
        return f"DictConfig({self.__dict__})"


def load_config(
    env_config: str = "configs/stick_reorder.yaml",
    td3_config: str = "configs/residual_td3.yaml",
) -> dict:
    cfg = {}
    for path in [env_config, td3_config]:
        with open(path) as f:
            cfg.update(yaml.safe_load(f))
    return cfg


# ── Environment creation ────────────────────────────────────────────────────

def make_gym_env(env_cfg: dict):
    raw_env = rrl_env.make_rrl_gym_env(env_cfg)
    return GymWrapper(raw_env)



def add_transition_to_buffer(
    rb,
    obs: np.ndarray,
    action: np.ndarray,
    action_base: np.ndarray,
    next_obs: np.ndarray,
    next_action_base: np.ndarray,
    reward: float,
    done: bool,
):
    """Helper function taht adds a single transition to the replay buffer as a TensorDict.
    """
    from tensordict import TensorDict

    td = TensorDict(
        {
            "obs": torch.tensor(obs, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "action_base": torch.tensor(action_base, dtype=torch.float32),
            "next": TensorDict(
                {
                    "obs": torch.tensor(next_obs, dtype=torch.float32),
                    "action_base": torch.tensor(next_action_base, dtype=torch.float32),
                    "done": torch.tensor(done, dtype=torch.bool),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                },
                batch_size=[],
            ),
            "_priority": torch.tensor(10.0, dtype=torch.float32),
        },
        batch_size=[],
    ).unsqueeze(0)
    rb.add(td)


def make_replay_buffer(buffer_size: int, batch_size: int, n_step: int, gamma: float):
    """Create a TensorDictPrioritizedReplayBuffer with TD(n_step) transform.
    """
    from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
    from wire_untangling.utils.rb_transforms import MultiStepTransform

    return TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=buffer_size, device="cpu"),
        alpha=0.0,  # uniform sampling (no prioritization for now)
        beta=0.0,
        eps=1e-6,
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=n_step, gamma=gamma),
        pin_memory=True,
        batch_size=batch_size,
    )


def _demo_stick_order(demo, order_schedule: StickOrderScheduler | None, episode_index: int):
    """Return the order associated with a demo, falling back to the configured schedule."""
    if "stick_order" in demo.attrs:
        return tuple(int(i) for i in np.asarray(demo.attrs["stick_order"]).tolist())
    if order_schedule is not None:
        return order_schedule.order_for(episode_index)
    return None


# ── Offline buffer population ───────────────────────────────────────────────
# Ref: train_residual_td3.py:513-631 (_populate_offline_buffer)

def populate_offline_buffer(
    demos_path: str,
    offline_rb,
    base_policy: DPFMModelPolicy,
    obs_norm,
    action_norm,
    gamma: float,
    max_transitions: int | None = None,
    order_schedule: StickOrderScheduler | None = None,
):
    """Load demonstrations from HDF5 and fill the offline buffer with normalized transitions.

    For each demo timestep:
    - runs the frozen base policy to get base_action (what it would predict for that observation). Stores the GT demo action as the executed
    action. This teaches the critic the value landscape around demonstrated behavior.

    Ref: train_residual_td3.py:513-631
    """
    import h5py

    print("Populating offline buffer from demos...")
    transitions = 0

    with h5py.File(demos_path, "r") as f:
        data_grp = f["data"]
        num_episodes = len(data_grp.keys())
        for ep_idx, ep_key in enumerate(sorted(data_grp.keys())):
            if max_transitions is not None and transitions >= max_transitions:
                print(f"  Reached max_transitions={max_transitions}, stopping.")
                break
            if (ep_idx + 1) % 50 == 0 or ep_idx == 0:
                print(f"  Episode {ep_idx + 1}/{num_episodes} ({transitions} transitions so far)")
            demo = data_grp[ep_key]
            obs_all = demo["obs"][:]
            actions_all = demo["actions"][:]
            rewards_all = demo["rewards"][:]
            T = len(obs_all)

            # Reset the base policy at the start of each episode so its action
            # chunk state and optional phase/active tracker match online rollout.
            stick_order = _demo_stick_order(demo, order_schedule, ep_idx)
            base_policy.reset(stick_order=stick_order)

            # Get base policy's action for this observation.
            # predict_norm advances the chunk counter by 1, matching online behavior.
            _, base_naction, nobs = base_policy.predict_norm_with_state(obs_all[0])
            # GT action from the demonstration (normalized)
            gt_naction = action_norm.normalize(actions_all[0])
            reward = float(rewards_all[0])
            # 1st state can't be a terminal state!
            assert not demo["dones"][0]
            done = demo["dones"][0]

            for t in range(1, T):
                # We must advance the base policy only once per step. It maintainss the internal chunk
                # buffer, so we must call predict_norm exactly once per replay advance.
                # So here we maintain a "rolling window" of the (previous, current) observations and actions.
                _, next_base_naction, next_nobs = base_policy.predict_norm_with_state(obs_all[t])

                # We add to the buffer results of the "previous" step as current observation/action,
                # and the currently sampled obs/action as the next obs/action
                add_transition_to_buffer(
                    rb=offline_rb,
                    obs=nobs,
                    action=gt_naction,
                    action_base=base_naction,
                    next_obs=next_nobs,
                    next_action_base=next_base_naction,
                    reward=reward,
                    done=done,
                )
                # Save data into the "current" step
                nobs = next_nobs
                base_naction = next_base_naction
                gt_naction = action_norm.normalize(actions_all[t])
                reward = float(rewards_all[t])
                done = demo["dones"][t]
                transitions += 1

            # This is the terminal state.  We must do exactly one more advance to account for a final step.
            assert done
            add_transition_to_buffer(
                rb=offline_rb,
                obs=nobs,
                action=gt_naction,
                action_base=base_naction,
                next_obs=np.zeros_like(next_nobs),
                next_action_base=np.zeros_like(next_base_naction),
                reward=reward,
                done=done
            )
            transitions += 1


    print(f"Offline buffer populated: {transitions} transitions from {num_episodes} episodes")
    return transitions


def sample_buffers(offline_buffer, replay_buffer, offline_fraction, device):
    """Sample a batch of data from a mix of offline buffer and replay buffer."""
    if offline_buffer is not None and offline_fraction > 0.0:
        online_batch = replay_buffer.sample().to(device)
        offline_batch = offline_buffer.sample().to(device)
        batch = torch.cat([online_batch, offline_batch], dim=0)
    else:
        batch = replay_buffer.sample().to(device)
    return batch


# ── Noise schedule ──────────────────────────────────────────────────────────

def linear_schedule(step: int, max_val: float, min_val: float, decay_steps: int) -> float:
    """Linearly decay from max_val to min_val over decay_steps."""
    frac = min(1.0, step / max(decay_steps, 1))
    return max_val + frac * (min_val - max_val)


def _checkpoint_step(path: str) -> int | None:
    match = re.fullmatch(r"td3_step(\d+)\.pt", os.path.basename(path))
    return int(match.group(1)) if match else None


def find_latest_training_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the highest numbered periodic TD3 checkpoint in a directory."""
    if not os.path.isdir(checkpoint_dir):
        return None

    candidates: list[tuple[int, str]] = []
    for name in os.listdir(checkpoint_dir):
        step = _checkpoint_step(name)
        if step is not None:
            candidates.append((step, os.path.join(checkpoint_dir, name)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _estimated_critic_updates(global_step: int, td3_cfg: DictConfig) -> int:
    if global_step < td3_cfg.learning_starts:
        return 0
    update_steps = (global_step - td3_cfg.learning_starts) // td3_cfg.gradient_update_per_env_steps + 1
    return int(update_steps * td3_cfg.critic_updates_per_gradient_step)


def _save_training_checkpoint(
    save_path: str,
    *,
    agent: TD3Agent,
    state_dim: int,
    action_dim: int,
    global_step: int,
    episode_count: int,
    critic_updates: int,
    obs_norm,
    action_norm,
    td3_raw: dict,
    env_cfg: dict,
    expert_cfg: dict,
    dpfm_checkpoint: str,
    dpfm_stochastic: bool,
    dpfm_execute_steps: int,
    dpfm_replan_on_context_change: bool,
    checkpoint_callback: Callable[[str, int], None] | None = None,
):
    payload = {
        "model_state_dict": agent.state_dict(),
        "critic_optimizer_state_dict": agent.critic_opt.state_dict(),
        "actor_optimizer_state_dict": agent.actor_opt.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "global_step": global_step,
        "episode_count": episode_count,
        "critic_updates": critic_updates,
        "obs_norm": obs_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "rrl_config": td3_raw,
        "env_config": env_cfg,
        "expert_config": expert_cfg,
        "base_dpfm_checkpoint": dpfm_checkpoint,
        "dpfm_stochastic": dpfm_stochastic,
        "dpfm_execute_steps": dpfm_execute_steps,
        "dpfm_replan_on_context_change": dpfm_replan_on_context_change,
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tmp_path = f"{save_path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, save_path)
    print(f"  Checkpoint saved: {save_path}")
    if checkpoint_callback is not None:
        checkpoint_callback(save_path, global_step)


def _load_training_checkpoint(
    checkpoint_path: str,
    *,
    agent: TD3Agent,
    device: torch.device,
    state_dim: int,
    action_dim: int,
    td3_cfg: DictConfig,
) -> tuple[int, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    ckpt_state_dim = int(checkpoint.get("state_dim", state_dim))
    ckpt_action_dim = int(checkpoint.get("action_dim", action_dim))
    if ckpt_state_dim != state_dim:
        raise ValueError(
            f"Resume checkpoint state_dim={ckpt_state_dim} does not match current state_dim={state_dim}"
        )
    if ckpt_action_dim != action_dim:
        raise ValueError(
            f"Resume checkpoint action_dim={ckpt_action_dim} does not match current action_dim={action_dim}"
        )

    agent.load_state_dict(checkpoint["model_state_dict"])
    if "critic_optimizer_state_dict" in checkpoint:
        agent.critic_opt.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    else:
        print("WARNING: resume checkpoint has no critic optimizer state; resuming critic optimizer fresh.")
    if "actor_optimizer_state_dict" in checkpoint:
        agent.actor_opt.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    else:
        print("WARNING: resume checkpoint has no actor optimizer state; resuming actor optimizer fresh.")

    global_step = int(checkpoint.get("global_step", 0))
    episode_count = int(checkpoint.get("episode_count", 0))
    critic_updates = checkpoint.get("critic_updates")
    if critic_updates is None:
        critic_updates = _estimated_critic_updates(global_step, td3_cfg)
        print(
            "WARNING: resume checkpoint has no critic_updates; "
            f"estimated critic_updates={critic_updates} from global_step."
        )
    else:
        critic_updates = int(critic_updates)
    return global_step, episode_count, critic_updates


def train(
    config: dict,
    dpfm_checkpoint: str,
    seed: int = None,
    use_wandb: bool = True,
    checkpoint_dir: str = "checkpoints/td3",
    dpfm_stochastic: bool = True,
    dpfm_execute_steps: int | None = None,
    dpfm_replan_on_context_change: bool = False,
    resume_checkpoint: str | None = None,
    auto_resume: bool = False,
    checkpoint_callback: Callable[[str, int], None] | None = None,
):
    td3_raw = config.get("residual_td3", {})
    env_cfg = config.get("env", {})
    expert_cfg = config.get("expert", {})

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    gym_env = make_gym_env(env_cfg)
    env_num_sticks = int(getattr(gym_env.env, "num_sticks", env_cfg.get("num_sticks", 1)))
    order_schedule = StickOrderScheduler(expert_cfg, env_num_sticks)

    # Load and initialize base DPFM policy
    base_policy = DPFMModelPolicy(
        dpfm_checkpoint,
        gym_env,
        execute_steps=dpfm_execute_steps,
        stochastic=dpfm_stochastic,
        replan_on_context_change=dpfm_replan_on_context_change,
    )
    base_policy.set_gym_env(gym_env, expert_cfg=expert_cfg)
    obs_norm = base_policy.obs_norm
    action_norm = base_policy.action_norm
    state_dim = int(base_policy.obs_norm.loc.shape[0])
    action_dim = int(base_policy.action_dim)

    # Create TD3 agent. We expect that during the training configured action/state dimensions
    if td3_raw["state_dim"] is None:
        td3_raw["state_dim"]= state_dim
    else:
        assert td3_raw["state_dim"] == state_dim

    if td3_raw["action_dim"] is None:
        td3_raw["action_dim"] = action_dim
    else:
        assert td3_raw["action_dim"] == action_dim
    td3_cfg = DictConfig(td3_raw)
    agent = TD3Agent(td3_cfg)

    # Set up replay buffers
    offline_fraction = float(td3_raw.get("offline_fraction", 0.0))
    offline_demos_path = td3_raw.get("offline_demos_path")
    online_batch_size = int(td3_cfg.batch_size * (1 - offline_fraction))
    offline_batch_size = td3_cfg.batch_size - online_batch_size

    # Online buffer — filled during training rollouts
    replay_buffer = make_replay_buffer(
        buffer_size=td3_cfg.replay_buffer_size,
        batch_size=online_batch_size,
        n_step=td3_cfg.n_step,
        gamma=td3_cfg.gamma,
    )

    # Offline buffer — filled from demo data before training starts
    offline_buffer = None
    if offline_fraction > 0.0 and offline_demos_path is not None:
        offline_buffer = make_replay_buffer(
            buffer_size=td3_cfg.replay_buffer_size,
            batch_size=offline_batch_size,
            n_step=td3_cfg.n_step,
            gamma=td3_cfg.gamma,
        )
        offline_max = td3_raw.get("offline_max_transitions")
        if offline_max is not None:
            offline_max = int(offline_max)
        populate_offline_buffer(
            demos_path=offline_demos_path,
            offline_rb=offline_buffer,
            base_policy=base_policy,
            obs_norm=obs_norm,
            action_norm=action_norm,
            gamma=td3_cfg.gamma,
            max_transitions=offline_max,
            order_schedule=order_schedule,
        )
        # Reset base policy after offline population so online rollouts start fresh
        base_policy.reset()
        print(f"Mixed training: {online_batch_size} online + {offline_batch_size} offline per batch")
    elif offline_fraction > 0.0:
        print(f"WARNING: offline_fraction={offline_fraction} but no offline_demos_path set. Running online-only.")
        offline_fraction = 0.0

    # Logging / checkpointing intervals
    log_freq = 1000
    eval_freq = 10_000

    os.makedirs(checkpoint_dir, exist_ok=True)
    resume_path = resume_checkpoint
    if auto_resume:
        latest_path = find_latest_training_checkpoint(checkpoint_dir)
        latest_step = _checkpoint_step(latest_path) if latest_path else None
        resume_step = _checkpoint_step(resume_path) if resume_path else None
        if latest_path and (resume_step is None or latest_step >= resume_step):
            resume_path = latest_path

    resume_global_step = 0
    resume_episode_count = 0
    resume_critic_updates = 0
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming residual TD3 training from {resume_path}")
        resume_global_step, resume_episode_count, resume_critic_updates = _load_training_checkpoint(
            resume_path,
            agent=agent,
            device=device,
            state_dim=state_dim,
            action_dim=action_dim,
            td3_cfg=td3_cfg,
        )
        print(
            f"  Resume state: global_step={resume_global_step}, "
            f"episodes={resume_episode_count}, critic_updates={resume_critic_updates}"
        )
    elif auto_resume:
        print(f"Auto-resume enabled, but no td3_step*.pt checkpoint found in {checkpoint_dir}.")

    # Wandb initialization
    if use_wandb:
        import wandb
        run = wandb.init(
            project="cs224r-wire-untangling",
            config={
                **config,
                "seed": seed,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "base_dpfm_checkpoint": dpfm_checkpoint,
                "dpfm_stochastic": dpfm_stochastic,
                "dpfm_execute_steps": base_policy.execute_steps,
                "dpfm_replan_on_context_change": dpfm_replan_on_context_change,
                "resume_checkpoint": resume_path,
                "auto_resume": auto_resume,
            },
            tags=["residual-td3"],
        )
    else:
        run = None

    # ── Collect + train loop ────────────────────────────────────────────
    global_step = resume_global_step
    episode_count = resume_episode_count
    episode_reward = 0.0
    recent_rewards = deque(maxlen=20)
    critic_updates = resume_critic_updates
    train_start = time.time()
    run_start_step = global_step

    current_stick_order = order_schedule.order_for(episode_count)
    obs_raw, _ = gym_env.reset()
    base_policy.reset(stick_order=current_stick_order)


    print(f"Starting residual TD3 training for {td3_cfg.total_timesteps} steps")
    print(f"  state_dim={state_dim}, action_dim={action_dim}")
    print(f"  num_sticks={env_num_sticks}, order_mode={order_schedule.mode}")
    print(f"  dpfm_stochastic={dpfm_stochastic}, dpfm_execute_steps={base_policy.execute_steps}")
    print(f"  learning_starts={td3_cfg.learning_starts}, batch_size={td3_cfg.batch_size}")
    print(f"  critic_warmup={td3_cfg.critic_warmup_steps}, actor_update_every={td3_cfg.actor_update_every}")
    if resume_path:
        print("  online replay buffer is rebuilt after resume; updates wait until it has a batch.")

    agent.eval()
    # Query the base policy once for the initial action; subsequent steps
    # reuse next_base_naction to avoid advancing the chunk index twice per step.
    base_naction = None
    nobs = None

    exploration_stddev = td3_cfg.exploration_stddev_max
    critic_smoothing_stddev = td3_cfg.smoothing_stddev_max

    while global_step < td3_cfg.total_timesteps:
        # A policy rollout (i.e. "exploration") phase.
        # Only query the base policy on the first step or after an episode reset
        if base_naction is None:
            _, base_naction, nobs = base_policy.predict_norm_with_state(obs_raw)

        combined_action = None
        residual_naction = None
        # Action selection: random during warmup, policy + noise after
        if global_step < td3_cfg.learning_starts:
            # Uniform random residual bounded by action_scale for initial exploration
            residual_naction = np.random.uniform(
                -td3_cfg.actor.action_scale, td3_cfg.actor.action_scale, size=action_dim
            ).astype(np.float32)
            combined_action, _, _ = ResidualRLPolicy.combine_actions(action_norm, base_naction, residual_naction)
        else:
            # Decay exploration noise linearly over training
            exploration_stddev = linear_schedule(
                global_step - td3_cfg.learning_starts,
                td3_cfg.exploration_stddev_max, td3_cfg.exploration_stddev_min, td3_cfg.exploration_stddev_decay_steps,
            )
            with torch.no_grad():
                # Sample a residual action from the agent.
                residual_naction = agent.act(
                    torch.tensor(nobs, dtype=torch.float32, device=device),
                    torch.tensor(base_naction, dtype=torch.float32, device=device),
                    eval_mode=False,
                    stddev=exploration_stddev,
                    clip=td3_cfg.exploration_stddev_clip
                ).numpy()
                # Combine the residual action with the base action from BC policy and denormalize it.
                combined_action, _, _ = ResidualRLPolicy.combine_actions(action_norm, base_naction, residual_naction)

        # Step the environment with the denormalized action
        assert combined_action is not None
        assert residual_naction is not None
        next_obs_raw, reward, terminated, truncated, info = gym_env.step(combined_action)
        done = terminated or truncated
        episode_reward += reward

        # Prepare next-state quantities for the replay buffer.
        # Advance the base policy chunk by one step to get the next base action.
        if done:
            # Terminal: next base action irrelevant (discount will zero it out)
            next_base_naction = np.zeros(action_dim, dtype=np.float32)
            next_nobs = np.zeros_like(nobs, dtype=np.float32)
        else:
            _, next_base_naction, next_nobs = base_policy.predict_norm_with_state(next_obs_raw)

        # Store the transition — everything is in normalized space
        add_transition_to_buffer(
            rb=replay_buffer,
            obs=nobs,
            action=TD3Agent.get_combined_action_numpy(base_naction, residual_naction),
            action_base=base_naction,
            next_obs=next_nobs,
            next_action_base=next_base_naction,
            reward=reward,
            done=done,
        )

        # Handle episode resets and carry forward the next base action
        if done:
            recent_rewards.append(episode_reward)
            episode_count += 1
            current_stick_order = order_schedule.order_for(episode_count)
            obs_raw, _ = gym_env.reset()
            base_policy.reset(stick_order=current_stick_order)
            episode_reward = 0.0
            base_naction = None
            nobs = None
        else:
            obs_raw = next_obs_raw
            base_naction = next_base_naction
            nobs = next_nobs

        global_step += 1

        # Gradient updates phase: start after warmup, every update_every_n_steps
        metrics = None
        online_ready = len(replay_buffer) >= max(1, online_batch_size)
        offline_ready = (
            offline_buffer is None
            or offline_fraction <= 0.0
            or len(offline_buffer) >= max(1, offline_batch_size)
        )
        if (
            global_step >= td3_cfg.learning_starts
            and global_step % td3_cfg.gradient_update_per_env_steps == 0
            and online_ready
            and offline_ready
        ):
            for _ in range(td3_cfg.critic_updates_per_gradient_step):
                # Allow std. deviation of actions to be high initially then diminish
                critic_smoothing_stddev = linear_schedule(
                    global_step - td3_cfg.learning_starts,
                    td3_cfg.smoothing_stddev_max, td3_cfg.smoothing_stddev_min, td3_cfg.critic_stddev_decay_steps,
                )
                # Sample a batch of data from both offline and online buffers
                batch = sample_buffers(offline_buffer, replay_buffer, offline_fraction, device)
                agent.train()
                metrics = agent.update(batch, update_critic=True, update_actor=False, critic_smoothing_stddev=critic_smoothing_stddev)
                agent.eval()
                critic_updates += 1
            if critic_updates >= td3_cfg.critic_warmup_steps:
                for _ in range(td3_cfg.actor_updates_per_gradient_step):
                    # Note: unlike ResFiT we sample a different batch for training the actor
                    batch = sample_buffers(offline_buffer, replay_buffer, offline_fraction, device)
                    agent.train()
                    actor_metrics = agent.update(batch, update_critic=False, update_actor=True, critic_smoothing_stddev=critic_smoothing_stddev)
                    agent.eval()
                    if metrics is not None:
                        metrics.update(actor_metrics)
                    else:
                        metrics = actor_metrics

        # Periodic console + wandb logging
        if global_step % log_freq == 0 and global_step > 0:
            elapsed = time.time() - train_start
            sps = int(max(0, global_step - run_start_step) / elapsed) if elapsed > 0 else 0
            mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            phase = "warmup" if critic_updates < td3_cfg.critic_warmup_steps else "training"

            # Build console message
            print_msg = (
                f"[{global_step}/{td3_cfg.total_timesteps}] episodes={episode_count} "
                f"mean_reward={mean_reward:.3f} sps={sps} phase={phase}"
            )
            if metrics is not None:
                print_msg += f" critic_loss={metrics['train/critic_loss']:.4f}"
                print_msg += f" critic_qt={metrics['train/critic_qt']:.4f}"
                if "train/actor_loss" in metrics:
                    print_msg += f" actor_loss={metrics['train/actor_loss']:.4f}"
                if "train/actor_grad_norm" in metrics:
                    print_msg += f" actor_grad={metrics['train/actor_grad_norm']:.4f}"
                if "_actions" in metrics:
                    residual_l1 = torch.mean(torch.abs(metrics["_actions"])).item()
                    residual_l2 = torch.mean(torch.square(metrics["_actions"])).item()
                    print_msg += f" res_l1={residual_l1:.4f} res_l2={residual_l2:.4f}"
            print(print_msg)

            if run is not None:
                import wandb
                log_dict = {
                    "training/global_step": global_step,
                    "training/episodes": episode_count,
                    "training/mean_episode_reward": mean_reward,
                    "training/exploration_stddev": exploration_stddev,
                    "training/critic_smoothing_stddev": critic_smoothing_stddev,
                    "training/SPS": sps,
                    "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
                    "buffer/online_size": len(replay_buffer),
                    "buffer/offline_size": len(offline_buffer) if offline_buffer is not None else 0,
                    "buffer/critic_updates": critic_updates,
                }
                if metrics is not None:
                    # Add scalar metrics (skip internal tensors prefixed with _)
                    for k, v in metrics.items():
                        if not k.startswith("_"):
                            log_dict[k] = v
                    # Residual action magnitude stats
                    if "_actions" in metrics:
                        actions = metrics["_actions"]
                        log_dict["train/residual_l1_magnitude"] = torch.mean(torch.abs(actions)).item()
                        log_dict["train/residual_l2_magnitude"] = torch.mean(torch.square(actions)).item()
                        log_dict["histograms/residual_actions"] = wandb.Histogram(actions.numpy().reshape(-1))
                    # Target Q distribution
                    if "_target_q" in metrics:
                        log_dict["histograms/critic_qt"] = wandb.Histogram(metrics["_target_q"].numpy().reshape(-1))
                wandb.log(log_dict, step=global_step)

        # Periodic checkpointing
        if global_step % eval_freq == 0 and global_step > 0:
            save_path = os.path.join(checkpoint_dir, f"td3_step{global_step}.pt")
            _save_training_checkpoint(
                save_path,
                agent=agent,
                state_dim=state_dim,
                action_dim=action_dim,
                global_step=global_step,
                episode_count=episode_count,
                critic_updates=critic_updates,
                obs_norm=obs_norm,
                action_norm=action_norm,
                td3_raw=td3_raw,
                env_cfg=env_cfg,
                expert_cfg=expert_cfg,
                dpfm_checkpoint=dpfm_checkpoint,
                dpfm_stochastic=dpfm_stochastic,
                dpfm_execute_steps=base_policy.execute_steps,
                dpfm_replan_on_context_change=dpfm_replan_on_context_change,
                checkpoint_callback=checkpoint_callback,
            )

    # ── Final save ──────────────────────────────────────────────────────
    save_path = os.path.join(checkpoint_dir, "td3_final.pt")
    _save_training_checkpoint(
        save_path,
        agent=agent,
        state_dim=state_dim,
        action_dim=action_dim,
        global_step=global_step,
        episode_count=episode_count,
        critic_updates=critic_updates,
        obs_norm=obs_norm,
        action_norm=action_norm,
        td3_raw=td3_raw,
        env_cfg=env_cfg,
        expert_cfg=expert_cfg,
        dpfm_checkpoint=dpfm_checkpoint,
        dpfm_stochastic=dpfm_stochastic,
        dpfm_execute_steps=base_policy.execute_steps,
        dpfm_replan_on_context_change=dpfm_replan_on_context_change,
        checkpoint_callback=checkpoint_callback,
    )
    print(f"Final model saved: {save_path}")

    if run is not None:
        import wandb
        # artifact = wandb.Artifact("residual-td3-policy", type="model")
        # artifact.add_file(save_path)
        # run.log_artifact(artifact)
        run.finish()

    gym_env.close()


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/stick_reorder.yaml")
    parser.add_argument("--td3-config", default="configs/residual_td3.yaml")
    parser.add_argument("--dpfm-checkpoint", required=True,
                        help="Path to trained DPFM checkpoint (.pt)")
    parser.add_argument("--num-sticks", type=int, default=None,
                        help="Override num_sticks from env config")
    parser.add_argument("--reward-shaping", action=argparse.BooleanOptionalAction, default=None,
                        help="Override reward_shaping from env config (--reward-shaping / --no-reward-shaping)")
    parser.add_argument("--demos-path", type=str, default=None,
                        help="Override offline_demos_path from config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/td3")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Resume residual TD3 weights/optimizer state from this checkpoint")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Resume from the highest td3_step*.pt in --checkpoint-dir if present")
    # Hyperparameter overrides
    parser.add_argument("--action-scale", type=float, default=None, help="Override actor.action_scale")
    parser.add_argument("--actor-lr", type=float, default=None, help="Override actor.lr")
    parser.add_argument("--critic-lr", type=float, default=None, help="Override critic.lr")
    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=None,
                        help="Override actor.hidden_dims (sequence of layer widths, e.g. 512 512)")
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=None,
                        help="Override critic.hidden_dims (sequence of layer widths, e.g. 512 512)")
    parser.add_argument("--offline-fraction", type=float, default=None, help="Override offline_fraction")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total_timesteps")
    parser.add_argument("--dpfm-execute-steps", type=int, default=None,
                        help="Override DPFM chunk actions executed before re-planning")
    parser.add_argument("--dpfm-stochastic", action="store_true", default=True,
                        help="Use random Flow Matching initial noise for the frozen base policy")
    parser.add_argument("--dpfm-deterministic", dest="dpfm_stochastic", action="store_false",
                        help="Use zero initial noise for deterministic frozen-base DPFM sampling")
    parser.add_argument("--dpfm-replan-on-context-change", action="store_true",
                        help=("For phase-active DPFM, discard cached actions and re-sample "
                              "when tracked (phase, active_stick) changes."))
    args = parser.parse_args()

    cfg = load_config(args.env_config, args.td3_config)
    if args.demos_path is not None:
        cfg.setdefault("residual_td3", {})["offline_demos_path"] = args.demos_path
    if args.num_sticks is not None:
        cfg["env"]["num_sticks"] = args.num_sticks
    if args.reward_shaping is not None:
        cfg["env"]["reward_shaping"] = args.reward_shaping
    td3 = cfg.setdefault("residual_td3", {})
    if args.action_scale is not None:
        td3.setdefault("actor", {})["action_scale"] = args.action_scale
    if args.actor_lr is not None:
        td3.setdefault("actor", {})["lr"] = args.actor_lr
    if args.critic_lr is not None:
        td3.setdefault("critic", {})["lr"] = args.critic_lr
    if args.actor_hidden_dims is not None:
        td3.setdefault("actor", {})["hidden_dims"] = args.actor_hidden_dims
    if args.critic_hidden_dims is not None:
        td3.setdefault("critic", {})["hidden_dims"] = args.critic_hidden_dims
    if args.offline_fraction is not None:
        td3["offline_fraction"] = args.offline_fraction
    if args.total_timesteps is not None:
        td3["total_timesteps"] = args.total_timesteps
    seed = args.seed if args.seed is not None else 42

    train(
        cfg,
        dpfm_checkpoint=args.dpfm_checkpoint,
        seed=seed,
        use_wandb=not args.no_wandb,
        checkpoint_dir=args.checkpoint_dir,
        dpfm_stochastic=args.dpfm_stochastic,
        dpfm_execute_steps=args.dpfm_execute_steps,
        dpfm_replan_on_context_change=args.dpfm_replan_on_context_change,
        resume_checkpoint=args.resume_checkpoint,
        auto_resume=args.auto_resume,
    )


if __name__ == "__main__":
    main()
