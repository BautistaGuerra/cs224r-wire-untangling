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
import time
from collections import deque

import numpy as np
import yaml
import torch

from robosuite.wrappers import GymWrapper

from wire_untangling.envs import StickReorderEnv
from wire_untangling.policies.rl.agent import TD3Agent
from wire_untangling.policies.policy_inference_wrappers import DPFMModelPolicy, ResidualRLPolicy
from wire_untangling.utils.normalizer import Normalizer, DEFAULT_SCALE_OBSERVATIONS, DEFAULT_SCALE_ACTIONS
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
    raw_env = rrl_env.make_rrl_gym_env_1stick(env_cfg)
    return GymWrapper(raw_env)


# ── Replay buffer ───────────────────────────────────────────────────────────

class ReplayBuffer:
    """Simple numpy replay buffer storing flat tensors."""

    def __init__(self, obs_dim: int, action_dim: int, max_size: int):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.action_base = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_action_base = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros(max_size, dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)
        self.gamma = np.zeros(max_size, dtype=np.float32)

    def add(self, obs, action, action_base, next_obs, next_action_base, reward, done, gamma):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.action_base[self.ptr] = action_base
        self.next_obs[self.ptr] = next_obs
        self.next_action_base[self.ptr] = next_action_base
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.gamma[self.ptr] = gamma

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32, device=device),
            "action": torch.tensor(self.action[idx], dtype=torch.float32, device=device),
            "action_base": torch.tensor(self.action_base[idx], dtype=torch.float32, device=device),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "next_action_base": torch.tensor(self.next_action_base[idx], dtype=torch.float32, device=device),
            "next_reward": torch.tensor(self.reward[idx], dtype=torch.float32, device=device),
            "next_done": torch.tensor(self.done[idx], dtype=torch.float32, device=device),
            "gamma": torch.tensor(self.gamma[idx], dtype=torch.float32, device=device),
        }


# ── Noise schedule ──────────────────────────────────────────────────────────

def linear_schedule(step: int, max_val: float, min_val: float, decay_steps: int) -> float:
    """Linearly decay from max_val to min_val over decay_steps."""
    frac = min(1.0, step / max(decay_steps, 1))
    return max_val + frac * (min_val - max_val)


# ── Training loop ───────────────────────────────────────────────────────────

def train(
    config: dict,
    dpfm_checkpoint: str,
    seed: int = None,
    use_wandb: bool = True,
    checkpoint_dir: str = "checkpoints/td3",
):
    td3_raw = config.get("residual_td3", {})
    env_cfg = config.get("env", {})

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    gym_env = make_gym_env(env_cfg)

    # Load and initialize base DPFM policy
    base_policy = DPFMModelPolicy(dpfm_checkpoint, gym_env)
    obs_norm = base_policy.obs_norm
    action_norm = base_policy.action_norm
    state_dim = int(base_policy.obs_norm.loc.shape[0])
    action_dim = int(base_policy.action_dim)

    # Create TD3 agent
    td3_raw["state_dim"] = (state_dim,)
    td3_raw["action_dim"] = (action_dim,)
    td3_cfg = DictConfig(td3_raw)
    agent = TD3Agent(td3_cfg)

    # Replay buffer holds transitions in normalized space
    replay_buffer = ReplayBuffer(
        obs_dim=state_dim, action_dim=action_dim, max_size=td3_cfg.replay_buffer_size
    )

    # Logging / checkpointing intervals
    log_freq = 1000
    eval_freq = 10_000

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Wandb initialization
    if use_wandb:
        import wandb
        run = wandb.init(
            project="cs224r-wire-untangling",
            config={**config, "seed": seed, "state_dim": state_dim, "action_dim": action_dim},
            tags=["residual-td3"],
        )
    else:
        run = None

    # ── Collect + train loop ────────────────────────────────────────────
    obs_raw, _ = gym_env.reset()
    base_policy.reset()

    global_step = 0
    episode_count = 0
    episode_reward = 0.0
    recent_rewards = deque(maxlen=20)
    critic_updates = 0
    train_start = time.time()


    print(f"Starting residual TD3 training for {td3_cfg.total_timesteps} steps")
    print(f"  state_dim={state_dim}, action_dim={action_dim}")
    print(f"  learning_starts={td3_cfg.learning_starts}, batch_size={td3_cfg.batch_size}")
    print(f"  critic_warmup={td3_cfg.critic_warmup_steps}, actor_update_every={td3_cfg.actor_update_every}")

    agent.eval()
    # Query the base policy once for the initial action; subsequent steps
    # reuse next_base_naction to avoid advancing the chunk index twice per step.
    base_naction = None
    stddev = td3_cfg.stddev_max

    while global_step < td3_cfg.total_timesteps:
        nobs = obs_norm.normalize(obs_raw)
        # Only query the base policy on the first step or after an episode reset
        if base_naction is None:
            _, base_naction = base_policy.predict_norm(obs_raw)

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
            stddev = linear_schedule(
                global_step - td3_cfg.learning_starts,
                td3_cfg.stddev_max, td3_cfg.stddev_min, td3_cfg.stddev_decay_steps,
            )
            with torch.no_grad():
                # Sample a residual action from the agent.
                residual_naction = agent.act(
                    torch.tensor(nobs, dtype=torch.float32, device=device),
                    torch.tensor(base_naction, dtype=torch.float32, device=device),
                    eval_mode=False,
                    stddev=stddev,
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
        next_nobs = obs_norm.normalize(next_obs_raw)
        if done:
            # Terminal: next base action irrelevant (discount will zero it out)
            next_base_naction = np.zeros(action_dim, dtype=np.float32)
        else:
            _, next_base_naction = base_policy.predict_norm(next_obs_raw)

        # Store the transition — everything is in z-score normalized space
        replay_buffer.add(
            obs=nobs,
            action=base_naction+residual_naction,
            action_base=base_naction,
            next_obs=next_nobs,
            next_action_base=next_base_naction,
            reward=reward,
            done=float(done),
            gamma=td3_cfg.gamma,
        )

        # Handle episode resets and carry forward the next base action
        if done:
            recent_rewards.append(episode_reward)
            episode_count += 1
            obs_raw, _ = gym_env.reset()
            base_policy.reset()
            episode_reward = 0.0
            base_naction = None
        else:
            obs_raw = next_obs_raw
            base_naction = next_base_naction

        global_step += 1

        # Gradient updates: start after warmup, every update_every_n_steps
        metrics = None
        if global_step >= td3_cfg.learning_starts and global_step % td3_cfg.update_every_n_steps == 0:
            # Allow std. deviation of actions to be high initially then diminish
            stddev = linear_schedule(
                global_step - td3_cfg.learning_starts,
                td3_cfg.stddev_max, td3_cfg.stddev_min, td3_cfg.stddev_decay_steps,
            )
            batch = replay_buffer.sample(td3_cfg.batch_size, device)

            # Delayed policy update: critic trains every step, actor only every N critic updates
            update_actor = (
                critic_updates >= td3_cfg.critic_warmup_steps
                and critic_updates % td3_cfg.actor_update_every == 0
            )
            agent.train()
            metrics = agent.update(batch, update_actor=update_actor, stddev=stddev)
            agent.eval()
            critic_updates += 1

        # Periodic console + wandb logging
        if global_step % log_freq == 0 and global_step > 0:
            elapsed = time.time() - train_start
            sps = int(global_step / elapsed)
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
                    "training/stddev": stddev if global_step >= td3_cfg.learning_starts else td3_cfg.stddev_max,
                    "training/SPS": sps,
                    "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
                    "buffer/size": replay_buffer.size,
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
            torch.save({
                "model_state_dict": agent.state_dict(),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "global_step": global_step,
                "obs_norm": obs_norm.state_dict(),
                "action_norm": action_norm.state_dict(),
            }, save_path)
            print(f"  Checkpoint saved: {save_path}")

    # ── Final save ──────────────────────────────────────────────────────
    save_path = os.path.join(checkpoint_dir, "td3_final.pt")
    torch.save({
        "model_state_dict": agent.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "global_step": global_step,
        "obs_norm": obs_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
    }, save_path)
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
                        help="Override num_sticks from env config (currently only 1 is supported)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/td3")
    args = parser.parse_args()

    cfg = load_config(args.env_config, args.td3_config)
    if args.num_sticks is not None:
        cfg["env"]["num_sticks"] = args.num_sticks
    seed = args.seed if args.seed is not None else 42

    train(
        cfg,
        dpfm_checkpoint=args.dpfm_checkpoint,
        seed=seed,
        use_wandb=not args.no_wandb,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
