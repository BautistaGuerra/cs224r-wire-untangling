"""End-to-end script: collect demos, train MLP-BC and DPFM, evaluate both.

Runs the full pipeline with default hyperparameters and reports success rates.
The pipeline has 5 stages:
  1. Collect expert demonstrations using the scripted pick-and-place policy
  2. Train an MLP behavior cloning (BC) policy on those demos
  3. Train a Diffusion Policy with Flow Matching (DPFM) on the same demos
  4. Evaluate MLP-BC in the single-stick environment and report success rate
  5. Evaluate DPFM in the same environment and report success rate

Usage:
    python -m scripts.e2e_train_eval
    python -m scripts.e2e_train_eval --num-demos 100 --eval-episodes 20
    python -m scripts.e2e_train_eval --skip-collect   # reuse existing demos
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from robosuite.wrappers import GymWrapper

from wire_untangling.utils.seeding import resolve_seed
from wire_untangling.policies.policy_inference_wrappers import (
    DPFMModelPolicy,
    MLPBCModelPolicy,
)
import scripts.rrl_env_creation as rrl_env


def load_env_config(path: str = "configs/stick_reorder.yaml") -> dict:
    """Load the 'env' section from the YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f).get("env", {})


def make_eval_env(env_cfg: dict):
    """Create a single-stick Robosuite environment wrapped in Gymnasium interface.

    Uses the same factory as residual RL training so that evaluation conditions
    match training conditions exactly (controller gains, horizon, thresholds).
    """
    raw_env = rrl_env.make_rrl_gym_env_1stick(env_cfg)
    return GymWrapper(raw_env)


def evaluate(policy, env_cfg: dict, n_episodes: int, label: str) -> dict:
    """Roll out a policy for n_episodes and compute aggregate metrics.

    Args:
        policy: Any object with .predict(obs) -> action and .reset().
                Works with MLPBCModelPolicy and DPFMModelPolicy.
        env_cfg: Environment config dict (num_sticks, thresholds, etc.).
        n_episodes: How many episodes to run.
        label: Human-readable name for this policy (used in printed output).

    Returns:
        Dict with success_rate, mean_reward, std_reward, mean_length.
    """
    env = make_eval_env(env_cfg)

    successes, rewards, lengths = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        # Reset internal policy state (e.g. DPFM's cached action chunk index)
        policy.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        last_info = {}

        while not done:
            # Policy produces a 7-dim action in [-1, 1] (OSC_POSE deltas + gripper)
            action = policy.predict(obs)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += r
            steps += 1
            last_info = info

        # "is_success" is set by StickReorderEnv when all sticks are within
        # success_threshold of their goals and yaw error < orientation_threshold
        successes.append(bool(last_info.get("is_success", False)))
        rewards.append(ep_reward)
        lengths.append(steps)

    env.close()

    success_rate = float(np.mean(successes))
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_length = float(np.mean(lengths))

    print(f"\n{'='*60}")
    print(f"  {label} — {n_episodes} episodes")
    print(f"  Success rate : {success_rate:.1%}")
    print(f"  Mean reward  : {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Mean length  : {mean_length:.1f}")
    print(f"{'='*60}\n")

    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
    }


def run_step(description: str, cmd: list[str]):
    """Run a training subprocess, print the command, and time it."""
    print(f"\n{'─'*60}")
    print(f"  STEP: {description}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'─'*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-demos", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-collect", action="store_true", default=True,
                        help="Skip demo collection (reuse existing data/demos.hdf5)")
    parser.add_argument("--demos-path", default="data/demos.hdf5")
    parser.add_argument("--env-config", default="configs/stick_reorder.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (e.g. cpu, cuda, cuda:0, cuda:1). Default: auto-detect")
    args = parser.parse_args()

    # Checkpoint paths for both policies — training scripts write here,
    # evaluation loads from the same paths
    bc_ckpt_dir = "checkpoints/mlp_bc"
    fm_ckpt_dir = "checkpoints/flow_matching"
    bc_ckpt = os.path.join(bc_ckpt_dir, "mlp_bc_policy.pt")
    fm_ckpt = os.path.join(fm_ckpt_dir, "flow_matching_policy.pt")

    # Load environment config and force single-stick mode for this pipeline
    args.seed = resolve_seed(args.seed)
    env_cfg = load_env_config(args.env_config)
    env_cfg["num_sticks"] = 1

    # ── 1. Collect demos ─────────────────────────────────────────────
    # Run the scripted expert policy to collect (obs, action) demonstration
    # pairs, saved to an HDF5 file. Each demo is one successful episode.
    if not args.skip_collect:
        run_step("Collect expert demonstrations", [
            sys.executable, "-m", "scripts.collect_demos",
            "--num-demos", str(args.num_demos),
            "--output", args.demos_path,
            "--seed", str(args.seed),
        ])
    else:
        print(f"\nSkipping demo collection, reusing {args.demos_path}")

    # ── 2. Train MLP-BC ──────────────────────────────────────────────
    # Trains a 3-layer MLP (256-256-256, ReLU, tanh output) with MSE loss
    # on single (state, action) pairs. Observations are z-score normalized.
    # Defaults: 500 epochs, lr=1e-3, batch_size=256.
    bc_cmd = [
        sys.executable, "-m", "scripts.train_bc",
        "--demos-path", args.demos_path,
        "--checkpoint-dir", bc_ckpt_dir,
        "--no-wandb",
        "--seed", str(args.seed),
    ]
    if args.device:
        bc_cmd += ["--device", args.device]
    run_step("Train MLP-BC policy", bc_cmd)

    # ── 3. Train DPFM ────────────────────────────────────────────────
    # Trains a flow matching policy with a temporal 1D U-Net that predicts
    # action chunks (horizon=10). Uses conditional OT interpolation and
    # MSE on velocity targets. Defaults: 200 epochs, lr=1e-4, batch_size=2048.
    fm_cmd = [
        sys.executable, "-m", "scripts.train_flow_matching",
        "--demos-path", args.demos_path,
        "--checkpoint-dir", fm_ckpt_dir,
        "--no-wandb",
        "--seed", str(args.seed),
    ]
    if args.device:
        fm_cmd += ["--device", args.device]
    run_step("Train Flow Matching (DPFM) policy", fm_cmd)

    # ── 4. Evaluate MLP-BC ───────────────────────────────────────────
    # Load the trained MLP-BC checkpoint. MLPBCModelPolicy handles
    # observation normalization internally using saved corpus statistics.
    print("\nLoading MLP-BC checkpoint for evaluation...")
    bc_policy = MLPBCModelPolicy(bc_ckpt, device=args.device)
    bc_results = evaluate(bc_policy, env_cfg, args.eval_episodes, "MLP-BC")

    # ── 5. Evaluate DPFM ─────────────────────────────────────────────
    # Load the trained DPFM checkpoint. DPFMModelPolicy manages action
    # chunk caching: it samples a full chunk, executes execute_steps
    # actions from it, then re-plans from the current observation.
    # Needs a gym_env reference for action space bounds.
    print("Loading DPFM checkpoint for evaluation...")
    fm_env = make_eval_env(env_cfg)
    fm_policy = DPFMModelPolicy(fm_ckpt, fm_env, device=args.device)
    fm_env.close()
    fm_results = evaluate(fm_policy, env_cfg, args.eval_episodes, "DPFM (Flow Matching)")

    # ── Summary ──────────────────────────────────────────────────────
    # Print a side-by-side comparison table of both policies
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON ({args.eval_episodes} episodes each)")
    print(f"{'='*60}")
    print(f"  {'Policy':<25} {'Success':>10} {'Reward':>15}")
    print(f"  {'─'*25} {'─'*10} {'─'*15}")
    print(f"  {'MLP-BC':<25} {bc_results['success_rate']:>9.1%} {bc_results['mean_reward']:>15.2f}")
    print(f"  {'DPFM (Flow Matching)':<25} {fm_results['success_rate']:>9.1%} {fm_results['mean_reward']:>15.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
