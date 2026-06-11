"""
Sanity-check and visualization script: instantiate StickReorderEnv,
run random, trained-policy, or scripted expert actions, and optionally render.

Usage:
    # Headless random actions — just checks everything loads and steps correctly
    python scripts/play_env.py

    # Random actions with MuJoCo viewer (Linux: python, macOS: mjpython)
    python scripts/play_env.py --render
    python scripts/play_env.py --render --fps 20

    # Visualize a trained policy
    python scripts/play_env.py --render --checkpoint checkpoints/best/best_model.zip

    # Visualize the scripted expert policy (single stick)
    python scripts/play_env.py --render --expert
    python scripts/play_env.py --expert --episodes 10   # headless success rate check

    # Record video to disk (no GUI needed)
    python scripts/play_env.py --record videos/expert_demo.mp4 --expert
    python scripts/play_env.py --record videos/policy.mp4 --checkpoint checkpoints/best/best_model.zip

    # Wrap as Gymnasium env and print observation/action spaces
    python scripts/play_env.py --gym
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import imageio
import numpy as np
import yaml
from robosuite.wrappers import GymWrapper

from wire_untangling.envs import StickReorderEnv
from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map
from wire_untangling.utils.stick_order import StickOrderScheduler

from wire_untangling.policies.policy_inference_wrappers import (
    DPFMModelPolicy,
    MLPBCModelPolicy,
    ModelPolicy,
    ResidualRLPolicy,
    SACModelPolicy,
)


def _make_writer(path: str, fps: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=8)


def _grab_frame(env):
    return env.sim.render(width=1280, height=720, camera_name="frontview")[::-1]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _collect_stick_metrics(env) -> dict:
    metrics = {}
    goal_positions = getattr(env, "_goal_positions", None)
    for i, body_id in enumerate(getattr(env, "stick_body_ids", [])):
        pos = env.sim.data.body_xpos[body_id]
        dist = float(np.linalg.norm(pos - goal_positions[i])) if goal_positions is not None else np.nan
        yaw_err = float(env._yaw_error(body_id)) if hasattr(env, "_yaw_error") else np.nan
        placed = bool(
            dist <= getattr(env, "success_threshold", np.inf)
            and yaw_err <= getattr(env, "orientation_threshold", np.inf)
        )
        metrics[f"stick{i}_dist"] = dist
        metrics[f"stick{i}_yaw_err"] = yaw_err
        metrics[f"stick{i}_placed"] = int(placed)
    return metrics


def make_env(
    render: bool = False,
    record: bool = False,
    num_sticks: int | None = None,
    env_cfg: dict | None = None,
):
    env_cfg = dict(env_cfg or {})
    if num_sticks is not None:
        env_cfg["num_sticks"] = num_sticks

    kwargs = dict(
        robots=env_cfg.get("robot", "Panda"),
        num_sticks=env_cfg.get("num_sticks", 3),
        stick_length=env_cfg.get("stick_length", 0.20),
        stick_radius=env_cfg.get("stick_radius", 0.0075),
        goal_spacing=env_cfg.get("goal_spacing", 0.06),
        success_threshold=env_cfg.get("success_threshold", 0.03),
        orientation_threshold=env_cfg.get("orientation_threshold", np.deg2rad(10.0)),
        lambda_rot=env_cfg.get("lambda_rot", 0.1),
        goal_yaw=env_cfg.get("goal_yaw", 0.0),
        reward_shaping=env_cfg.get("reward_shaping", True),
        success_bonus=env_cfg.get("success_bonus", 1.0),
        terminate_on_success=env_cfg.get("terminate_on_success", True),
        has_renderer=render,
        has_offscreen_renderer=record,
        use_camera_obs=False,
        control_freq=20,
        horizon=env_cfg.get("horizon", 500),
        camera_names="agentview",
        camera_heights=720,
        camera_widths=1280,
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

    return StickReorderEnv(
        **kwargs,
    )


def run_random(env, n_episodes: int = 2, render: bool = False, fps: int = 20, record_path: str = None):
    sleep_time = 1.0 / fps if render else 0.0
    writer = _make_writer(record_path, fps) if record_path else None

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        for i, body_id in enumerate(env.stick_body_ids):
            pos = env.sim.data.body_xpos[body_id]
            print(f"  stick{i} initial pos: {pos}")

        while not done:
            low, high = env.action_spec
            action = np.random.uniform(low, high)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            if render:
                env.render()
                if sleep_time:
                    time.sleep(sleep_time)
            if writer:
                writer.append_data(_grab_frame(env))

        print(f"Episode {ep + 1}: steps={step}  total_reward={total_reward:.3f}  success={info.get('success', False)}")

    if writer:
        writer.close()
        print(f"Video saved to {record_path}")
    env.close()


def run_policy(
    env,
    policy: ModelPolicy,
    n_episodes: int = 2,
    render: bool = False,
    fps: int = 20,
    record_path: str = None,
    expert_cfg: dict | None = None,
    results_file: str = None,
    results_metadata: dict | None = None,
    step_diagnostics_file: str = None,
):
    """Run a trained policy in the environment and report success rate.
    Uses GymWrapper to produce the flat obs vector the policy expects,
    while keeping the underlying Robosuite renderer active."""
    gym_env = GymWrapper(env)
    expert_cfg = dict(expert_cfg or {})
    order_schedule = StickOrderScheduler(expert_cfg, env.num_sticks)
    if hasattr(policy, "set_gym_env"):
        policy.set_gym_env(gym_env, expert_cfg=expert_cfg)
    sleep_time = 1.0 / fps if render else 0.0
    writer = _make_writer(record_path, fps) if record_path else None

    successes = 0
    successes_by_order: dict[tuple[int, ...], int] = {}
    attempts_by_order: dict[tuple[int, ...], int] = {}
    total_rewards = []
    episode_records = []
    step_writer = None
    step_fh = None
    if step_diagnostics_file:
        os.makedirs(os.path.dirname(step_diagnostics_file) or ".", exist_ok=True)
        step_fh = open(step_diagnostics_file, "w", newline="")
        step_fields = [
            "episode",
            "timestep",
            "order_str",
            "reward",
            "cumulative_reward",
            "terminated",
            "truncated",
            "success",
            "q_final_mean",
            "q_final_min",
            "q_base_mean",
            "q_base_min",
            "q_advantage_mean",
            "q_advantage_min",
            "residual_l1",
            "residual_l2",
            "base_action_l2",
            "final_action_l2",
            "phase",
            "active_stick",
        ]
        for i in range(env.num_sticks):
            step_fields.extend([f"stick{i}_dist", f"stick{i}_yaw_err", f"stick{i}_placed"])
        step_writer = csv.DictWriter(step_fh, fieldnames=step_fields)
        step_writer.writeheader()

    for ep in range(n_episodes):
        stick_order = order_schedule.order_for(ep)
        attempts_by_order[stick_order] = attempts_by_order.get(stick_order, 0) + 1
        obs, _ = gym_env.reset()
        policy.reset(stick_order=stick_order)
        total_reward = 0.0
        done = False
        step = 0

        for i, body_id in enumerate(env.stick_body_ids):
            pos = env.sim.data.body_xpos[body_id]
            print(f"  stick{i} initial pos: {pos}")

        while not done:
            diagnostics = {}
            if step_writer is not None and hasattr(policy, "predict_with_diagnostics"):
                action, diagnostics = policy.predict_with_diagnostics(obs)
            else:
                action = policy.predict(obs)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            if step_writer is not None:
                row = {
                    "episode": ep + 1,
                    "timestep": step,
                    "order_str": StickOrderScheduler.format_order(stick_order),
                    "reward": float(reward),
                    "cumulative_reward": float(total_reward),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "success": int(info.get("is_success", False)),
                }
                row.update(diagnostics)
                row.update(_collect_stick_metrics(env))
                step_writer.writerow(row)

            if render:
                env.render()
                if sleep_time:
                    time.sleep(sleep_time)
            if writer:
                writer.append_data(_grab_frame(env))

        success = info.get("is_success", False)
        successes += int(success)
        if success:
            successes_by_order[stick_order] = successes_by_order.get(stick_order, 0) + 1
        total_rewards.append(total_reward)
        episode_records.append(
            {
                "episode": ep + 1,
                "order": list(stick_order),
                "order_str": StickOrderScheduler.format_order(stick_order),
                "steps": int(step),
                "total_reward": float(total_reward),
                "success": bool(success),
            }
        )
        print(
            f"Episode {ep + 1}: order={StickOrderScheduler.format_order(stick_order)} "
            f"steps={step}  total_reward={total_reward:.3f}  success={success}"
        )

    mean_reward = float(np.mean(total_rewards))
    std_reward = float(np.std(total_rewards))
    success_rate = float(successes / n_episodes)
    print(f"\nSuccess rate: {successes}/{n_episodes} ({success_rate:.0%})")
    print("Per-order success:")
    per_order_records = []
    for order in order_schedule.order_choices:
        order_successes = successes_by_order.get(order, 0)
        order_attempts = attempts_by_order.get(order, 0)
        order_rate = order_successes / order_attempts if order_attempts else 0.0
        per_order_records.append(
            {
                "order": list(order),
                "order_str": StickOrderScheduler.format_order(order),
                "successes": int(order_successes),
                "attempts": int(order_attempts),
                "success_rate": float(order_rate),
            }
        )
        print(
            f"  {StickOrderScheduler.format_order(order)}: "
            f"{order_successes}/{order_attempts} ({order_rate:.0%})"
        )
    print(f"Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    if hasattr(policy, "context_diagnostics"):
        diag = policy.context_diagnostics()
        if int(diag.get("steps", 0)):
            print(
                "Learned-vs-oracle context disagreement: "
                f"phase={diag['phase_disagreements']}/{diag['steps']} "
                f"({diag['phase_disagreement_rate']:.1%}), "
                f"active={diag['active_stick_disagreements']}/{diag['steps']} "
                f"({diag['active_stick_disagreement_rate']:.1%}), "
                f"joint={diag['joint_disagreements']}/{diag['steps']} "
                f"({diag['joint_disagreement_rate']:.1%})"
            )

    summary = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **dict(results_metadata or {}),
        },
        "aggregate": {
            "episodes": int(n_episodes),
            "successes": int(successes),
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_steps": float(np.mean([record["steps"] for record in episode_records])),
            "std_steps": float(np.std([record["steps"] for record in episode_records])),
        },
        "per_order": per_order_records,
        "episodes": episode_records,
        "artifacts": {},
    }
    if step_diagnostics_file:
        summary["artifacts"]["step_diagnostics_csv"] = step_diagnostics_file

    if results_file:
        os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
        with open(results_file, "w") as f:
            f.write(f"success_rate: {success_rate:.4f}\n")
            f.write(f"successes: {successes}/{n_episodes}\n")
            f.write(f"mean_reward: {mean_reward:.4f}\n")
            f.write(f"std_reward: {std_reward:.4f}\n")
        print(f"Results saved to {results_file}")
        summary["artifacts"]["text"] = results_file

        results_path = Path(results_file)
        json_path = results_path.with_suffix(".json")
        episodes_path = results_path.with_name(f"{results_path.stem}_episodes.csv")
        summary["artifacts"]["json"] = str(json_path)
        summary["artifacts"]["episodes_csv"] = str(episodes_path)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        with open(episodes_path, "w", newline="") as f:
            episode_writer = csv.DictWriter(
                f,
                fieldnames=["episode", "order_str", "steps", "total_reward", "success"],
            )
            episode_writer.writeheader()
            for record in episode_records:
                episode_writer.writerow(
                    {
                        "episode": record["episode"],
                        "order_str": record["order_str"],
                        "steps": record["steps"],
                        "total_reward": f"{record['total_reward']:.6f}",
                        "success": int(record["success"]),
                    }
                )
        print(f"Structured results saved to {json_path}")
        print(f"Episode results saved to {episodes_path}")

    if step_fh is not None:
        step_fh.close()
        print(f"Step diagnostics saved to {step_diagnostics_file}")

    if writer:
        writer.close()
        print(f"Video saved to {record_path}")
    gym_env.close()
    return summary


def run_expert(
    env,
    n_episodes: int = 2,
    render: bool = False,
    fps: int = 20,
    record_path: str = None,
    expert_cfg: dict | None = None,
):
    """Run the scripted pick-and-place expert policy.
    Uses GymWrapper for flat observations + underlying Robosuite renderer."""
    gym_env = GymWrapper(env)
    obs_map = build_obs_index_map(gym_env)
    expert_cfg = dict(expert_cfg or {})
    order_schedule = StickOrderScheduler(expert_cfg, env.num_sticks)
    expert = PickPlaceExpertPolicy(
        obs_map,
        goal_yaw=getattr(env, "goal_yaw", 0.0),
        # TODO(alexta): previously, it was stick_order=expert_cfg.get("stick_order"),
        # In both main and dev branches. Not sure where "stick_order" was coming from.
        stick_order=order_schedule.order_for(0),
    )
    sleep_time = 1.0 / fps if render else 0.0
    writer = _make_writer(record_path, fps) if record_path else None

    successes = 0
    successes_by_order: dict[tuple[int, ...], int] = {}
    attempts_by_order: dict[tuple[int, ...], int] = {}
    for ep in range(n_episodes):
        stick_order = order_schedule.order_for(ep)
        attempts_by_order[stick_order] = attempts_by_order.get(stick_order, 0) + 1
        obs, _ = gym_env.reset()
        expert.reset(stick_order=stick_order)
        total_reward = 0.0
        done = False
        step = 0

        for i, body_id in enumerate(env.stick_body_ids):
            pos = env.sim.data.body_xpos[body_id]
            print(f"  stick{i} initial pos: {pos}")

        while not done:
            action, _ = expert.predict(obs)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            if render:
                env.render()
                if sleep_time:
                    time.sleep(sleep_time)
            if writer:
                writer.append_data(_grab_frame(env))

        success = info.get("is_success", False)
        successes += int(success)
        if success:
            successes_by_order[stick_order] = successes_by_order.get(stick_order, 0) + 1
        print(
            f"Episode {ep + 1}: order={StickOrderScheduler.format_order(stick_order)} "
            f"steps={step}  total_reward={total_reward:.3f}  "
            f"success={success}  phase={expert._phase.name}"
        )

    print(f"\nSuccess rate: {successes}/{n_episodes} ({successes/n_episodes:.0%})")
    print("Per-order success:")
    for order in order_schedule.order_choices:
        order_successes = successes_by_order.get(order, 0)
        order_attempts = attempts_by_order.get(order, 0)
        order_rate = order_successes / order_attempts if order_attempts else 0.0
        print(
            f"  {StickOrderScheduler.format_order(order)}: "
            f"{order_successes}/{order_attempts} ({order_rate:.0%})"
        )
    if writer:
        writer.close()
        print(f"Video saved to {record_path}")
    gym_env.close()


def print_gym_spaces(env):
    """Wrap in GymWrapper to show what SB3 sees: flat observation and action spaces."""
    gym_env = GymWrapper(env)
    print("Observation space:", gym_env.observation_space)
    print("Action space:     ", gym_env.action_space)
    gym_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer (use mjpython on macOS)")
    parser.add_argument("--record", type=str, default=None, metavar="PATH", help="Save video to .mp4 file (offscreen, no GUI needed)")
    parser.add_argument("--fps", type=int, default=20, help="Target render FPS (default 20)")
    parser.add_argument("--gym", action="store_true", help="Print Gymnasium spaces")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--results-file", type=str, default=None,
                        help="Save success rate and reward stats to a text file")
    parser.add_argument("--step-diagnostics-file", type=str, default=None,
                        help="Save per-timestep eval diagnostics to a CSV file")
    parser.add_argument("--config", default=None,
                        help="Optional YAML config for env/expert settings, e.g. configs/stick_reorder_n2.yaml")
    parser.add_argument("--bc_checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint for trained MLP-BC policy")
    parser.add_argument("--context-predictor-checkpoint", type=str, default=None,
                        help="Optional learned context predictor for phase-active MLP-BC")
    parser.add_argument("--compare-oracle-context", action="store_true",
                        help="Track oracle-vs-learned context disagreement while MLP-BC consumes learned context")
    parser.add_argument("--sac_checkpoint", type=str, default=None, help="Path to SB3 .zip checkpoint for trained policy")
    parser.add_argument("--dpfm_checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint for trained DPFM policy")
    parser.add_argument("--dpfm-execute-steps", type=int, default=None,
                        help="Override DPFM chunk actions executed before re-planning")
    parser.add_argument("--dpfm-stochastic", action="store_true", default=True,
                        help="Use random Flow Matching initial noise instead of deterministic zero-noise sampling")
    parser.add_argument("--dpfm-deterministic", dest="dpfm_stochastic", action="store_false",
                        help="Use zero initial noise for deterministic Flow Matching sampling")
    parser.add_argument("--dpfm-replan-on-context-change", action="store_true",
                        help=("For phase-active DPFM, discard cached actions and re-sample "
                              "when the tracked (phase, active_stick) changes."))
    parser.add_argument("--rrl-checkpoint", type=str, default=None,
                        help="Path to residual RL (TD3) checkpoint (.pt)")
    parser.add_argument("--rrl-config", type=str, default=None,
                        help="Path to residual TD3 YAML config (optional for checkpoints that embed config)")
    parser.add_argument("--action-scale", type=float, default=None,
                        help="Override actor.action_scale for the RRL policy (only needed with --rrl-config)")
    parser.add_argument("--expert", action="store_true", help="Run scripted pick-and-place expert (single stick)")
    parser.add_argument("--num-sticks", type=int, default=None, help="Override number of sticks")
    parser.add_argument("--reward-shaping", action=argparse.BooleanOptionalAction, default=None,
                        help="Override reward_shaping from env config (--reward-shaping / --no-reward-shaping)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (e.g. cpu, cuda, cuda:0, cuda:1). Default: auto-detect")
    args = parser.parse_args()
    if args.context_predictor_checkpoint and not args.bc_checkpoint:
        parser.error("--context-predictor-checkpoint is valid only with --bc_checkpoint")
    if args.compare_oracle_context and not args.context_predictor_checkpoint:
        parser.error("--compare-oracle-context requires --context-predictor-checkpoint")

    cfg = load_config(args.config) if args.config else {}
    env_cfg = cfg.get("env", {})
    expert_cfg = cfg.get("expert", {})

    if args.reward_shaping is not None:
        env_cfg["reward_shaping"] = args.reward_shaping

    if args.num_sticks is not None:
        num_sticks = args.num_sticks
    elif args.config:
        num_sticks = None
    else:
        # Preserve the old one-stick default for BC / expert checkpoint smoke tests.
        num_sticks = 1 if (args.expert or args.dpfm_checkpoint or args.bc_checkpoint or args.rrl_checkpoint) else 3
    env = make_env(
        render=args.render,
        record=bool(args.record),
        num_sticks=num_sticks,
        env_cfg=env_cfg,
    )

    if args.gym:
        print_gym_spaces(env)
    elif args.expert:
        run_expert(
            env,
            n_episodes=args.episodes,
            render=args.render,
            fps=args.fps,
            record_path=args.record,
            expert_cfg=expert_cfg,
        )
    elif args.bc_checkpoint:
        policy = MLPBCModelPolicy(args.bc_checkpoint, env, device=args.device)
        run_policy(
            env,
            policy,
            n_episodes=args.episodes,
            render=args.render,
            fps=args.fps,
            record_path=args.record,
            expert_cfg=expert_cfg,
            results_file=args.results_file,
            step_diagnostics_file=args.step_diagnostics_file,
            context_predictor_checkpoint=args.context_predictor_checkpoint,
            compare_oracle_context=args.compare_oracle_context,
        )
    elif args.sac_checkpoint:
        policy = SACModelPolicy(args.sac_checkpoint, env)
        run_policy(
            env,
            policy,
            n_episodes=args.episodes,
            render=args.render,
            fps=args.fps,
            record_path=args.record,
            expert_cfg=expert_cfg,
            results_file=args.results_file,
            step_diagnostics_file=args.step_diagnostics_file,
        )
    elif args.rrl_checkpoint:
        if not args.dpfm_checkpoint:
            parser.error("--rrl-checkpoint requires --dpfm_checkpoint (the base DPFM policy)")
        base_policy = DPFMModelPolicy(
            args.dpfm_checkpoint,
            env,
            execute_steps=args.dpfm_execute_steps,
            stochastic=args.dpfm_stochastic,
            replan_on_context_change=args.dpfm_replan_on_context_change,
            device=args.device,
        )
        rrl_cfg = None
        if args.rrl_config:
            from scripts.train_residual_rl import DictConfig
            with open(args.rrl_config) as f:
                rrl_raw = yaml.safe_load(f).get("residual_td3", {})
            if args.action_scale is not None:
                rrl_raw.setdefault("actor", {})["action_scale"] = args.action_scale
            rrl_cfg = DictConfig(rrl_raw)
        policy = ResidualRLPolicy(
            rl_model_path=args.rrl_checkpoint,
            base_model_path=args.dpfm_checkpoint,
            base_policy=base_policy,
            gym_env=env,
            rrl_cfg=rrl_cfg,
            device=args.device,
        )
        run_policy(
            env,
            policy,
            n_episodes=args.episodes,
            render=args.render,
            fps=args.fps,
            record_path=args.record,
            expert_cfg=expert_cfg,
            results_file=args.results_file,
            step_diagnostics_file=args.step_diagnostics_file,
        )
    elif args.dpfm_checkpoint:
        policy = DPFMModelPolicy(
            args.dpfm_checkpoint,
            env,
            execute_steps=args.dpfm_execute_steps,
            stochastic=args.dpfm_stochastic,
            replan_on_context_change=args.dpfm_replan_on_context_change,
            device=args.device,
        )
        run_policy(
            env,
            policy,
            n_episodes=args.episodes,
            render=args.render,
            fps=args.fps,
            record_path=args.record,
            expert_cfg=expert_cfg,
            results_file=args.results_file,
            step_diagnostics_file=args.step_diagnostics_file,
        )
    else:
        run_random(env, n_episodes=args.episodes, render=args.render, fps=args.fps, record_path=args.record)
