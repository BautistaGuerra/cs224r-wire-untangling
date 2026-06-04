"""
Modal entrypoint for headless policy evaluation.

Examples:
    # Evaluate an N=2 residual-RL checkpoint from the Modal checkpoint volume.
    modal run modal_eval_policy.py \
        --config configs/stick_reorder_n2_random_order.yaml \
        --rrl-checkpoint checkpoints/rrl_n2_random_obs_h4_e1_i10_as0.2_of0.5_e1m/td3_step360000.pt \
        --dpfm-checkpoint checkpoints/dpfm_pr14_n2_random_obs_zscore_h4_e1_i10/flow_matching_policy.pt \
        --episodes 100

    # Evaluate the frozen DPFM base policy only.
    modal run modal_eval_policy.py \
        --config configs/stick_reorder_n2_random_order.yaml \
        --dpfm-checkpoint checkpoints/dpfm_pr14_n2_random_obs_zscore_h4_e1_i10/flow_matching_policy.pt \
        --episodes 100
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import modal
import yaml


DEFAULT_CONFIG = "configs/stick_reorder_n2_random_order.yaml"
DEFAULT_DPFM_CHECKPOINT = (
    "checkpoints/dpfm_pr14_n2_random_obs_zscore_h4_e1_i10/flow_matching_policy.pt"
)
CHECKPOINT_MOUNT = "/checkpoints"


with open(DEFAULT_CONFIG) as _f:
    _cfg = yaml.safe_load(_f)
_modal_cfg = _cfg.get("modal", {})
_GPU_TYPE = os.environ.get("MODAL_GPU", _modal_cfg.get("gpu", "A10G"))
_CPU_CORES = float(os.environ.get("MODAL_CPU", _modal_cfg.get("eval_cpu", 4.0)))
_TIMEOUT = _modal_cfg.get("eval_timeout", max(int(_modal_cfg.get("timeout", 3 * 3600)), 6 * 3600))


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libosmesa6",
        "libglfw3",
        "patchelf",
    )
    .pip_install(
        "robosuite>=1.4.0",
        "mujoco>=3.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchrl",
        "tensordict",
        "stable-baselines3>=2.0.0",
        "pyyaml",
        "h5py",
        "imageio",
        "wandb",
    )
    .run_commands("pip uninstall -y triton")
    .env(
        {
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "TORCH_COMPILE_DISABLE": "1",
        }
    )
    .add_local_python_source("wire_untangling")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_dir("configs", remote_path="/root/configs")
)

app = modal.App("cs224r-wire-untangling-eval", image=image)

checkpoint_volume = modal.Volume.from_name("cs224r-checkpoints", create_if_missing=True)


def _under_mount(path: str, mount: str) -> bool:
    return path == mount or path.startswith(f"{mount}/")


def _volume_path(path: str, mount: str, local_prefix: str) -> str:
    if not path:
        return ""
    if _under_mount(path, mount):
        return path
    if path.startswith("/"):
        raise ValueError(f"Remote path must live under {mount}; got {path!r}.")

    rel = path
    prefix = f"{local_prefix}/"
    if rel.startswith(prefix):
        rel = rel[len(prefix):]
    return f"{mount}/{rel}"


def _checkpoint_step(path: str) -> int | None:
    match = re.fullmatch(r"td3_step(\d+)\.pt", os.path.basename(path))
    return int(match.group(1)) if match else None


def _latest_td3_checkpoint(directory: str) -> str | None:
    candidates: list[tuple[int, str]] = []
    for name in os.listdir(directory):
        step = _checkpoint_step(name)
        if step is not None:
            candidates.append((step, os.path.join(directory, name)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _resolve_dpfm_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "flow_matching_policy.pt")
        if os.path.exists(candidate):
            print(f"Resolved DPFM checkpoint directory to {candidate}")
            return candidate
    return path


def _resolve_rrl_checkpoint(path: str) -> str:
    if path and os.path.isdir(path):
        latest = _latest_td3_checkpoint(path)
        if latest:
            print(f"Resolved RRL checkpoint directory to {latest}")
            return latest
    return path


def _default_results_file(
    rrl_checkpoint: str,
    dpfm_checkpoint: str,
    episodes: int,
) -> str:
    checkpoint = rrl_checkpoint or dpfm_checkpoint
    checkpoint_path = Path(checkpoint)
    return str(checkpoint_path.with_name(f"eval_{checkpoint_path.stem}_episodes{episodes}.txt"))


@app.function(
    gpu=_GPU_TYPE,
    cpu=_CPU_CORES,
    timeout=_TIMEOUT,
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def eval_policy_remote(
    config: str,
    dpfm_checkpoint: str,
    rrl_checkpoint: str,
    rrl_config: str,
    episodes: int,
    seed: int | None,
    dpfm_stochastic: bool,
    dpfm_execute_steps: int | None,
    dpfm_replan_on_context_change: bool,
    results_file: str,
    step_diagnostics_file: str,
    use_wandb: bool,
    wandb_project: str,
    wandb_run_id: str,
    wandb_name: str,
    wandb_step: int | None,
):
    import os
    import random
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    os.chdir("/root")

    checkpoint_volume.reload()

    dpfm_checkpoint = _resolve_dpfm_checkpoint(dpfm_checkpoint)
    rrl_checkpoint = _resolve_rrl_checkpoint(rrl_checkpoint)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if not os.path.exists(dpfm_checkpoint):
        raise FileNotFoundError(f"DPFM checkpoint not found on Modal: {dpfm_checkpoint}")
    if os.path.isdir(dpfm_checkpoint):
        raise IsADirectoryError(
            f"DPFM checkpoint path is a directory and no flow_matching_policy.pt was found: {dpfm_checkpoint}"
        )
    if rrl_checkpoint and not os.path.exists(rrl_checkpoint):
        raise FileNotFoundError(f"RRL checkpoint not found on Modal: {rrl_checkpoint}")
    if rrl_checkpoint and os.path.isdir(rrl_checkpoint):
        raise IsADirectoryError(
            f"RRL checkpoint path is a directory and no td3_step*.pt was found: {rrl_checkpoint}"
        )
    if rrl_config and not os.path.exists(rrl_config):
        raise FileNotFoundError(f"RRL config not found in image: {rrl_config}")

    from scripts.play_env import load_config, make_env, run_policy
    from wire_untangling.policies.policy_inference_wrappers import DPFMModelPolicy, ResidualRLPolicy

    cfg = load_config(config)
    env_cfg = cfg.get("env", {})
    expert_cfg = cfg.get("expert", {})
    env = make_env(render=False, record=False, num_sticks=None, env_cfg=env_cfg)

    base_policy = DPFMModelPolicy(
        dpfm_checkpoint,
        env,
        execute_steps=dpfm_execute_steps,
        stochastic=dpfm_stochastic,
        replan_on_context_change=dpfm_replan_on_context_change,
    )
    if rrl_checkpoint:
        rrl_cfg = None
        if rrl_config:
            from scripts.train_residual_rl import DictConfig

            with open(rrl_config) as f:
                rrl_raw = yaml.safe_load(f).get("residual_td3", {})
            rrl_cfg = DictConfig(rrl_raw)
        policy = ResidualRLPolicy(
            rl_model_path=rrl_checkpoint,
            base_model_path=dpfm_checkpoint,
            base_policy=base_policy,
            gym_env=env,
            rrl_cfg=rrl_cfg,
        )
    else:
        policy = base_policy

    checkpoint_step = wandb_step
    if checkpoint_step is None and rrl_checkpoint:
        checkpoint_step = _checkpoint_step(rrl_checkpoint)
    metadata = {
        "config": config,
        "dpfm_checkpoint": dpfm_checkpoint,
        "rrl_checkpoint": rrl_checkpoint,
        "episodes": episodes,
        "seed": seed,
        "dpfm_stochastic": dpfm_stochastic,
        "dpfm_execute_steps": dpfm_execute_steps,
        "dpfm_replan_on_context_change": dpfm_replan_on_context_change,
        "checkpoint_step": checkpoint_step,
    }

    summary = run_policy(
        env,
        policy,
        n_episodes=episodes,
        render=False,
        fps=20,
        record_path=None,
        expert_cfg=expert_cfg,
        results_file=results_file,
        results_metadata=metadata,
        step_diagnostics_file=step_diagnostics_file,
    )

    if use_wandb:
        import wandb

        if wandb_run_id:
            os.environ["WANDB_RUN_ID"] = wandb_run_id
            os.environ["WANDB_RESUME"] = "allow"
        run = wandb.init(
            project=wandb_project,
            name=wandb_name or Path(results_file).stem,
            id=wandb_run_id or None,
            resume="allow" if wandb_run_id else None,
            config=metadata,
            tags=["eval", "residual-td3" if rrl_checkpoint else "dpfm"],
        )
        aggregate = summary["aggregate"]
        log_dict = {
            "eval/success_rate": aggregate["success_rate"],
            "eval/successes": aggregate["successes"],
            "eval/episodes": aggregate["episodes"],
            "eval/mean_reward": aggregate["mean_reward"],
            "eval/std_reward": aggregate["std_reward"],
            "eval/mean_steps": aggregate["mean_steps"],
            "eval/std_steps": aggregate["std_steps"],
        }
        for order_record in summary["per_order"]:
            order_key = order_record["order_str"].replace("[", "").replace("]", "").replace(", ", "_")
            log_dict[f"eval/order_{order_key}_success_rate"] = order_record["success_rate"]
            log_dict[f"eval/order_{order_key}_successes"] = order_record["successes"]
            log_dict[f"eval/order_{order_key}_attempts"] = order_record["attempts"]
        per_order_table = wandb.Table(
            columns=["order", "successes", "attempts", "success_rate"],
            data=[
                [
                    record["order_str"],
                    record["successes"],
                    record["attempts"],
                    record["success_rate"],
                ]
                for record in summary["per_order"]
            ],
        )
        episode_table = wandb.Table(
            columns=["episode", "order", "steps", "total_reward", "success"],
            data=[
                [
                    record["episode"],
                    record["order_str"],
                    record["steps"],
                    record["total_reward"],
                    record["success"],
                ]
                for record in summary["episodes"]
            ],
        )
        log_dict["eval/per_order_table"] = per_order_table
        log_dict["eval/episode_table"] = episode_table
        if checkpoint_step is not None:
            wandb.log(log_dict, step=checkpoint_step)
        else:
            wandb.log(log_dict)
        for artifact_path in summary.get("artifacts", {}).values():
            if artifact_path and os.path.exists(artifact_path):
                wandb.save(artifact_path)
        run.finish()
    checkpoint_volume.commit()


@app.local_entrypoint()
def main(
    config: str = DEFAULT_CONFIG,
    dpfm_checkpoint: str = DEFAULT_DPFM_CHECKPOINT,
    rrl_checkpoint: str = "",
    rrl_config: str = "",
    episodes: int = 100,
    seed: int | None = 42,
    dpfm_execute_steps: int | None = None,
    dpfm_deterministic: bool = False,
    dpfm_replan_on_context_change: bool = False,
    results_file: str = "",
    save_step_diagnostics: bool = False,
    step_diagnostics_file: str = "",
    wandb: bool = False,
    wandb_project: str = "cs224r-wire-untangling",
    wandb_run_id: str = "",
    wandb_name: str = "",
    wandb_step: int | None = None,
):
    remote_dpfm_checkpoint = _volume_path(dpfm_checkpoint, CHECKPOINT_MOUNT, "checkpoints")
    remote_rrl_checkpoint = _volume_path(rrl_checkpoint, CHECKPOINT_MOUNT, "checkpoints")
    remote_results_file = (
        _volume_path(results_file, CHECKPOINT_MOUNT, "checkpoints")
        if results_file
        else _default_results_file(remote_rrl_checkpoint, remote_dpfm_checkpoint, episodes)
    )
    if step_diagnostics_file:
        remote_step_diagnostics_file = _volume_path(
            step_diagnostics_file,
            CHECKPOINT_MOUNT,
            "checkpoints",
        )
    elif save_step_diagnostics:
        results_path = Path(remote_results_file)
        remote_step_diagnostics_file = str(results_path.with_name(f"{results_path.stem}_steps.csv"))
    else:
        remote_step_diagnostics_file = ""
    dpfm_stochastic = not dpfm_deterministic

    print(
        "Launching Modal policy evaluation: "
        f"config={config}, dpfm={remote_dpfm_checkpoint}, "
        f"rrl={remote_rrl_checkpoint or '<none>'}, episodes={episodes}, "
        f"dpfm_stochastic={dpfm_stochastic}, gpu={_GPU_TYPE}, cpu={_CPU_CORES}, "
        f"results_file={remote_results_file}, "
        f"step_diagnostics_file={remote_step_diagnostics_file or '<none>'}, "
        f"wandb={wandb}"
    )
    eval_policy_remote.remote(
        config=config,
        dpfm_checkpoint=remote_dpfm_checkpoint,
        rrl_checkpoint=remote_rrl_checkpoint,
        rrl_config=rrl_config,
        episodes=episodes,
        seed=seed,
        dpfm_stochastic=dpfm_stochastic,
        dpfm_execute_steps=dpfm_execute_steps,
        dpfm_replan_on_context_change=dpfm_replan_on_context_change,
        results_file=remote_results_file,
        step_diagnostics_file=remote_step_diagnostics_file,
        use_wandb=wandb,
        wandb_project=wandb_project,
        wandb_run_id=wandb_run_id,
        wandb_name=wandb_name,
        wandb_step=wandb_step,
    )
