"""
Modal GPU entrypoint for residual TD3 training on a frozen DPFM base.

Example:
    modal run --detach modal_train_residual_rl.py \
        --env-config configs/stick_reorder_n2_random_order.yaml \
        --td3-config configs/residual_td3_n2_random_order.yaml \
        --dpfm-checkpoint checkpoints/dpfm_pr14_n2_random_obs_zscore_h4_e1_i10/flow_matching_policy.pt \
        --demos-path /data/stick_n2_random_order_demos.hdf5 \
        --checkpoint-dir checkpoints/rrl_n2_random_obs_h4_e1_i10_as0.1_of0.5_e1m
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import modal
import yaml


DEFAULT_ENV_CONFIG = "configs/stick_reorder_n2_random_order.yaml"
DEFAULT_TD3_CONFIG = "configs/residual_td3_n2_random_order.yaml"
DEFAULT_DPFM_CHECKPOINT = (
    "checkpoints/dpfm_pr14_n2_random_obs_zscore_h4_e1_i10/flow_matching_policy.pt"
)
DEFAULT_DEMOS_PATH = "/data/stick_n2_random_order_demos.hdf5"
DEFAULT_CHECKPOINT_DIR = "checkpoints/rrl_n2_random_obs_h4_e1_i10_as0.1_of0.5_e1m"
DATA_MOUNT = "/data"
CHECKPOINT_MOUNT = "/checkpoints"


with open(DEFAULT_ENV_CONFIG) as _f:
    _cfg = yaml.safe_load(_f)
_modal_cfg = _cfg.get("modal", {})
_GPU_TYPE = os.environ.get("MODAL_GPU", _modal_cfg.get("gpu", "A10G"))
_CPU_CORES = float(os.environ.get("MODAL_CPU", _modal_cfg.get("rrl_cpu", 4.0)))
_TIMEOUT = _modal_cfg.get("rrl_timeout", max(int(_modal_cfg.get("timeout", 3 * 3600)), 24 * 3600))


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
        "wandb",
        "tensorboard",
        "pyyaml",
        "h5py",
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

app = modal.App("cs224r-wire-untangling-rrl", image=image)

data_volume = modal.Volume.from_name("cs224r-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cs224r-checkpoints", create_if_missing=True)


def _under_mount(path: str, mount: str) -> bool:
    return path == mount or path.startswith(f"{mount}/")


def _volume_path(path: str, mount: str, local_prefix: str) -> str:
    if _under_mount(path, mount):
        return path
    if path.startswith("/"):
        raise ValueError(f"Remote path must live under {mount}; got {path!r}.")

    rel = path
    prefix = f"{local_prefix}/"
    if rel.startswith(prefix):
        rel = rel[len(prefix):]
    return f"{mount}/{rel}"


def _checkpoint_step(name: str) -> int | None:
    import re

    match = re.fullmatch(r"td3_step(\d+)\.pt", os.path.basename(name))
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


def _resolve_resume_checkpoint(path: str) -> str:
    if path and os.path.isdir(path):
        latest = _latest_td3_checkpoint(path)
        if latest:
            print(f"Resolved RRL resume checkpoint directory to {latest}")
            return latest
    return path


@app.function(
    gpu=_GPU_TYPE,
    cpu=_CPU_CORES,
    timeout=_TIMEOUT,
    volumes={
        DATA_MOUNT: data_volume,
        CHECKPOINT_MOUNT: checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("wandb")],
)
def train_residual_rl_remote(
    env_config: str,
    td3_config: str,
    dpfm_checkpoint: str,
    demos_path: str,
    checkpoint_dir: str,
    resume_checkpoint: str,
    auto_resume: bool,
    seed: int | None,
    use_wandb: bool,
    total_timesteps: int | None,
    action_scale: float | None,
    actor_lr: float | None,
    critic_lr: float | None,
    offline_fraction: float | None,
    dpfm_stochastic: bool,
    dpfm_execute_steps: int | None,
    dpfm_replan_on_context_change: bool,
    wandb_run_id: str,
    wandb_name: str,
):
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")

    checkpoint_volume.reload()

    dpfm_checkpoint = _resolve_dpfm_checkpoint(dpfm_checkpoint)
    resume_checkpoint = _resolve_resume_checkpoint(resume_checkpoint)

    if not os.path.exists(dpfm_checkpoint):
        raise FileNotFoundError(f"DPFM checkpoint not found on Modal: {dpfm_checkpoint}")
    if os.path.isdir(dpfm_checkpoint):
        raise IsADirectoryError(
            f"DPFM checkpoint path is a directory and no flow_matching_policy.pt was found: {dpfm_checkpoint}"
        )
    if demos_path and not os.path.exists(demos_path):
        raise FileNotFoundError(f"Demo file not found on Modal: {demos_path}")
    if resume_checkpoint and not os.path.exists(resume_checkpoint):
        raise FileNotFoundError(f"Resume checkpoint not found on Modal: {resume_checkpoint}")
    if resume_checkpoint and os.path.isdir(resume_checkpoint):
        raise IsADirectoryError(
            f"RRL resume checkpoint path is a directory and no td3_step*.pt was found: {resume_checkpoint}"
        )

    if wandb_run_id:
        os.environ["WANDB_RUN_ID"] = wandb_run_id
        os.environ["WANDB_RESUME"] = "allow"
    if wandb_name:
        os.environ["WANDB_NAME"] = wandb_name

    from scripts.train_residual_rl import load_config, train

    cfg = load_config(env_config, td3_config)
    td3 = cfg.setdefault("residual_td3", {})
    if demos_path:
        td3["offline_demos_path"] = demos_path
    if total_timesteps is not None:
        td3["total_timesteps"] = total_timesteps
    if action_scale is not None:
        td3.setdefault("actor", {})["action_scale"] = action_scale
    if actor_lr is not None:
        td3.setdefault("actor", {})["lr"] = actor_lr
    if critic_lr is not None:
        td3.setdefault("critic", {})["lr"] = critic_lr
    if offline_fraction is not None:
        td3["offline_fraction"] = offline_fraction

    def _commit_checkpoint(path: str, step: int):
        checkpoint_volume.commit()
        print(f"  Modal checkpoint volume committed after step {step}: {path}")

    train(
        cfg,
        dpfm_checkpoint=dpfm_checkpoint,
        seed=seed,
        use_wandb=use_wandb,
        checkpoint_dir=checkpoint_dir,
        dpfm_stochastic=dpfm_stochastic,
        dpfm_execute_steps=dpfm_execute_steps,
        dpfm_replan_on_context_change=dpfm_replan_on_context_change,
        resume_checkpoint=resume_checkpoint or None,
        auto_resume=auto_resume,
        checkpoint_callback=_commit_checkpoint,
    )

    checkpoint_volume.commit()


@app.local_entrypoint()
def main(
    env_config: str = DEFAULT_ENV_CONFIG,
    td3_config: str = DEFAULT_TD3_CONFIG,
    dpfm_checkpoint: str = DEFAULT_DPFM_CHECKPOINT,
    demos_path: str = DEFAULT_DEMOS_PATH,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    seed: int | None = 42,
    no_wandb: bool = False,
    total_timesteps: int | None = None,
    action_scale: float | None = None,
    actor_lr: float | None = None,
    critic_lr: float | None = None,
    offline_fraction: float | None = None,
    dpfm_execute_steps: int | None = None,
    dpfm_deterministic: bool = False,
    dpfm_replan_on_context_change: bool = False,
    resume_checkpoint: str = "",
    no_resume: bool = False,
    wandb_run_id: str = "",
    wandb_name: str = "",
):
    remote_demo = _volume_path(demos_path, DATA_MOUNT, "data") if demos_path else ""
    remote_dpfm_checkpoint = _volume_path(dpfm_checkpoint, CHECKPOINT_MOUNT, "checkpoints")
    remote_checkpoint_dir = _volume_path(checkpoint_dir, CHECKPOINT_MOUNT, "checkpoints")
    remote_resume_checkpoint = (
        _volume_path(resume_checkpoint, CHECKPOINT_MOUNT, "checkpoints")
        if resume_checkpoint
        else ""
    )
    auto_resume = not no_resume
    dpfm_stochastic = not dpfm_deterministic

    if not wandb_run_id:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        fingerprint = hashlib.sha1(
            "|".join(
                [
                    env_config,
                    td3_config,
                    remote_dpfm_checkpoint,
                    remote_demo,
                    remote_checkpoint_dir,
                    str(seed),
                    str(total_timesteps),
                    str(action_scale),
                    str(offline_fraction),
                    str(dpfm_stochastic),
                    timestamp,
                ]
            ).encode()
        ).hexdigest()[:8]
        wandb_run_id = f"rrl-{timestamp}-{fingerprint}"
    if not wandb_name:
        wandb_name = Path(remote_checkpoint_dir).name

    print(
        "Launching Modal residual-RL training: "
        f"env_config={env_config}, td3_config={td3_config}, "
        f"dpfm={remote_dpfm_checkpoint}, demos={remote_demo}, "
        f"checkpoint_dir={remote_checkpoint_dir}, total_timesteps={total_timesteps}, "
        f"action_scale={action_scale}, offline_fraction={offline_fraction}, "
        f"dpfm_stochastic={dpfm_stochastic}, seed={seed}, gpu={_GPU_TYPE}, cpu={_CPU_CORES}, "
        f"auto_resume={auto_resume}, resume_checkpoint={remote_resume_checkpoint or '<latest>'}, "
        f"timeout={_TIMEOUT}, wandb_run_id={wandb_run_id}"
    )
    call = train_residual_rl_remote.spawn(
        env_config=env_config,
        td3_config=td3_config,
        dpfm_checkpoint=remote_dpfm_checkpoint,
        demos_path=remote_demo,
        checkpoint_dir=remote_checkpoint_dir,
        resume_checkpoint=remote_resume_checkpoint,
        auto_resume=auto_resume,
        seed=seed,
        use_wandb=not no_wandb,
        total_timesteps=total_timesteps,
        action_scale=action_scale,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        offline_fraction=offline_fraction,
        dpfm_stochastic=dpfm_stochastic,
        dpfm_execute_steps=dpfm_execute_steps,
        dpfm_replan_on_context_change=dpfm_replan_on_context_change,
        wandb_run_id=wandb_run_id,
        wandb_name=wandb_name,
    )
    call.get()
