"""
Modal GPU entrypoint for DPFM / flow-matching behavior cloning.

Examples:
    # One-time setup:
    #   modal setup
    #   modal secret create wandb WANDB_API_KEY=<token>

    # Upload local demos and train the PR #14 N=1 reproduction config.
    modal run --detach modal_train_flow_matching.py \
        --env-config configs/stick_reorder_n1.yaml \
        --demos-path data/stick_n1_orientation_demos.hdf5 \
        --checkpoint-dir checkpoints/dpfm_n1_pr14_zscore_h8_e4_i20 \
        --action-normalizer zscore \
        --conditioning obs \
        --action-chunk-horizon 8 \
        --execute-steps 4 \
        --integration-steps 20 \
        --upload-demos

    # Reuse an already-uploaded demo file from /data.
    modal run --detach modal_train_flow_matching.py \
        --env-config configs/stick_reorder_n1.yaml \
        --demos-path /data/stick_n1_orientation_demos.hdf5 \
        --checkpoint-dir /checkpoints/dpfm_n1_pr14_zscore_h8_e4_i20 \
        --action-normalizer zscore \
        --conditioning obs \
        --action-chunk-horizon 8 \
        --execute-steps 4 \
        --integration-steps 20

    # Train N=2 random-order with oracle phase/active-stick context.
    modal run --detach modal_train_flow_matching.py \
        --env-config configs/stick_reorder_n2_random_order.yaml \
        --demos-path /data/stick_n2_random_order_demos.hdf5 \
        --checkpoint-dir /checkpoints/dpfm_n2_random_phase_active \
        --action-normalizer zscore \
        --conditioning phase-active \
        --action-chunk-horizon 8 \
        --execute-steps 4 \
        --integration-steps 10
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import modal
import yaml


DEFAULT_CONFIG = "configs/stick_reorder_n1.yaml"
DATA_MOUNT = "/data"
CHECKPOINT_MOUNT = "/checkpoints"


with open(DEFAULT_CONFIG) as _f:
    _cfg = yaml.safe_load(_f)
_modal_cfg = _cfg.get("modal", {})
_GPU_TYPE = _modal_cfg.get("gpu", "A10G")
_TIMEOUT = _modal_cfg.get("timeout", 3 * 3600)


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

app = modal.App("cs224r-wire-untangling-dpfm", image=image)

data_volume = modal.Volume.from_name("cs224r-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cs224r-checkpoints", create_if_missing=True)


def _under_mount(path: str, mount: str) -> bool:
    return path == mount or path.startswith(f"{mount}/")


def _volume_path(path: str, mount: str, local_prefix: str) -> str:
    """Map local repo-style paths to Modal volume mount paths."""
    if _under_mount(path, mount):
        return path
    if path.startswith("/"):
        raise ValueError(
            f"Remote path must live under {mount}; got {path!r}. "
            "Pass a mounted path or use --upload-demos for local demo files."
        )

    rel = path
    prefix = f"{local_prefix}/"
    if rel.startswith(prefix):
        rel = rel[len(prefix):]
    return f"{mount}/{rel}"


@app.function(
    gpu=_GPU_TYPE,
    timeout=_TIMEOUT,
    volumes={
        DATA_MOUNT: data_volume,
        CHECKPOINT_MOUNT: checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("wandb")],
)
def train_flow_matching_remote(
    env_config: str,
    dpfm_config: str,
    demos_path: str,
    checkpoint_dir: str,
    seed: int | None,
    use_wandb: bool,
    action_normalizer: str,
    conditioning: str,
    action_chunk_horizon: int | None,
    execute_steps: int | None,
    integration_steps: int | None,
    wandb_run_id: str,
    wandb_name: str,
):
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")

    if not os.path.exists(demos_path):
        raise FileNotFoundError(
            f"Demo file {demos_path!r} was not found on Modal. "
            "Upload it with --upload-demos or modal volume put cs224r-data ..."
        )

    if wandb_run_id:
        os.environ["WANDB_RUN_ID"] = wandb_run_id
        os.environ["WANDB_RESUME"] = "allow"
    if wandb_name:
        os.environ["WANDB_NAME"] = wandb_name

    from scripts.train_flow_matching import load_config, train

    cfg = load_config(env_config, dpfm_config)
    if action_chunk_horizon is not None:
        cfg.setdefault("dpfm", {})["action_chunk_horizon"] = action_chunk_horizon
    if execute_steps is not None:
        cfg.setdefault("dpfm", {})["execute_steps"] = execute_steps
    if integration_steps is not None:
        cfg.setdefault("dpfm", {})["integration_steps"] = integration_steps

    train(
        cfg,
        demos_path=demos_path,
        seed=seed,
        use_wandb=use_wandb,
        checkpoint_dir=checkpoint_dir,
        action_normalizer_type=action_normalizer,
        conditioning=conditioning,
        wandb_run_id=wandb_run_id or None,
        wandb_name=wandb_name or None,
    )

    checkpoint_volume.commit()


@app.local_entrypoint()
def main(
    env_config: str = DEFAULT_CONFIG,
    dpfm_config: str = "configs/flow_matching.yaml",
    demos_path: str = "data/stick_n1_orientation_demos.hdf5",
    checkpoint_dir: str = "checkpoints/dpfm_n1_pr14_zscore_h8_e4_i20",
    seed: int | None = None,
    no_wandb: bool = False,
    action_normalizer: str = "zscore",
    conditioning: str = "obs",
    action_chunk_horizon: int | None = 8,
    execute_steps: int | None = 4,
    integration_steps: int | None = 20,
    upload_demos: bool = False,
    remote_demos_path: str = "",
    wandb_run_id: str = "",
    wandb_name: str = "",
):
    if action_normalizer not in {"zscore", "minmax", "identity"}:
        raise ValueError("action_normalizer must be 'zscore', 'minmax', or 'identity'")
    if conditioning not in {"obs", "phase-active"}:
        raise ValueError("conditioning must be 'obs' or 'phase-active'")

    if remote_demos_path:
        remote_demo = _volume_path(remote_demos_path, DATA_MOUNT, "data")
    elif upload_demos and Path(demos_path).is_absolute():
        remote_demo = f"{DATA_MOUNT}/{Path(demos_path).name}"
    else:
        remote_demo = _volume_path(demos_path, DATA_MOUNT, "data")
    remote_checkpoint_dir = _volume_path(checkpoint_dir, CHECKPOINT_MOUNT, "checkpoints")
    if not wandb_run_id:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        fingerprint = hashlib.sha1(
            "|".join(
                [
                    env_config,
                    dpfm_config,
                    remote_demo,
                    remote_checkpoint_dir,
                    action_normalizer,
                    conditioning,
                    str(action_chunk_horizon),
                    str(execute_steps),
                    str(integration_steps),
                    str(seed),
                    timestamp,
                ]
            ).encode()
        ).hexdigest()[:8]
        wandb_run_id = f"dpfm-{timestamp}-{fingerprint}"
    if not wandb_name:
        wandb_name = Path(remote_checkpoint_dir).name

    if upload_demos:
        local_demo = Path(demos_path)
        if not local_demo.exists():
            raise FileNotFoundError(f"Local demo file not found: {local_demo}")
        remote_volume_path = remote_demo.removeprefix(DATA_MOUNT).lstrip("/")
        print(f"Uploading {local_demo} to Modal volume cs224r-data:{remote_demo}")
        try:
            with data_volume.batch_upload() as batch:
                batch.put_file(local_demo, remote_volume_path)
        except FileExistsError:
            print(f"Demo already exists on Modal volume at {remote_demo}; reusing it.")

    print(
        "Launching Modal DPFM training: "
        f"normalizer={action_normalizer}, conditioning={conditioning}, "
        f"horizon={action_chunk_horizon}, "
        f"execute_steps={execute_steps}, integration_steps={integration_steps}, "
        f"seed={seed}, gpu={_GPU_TYPE}, demos={remote_demo}, "
        f"checkpoint_dir={remote_checkpoint_dir}, wandb_run_id={wandb_run_id}"
    )
    call = train_flow_matching_remote.spawn(
        env_config=env_config,
        dpfm_config=dpfm_config,
        demos_path=remote_demo,
        checkpoint_dir=remote_checkpoint_dir,
        seed=seed,
        use_wandb=not no_wandb,
        action_normalizer=action_normalizer,
        conditioning=conditioning,
        action_chunk_horizon=action_chunk_horizon,
        execute_steps=execute_steps,
        integration_steps=integration_steps,
        wandb_run_id=wandb_run_id,
        wandb_name=wandb_name,
    )
    call.get()
