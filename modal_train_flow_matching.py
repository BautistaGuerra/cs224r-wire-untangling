"""
Modal GPU entrypoint for DPFM / flow-matching behavior cloning.

Examples:
    # One-time setup:
    #   modal setup
    #   modal secret create wandb WANDB_API_KEY=<token>

    # Upload a local demo file to the Modal data volume and train.
    modal run modal_train_flow_matching.py \
        --env-config configs/stick_reorder_n2_random_order.yaml \
        --demos-path data/stick_n2_random_order_demos.hdf5 \
        --checkpoint-dir checkpoints/dpfm_n2_random_phase_active \
        --conditioning phase-active \
        --upload-demos

    # Reuse an already-uploaded demo file from /data.
    modal run modal_train_flow_matching.py \
        --env-config configs/stick_reorder_n2_random_order.yaml \
        --demos-path /data/stick_n2_random_order_demos.hdf5 \
        --checkpoint-dir /checkpoints/dpfm_n2_random_obs \
        --conditioning obs
"""

import hashlib
from pathlib import Path
from datetime import datetime, timezone

import modal
import yaml


DEFAULT_CONFIG = "configs/stick_reorder_n2_random_order.yaml"
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
    conditioning: str,
    seed: int,
    use_wandb: bool,
    obs_std_floor: float,
    action_std_floor: float,
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

    from scripts.train_flow_matching import load_config, train

    cfg = load_config(env_config, dpfm_config)
    train(
        cfg,
        demos_path=demos_path,
        seed=seed,
        use_wandb=use_wandb,
        checkpoint_dir=checkpoint_dir,
        conditioning=conditioning,
        obs_std_floor=obs_std_floor,
        action_std_floor=action_std_floor,
        wandb_run_id=wandb_run_id or None,
        wandb_name=wandb_name or None,
    )

    checkpoint_volume.commit()


@app.local_entrypoint()
def main(
    env_config: str = DEFAULT_CONFIG,
    dpfm_config: str = "configs/flow_matching.yaml",
    demos_path: str = "data/stick_n2_random_order_demos.hdf5",
    checkpoint_dir: str = "checkpoints/dpfm_n2_random_obs",
    conditioning: str = "obs",
    seed: int = 42,
    no_wandb: bool = False,
    obs_std_floor: float = 1e-6,
    action_std_floor: float = 1e-6,
    upload_demos: bool = False,
    remote_demos_path: str = "",
    wandb_run_id: str = "",
    wandb_name: str = "",
):
    if conditioning not in {"obs", "phase-active"}:
        raise ValueError("conditioning must be 'obs' or 'phase-active'")

    remote_demo = _volume_path(remote_demos_path or demos_path, DATA_MOUNT, "data")
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
                    conditioning,
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
        print(f"Uploading {local_demo} to Modal volume cs224r-data:{remote_demo}")
        with data_volume.batch_upload() as batch:
            batch.put_file(local_demo, remote_demo.removeprefix(DATA_MOUNT))

    print(
        "Launching Modal DPFM training: "
        f"conditioning={conditioning}, seed={seed}, gpu={_GPU_TYPE}, "
        f"demos={remote_demo}, checkpoint_dir={remote_checkpoint_dir}, "
        f"wandb_run_id={wandb_run_id}"
    )
    call = train_flow_matching_remote.spawn(
        env_config=env_config,
        dpfm_config=dpfm_config,
        demos_path=remote_demo,
        checkpoint_dir=remote_checkpoint_dir,
        conditioning=conditioning,
        seed=seed,
        use_wandb=not no_wandb,
        obs_std_floor=obs_std_floor,
        action_std_floor=action_std_floor,
        wandb_run_id=wandb_run_id,
        wandb_name=wandb_name,
    )
    call.get()
