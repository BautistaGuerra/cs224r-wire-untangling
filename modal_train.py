"""
Modal GPU training entrypoint.

Usage:
    # Short smoke test (1000 steps, runs on Modal GPU)
    modal run modal_train.py --total-timesteps 1000

    # Full training run
    modal run modal_train.py

    # Custom config
    modal run modal_train.py --config configs/stick_reorder.yaml --total-timesteps 500000 --seed 1

GPU type and timeout are read from configs/stick_reorder.yaml (modal.gpu / modal.timeout)
at import time, so changing the YAML is all that's needed to switch hardware.
"""

import yaml
import modal

# Read modal-specific params from config at import time so the decorator picks them up.
with open("configs/stick_reorder.yaml") as _f:
    _cfg = yaml.safe_load(_f)
_modal_cfg = _cfg.get("modal", {})
_GPU_TYPE = _modal_cfg.get("gpu", "A10G")
_TIMEOUT  = _modal_cfg.get("timeout", 3 * 3600)

# Image: debian + osmesa for headless MuJoCo on Linux + all Python deps.
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
        "stable-baselines3>=2.0.0",
        "wandb",
        "tensorboard",
        "pyyaml",
        "h5py",
    )
    .env({
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
    })
    .add_local_python_source("wire_untangling")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_dir("configs", remote_path="/root/configs")
)

app = modal.App("cs224r-wire-untangling", image=image)

# Persistent volume for checkpoints that survives across runs
volume = modal.Volume.from_name("cs224r-checkpoints", create_if_missing=True)


@app.function(
    gpu=_GPU_TYPE,
    timeout=_TIMEOUT,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("wandb")],  # created via: modal secret create wandb WANDB_API_KEY=<token>
)
def train_remote(
    config_path: str = "configs/stick_reorder.yaml",
    total_timesteps: int = 1_000_000,
    seed: int = 42,
):
    import sys
    sys.path.insert(0, "/root")

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from scripts.train import train
    train(
        config,
        total_timesteps=total_timesteps,
        seed=seed,
        use_wandb=True,
        checkpoint_dir="/checkpoints",
    )

    # Commit volume so checkpoints are visible outside the function
    volume.commit()


@app.local_entrypoint()
def main(
    config: str = "configs/stick_reorder.yaml",
    total_timesteps: int = 1_000_000,
    seed: int = 42,
):
    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)

    ts = cfg.get("training", {}).get("total_timesteps", total_timesteps)
    sd = cfg.get("training", {}).get("seed", seed)

    print(f"Launching Modal training: {ts} steps, seed={sd}, gpu={_GPU_TYPE}")
    train_remote.remote(
        config_path=config,
        total_timesteps=ts,
        seed=sd,
    )
