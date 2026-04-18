"""
Modal GPU training entrypoint.

Usage:
    # Short smoke test (1000 steps, runs on Modal GPU)
    modal run modal_train.py --total-timesteps 1000

    # Full training run
    modal run modal_train.py

    # Custom config
    modal run modal_train.py --config configs/stick_reorder.yaml --total-timesteps 500000 --seed 1

GPU type and timeout are read from the config file (modal.gpu / modal.timeout).
To switch GPU, change configs/stick_reorder.yaml modal.gpu and re-run, no code changes needed.
"""

import modal

# Image: debian + osmesa for headless MuJoCo on Linux + all Python deps
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
        "pyyaml",
    )
    .env({
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
    })
)

app = modal.App("cs224r-wire-untangling", image=image)

# Persistent volume for checkpoints that survives across runs
volume = modal.Volume.from_name("cs224r-checkpoints", create_if_missing=True)

# Mount local package so Modal can import wire_untangling and scripts
local_mount = modal.Mount.from_local_python_packages("wire_untangling")
scripts_mount = modal.Mount.from_local_dir("scripts", remote_path="/root/scripts")
configs_mount = modal.Mount.from_local_dir("configs", remote_path="/root/configs")


# Default resources are overridden at call time via .with_options() in local_entrypoint
# so that gpu type and timeout are always driven by the config file, not this decorator.
@app.function(
    gpu="A10G",
    timeout=3 * 3600,
    mounts=[local_mount, scripts_mount, configs_mount],
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


# this local entrypoint parses CLI args and dispatches to Modal
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

    # GPU type and timeout come from config, change them there, not here
    gpu_type = cfg.get("modal", {}).get("gpu", "A10G")
    timeout = cfg.get("modal", {}).get("timeout", 3 * 3600)

    print(f"Launching Modal training: {ts} steps, seed={sd}, gpu={gpu_type}")
    train_remote.with_options(gpu=gpu_type, timeout=timeout).remote(
        config_path=config,
        total_timesteps=ts,
        seed=sd,
    )
