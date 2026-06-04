"""Modal entrypoint for the demo-seeded SAC baseline.

Examples:
    # Override Modal resources at launch time.
    MODAL_GPU=A100 MODAL_CPU=8 modal run --detach modal_train_sac_baseline.py \
        --demos-path /data/stick_n1_orientation_demos.hdf5

    # Upload local demos and run a short smoke job.
    modal run --detach modal_train_sac_baseline.py \
        --total-timesteps 1000 \
        --upload-demos \
        --no-wandb

    # Full one-stick SAC baseline using demos already on the data volume.
    modal run --detach modal_train_sac_baseline.py \
        --demos-path /data/stick_n1_orientation_demos.hdf5 \
        --checkpoint-dir checkpoints/sac_demo_seeded_n1

    # Pure online SAC ablation.
    modal run --detach modal_train_sac_baseline.py \
        --no-demo-seed \
        --checkpoint-dir checkpoints/sac_online_n1
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
import yaml


DEFAULT_ENV_CONFIG = "configs/stick_reorder.yaml"
DEFAULT_SAC_CONFIG = "configs/sac_baseline.yaml"
DATA_MOUNT = "/data"
CHECKPOINT_MOUNT = "/checkpoints"


with open(DEFAULT_ENV_CONFIG) as _f:
    _cfg = yaml.safe_load(_f)
_modal_cfg = _cfg.get("modal", {})
_GPU_TYPE = os.environ.get("MODAL_GPU", _modal_cfg.get("gpu", "A10G"))
_CPU_CORES = float(os.environ.get("MODAL_CPU", _modal_cfg.get("sac_cpu", 4.0)))
_TIMEOUT = _modal_cfg.get("sac_timeout", max(int(_modal_cfg.get("timeout", 3 * 3600)), 24 * 3600))


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

app = modal.App("cs224r-wire-untangling-sac-baseline", image=image)

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
def train_sac_baseline_remote(
    env_config: str,
    sac_config: str,
    demos_path: str,
    checkpoint_dir: str,
    total_timesteps: int | None,
    seed: int | None,
    use_wandb: bool,
    demo_seed: bool,
    num_sticks: int,
    reward_shaping: bool | None,
):
    import os
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")

    if demo_seed and not os.path.exists(demos_path):
        raise FileNotFoundError(
            f"Demo file {demos_path!r} was not found on Modal. "
            "Upload it with --upload-demos or modal volume put cs224r-data ..."
        )

    from scripts.train_sac_baseline import load_config, train

    cfg = load_config(env_config, sac_config)
    env_cfg = cfg.setdefault("env", {})
    sac_cfg = cfg.setdefault("sac_baseline", {})
    demo_cfg = sac_cfg.setdefault("demo_seed", {})

    env_cfg["num_sticks"] = num_sticks
    if reward_shaping is not None:
        env_cfg["reward_shaping"] = reward_shaping
    demo_cfg["enabled"] = demo_seed
    if demo_seed:
        demo_cfg["demos_path"] = demos_path
    if total_timesteps is not None:
        sac_cfg["total_timesteps"] = total_timesteps

    train(
        cfg,
        seed=seed,
        use_wandb=use_wandb,
        checkpoint_dir=checkpoint_dir,
    )

    checkpoint_volume.commit()


@app.local_entrypoint()
def main(
    env_config: str = DEFAULT_ENV_CONFIG,
    sac_config: str = DEFAULT_SAC_CONFIG,
    demos_path: str = "data/stick_n1_orientation_demos.hdf5",
    checkpoint_dir: str = "checkpoints/sac_demo_seeded_n1",
    total_timesteps: int | None = None,
    seed: int | None = None,
    no_wandb: bool = False,
    no_demo_seed: bool = False,
    num_sticks: int = 1,
    reward_shaping: bool | None = None,
    upload_demos: bool = False,
    remote_demos_path: str = "",
):
    demo_seed = not no_demo_seed
    if not demo_seed:
        remote_demo = ""
    elif remote_demos_path:
        remote_demo = _volume_path(remote_demos_path, DATA_MOUNT, "data")
    elif upload_demos and Path(demos_path).is_absolute():
        remote_demo = f"{DATA_MOUNT}/{Path(demos_path).name}"
    else:
        remote_demo = _volume_path(demos_path, DATA_MOUNT, "data")
    remote_checkpoint_dir = _volume_path(checkpoint_dir, CHECKPOINT_MOUNT, "checkpoints")

    if upload_demos and demo_seed:
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
        "Launching Modal SAC baseline training: "
        f"timesteps={total_timesteps}, seed={seed}, gpu={_GPU_TYPE}, cpu={_CPU_CORES}, "
        f"num_sticks={num_sticks}, reward_shaping={reward_shaping}, "
        f"demo_seed={demo_seed}, demos={remote_demo}, "
        f"checkpoint_dir={remote_checkpoint_dir}, timeout={_TIMEOUT}"
    )
    call = train_sac_baseline_remote.spawn(
        env_config=env_config,
        sac_config=sac_config,
        demos_path=remote_demo,
        checkpoint_dir=remote_checkpoint_dir,
        total_timesteps=total_timesteps,
        seed=seed,
        use_wandb=not no_wandb,
        demo_seed=demo_seed,
        num_sticks=num_sticks,
        reward_shaping=reward_shaping,
    )
    call.get()
