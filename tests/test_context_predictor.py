import json

import h5py
import numpy as np
import pytest
import torch

from scripts.train_context_predictor import (
    NUM_PHASES,
    compute_metrics,
    load_context_dataset,
    split_demo_keys,
    train,
)
from wire_untangling.policies.context_predictor import ContextPredictor


def _write_context_demos(path, num_demos=5, steps=6, obs_dim=4, num_sticks=2):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs["env_config"] = json.dumps({"num_sticks": num_sticks})
        data = f.create_group("data")
        for demo_idx in range(num_demos):
            grp = data.create_group(f"demo_{demo_idx}")
            obs = rng.standard_normal((steps, obs_dim)).astype(np.float32)
            phase = ((np.arange(steps) + demo_idx) % NUM_PHASES).astype(np.int8)
            active = ((np.arange(steps) + demo_idx) % num_sticks).astype(np.int8)
            grp.create_dataset("obs", data=obs)
            grp.create_dataset("phase", data=phase)
            grp.create_dataset("active_stick", data=active)


def test_context_data_loading_requires_labels(tmp_path):
    path = tmp_path / "demos.hdf5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("data/demo_0")
        grp.create_dataset("obs", data=np.zeros((3, 4), dtype=np.float32))
        grp.create_dataset("phase", data=np.zeros(3, dtype=np.int8))

    with pytest.raises(ValueError, match="active_stick"):
        load_context_dataset(str(path), ["demo_0"], num_sticks=2)


def test_context_episode_split_is_reproducible_and_disjoint():
    keys = [f"demo_{i}" for i in range(10)]
    split_a = split_demo_keys(keys, split_seed=42)
    split_b = split_demo_keys(reversed(keys), split_seed=42)

    assert split_a == split_b
    assert len(split_a["train"]) == 8
    assert len(split_a["val"]) == 1
    assert len(split_a["test"]) == 1

    train = set(split_a["train"])
    val = set(split_a["val"])
    test = set(split_a["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(keys)


def test_context_predictor_forward_shapes():
    model = ContextPredictor(
        obs_dim=5,
        num_phases=NUM_PHASES,
        num_sticks=2,
        hidden_dims=(8, 8),
    )
    phase_logits, active_logits = model(torch.randn(3, 5))

    assert phase_logits.shape == (3, NUM_PHASES)
    assert active_logits.shape == (3, 2)


def test_context_tiny_training_writes_checkpoints_and_metrics(tmp_path):
    demos_path = tmp_path / "demos.hdf5"
    ckpt_dir = tmp_path / "context_ckpt"
    _write_context_demos(demos_path, num_demos=5, steps=5, obs_dim=4, num_sticks=2)

    metrics = train(
        demos_path=str(demos_path),
        checkpoint_dir=str(ckpt_dir),
        epochs=1,
        batch_size=4,
        lr=1e-3,
        hidden_dims=(8,),
        dropout=0.0,
        seed=0,
        split_seed=0,
        use_wandb=False,
    )

    best_path = ckpt_dir / "context_predictor_best.pt"
    last_path = ckpt_dir / "context_predictor_last.pt"
    metrics_path = ckpt_dir / "metrics.json"
    assert best_path.exists()
    assert last_path.exists()
    assert metrics_path.exists()

    ckpt = torch.load(str(best_path), map_location="cpu", weights_only=True)
    assert int(ckpt["obs_dim"]) == 4
    assert int(ckpt["num_phases"]) == NUM_PHASES
    assert int(ckpt["num_sticks"]) == 2
    assert ckpt["obs_mean"].shape == (4,)
    assert ckpt["obs_std"].shape == (4,)

    with open(metrics_path) as f:
        saved = json.load(f)
    assert saved["design_choices"]["stateless"] is True
    assert set(saved["metrics"].keys()) == {"train", "val", "test"}
    assert metrics["best_epoch"] == saved["best_epoch"]


def test_context_metrics_include_joint_within_one_and_phase_names():
    metrics = compute_metrics(
        true_phase=np.array([0, 1, 2, 3]),
        true_active=np.array([0, 1, 1, 0]),
        pred_phase=np.array([0, 2, 4, 3]),
        pred_active=np.array([0, 1, 0, 0]),
        num_phases=NUM_PHASES,
    )

    assert metrics["phase_accuracy"] == pytest.approx(0.5)
    assert metrics["active_stick_accuracy"] == pytest.approx(0.75)
    assert metrics["joint_accuracy"] == pytest.approx(0.5)
    assert metrics["bc_input_agreement"] == pytest.approx(0.5)
    assert metrics["within_one_phase_accuracy"] == pytest.approx(0.75)
    assert metrics["phase_labels"][0] == "APPROACH"
    assert metrics["per_phase"]["APPROACH"]["recall"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"][2][4] == 1
