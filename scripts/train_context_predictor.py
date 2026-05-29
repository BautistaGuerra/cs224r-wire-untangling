"""Train a supervised phase / active-stick context predictor from HDF5 demos."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from wire_untangling.policies.context_predictor import ContextPredictor
from wire_untangling.policies.pick_place_expert import Phase


NUM_PHASES = 8
DESIGN_CHOICES = {
    "input": "current_raw_observation",
    "stateless": True,
    "context": "hard_argmax_one_hot",
    "loss": "unweighted_ce_phase_plus_ce_active",
    "sequence_smoothing": False,
    "order_id_input": False,
    "active_stick_classes": "num_sticks_without_none_or_done",
}


@dataclass(frozen=True)
class ContextDataset:
    obs: np.ndarray
    phase: np.ndarray
    active_stick: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.obs.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[1])


def _decode_json_attr(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(value)


def _infer_num_sticks(f: h5py.File) -> int:
    env_cfg = _decode_json_attr(f.attrs.get("env_config"))
    if env_cfg and "num_sticks" in env_cfg:
        return int(env_cfg["num_sticks"])

    max_stick = -1
    for key in f["data"].keys():
        demo = f["data"][key]
        if "active_stick" in demo:
            active = demo["active_stick"][:]
            if len(active):
                max_stick = max(max_stick, int(active.max()))
    return max_stick + 1 if max_stick >= 0 else 1


def list_demo_keys(demos_path: str) -> list[str]:
    with h5py.File(demos_path, "r") as f:
        if "data" not in f:
            raise ValueError(f"{demos_path} does not contain a data group")
        return sorted(f["data"].keys())


def _split_counts(
    n: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    if n <= 0:
        raise ValueError("Cannot split an empty demo set")
    fractions = np.array([train_fraction, val_fraction, test_fraction], dtype=np.float64)
    if np.any(fractions < 0.0) or float(fractions.sum()) <= 0.0:
        raise ValueError("Split fractions must be non-negative and sum to a positive value")

    raw = fractions / fractions.sum() * n
    counts = np.floor(raw).astype(np.int64)
    remainder = int(n - counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1

    positive_splits = int(np.count_nonzero(fractions > 0.0))
    if n >= positive_splits:
        for idx, frac in enumerate(fractions):
            if frac <= 0.0 or counts[idx] > 0:
                continue
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[idx] = 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def split_demo_keys(
    demo_keys: list[str],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    split_seed: int = 42,
) -> dict[str, list[str]]:
    shuffled = list(sorted(demo_keys))
    rng = np.random.default_rng(split_seed)
    rng.shuffle(shuffled)
    n_train, n_val, n_test = _split_counts(
        len(shuffled),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    train_end = n_train
    val_end = train_end + n_val
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:val_end + n_test],
    }


def load_context_dataset(
    demos_path: str,
    demo_keys: list[str],
    num_phases: int = NUM_PHASES,
    num_sticks: int | None = None,
) -> ContextDataset:
    all_obs: list[np.ndarray] = []
    all_phase: list[np.ndarray] = []
    all_active: list[np.ndarray] = []
    obs_dim = None

    with h5py.File(demos_path, "r") as f:
        if "data" not in f:
            raise ValueError(f"{demos_path} does not contain a data group")
        if num_sticks is None:
            num_sticks = _infer_num_sticks(f)

        for key in demo_keys:
            if key not in f["data"]:
                raise ValueError(f"Missing data/{key} in {demos_path}")
            demo = f["data"][key]
            missing = [name for name in ("obs", "phase", "active_stick") if name not in demo]
            if missing:
                raise ValueError(
                    f"data/{key} missing required dataset(s): {', '.join(missing)}"
                )

            obs = np.asarray(demo["obs"][:], dtype=np.float32)
            phase = np.asarray(demo["phase"][:], dtype=np.int64)
            active = np.asarray(demo["active_stick"][:], dtype=np.int64)
            if obs.ndim != 2:
                raise ValueError(f"data/{key}/obs must have shape (T, obs_dim)")
            if phase.ndim != 1 or active.ndim != 1:
                raise ValueError(f"data/{key}/phase and active_stick must have shape (T,)")
            if len(obs) != len(phase) or len(obs) != len(active):
                raise ValueError(
                    f"data/{key} length mismatch: obs={len(obs)}, "
                    f"phase={len(phase)}, active_stick={len(active)}"
                )
            if obs_dim is None:
                obs_dim = obs.shape[1]
            elif obs.shape[1] != obs_dim:
                raise ValueError(
                    f"data/{key}/obs_dim={obs.shape[1]} does not match {obs_dim}"
                )
            if np.any((phase < 0) | (phase >= num_phases)):
                raise ValueError(f"data/{key}/phase values must be in [0, {num_phases})")
            if np.any((active < 0) | (active >= int(num_sticks))):
                raise ValueError(
                    f"data/{key}/active_stick values must be in [0, {int(num_sticks)})"
                )

            all_obs.append(obs)
            all_phase.append(phase)
            all_active.append(active)

    if not all_obs:
        raise ValueError("No demos selected for this split")

    return ContextDataset(
        obs=np.concatenate(all_obs, axis=0),
        phase=np.concatenate(all_phase, axis=0),
        active_stick=np.concatenate(all_active, axis=0),
    )


def make_loader(
    dataset: ContextDataset,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    obs = ((dataset.obs - obs_mean) / obs_std).astype(np.float32)
    tensors = TensorDataset(
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor(dataset.phase, dtype=torch.long),
        torch.tensor(dataset.active_stick, dtype=torch.long),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def _phase_names(num_phases: int) -> list[str]:
    names = []
    for idx in range(num_phases):
        try:
            names.append(Phase(idx).name)
        except ValueError:
            names.append(f"PHASE_{idx}")
    return names


def compute_metrics(
    true_phase: np.ndarray,
    true_active: np.ndarray,
    pred_phase: np.ndarray,
    pred_active: np.ndarray,
    num_phases: int = NUM_PHASES,
) -> dict[str, Any]:
    true_phase = np.asarray(true_phase, dtype=np.int64)
    true_active = np.asarray(true_active, dtype=np.int64)
    pred_phase = np.asarray(pred_phase, dtype=np.int64)
    pred_active = np.asarray(pred_active, dtype=np.int64)
    if not (
        true_phase.shape == true_active.shape == pred_phase.shape == pred_active.shape
    ):
        raise ValueError("Metric arrays must have matching shape")

    n = int(true_phase.size)
    if n == 0:
        raise ValueError("Cannot compute metrics on an empty dataset")

    phase_correct = pred_phase == true_phase
    active_correct = pred_active == true_active
    joint_correct = phase_correct & active_correct
    confusion = np.zeros((num_phases, num_phases), dtype=np.int64)
    for target, pred in zip(true_phase, pred_phase):
        confusion[int(target), int(pred)] += 1

    names = _phase_names(num_phases)
    per_phase: dict[str, dict[str, float | int]] = {}
    for idx, name in enumerate(names):
        tp = int(confusion[idx, idx])
        predicted = int(confusion[:, idx].sum())
        actual = int(confusion[idx, :].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_phase[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": actual,
        }

    joint_accuracy = float(joint_correct.mean())
    return {
        "n_samples": n,
        "phase_accuracy": float(phase_correct.mean()),
        "active_stick_accuracy": float(active_correct.mean()),
        "joint_accuracy": joint_accuracy,
        "bc_input_agreement": joint_accuracy,
        "within_one_phase_accuracy": float((np.abs(pred_phase - true_phase) <= 1).mean()),
        "phase_labels": names,
        "per_phase": per_phase,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_model(
    model: ContextPredictor,
    dataset: ContextDataset,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_phases: int = NUM_PHASES,
) -> dict[str, Any]:
    loader = make_loader(dataset, obs_mean, obs_std, batch_size=batch_size, shuffle=False)
    pred_phase = []
    pred_active = []
    model.eval()
    with torch.no_grad():
        for obs_batch, _, _ in loader:
            obs_batch = obs_batch.to(device)
            phase_logits, active_logits = model(obs_batch)
            pred_phase.append(phase_logits.argmax(dim=1).cpu().numpy())
            pred_active.append(active_logits.argmax(dim=1).cpu().numpy())

    return compute_metrics(
        true_phase=dataset.phase,
        true_active=dataset.active_stick,
        pred_phase=np.concatenate(pred_phase, axis=0),
        pred_active=np.concatenate(pred_active, axis=0),
        num_phases=num_phases,
    )


def _checkpoint_dict(
    model: ContextPredictor,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    epoch: int,
    val_joint_accuracy: float,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "num_phases": model.num_phases,
        "num_sticks": model.num_sticks,
        "hidden_dims": list(model.hidden_dims),
        "dropout": model.dropout,
        "obs_mean": torch.from_numpy(obs_mean.astype(np.float32)),
        "obs_std": torch.from_numpy(obs_std.astype(np.float32)),
        "epoch": int(epoch),
        "val_joint_accuracy": float(val_joint_accuracy),
        "design_choices": dict(DESIGN_CHOICES),
    }


def load_model_from_checkpoint(path: str, device: torch.device | str = "cpu") -> tuple[ContextPredictor, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = ContextPredictor(
        obs_dim=int(ckpt["obs_dim"]),
        num_phases=int(ckpt["num_phases"]),
        num_sticks=int(ckpt["num_sticks"]),
        hidden_dims=tuple(int(h) for h in ckpt["hidden_dims"]),
        dropout=float(ckpt.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def train(
    demos_path: str,
    checkpoint_dir: str,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dims: tuple[int, ...] = (256, 256, 256),
    dropout: float = 0.0,
    seed: int = 42,
    split_seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    use_wandb: bool = True,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training device: {device}")

    demo_keys = list_demo_keys(demos_path)
    splits = split_demo_keys(
        demo_keys,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
    )
    with h5py.File(demos_path, "r") as f:
        num_sticks = _infer_num_sticks(f)

    train_data = load_context_dataset(demos_path, splits["train"], num_sticks=num_sticks)
    val_data = load_context_dataset(demos_path, splits["val"], num_sticks=num_sticks)
    test_data = load_context_dataset(demos_path, splits["test"], num_sticks=num_sticks)
    if train_data.obs_dim != val_data.obs_dim or train_data.obs_dim != test_data.obs_dim:
        raise ValueError("Train/val/test observation dimensions do not match")

    obs_mean = train_data.obs.mean(axis=0).astype(np.float32)
    obs_std = np.maximum(train_data.obs.std(axis=0).astype(np.float32), 1e-6)
    train_loader = make_loader(
        train_data,
        obs_mean=obs_mean,
        obs_std=obs_std,
        batch_size=batch_size,
        shuffle=True,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "context_predictor_best.pt")
    last_path = os.path.join(checkpoint_dir, "context_predictor_last.pt")
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")

    run = None
    if use_wandb:
        import wandb

        run = wandb.init(
            project="cs224r-wire-untangling",
            config={
                "policy": "context_predictor",
                "seed": seed,
                "split_seed": split_seed,
                "obs_dim": train_data.obs_dim,
                "num_phases": NUM_PHASES,
                "num_sticks": num_sticks,
                "hidden_dims": list(hidden_dims),
                "dropout": dropout,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "loss": "CE_phase + CE_active",
            },
            tags=["context-predictor", "supervised"],
        )

    model = ContextPredictor(
        obs_dim=train_data.obs_dim,
        num_phases=NUM_PHASES,
        num_sticks=num_sticks,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    phase_loss_fn = torch.nn.CrossEntropyLoss()
    active_loss_fn = torch.nn.CrossEntropyLoss()

    best_epoch = 0
    best_val_joint = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for obs_batch, phase_batch, active_batch in train_loader:
            obs_batch = obs_batch.to(device)
            phase_batch = phase_batch.to(device)
            active_batch = active_batch.to(device)
            phase_logits, active_logits = model(obs_batch)
            loss = phase_loss_fn(phase_logits, phase_batch) + active_loss_fn(
                active_logits, active_batch
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * obs_batch.size(0)
            total_examples += obs_batch.size(0)

        train_loss = total_loss / total_examples
        train_metrics = evaluate_model(
            model, train_data, obs_mean, obs_std, batch_size, device
        )
        val_metrics = evaluate_model(model, val_data, obs_mean, obs_std, batch_size, device)
        val_joint = float(val_metrics["joint_accuracy"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_joint_accuracy": train_metrics["joint_accuracy"],
                "val_joint_accuracy": val_joint,
            }
        )
        print(
            f"  Epoch {epoch}/{epochs}, loss={train_loss:.6f}, "
            f"train_joint={train_metrics['joint_accuracy']:.3f}, "
            f"val_joint={val_joint:.3f}"
        )

        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/phase_accuracy": train_metrics["phase_accuracy"],
                    "train/active_stick_accuracy": train_metrics["active_stick_accuracy"],
                    "train/joint_accuracy": train_metrics["joint_accuracy"],
                    "val/phase_accuracy": val_metrics["phase_accuracy"],
                    "val/active_stick_accuracy": val_metrics["active_stick_accuracy"],
                    "val/joint_accuracy": val_metrics["joint_accuracy"],
                }
            )

        if val_joint > best_val_joint:
            best_val_joint = val_joint
            best_epoch = epoch
            torch.save(
                _checkpoint_dict(model, obs_mean, obs_std, epoch, val_joint),
                best_path,
            )

    torch.save(
        _checkpoint_dict(model, obs_mean, obs_std, epochs, history[-1]["val_joint_accuracy"]),
        last_path,
    )

    best_model, best_ckpt = load_model_from_checkpoint(best_path, device=device)
    final_metrics = {
        "train": evaluate_model(best_model, train_data, obs_mean, obs_std, batch_size, device),
        "val": evaluate_model(best_model, val_data, obs_mean, obs_std, batch_size, device),
        "test": evaluate_model(best_model, test_data, obs_mean, obs_std, batch_size, device),
    }
    metrics = {
        "design_choices": dict(DESIGN_CHOICES),
        "demos_path": demos_path,
        "checkpoint_best": best_path,
        "checkpoint_last": last_path,
        "selection_metric": "val/joint_accuracy",
        "best_epoch": int(best_ckpt["epoch"]),
        "best_val_joint_accuracy": float(best_ckpt["val_joint_accuracy"]),
        "split_seed": split_seed,
        "splits": splits,
        "num_sticks": num_sticks,
        "num_phases": NUM_PHASES,
        "obs_dim": train_data.obs_dim,
        "history": history,
        "metrics": final_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Best checkpoint saved to {best_path}")
    print(f"Last checkpoint saved to {last_path}")
    print(f"Metrics saved to {metrics_path}")

    if run is not None:
        import wandb

        artifact = wandb.Artifact("context-predictor", type="model")
        artifact.add_file(best_path)
        artifact.add_file(last_path)
        artifact.add_file(metrics_path)
        run.log_artifact(artifact)
        run.finish()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos-path", default="data/demos.hdf5")
    parser.add_argument("--checkpoint-dir", default="checkpoints/context_predictor")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    train(
        demos_path=args.demos_path,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        seed=args.seed,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
