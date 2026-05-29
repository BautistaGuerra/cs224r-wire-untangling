import types

import numpy as np
import pytest
import torch

from wire_untangling.policies.context_predictor import ContextPredictor
from wire_untangling.policies.mlp_bc import MLPBCPolicy
from wire_untangling.policies.policy_inference_wrappers import (
    MLPBCModelPolicy,
    hard_context_from_logits,
    load_context_predictor_checkpoint,
)
from wire_untangling.utils.normalizer import Normalizer


def _save_bc_checkpoint(
    path,
    raw_obs_dim=4,
    num_phases=8,
    num_sticks=2,
    conditioning="phase-active",
):
    state_dim = raw_obs_dim
    if conditioning == "phase-active":
        state_dim = raw_obs_dim + num_phases + num_sticks
    policy = MLPBCPolicy(state_dim=state_dim, action_dim=2, hidden_dims=(8,))
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "state_dim": state_dim,
            "raw_obs_dim": raw_obs_dim,
            "action_dim": 2,
            "conditioning": conditioning,
            "num_phases": num_phases,
            "num_sticks": num_sticks,
            "hidden_dims": [8],
            "dropout": 0.0,
            "obs_norm": Normalizer(
                loc=np.zeros(state_dim, dtype=np.float32),
                scale=np.ones(state_dim, dtype=np.float32),
            ).state_dict(),
        },
        path,
    )


def _save_context_checkpoint(
    path,
    obs_dim=4,
    num_phases=8,
    num_sticks=2,
):
    model = ContextPredictor(
        obs_dim=obs_dim,
        num_phases=num_phases,
        num_sticks=num_sticks,
        hidden_dims=(8,),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": obs_dim,
            "num_phases": num_phases,
            "num_sticks": num_sticks,
            "hidden_dims": [8],
            "dropout": 0.0,
            "obs_mean": torch.zeros(obs_dim),
            "obs_std": torch.ones(obs_dim),
        },
        path,
    )


def test_context_predictor_checkpoint_loading(tmp_path):
    ckpt_path = tmp_path / "context.pt"
    _save_context_checkpoint(ckpt_path, obs_dim=5, num_sticks=3)

    model, obs_norm, meta = load_context_predictor_checkpoint(
        str(ckpt_path),
        torch.device("cpu"),
    )

    assert isinstance(model, ContextPredictor)
    assert meta["obs_dim"] == 5
    assert meta["num_sticks"] == 3
    np.testing.assert_allclose(obs_norm.loc, np.zeros(5, dtype=np.float32))


def test_hard_context_from_logits_builds_one_hot_features():
    phase_logits = torch.tensor([-1.0, 0.0, 5.0, 0.5, 0.1, 0.2, 0.3, 0.4])
    active_logits = torch.tensor([0.1, 3.0])

    phase, active_stick, features = hard_context_from_logits(
        phase_logits,
        active_logits,
        num_phases=8,
        num_sticks=2,
    )

    assert phase == 2
    assert active_stick == 1
    assert features.shape == (10,)
    assert features[2] == 1.0
    assert features[8 + 1] == 1.0
    assert features[:8].sum() == 1.0
    assert features[8:].sum() == 1.0


@pytest.mark.parametrize(
    ("context_kwargs", "match"),
    [
        ({"obs_dim": 5}, "obs_dim"),
        ({"num_phases": 9}, "num_phases"),
        ({"num_sticks": 3}, "num_sticks"),
    ],
)
def test_mlp_bc_rejects_context_metadata_mismatch(tmp_path, context_kwargs, match):
    bc_path = tmp_path / "bc.pt"
    context_path = tmp_path / "context.pt"
    _save_bc_checkpoint(bc_path)
    _save_context_checkpoint(context_path, **context_kwargs)

    with pytest.raises(ValueError, match=match):
        MLPBCModelPolicy(
            str(bc_path),
            context_predictor_checkpoint=str(context_path),
        )


def test_mlp_bc_rejects_context_predictor_for_obs_only_checkpoint(tmp_path):
    bc_path = tmp_path / "bc.pt"
    context_path = tmp_path / "context.pt"
    _save_bc_checkpoint(bc_path, conditioning="obs")
    _save_context_checkpoint(context_path)

    with pytest.raises(ValueError, match="phase-active"):
        MLPBCModelPolicy(
            str(bc_path),
            context_predictor_checkpoint=str(context_path),
        )


def test_mlp_bc_learned_context_builds_conditioned_state(tmp_path):
    bc_path = tmp_path / "bc.pt"
    context_path = tmp_path / "context.pt"
    _save_bc_checkpoint(bc_path)
    _save_context_checkpoint(context_path)
    policy = MLPBCModelPolicy(
        str(bc_path),
        context_predictor_checkpoint=str(context_path),
    )

    state = policy._build_state(np.zeros(4, dtype=np.float32))

    assert state.shape == (14,)
    np.testing.assert_allclose(state[:4], np.zeros(4, dtype=np.float32))
    assert state[4:12].sum() == 1.0
    assert state[12:].sum() == 1.0


def test_mlp_bc_learned_context_rejects_env_num_stick_mismatch(tmp_path):
    bc_path = tmp_path / "bc.pt"
    context_path = tmp_path / "context.pt"
    _save_bc_checkpoint(bc_path, num_sticks=2)
    _save_context_checkpoint(context_path, num_sticks=2)
    policy = MLPBCModelPolicy(
        str(bc_path),
        context_predictor_checkpoint=str(context_path),
    )
    gym_env = types.SimpleNamespace(env=types.SimpleNamespace(num_sticks=3))

    with pytest.raises(ValueError, match="env has num_sticks=3"):
        policy.set_gym_env(gym_env)
