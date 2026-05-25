"""Tests for wire_untangling.utils.normalizer.Normalizer.

Covers:
    - Statistics computation via from_data (mean, std, eps guard, clip storage)
    - Forward / inverse transforms in numpy (formula, roundtrip, shapes)
    - Clip-on-denormalize behavior (high, low, in-range, no-clip passthrough)
    - Zero-variance corner case (eps clamping, no inf, roundtrip)
    - Serialization via state_dict / from_state_dict and torch.save roundtrip
    - Torch code path parity with numpy, including clipping and zero-variance
"""

import numpy as np
import pytest
import torch

from wire_untangling.utils.normalizer import Normalizer


# ── from_data stats ──────────────────────────────────────────────────────

class TestFromData:
    """Verify that from_data computes correct mean/std and stores clip bounds."""

    def test_loc_is_mean(self):
        """loc should equal the per-column mean of the input data."""
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        n = Normalizer.from_data(data)
        np.testing.assert_allclose(n.loc, data.mean(axis=0))

    def test_scale_is_std(self):
        """scale should equal the per-column std of the input data."""
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        n = Normalizer.from_data(data)
        np.testing.assert_allclose(n.scale, data.std(axis=0))

    def test_zero_std_gets_eps(self):
        """A constant column (std=0) should have its scale clamped to EPS."""
        data = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float32)
        n = Normalizer.from_data(data)
        assert n.scale[0] == pytest.approx(Normalizer.EPS)
        assert n.scale[1] > Normalizer.EPS

    def test_default_scale_can_be_overridden(self):
        """Callers can request a larger local scale floor without changing EPS."""
        data = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float32)
        n = Normalizer.from_data(data, default_scale=0.1)
        assert Normalizer.EPS == pytest.approx(1e-6)
        assert n.scale[0] == pytest.approx(0.1)

    def test_clips_stored(self):
        """clip_low and clip_high should be stored when provided."""
        data = np.ones((3, 2), dtype=np.float32)
        low = np.array([-1, -1], dtype=np.float32)
        high = np.array([1, 1], dtype=np.float32)
        n = Normalizer.from_data(data, clip_low=low, clip_high=high)
        np.testing.assert_array_equal(n.clip_low, low)
        np.testing.assert_array_equal(n.clip_high, high)

    def test_no_clips_by_default(self):
        """Without explicit clip bounds, clip_low/high should be None."""
        n = Normalizer.from_data(np.ones((3, 2), dtype=np.float32))
        assert n.clip_low is None
        assert n.clip_high is None


# ── normalize / denormalize (numpy) ──────────────────────────────────────

class TestNumpyForwardInverse:
    """Test the numpy normalize and denormalize paths."""

    @pytest.fixture()
    def norm(self):
        """Normalizer with loc=[10,20], scale=[2,5], no clips."""
        return Normalizer(
            loc=np.array([10, 20], dtype=np.float32),
            scale=np.array([2, 5], dtype=np.float32),
        )

    def test_normalize_formula(self, norm):
        """normalize should compute (x - loc) / scale."""
        x = np.array([[12, 30]], dtype=np.float32)
        out = norm.normalize(x)
        np.testing.assert_allclose(out, [[1.0, 2.0]])

    def test_denormalize_formula(self, norm):
        """denormalize should compute x * scale + loc."""
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        out = norm.denormalize(x)
        np.testing.assert_allclose(out, [[12, 30]])

    def test_roundtrip(self, norm):
        """denormalize(normalize(x)) should recover x within float32 tolerance."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((100, 2)).astype(np.float32) * 5 + 15
        np.testing.assert_allclose(norm.denormalize(norm.normalize(x)), x, atol=1e-5)

    def test_batch_dims(self, norm):
        """Output shape should match input shape for batched input."""
        x = np.ones((4, 2), dtype=np.float32)
        assert norm.normalize(x).shape == (4, 2)

    def test_single_row(self, norm):
        """A 1-D input (single observation) should work without unsqueezing."""
        x = np.array([10, 20], dtype=np.float32)
        np.testing.assert_allclose(norm.normalize(x), [0, 0])


# ── clipping ─────────────────────────────────────────────────────────────

class TestClipping:
    """Verify that clip bounds are applied on denormalize only."""

    @pytest.fixture()
    def norm(self):
        """Identity normalizer (loc=0, scale=1) with clips at [-1, 1]."""
        return Normalizer(
            loc=np.array([0, 0], dtype=np.float32),
            scale=np.array([1, 1], dtype=np.float32),
            clip_low=np.array([-1, -1], dtype=np.float32),
            clip_high=np.array([1, 1], dtype=np.float32),
        )

    def test_denormalize_clips_high(self, norm):
        """Values above clip_high should be clamped down."""
        x = np.array([[5, 0]], dtype=np.float32)
        out = norm.denormalize(x)
        assert out[0, 0] == 1.0

    def test_denormalize_clips_low(self, norm):
        """Values below clip_low should be clamped up."""
        x = np.array([[-5, 0]], dtype=np.float32)
        out = norm.denormalize(x)
        assert out[0, 0] == -1.0

    def test_denormalize_no_clip_when_in_range(self, norm):
        """Values within [clip_low, clip_high] should pass through unchanged."""
        x = np.array([[0.5, -0.3]], dtype=np.float32)
        np.testing.assert_allclose(norm.denormalize(x), x)

    def test_normalize_never_clips(self, norm):
        """normalize should never clip, even when clip bounds are set."""
        x = np.array([[100, -100]], dtype=np.float32)
        out = norm.normalize(x)
        np.testing.assert_allclose(out, [[100, -100]])

    def test_no_clip_normalizer_passes_through(self):
        """A normalizer without clip bounds should not alter denormalize output."""
        n = Normalizer(loc=np.zeros(2), scale=np.ones(2))
        x = np.array([[100, -100]], dtype=np.float32)
        np.testing.assert_allclose(n.denormalize(x), x)


# ── clip violation warnings ─────────────────────────────────────────────

class TestClipViolationWarning:
    """Verify that denormalize logs warnings when values exceed clip bounds."""

    @pytest.fixture()
    def norm(self):
        """Identity normalizer (loc=0, scale=1) with clips at [-1, 1] and warnings enabled."""
        return Normalizer(
            loc=np.array([0, 0], dtype=np.float32),
            scale=np.array([1, 1], dtype=np.float32),
            clip_low=np.array([-1, -1], dtype=np.float32),
            clip_high=np.array([1, 1], dtype=np.float32),
            warn_on_clip=True,
        )

    def test_warns_on_overshoot_above(self, norm, caplog):
        """Denormalizing a value above clip_high should log a warning."""
        x = np.array([[1.5, 0.5]], dtype=np.float32)
        with caplog.at_level("WARNING", logger="wire_untangling.utils.normalizer"):
            norm.denormalize(x)
        assert "above max" in caplog.text
        assert "dim 0" in caplog.text

    def test_warns_on_overshoot_below(self, norm, caplog):
        """Denormalizing a value below clip_low should log a warning."""
        x = np.array([[0.0, -2.0]], dtype=np.float32)
        with caplog.at_level("WARNING", logger="wire_untangling.utils.normalizer"):
            norm.denormalize(x)
        assert "below min" in caplog.text
        assert "dim 1" in caplog.text

    def test_reports_overshoot_amount(self, norm, caplog):
        """The warning should contain the numeric overshoot magnitude."""
        x = np.array([[1.75, 0.0]], dtype=np.float32)
        with caplog.at_level("WARNING", logger="wire_untangling.utils.normalizer"):
            norm.denormalize(x)
        assert "0.7500" in caplog.text

    def test_no_warning_when_in_range(self, norm, caplog):
        """No warning should be emitted when all values are within bounds."""
        x = np.array([[0.5, -0.3]], dtype=np.float32)
        with caplog.at_level("WARNING", logger="wire_untangling.utils.normalizer"):
            norm.denormalize(x)
        assert caplog.text == ""

    def test_no_warning_without_clips(self, caplog):
        """No warning should be emitted when no clip bounds are set."""
        n = Normalizer(loc=np.zeros(2), scale=np.ones(2))
        x = np.array([[100, -100]], dtype=np.float32)
        with caplog.at_level("WARNING", logger="wire_untangling.utils.normalizer"):
            n.denormalize(x)
        assert caplog.text == ""

    def test_torch_warns_on_overshoot(self, norm, caplog):
        """The torch denormalize path should also log the clip violation warning."""
        x = torch.tensor([[2.0, -3.0]])
        with caplog.at_level("WARNING", logger="wire_untangling.utils.normalizer"):
            norm.denormalize_torch(x)
        assert "above max" in caplog.text
        assert "below min" in caplog.text


# ── zero-variance corner case ───────────────────────────────────────────

class TestZeroVariance:
    """Ensure the eps guard prevents inf/nan for constant-variance dimensions."""

    @pytest.fixture()
    def norm(self):
        """Normalizer where dim 0 has zero raw std (gets clamped to EPS)."""
        return Normalizer(
            loc=np.array([3.14, 0], dtype=np.float32),
            scale=np.array([0, 1], dtype=np.float32),
        )

    def test_scale_clamped_to_eps(self, norm):
        """A zero raw std should be replaced by EPS in the stored scale."""
        assert norm.scale[0] == pytest.approx(Normalizer.EPS)

    def test_normalize_does_not_produce_inf(self, norm):
        """Normalizing through a zero-std dim must not produce inf or nan."""
        x = np.array([[3.14, 1.0]], dtype=np.float32)
        out = norm.normalize(x)
        assert np.all(np.isfinite(out))

    def test_constant_input_normalizes_to_zero(self, norm):
        """Input equal to the mean on a zero-std dim should normalize to ~0."""
        x = np.array([[3.14, 1.0]], dtype=np.float32)
        out = norm.normalize(x)
        assert out[0, 0] == pytest.approx(0, abs=1e-5)

    def test_roundtrip_zero_variance_dim(self, norm):
        """denormalize(normalize(x)) should recover x even on zero-std dims."""
        x = np.array([[3.14, 5.0]], dtype=np.float32)
        np.testing.assert_allclose(norm.denormalize(norm.normalize(x)), x, atol=1e-5)

    def test_from_data_zero_variance(self):
        """from_data on data with a constant column should produce finite output."""
        data = np.array([[7, 1], [7, 2], [7, 3]], dtype=np.float32)
        n = Normalizer.from_data(data)
        out = n.normalize(data)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out[:, 0], 0, atol=1e-5)


# ── state_dict / from_state_dict ────────────────────────────────────────

class TestSerialization:
    """Test checkpoint serialization and deserialization."""

    def test_roundtrip_no_clips(self):
        """state_dict -> from_state_dict should preserve loc and scale exactly."""
        n = Normalizer(
            loc=np.array([1, 2, 3], dtype=np.float32),
            scale=np.array([4, 5, 6], dtype=np.float32),
        )
        n2 = Normalizer.from_state_dict(n.state_dict())
        np.testing.assert_array_equal(n.loc, n2.loc)
        np.testing.assert_array_equal(n.scale, n2.scale)
        assert n2.clip_low is None
        assert n2.clip_high is None

    def test_roundtrip_with_clips(self):
        """Clip bounds should survive the state_dict roundtrip."""
        n = Normalizer(
            loc=np.array([0], dtype=np.float32),
            scale=np.array([1], dtype=np.float32),
            clip_low=np.array([-1], dtype=np.float32),
            clip_high=np.array([1], dtype=np.float32),
        )
        n2 = Normalizer.from_state_dict(n.state_dict())
        np.testing.assert_array_equal(n2.clip_low, [-1])
        np.testing.assert_array_equal(n2.clip_high, [1])

    def test_state_dict_values_are_tensors(self):
        """state_dict values must be torch.Tensors for torch.save compatibility."""
        n = Normalizer(loc=np.zeros(2), scale=np.ones(2))
        sd = n.state_dict()
        assert isinstance(sd["loc"], torch.Tensor)
        assert isinstance(sd["scale"], torch.Tensor)

    def test_torch_save_load_roundtrip(self, tmp_path):
        """Full torch.save -> torch.load -> from_state_dict roundtrip."""
        n = Normalizer(
            loc=np.array([1, 2], dtype=np.float32),
            scale=np.array([0.5, 3], dtype=np.float32),
            clip_low=np.array([-1, -2], dtype=np.float32),
            clip_high=np.array([1, 2], dtype=np.float32),
        )
        path = tmp_path / "norm.pt"
        torch.save({"norm": n.state_dict()}, path)
        loaded = torch.load(path, weights_only=True)
        n2 = Normalizer.from_state_dict(loaded["norm"])
        x = np.array([[0.8, 10]], dtype=np.float32)
        np.testing.assert_allclose(n.normalize(x), n2.normalize(x))
        np.testing.assert_allclose(n.denormalize(x), n2.denormalize(x))

    def test_eps_preserved_through_roundtrip(self):
        """A zero raw scale should still be EPS after save/load."""
        n = Normalizer(loc=np.zeros(1), scale=np.array([0]))
        n2 = Normalizer.from_state_dict(n.state_dict())
        assert n2.scale[0] == pytest.approx(Normalizer.EPS)


# ── torch paths ──────────────────────────────────────────────────────────

class TestTorch:
    """Verify torch normalize/denormalize matches numpy and clips correctly."""

    @pytest.fixture()
    def norm(self):
        """Normalizer with loc=[10,20], scale=[2,5], clips at [5,10]-[15,30]."""
        return Normalizer(
            loc=np.array([10, 20], dtype=np.float32),
            scale=np.array([2, 5], dtype=np.float32),
            clip_low=np.array([5, 10], dtype=np.float32),
            clip_high=np.array([15, 30], dtype=np.float32),
        )

    def test_normalize_torch_matches_numpy(self, norm):
        """normalize_torch output should be identical to numpy normalize."""
        rng = np.random.default_rng(7)
        x_np = rng.standard_normal((10, 2)).astype(np.float32)
        x_t = torch.tensor(x_np)
        np.testing.assert_allclose(
            norm.normalize_torch(x_t).numpy(),
            norm.normalize(x_np),
            atol=1e-6,
        )

    def test_denormalize_torch_matches_numpy(self, norm):
        """denormalize_torch output should be identical to numpy denormalize."""
        rng = np.random.default_rng(8)
        x_np = rng.standard_normal((10, 2)).astype(np.float32)
        x_t = torch.tensor(x_np)
        np.testing.assert_allclose(
            norm.denormalize_torch(x_t).numpy(),
            norm.denormalize(x_np),
            atol=1e-6,
        )

    def test_denormalize_torch_clips(self, norm):
        """Extreme values should be clamped to clip bounds in torch path."""
        x = torch.tensor([[100.0, -100.0]])
        out = norm.denormalize_torch(x)
        assert out[0, 0].item() == 15.0
        assert out[0, 1].item() == 10.0

    def test_output_is_tensor(self, norm):
        """Both torch methods should return torch.Tensor."""
        x = torch.ones(2)
        assert isinstance(norm.normalize_torch(x), torch.Tensor)
        assert isinstance(norm.denormalize_torch(x), torch.Tensor)

    def test_zero_variance_torch(self):
        """Zero-std dim should not produce inf in the torch path."""
        n = Normalizer(loc=np.array([5.0]), scale=np.array([0.0]))
        x = torch.tensor([5.0])
        out = n.normalize_torch(x)
        assert torch.isfinite(out).all()
        assert out.item() == pytest.approx(0, abs=1e-5)
