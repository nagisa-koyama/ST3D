"""The learned point sampler (pcdet/datasets/processor/point_sampler.py, experiments_md/20261003_04):
at initialisation it IS the per-bin rule (the L0 ablation), it pickles into spawned workers, and the
processor step applies it in training mode only."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pcdet.datasets.processor.point_sampler import LearnedPointSampler, point_features, FEATURE_NAMES  # noqa: E402
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402


def _cloud(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(1, 70, n); az = rng.uniform(-np.pi, np.pi, n)
    return np.stack([r * np.cos(az), r * np.sin(az), rng.uniform(-2, 2, n)], 1).astype(np.float32)


def _rule_sampler(rate, max_dist=75.0, shift_z=1.7):
    f = point_features(_cloud(), shift_z)
    return LearnedPointSampler.from_rule(rate, max_dist, shift_z, f.mean(0), f.std(0) + 1e-6)


def test_features_shape_and_names():
    f = point_features(_cloud(), 1.7)
    assert f.shape == (5000, len(FEATURE_NAMES)) and np.isfinite(f).all()


def test_at_initialisation_the_sampler_is_the_rule():
    rate = np.linspace(0.05, 1.0, 50).astype(np.float32)
    s = _rule_sampler(rate)
    pts = _cloud(seed=1)
    r = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.floor(np.clip(r, 0, 75 - 1e-4) / 75 * 50).astype(int)
    np.testing.assert_allclose(s.keep_probability(pts), rate[idx], atol=1e-5)


def test_a_nonzero_last_layer_moves_the_probability():
    s = _rule_sampler(np.full(50, 0.5, np.float32))
    s.w3 = np.ones_like(s.w3)
    assert not np.allclose(s.keep_probability(_cloud()), 0.5)


def test_save_load_and_pickle_roundtrip(tmp_path):
    s = _rule_sampler(np.full(50, 0.3, np.float32)); s.w3 += 0.1
    s.save(tmp_path / 'w.npz')
    s2 = LearnedPointSampler.load(tmp_path / 'w.npz')
    s3 = pickle.loads(pickle.dumps(s))
    pts = _cloud(seed=2)
    np.testing.assert_allclose(s2.keep_probability(pts), s.keep_probability(pts), atol=1e-6)
    np.testing.assert_allclose(s3.keep_probability(pts), s.keep_probability(pts), atol=1e-6)


def test_sampling_keeps_about_the_expected_share():
    s = _rule_sampler(np.full(50, 0.25, np.float32))
    kept = s.sample(_cloud(n=20000), rng=np.random.RandomState(0))
    assert 0.23 < len(kept) / 20000 < 0.27


def _processor(tmp_path, training):
    s = _rule_sampler(np.full(50, 0.5, np.float32)); s.save(tmp_path / 'w.npz')
    cfg = [EasyDict(NAME='sample_points_learned', WEIGHTS=str(tmp_path / 'w.npz'))]
    return DataProcessor(cfg, point_cloud_range=np.array([-75.2, -75.2, -2, 75.2, 75.2, 4]), training=training, num_point_features=3)


def test_processor_step_samples_in_training_and_passes_through_in_eval(tmp_path):
    pts = _cloud(n=20000)
    out = _processor(tmp_path, True).forward({'points': pts.copy()})
    assert 0.45 < len(out['points']) / 20000 < 0.55
    out = _processor(tmp_path, False).forward({'points': pts.copy()})
    assert len(out['points']) == 20000


def test_processor_pickles_with_the_sampler(tmp_path):
    p = _processor(tmp_path, True)
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q.learned_sampler, LearnedPointSampler)


def test_both_sampling_steps_in_one_config_are_refused(tmp_path):
    s = _rule_sampler(np.full(50, 0.5, np.float32)); s.save(tmp_path / 'w.npz')
    cfg = [EasyDict(NAME='sample_points_hist_based'), EasyDict(NAME='sample_points_learned', WEIGHTS=str(tmp_path / 'w.npz'))]
    with pytest.raises(AssertionError):
        DataProcessor(cfg, point_cloud_range=np.array([-75.2, -75.2, -2, 75.2, 75.2, 4]), training=True, num_point_features=3)
