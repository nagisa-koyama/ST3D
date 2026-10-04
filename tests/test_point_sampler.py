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


def test_gate_leaves_the_rule_outside_the_observed_cells():
    from pcdet.datasets.processor.point_sampler import N_CELLS, cell_index
    rate = np.full(50, 0.4, np.float32)
    s = _rule_sampler(rate); s.w3 = np.ones_like(s.w3) * 0.5; s.b3 = np.ones_like(s.b3)
    pts = _cloud(seed=3)
    f = point_features(pts, s.shift_z)
    cells = cell_index(f[:, 0], np.arctan2(f[:, 7], f[:, 6]))
    mask = np.zeros(N_CELLS, bool); mask[: N_CELLS // 2] = True
    s.obs_mask = mask
    p = s.keep_probability(pts)
    inside = mask[cells]
    np.testing.assert_allclose(p[~inside], 0.4, atol=1e-5)       # the rule exactly where the target saw nothing
    assert not np.allclose(p[inside], 0.4)                       # the learned correction where it did


def test_weights_without_a_mask_load_ungated(tmp_path):
    s = _rule_sampler(np.full(50, 0.3, np.float32)); s.w3 += 0.1
    s.save(tmp_path / 'old.npz')                                 # obs_mask None -> not written, as before 2026-10-04
    assert 'obs_mask' not in np.load(tmp_path / 'old.npz').files
    assert LearnedPointSampler.load(tmp_path / 'old.npz').obs_mask is None
    s.obs_mask = np.ones(720, bool); s.obs_mask[:10] = False
    s.save(tmp_path / 'new.npz')
    np.testing.assert_array_equal(LearnedPointSampler.load(tmp_path / 'new.npz').obs_mask, s.obs_mask)


def test_target_cone_is_measured_from_cell_counts():
    sys.path.insert(0, str(ROOT / 'tools'))
    from analysis.train_point_sampler import target_cone
    from pcdet.datasets.processor.point_sampler import N_SECTORS, RANGE_EDGES
    n_rb = len(RANGE_EDGES) - 1
    full = np.ones((N_SECTORS, n_rb))
    assert target_cone(full.ravel()) is None                     # a 360 deg sensor: the rule is unchanged
    cone = np.zeros((N_SECTORS, n_rb)); cone[10:14] = 5.0        # sectors 10..13 = azimuth -30..+30 deg
    cone[9, 3] = 0.01                                            # a stray return beside the cone is not coverage
    fov, heading = target_cone(cone.ravel())
    assert fov == 60.0 and abs(heading) < 1e-9
    wrap = np.zeros((N_SECTORS, n_rb)); wrap[[22, 23, 0, 1]] = 1.0  # an arc across +-180 deg
    fov, heading = target_cone(wrap.ravel())
    assert fov == 60.0 and abs(abs(heading) - 180.0) < 1e-9
