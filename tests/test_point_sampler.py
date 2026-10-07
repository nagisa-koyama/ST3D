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


def test_voxel_structure_neighbours_and_rings():
    """The occupancy term's voxel graph: a 2 x 2 x 1 block of voxels has 3 occupied neighbours each (12 ordered
    pairs); points outside the detector grid get voxel -1; rings follow the voxel-centre planar range."""
    sys.path.insert(0, str(ROOT / 'tools'))
    from analysis.train_point_sampler import voxel_structure, VOXEL, PCR
    base = np.array([5.0, 0.0, 0.0])
    block = [base + (np.array([i, j, 0]) + 0.5) * VOXEL for i in (0, 1) for j in (0, 1)]
    far = np.array([60.05, 0.05, 0.0]); outside = np.array([100.0, 0.0, 0.0])
    xyz = np.array(block + [block[0] + 0.01, far, outside])
    vox, ring, r, az, pv, pu = voxel_structure(xyz)
    assert vox[-1] == -1 and vox[0] == vox[4] and len(set(vox[:4])) == 4
    occupied = np.bincount(pv, minlength=len(ring))
    assert occupied[vox[:4]].tolist() == [3, 3, 3, 3] and occupied[vox[5]] == 0 and len(pv) == 12
    assert set(zip(pv.tolist(), pu.tolist())) == set(zip(pu.tolist(), pv.tolist()))   # symmetric
    assert ring[vox[0]] == 0 and ring[vox[5]] == 5                                      # 0-10 m and 50+ m


# ---- the lattice (virtual range image) sampler, 20261003_04 §38 ------------------------------------------------------
INC = np.radians(np.linspace(2.0, -18.0, 64))  # top -> bottom, like Waymo TOP's published inclinations
SENSOR_Z = 2.184


def _at(el_deg, az_deg, r):
    e, a = np.radians(el_deg), np.radians(az_deg)
    return np.array([r * np.cos(a), r * np.sin(a), SENSOR_Z + r * np.tan(e)])


def test_lattice_pixels_rows_columns_and_the_fov_sentinel():
    from pcdet.datasets.processor.point_sampler import lattice_pixels
    pts = np.array([_at(2.0, 0.0, 20), _at(-18.0, 90.0, 20), _at(np.degrees(INC[10]), -90.0, 30), _at(10.0, 0.0, 20),
                    _at(-30.0, 0.0, 5)])
    pix, row, rng = lattice_pixels(pts, INC, 2650, SENSOR_Z)
    assert row[0] == 0 and row[1] == 63 and row[2] == 10                     # top beam, bottom beam, a middle beam
    assert pix[3] == -1 and pix[4] == -1 and row[3] == -1                    # above / below the vertical FOV: sentinel
    assert pix[0] // 2650 == 0 and abs(pix[0] % 2650 - 1325) <= 1           # azimuth 0 -> the middle column
    np.testing.assert_allclose(rng[0], 20 / np.cos(np.radians(2.0)), rtol=1e-6)


def test_zbuffer_keeps_only_the_nearest_point_of_each_pixel():
    from pcdet.datasets.processor.point_sampler import lattice_pixels, zbuffer_keep
    pts = np.array([_at(-5.0, 10.0, 30), _at(-5.0, 10.0, 20), _at(-5.0, 10.0, 25), _at(-5.0, 40.0, 50), _at(40.0, 0, 9)])
    pix, _, rng = lattice_pixels(pts, INC, 2650, SENSOR_Z)
    keep = zbuffer_keep(pix, rng)
    assert keep.tolist() == [False, True, False, True, False]                # nearest of the three; alone; outside FOV


def test_voxel_pixel_expectations_match_monte_carlo():
    import torch
    sys.path.insert(0, str(ROOT / 'tools'))
    from analysis.train_point_sampler import voxel_pixel_expectations
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 12); vox = np.array([0, 0, 0, 1, 1, 2, -1, 3, 3, 3, 3, 2]); pix = np.array([0, 0, 1, 1, -1, 2, 2, 2, 3, 3, 3, -1])
    P_occ, P_one, P_pix = voxel_pixel_expectations(torch.tensor(p, dtype=torch.float64), torch.tensor(vox), 4, torch.tensor(pix), 4)
    draws = rng.random((200000, len(p))) < p
    for v in range(4):
        k = draws[:, vox == v].sum(1)
        assert abs(float(P_occ[v]) - (k >= 1).mean()) < 0.005 and abs(float(P_one[v]) - (k == 1).mean()) < 0.005
    for q in range(4):
        assert abs(float(P_pix[q]) - (draws[:, pix == q].sum(1) >= 1).mean()) < 0.005


def test_zbuffer_sampler_init_is_the_rule_and_roundtrips(tmp_path):
    from pcdet.datasets.processor.point_sampler import (LearnedPointSampler, lattice_pixels, zbuffer_keep, FEATURE_NAMES,
                                                        LATTICE_FEATURE_NAMES)
    n_in = len(FEATURE_NAMES) + len(LATTICE_FEATURE_NAMES); h = 8; r = np.random.default_rng(1)
    s = LearnedPointSampler(np.ones(1), 75.0, 1.75, np.zeros(n_in), np.ones(n_in), r.normal(size=(n_in, h)), np.zeros(h),
                            r.normal(size=(h, h)), np.zeros(h), np.zeros((h, 1)), np.zeros(1), rule_kind='zbuffer',
                            lattice_inc=INC, lattice_cols=2650, lattice_sensor_z=SENSOR_Z, rule_margin=4.0)
    pts = np.array([_at(-5.0, 10.0, 30), _at(-5.0, 10.0, 20), _at(-5.0, 40.0, 50), _at(40.0, 0, 9)]).astype(np.float32)
    pts[:, 2] -= 0.0
    pix, _, rng = lattice_pixels(pts, INC, 2650, SENSOR_Z); zb = zbuffer_keep(pix, rng)
    np.testing.assert_allclose(s.keep_probability(pts), np.where(zb, 1 / (1 + np.exp(-4.0)), 1 / (1 + np.exp(4.0))), atol=1e-6)
    s.save(tmp_path / 'z.npz'); s2 = LearnedPointSampler.load(tmp_path / 'z.npz')
    assert s2.rule_kind == 'zbuffer' and s2.lattice_cols == 2650 and abs(s2.lattice_sensor_z - SENSOR_Z) < 1e-9
    np.testing.assert_allclose(s2.keep_probability(pts), s.keep_probability(pts), atol=1e-6)
    old = _rule_sampler(np.full(50, 0.3, np.float32)); old.save(tmp_path / 'old.npz')
    assert LearnedPointSampler.load(tmp_path / 'old.npz').rule_kind == 'rate'          # earlier weights unchanged
