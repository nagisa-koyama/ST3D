"""DRAW, the redraw key of the evaluation-time scan-pattern cuts (pcdet/datasets/eval_cut_rng.py; experiments_md
20261003_05 §5 item 30, the random-cut redraws of the contribution-1 lines-vs-random D). CPU only, synthetic frames,
no dataset files.

Each of the four loaders is run on a bare instance through its own seeding code and the real mask function (Waymo
EVAL_TOP_THIN -> waymo_rings.eval_top_thin_points; nuScenes EVAL_RING_THIN -> beam_downsample_utils.ring_thin_mask;
KITTI -> kitti_rings.eval_ring_thin_mask; PandaSet -> pandaset_rings.eval_thin_mask), and checked for:
  - DRAW absent and DRAW 0 give exactly the mask of the seeding the loader had before the key (written out here);
  - DRAW 1 and 2 give other masks, different from draw 0 and from each other;
  - every draw keeps exactly as many points as the structured twin ('rows', the lines-halved arm);
  - every draw is deterministic, and an empty frame still loads.
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import _init_path  # noqa: E402,F401
from pcdet.datasets.eval_cut_rng import eval_cut_rng  # noqa: E402

DRAWS = (None, 0, 1, 2)


def pre_change_rng(key_str):
    """The seeding every loader used before DRAW existed: default_rng(crc32 of the frame key)."""
    return np.random.default_rng(zlib.crc32(key_str.encode()))


def cut_cfg(mode, draw, **extra):
    cfg = dict(MODE=mode, STRIDE=2, **extra)
    if draw is not None:
        cfg['DRAW'] = draw
    return EasyDict(cfg)


def check_draws(masks, rows_count):
    """masks: {draw: keep mask (or kept points)} for DRAWS; rows_count: the structured twin's kept count."""
    ref0 = masks[None]
    assert np.array_equal(masks[0], ref0)
    for d in DRAWS:
        assert len(masks[d]) == len(ref0)
        kept = masks[d].sum() if masks[d].dtype == bool else len(masks[d])
        assert kept == rows_count, (d, kept, rows_count)
    assert not np.array_equal(masks[1], ref0)
    assert not np.array_equal(masks[2], ref0)
    assert not np.array_equal(masks[1], masks[2])


# ---- the helper -------------------------------------------------------------------------------------------------------

def test_draw_absent_or_zero_is_the_original_seed_bit_for_bit():
    for key in ('segment-1_000_with_camera_labels_042', 'samples/LIDAR_TOP/x.pcd.bin', '000042', '053/07'):
        want = pre_change_rng(key).bit_generator.state
        for d in (None, 0, np.int64(0)):
            assert eval_cut_rng(key, d).bit_generator.state == want


def test_draws_are_distinct_streams_and_reproducible():
    key = '000042'
    states = [eval_cut_rng(key, d).random(8) for d in (0, 1, 2, 3)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(states[i], states[j])
    assert np.array_equal(eval_cut_rng(key, 2).random(8), eval_cut_rng(key, 2).random(8))
    assert np.array_equal(eval_cut_rng(42, 1).random(4), eval_cut_rng('42', 1).random(4))   # the key is str()-ed


@pytest.mark.parametrize('bad', [-1, 1.5, True, '1'])
def test_draw_must_be_a_non_negative_integer(bad):
    with pytest.raises(ValueError):
        eval_cut_rng('000042', bad)


# ---- Waymo EVAL_TOP_THIN ----------------------------------------------------------------------------------------------

SEQ = 'segment-synthetic_with_camera_labels'
FRAME_ID = SEQ + '_042'


def waymo_frame(n_beams=16, n_cols=400, n_side=40, seed=0):
    """A TOP block stored row-major (top beam first, azimuth falling along a row, as the range image), side lidars after
    it; every point outside the no-label zone (column 5 = -1) so counts are exact."""
    rng = np.random.default_rng(seed)
    inc = np.radians(np.linspace(2.0, -17.0, n_beams))
    top = []
    for el in inc:
        az = np.linspace(np.pi, -np.pi, n_cols, endpoint=False) - 1e-3
        az = az[rng.random(n_cols) > 0.1]
        r = rng.uniform(5.0, 70.0, len(az))
        top.append(np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], 1))
    top = np.concatenate(top)
    raw = np.zeros((len(top) + n_side, 6), np.float32)
    raw[:, :3] = np.concatenate([top, rng.uniform(-5.0, 5.0, (n_side, 3))])
    raw[:, 3] = rng.random(len(raw))
    raw[:, 5] = -1
    calib = {'extrinsic': np.eye(4), 'inclinations': inc[::-1].copy()}
    return raw, np.array([len(top), n_side, 0, 0, 0]), calib


def waymo_loader(raw, counts, calib, mode, draw):
    from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
    ds = object.__new__(WaymoDataset)
    ds.eval_top_thin_cfg = cut_cfg(mode, draw, TOP_CALIB='unused.pkl')
    ds._top_calibration = lambda path: {SEQ: calib}
    ds._raw_frame = lambda info: raw
    info = {'frame_id': FRAME_ID, 'point_cloud': {'lidar_sequence': SEQ}, 'num_points_of_each_lidar': counts}
    return ds.get_lidar_eval_top_thin(info)


def test_waymo_eval_top_thin_draws():
    from pcdet.datasets.waymo.waymo_rings import eval_top_thin_points
    raw, counts, calib = waymo_frame()
    old = eval_top_thin_points(raw, counts, calib, 'random', stride=2, rng=pre_change_rng(str(FRAME_ID)))
    out = {d: waymo_loader(raw, counts, calib, 'random', d) for d in DRAWS}
    assert np.array_equal(out[None], old)
    rows = waymo_loader(raw, counts, calib, 'rows', None)
    assert 0.4 < (len(rows) - counts[1]) / counts[0] < 0.6          # the twin halves the TOP block
    assert np.array_equal(waymo_loader(raw, counts, calib, 'rows', 2), rows)   # a structured cut draws nothing
    check_draws(out, len(rows))
    for d in (1, 2):
        assert np.array_equal(waymo_loader(raw, counts, calib, 'random', d), out[d])
        assert np.array_equal(out[d][-counts[1]:, :3], raw[-counts[1]:, :3])      # side lidars untouched


def test_waymo_empty_frames_load_at_every_draw():
    _, _, calib = waymo_frame()
    for counts, raw in ((np.zeros(5, np.int64), np.zeros((0, 6), np.float32)),
                        (np.array([0, 3, 0, 0, 0]), np.c_[np.ones((3, 5), np.float32), -np.ones((3, 1), np.float32)])):
        for d in DRAWS:
            assert len(waymo_loader(raw, counts, calib, 'random', d)) == counts.sum()


# ---- nuScenes EVAL_RING_THIN ------------------------------------------------------------------------------------------

LIDAR_PATH = 'samples/LIDAR_TOP/n008-synthetic__LIDAR_TOP__1533151603547590.pcd.bin'


def nuscenes_cloud(seed=0, n_per_ring=150):
    """32 rings (the .bin's 5th column), every point outside the 1.5 m ego box so remove_ego_points keeps it."""
    rng = np.random.default_rng(seed)
    ring = np.repeat(np.arange(32), n_per_ring)
    az = rng.uniform(-np.pi, np.pi, len(ring))
    r = rng.uniform(3.0, 60.0, len(ring))
    return np.stack([r * np.cos(az), r * np.sin(az), rng.normal(0, 1, len(ring)), rng.random(len(ring)) * 255,
                     ring], 1).astype(np.float32)


def nuscenes_loader(tmp_path, monkeypatch, points, mode, draw):
    """The loader's keep mask (captured from the real ring_thin_mask it calls) and its returned cloud."""
    import pcdet.utils.beam_downsample_utils as bdu
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    path = tmp_path / LIDAR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    points.astype(np.float32).tofile(str(path))
    seen, real = [], bdu.ring_thin_mask

    def spy(*args, **kwargs):
        seen.append(real(*args, **kwargs))
        return seen[-1]
    monkeypatch.setattr(bdu, 'ring_thin_mask', spy)
    ds = object.__new__(NuScenesDataset)
    ds.root_path, ds.training, ds._sweep_compensator = tmp_path, False, None
    ds.infos = [{'lidar_path': LIDAR_PATH, 'sweeps': []}]
    ds.dataset_cfg = EasyDict(EVAL_RING_THIN=cut_cfg(mode, draw))
    out = ds.get_lidar_with_sweeps(0, max_sweeps=1)
    assert len(seen) == 1
    return seen[0], out


def test_nuscenes_eval_ring_thin_draws(tmp_path, monkeypatch):
    from pcdet.utils.beam_downsample_utils import ring_thin_mask
    pts = nuscenes_cloud()
    old = ring_thin_mask(pts, pts[:, 4], 'random', stride=2, rng=pre_change_rng(str(LIDAR_PATH)))
    masks, outs = {}, {}
    for d in DRAWS:
        masks[d], outs[d] = nuscenes_loader(tmp_path, monkeypatch, pts, 'random', d)
    assert np.array_equal(masks[None], old)
    rows, _ = nuscenes_loader(tmp_path, monkeypatch, pts, 'rows', None)
    assert rows.sum() == (pts[:, 4] % 2 == 0).sum()
    check_draws(masks, int(rows.sum()))
    for d in (1, 2):
        again, out = nuscenes_loader(tmp_path, monkeypatch, pts, 'random', d)
        assert np.array_equal(again, masks[d]) and np.array_equal(out, outs[d])
        assert np.array_equal(out[:, :4], pts[masks[d], :4])        # the cloud served is the masked one


def test_nuscenes_empty_frame_loads_at_every_draw(tmp_path, monkeypatch):
    for d in DRAWS:
        mask, out = nuscenes_loader(tmp_path, monkeypatch, np.zeros((0, 5), np.float32), 'random', d)
        assert mask.shape == (0,) and out.shape == (0, 5)


# ---- KITTI EVAL_RING_THIN ---------------------------------------------------------------------------------------------

SAMPLE_IDX = '000042'


def kitti_scan(seed=0, step=0.5):
    """A raw HDL-64E-like scan, laser-major, each laser one turn from azimuth ~0 (what kitti_rings recovers rings from)."""
    rng = np.random.default_rng(seed)
    pts = []
    for el in np.radians(np.linspace(2.0, -24.8, 64)):
        az = np.radians((np.arange(0.01, 360.0, step) + 180.0) % 360.0 - 180.0)
        az = az[rng.random(len(az)) > 0.15]
        r = rng.uniform(5.0, 70.0, len(az))
        pts.append(np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el),
                             rng.random(len(az))], 1))
    return np.concatenate(pts).astype(np.float32)


class _Stop(Exception):
    pass


def kitti_loader(monkeypatch, points, mode, draw):
    """The keep mask KittiDataset.__getitem__ applies (the real mask function on the generator the loader built); the
    item is stopped right there, before calibration files would be read."""
    import pcdet.datasets.kitti.kitti_rings as kr
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset

    def spy(pts, cfg, rng):
        raise _Stop(kr.eval_ring_thin_mask(pts, cfg, rng))
    monkeypatch.setattr(kr, 'eval_ring_thin_points', spy)
    ds = object.__new__(KittiDataset)
    ds._merge_all_iters_to_one_epoch, ds.training = False, False
    ds.kitti_infos = [{'point_cloud': {'lidar_idx': SAMPLE_IDX}}]
    ds.get_lidar = lambda idx: points
    ds.dataset_cfg = EasyDict(EVAL_RING_THIN=cut_cfg(mode, draw))
    with pytest.raises(_Stop) as stop:
        ds[0]
    return stop.value.args[0]


def test_kitti_eval_ring_thin_draws(monkeypatch):
    from pcdet.datasets.kitti.kitti_rings import eval_ring_thin_mask
    pts = kitti_scan()
    old = eval_ring_thin_mask(pts, {'MODE': 'random', 'STRIDE': 2}, pre_change_rng(str(SAMPLE_IDX)))
    masks = {d: kitti_loader(monkeypatch, pts, 'random', d) for d in DRAWS}
    assert np.array_equal(masks[None], old)
    rows = kitti_loader(monkeypatch, pts, 'rows', None)
    assert 0.45 < rows.mean() < 0.55
    assert np.array_equal(kitti_loader(monkeypatch, pts, 'rows', 1), rows)
    check_draws(masks, int(rows.sum()))
    for d in (1, 2):
        assert np.array_equal(kitti_loader(monkeypatch, pts, 'random', d), masks[d])


def test_kitti_empty_frame_at_every_draw(monkeypatch):
    for d in DRAWS:
        assert kitti_loader(monkeypatch, np.zeros((0, 4), np.float32), 'random', d).shape == (0,)


# ---- PandaSet EVAL_RING_THIN ------------------------------------------------------------------------------------------

SEQUENCE, FRAME_IDX = '053', 7


def pandaset_loader(tmp_path, ego_points, labels, mode, draw):
    """PandasetDataset._eval_ring_thin_mask on a bare instance, its channel labels read from a real label-cache file."""
    from pcdet.datasets.pandaset import pandaset_rings as pr
    from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset
    path = Path(pr.cache_path(str(tmp_path), SEQUENCE, FRAME_IDX))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(labels, np.int64))
    ds = object.__new__(PandasetDataset)
    ds.eval_ring_thin_cfg = cut_cfg(mode, draw, LABEL_CACHE=str(tmp_path))
    return ds._eval_ring_thin_mask({'sequence': SEQUENCE, 'frame_idx': FRAME_IDX}, ego_points)


def test_pandaset_eval_ring_thin_draws(tmp_path):
    from pcdet.datasets.pandaset.pandaset_rings import eval_thin_mask
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(64), 120)
    xyz = rng.uniform(-50.0, 50.0, (len(labels), 3))
    old = eval_thin_mask(xyz, labels, {'MODE': 'random', 'STRIDE': 2}, pre_change_rng('%s/%02d' % (SEQUENCE, FRAME_IDX)))
    masks = {d: pandaset_loader(tmp_path, xyz, labels, 'random', d) for d in DRAWS}
    assert np.array_equal(masks[None], old)
    rows = pandaset_loader(tmp_path, xyz, labels, 'rows', None)
    assert rows.sum() == (labels % 2 == 0).sum()
    assert np.array_equal(pandaset_loader(tmp_path, xyz, labels, 'rows', 2), rows)
    check_draws(masks, int(rows.sum()))
    for d in (1, 2):
        assert np.array_equal(pandaset_loader(tmp_path, xyz, labels, 'random', d), masks[d])


def test_pandaset_empty_frame_at_every_draw(tmp_path):
    for d in DRAWS:
        assert pandaset_loader(tmp_path, np.zeros((0, 3)), np.zeros(0, np.int64), 'random', d).shape == (0,)
