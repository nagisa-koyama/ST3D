"""Waymo TOP rows from the stored point ORDER and the RING_PATTERN thinning (pcdet/datasets/waymo/waymo_rings.py).

A synthetic TOP scan is built the way the re-extractor stores it - valid range-image pixels row-major, TOP first,
in the vehicle frame through a yawed, offset extrinsic - so the recovery is tested against known rows, including the
cases the real data contain: open-sky rows with few or NO valid pixels, points wrapped across the range-image seam by
compensation, and no-label-zone points filtered after recovery.
"""
import sys
from pathlib import Path

import numpy as np
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pcdet.datasets.waymo.waymo_rings import (W_TOP, match_rows_to_beams, ring_pattern_mask,  # noqa: E402
                                              ring_pattern_points, rows_from_order, top_beam_ids)

# Waymo-like non-uniform inclinations, +2.4 .. -17.6 deg, dense near the horizon
INC_DEG = np.concatenate([np.linspace(2.4, -2.0, 24), np.linspace(-2.25, -8.0, 20), np.linspace(-8.5, -17.6, 20)])
INC = np.radians(INC_DEG)


def _extrinsic(yaw_deg=148.0, t=(1.43, 0.0, 2.18)):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    E = np.eye(4); E[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; E[:3, 3] = t
    return E


def _scan(E, valid_frac=0.6, empty_rows=(), sky_rows=(), seam_jitter=0, seed=0):
    """(xyz vehicle frame, true beam id per point) for a row-major scan, as the re-extractor writes it."""
    rng = np.random.default_rng(seed)
    yaw = np.arctan2(E[1, 0], E[0, 0])
    pts, beam = [], []
    for b, inc in enumerate(INC):
        if b in empty_rows:
            continue
        frac = 0.02 if b in sky_rows else valid_frac
        cols = np.nonzero(rng.random(W_TOP) < frac)[0]
        az = ((W_TOP - cols - 0.5) / W_TOP * 2 - 1) * np.pi - yaw          # reconstruct(): sensor-frame azimuth
        if seam_jitter:
            near = (cols < 3) | (cols > W_TOP - 4)
            az[near] += rng.uniform(-seam_jitter, seam_jitter, near.sum()) * 2 * np.pi / W_TOP
        r = rng.uniform(3, 70, len(cols))
        p = np.stack([r * np.cos(inc) * np.cos(az), r * np.cos(inc) * np.sin(az), r * np.sin(inc)], 1)
        pts.append(p @ E[:3, :3].T + E[:3, 3]); beam.append(np.full(len(cols), b))
    return np.concatenate(pts), np.concatenate(beam)


def test_rows_recovered_from_order():
    E = _extrinsic()
    xyz, true = _scan(E)
    beam, _, _, _, diag = top_beam_ids(xyz, E, INC)
    assert diag['rows'] == 64 and diag['strict']
    assert np.array_equal(beam, true)


def test_seam_wrapped_points_do_not_split_rows():
    E = _extrinsic()
    xyz, true = _scan(E, seam_jitter=2.5, seed=1)
    beam, _, _, _, diag = top_beam_ids(xyz, E, INC)
    assert diag['rows'] == 64
    assert np.mean(beam == true) > 0.999          # only a few seam points may land in the neighbouring row


def test_sky_rows_and_a_missing_top_row_are_mapped_by_elevation():
    E = _extrinsic()
    xyz, true = _scan(E, empty_rows=(0,), sky_rows=(1, 2, 3), seed=2)
    beam, _, _, _, diag = top_beam_ids(xyz, E, INC)
    assert diag['rows'] == 63                      # the empty row is invisible in the order...
    assert np.array_equal(beam, true)              # ...and the monotone match skips it


def test_match_is_identity_when_one_row_per_beam():
    d = np.sort(INC)[::-1]
    perturbed = d + np.radians(0.3)                # all medians shifted (compensation over a bump)
    assert np.array_equal(match_rows_to_beams(perturbed, d), np.arange(64))


def test_empty_top_block():
    E = _extrinsic()
    assert len(rows_from_order(np.zeros(0))) == 0
    raw = np.zeros((5, 6), np.float32); raw[:, 5] = -1
    out = ring_pattern_points(raw, [0, 5, 0, 0, 0], {'extrinsic': E, 'inclinations': INC},
                              EasyDict(SPACING_DEG=1.33, AZ_RES_DEG=0.332, TOP_ONLY=True))
    assert out.shape == (0, 5)


def test_pattern_spacing_and_one_point_per_bin():
    E = _extrinsic()
    xyz, _ = _scan(E, valid_frac=0.9, seed=3)
    beam, col, _, inc_desc, _ = top_beam_ids(xyz, E, INC)
    keep, kept = ring_pattern_mask(beam, col, inc_desc, 1.33, 0.332, 0.4, 0.7)
    gaps = -np.diff(np.degrees(inc_desc[kept]))
    assert 13 <= len(kept) <= 17                   # (2.4 + 17.6) / 1.33 ~ 15 lines
    assert gaps.min() > 1.33 - 0.5 and gaps.max() < 1.33 + 0.5
    n_bins = int(round(360 / 0.332))
    b = np.floor((col / W_TOP + 0.7 / n_bins) * n_bins).astype(int) % n_bins
    key = beam[keep] * n_bins + b[keep]
    assert len(np.unique(key)) == keep.sum()       # at most one point per row per azimuth bin
    assert keep.sum() <= len(kept) * n_bins
    keep2, kept2 = ring_pattern_mask(beam, col, inc_desc, 1.33, 0.332, 0.9, 0.7)
    assert not np.array_equal(kept, kept2)         # the vertical phase moves the selected rows


def test_points_nlz_filter_side_lidars_and_columns():
    E = _extrinsic()
    xyz, _ = _scan(E, seed=4)
    rng = np.random.default_rng(0)
    top = np.concatenate([xyz, rng.random((len(xyz), 2)), np.full((len(xyz), 1), -1.0)], 1)
    top[::7, 5] = 1.0                               # no-label-zone points, interleaved in the row order
    side = np.concatenate([rng.random((50, 5)), np.full((50, 1), -1.0)], 1)
    raw = np.concatenate([top, side]).astype(np.float32)
    calib = {'extrinsic': E, 'inclinations': INC}
    cfg = EasyDict(SPACING_DEG=1.33, AZ_RES_DEG=0.332, TOP_ONLY=True)
    out = ring_pattern_points(raw, [len(top), 50, 0, 0, 0], calib, cfg, rng=np.random.default_rng(1))
    assert out.shape[1] == 5 and 0 < len(out) < 0.4 * len(top)
    assert np.all(out[:, 3] <= np.tanh(1.0) + 1e-6)          # intensity squashed as get_lidar does
    top32 = raw[:len(top)]
    top_xyz = {tuple(p) for p in top32[top32[:, 5] == -1, :3]}
    assert all(tuple(p) in top_xyz for p in out[:, :3])   # TOP only, and no NLZ point survives
    cfg_all = EasyDict(SPACING_DEG=1.33, AZ_RES_DEG=0.332, TOP_ONLY=False)
    out_all = ring_pattern_points(raw, [len(top), 50, 0, 0, 0], calib, cfg_all, rng=np.random.default_rng(1))
    assert len(out_all) == len(out) + 50


def test_counts_must_match_the_file():
    E = _extrinsic()
    raw = np.zeros((10, 6), np.float32); raw[:, 5] = -1
    try:
        ring_pattern_points(raw, [4, 5, 0, 0, 0], {'extrinsic': E, 'inclinations': INC},
                            EasyDict(SPACING_DEG=1.33, AZ_RES_DEG=0.332))
    except AssertionError:
        return
    raise AssertionError('mismatched per-lidar counts must be refused')


def test_absent_key_leaves_the_loader_unchanged():
    from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
    ds = object.__new__(WaymoDataset)
    ds.dataset_cfg = EasyDict(MAX_SWEEPS=1); ds.training = True; ds.ring_pattern_cfg = None
    ds.eval_top_thin_cfg = None; ds.train_top_thin_cfg = None   # the other opt-in TOP cuts (2d7892d), absent here too
    sentinel = np.ones((3, 5), np.float32)
    ds.get_lidar = lambda seq, idx, *counts: sentinel
    info = {'point_cloud': {'lidar_sequence': 's', 'sample_idx': 0}}
    assert ds.get_lidar_with_sweeps(info) is sentinel
    ds.ring_pattern_cfg = EasyDict(TOP_CALIB='x'); ds.training = False      # evaluation: untouched
    assert ds.get_lidar_with_sweeps(info) is sentinel


def test_ring_labels_for_beam_distillation():
    from pcdet.datasets.waymo.waymo_rings import ring_labelled_points
    from pcdet.utils.beam_downsample_utils import generate_mask
    E = _extrinsic()
    xyz, true = _scan(E, seed=5)
    rng = np.random.default_rng(0)
    top = np.concatenate([xyz, rng.random((len(xyz), 2)), np.full((len(xyz), 1), -1.0)], 1)
    top[::5, 5] = 1.0
    side = np.concatenate([rng.random((40, 5)), np.full((40, 1), -1.0)], 1)
    raw = np.concatenate([top, side]).astype(np.float32)
    pts, label = ring_labelled_points(raw, [len(top), 40, 0, 0, 0], {'extrinsic': E, 'inclinations': INC})
    keep = raw[:, 5] == -1
    assert len(pts) == len(label) == keep.sum()
    expect = np.concatenate([63 - true, np.full(40, -1)])[keep]   # ascending elevation, side lidars -1
    assert np.array_equal(label, expect)
    m = generate_mask(np.zeros(len(label)), label, 64, beam_ratio=2)
    assert set(np.unique(label[m])) == set(range(0, 64, 2))        # the student keeps every other TOP row, no side


def test_waymo_attach_ring_labels_override():
    from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
    ds = object.__new__(WaymoDataset)
    d = {'points': np.zeros((4, 5), np.float32), 'waymo_ring_label': np.array([0, 1, -1, 63])}
    out = ds._attach_ring_labels(d, EasyDict(NUM_BEAMS=64))
    assert out['points'].shape == (4, 6) and 'waymo_ring_label' not in out
    assert np.array_equal(out['points'][:, -1], [0, 1, -1, 63])
    try:
        ds._attach_ring_labels({'points': np.zeros((4, 5), np.float32)}, EasyDict(NUM_BEAMS=64))
    except AssertionError:
        return
    raise AssertionError('Waymo must refuse to fall back to elevation clustering')


# ---- AZ_RES_DEG: null, the vertical-only ablation of 27807 (experiments_md 20261005_01 §16) ----

def _mask_as_27807(beam, col, inc_desc, spacing_deg, az_res_deg, phase_v, phase_h, width=W_TOP):
    """ring_pattern_mask as 27807 / 27840 ran it (ST3D b3c3eec), transcribed literally: the reference for the
    'numeric key unchanged' tests below."""
    lo, hi = inc_desc.min(), inc_desc.max()
    sp = np.radians(spacing_deg)
    j0 = int(np.floor((lo - phase_v * sp) / sp)) - 1
    j1 = int(np.ceil((hi - phase_v * sp) / sp)) + 1
    targets = (np.arange(j0, j1 + 1) + phase_v) * sp
    targets = targets[(targets >= lo) & (targets <= hi)]
    kept_beams = np.unique(np.argmin(np.abs(inc_desc[None, :] - targets[:, None]), axis=1))
    keep = np.isin(beam, kept_beams)
    n_bins = int(round(360.0 / az_res_deg))
    b = np.floor((col / width + phase_h / n_bins) * n_bins).astype(np.int64) % n_bins
    key = beam.astype(np.int64) * n_bins + b
    idx = np.nonzero(keep)[0]
    _, first = np.unique(key[idx], return_index=True)
    out = np.zeros(len(beam), dtype=bool)
    out[idx[first]] = True
    return out, kept_beams


def _raw_frame(seed, nlz_every=7, n_side=50):
    E = _extrinsic()
    xyz, true = _scan(E, valid_frac=0.8, seed=seed)
    rng = np.random.default_rng(seed + 100)
    top = np.concatenate([xyz, rng.random((len(xyz), 2)), np.full((len(xyz), 1), -1.0)], 1)
    top[::nlz_every, 5] = 1.0
    side = np.concatenate([rng.random((n_side, 5)), np.full((n_side, 1), -1.0)], 1)
    return np.concatenate([top, side]).astype(np.float32), [len(top), n_side, 0, 0, 0], \
        {'extrinsic': E, 'inclinations': INC}, true


def _ringpattern_cfg(name):
    import os
    from pcdet.config import cfg_from_yaml_file
    prev = os.getcwd()
    os.chdir(ROOT / 'tools')                       # _BASE_CONFIG_ paths are tools/-relative
    try:
        return cfg_from_yaml_file('cfgs/da-ieee-access/%s.yaml' % name, EasyDict())
    finally:
        os.chdir(prev)


def test_numeric_resolution_is_bit_identical_to_27807():
    E = _extrinsic()
    for seed, (pv, ph) in zip((6, 7, 8), ((0.0, 0.0), (0.37, 0.81), (0.999, 0.5))):
        xyz, _ = _scan(E, valid_frac=0.9, seed=seed)
        beam, col, _, inc_desc, _ = top_beam_ids(xyz, E, INC)
        keep, kept = ring_pattern_mask(beam, col, inc_desc, 1.33, 0.332, pv, ph)
        ref, ref_kept = _mask_as_27807(beam, col, inc_desc, 1.33, 0.332, pv, ph)
        assert np.array_equal(keep, ref) and np.array_equal(kept, ref_kept)


def test_27807_config_cloud_is_bit_identical():
    """27807's resolved RING_PATTERN block through ring_pattern_points, against the transcribed mask on the same
    draws: the thinned training cloud is unchanged byte for byte."""
    rp = _ringpattern_cfg('centerpoint-ringpattern-waymo2nuscenes').DATA_CONFIG.RING_PATTERN
    assert rp.AZ_RES_DEG == 0.332
    for seed in (9, 10):
        raw, counts, calib, _ = _raw_frame(seed)
        out = ring_pattern_points(raw, counts, calib, rp, rng=np.random.default_rng(seed))
        draw = np.random.default_rng(seed)
        pv, ph = draw.random(), draw.random()
        top = raw[:counts[0]]
        beam, col, _, inc_desc, _ = top_beam_ids(top[:, :3], calib['extrinsic'], calib['inclinations'])
        ref, _ = _mask_as_27807(beam, col, inc_desc, rp.SPACING_DEG, rp.AZ_RES_DEG, pv, ph)
        pts = top[ref]
        pts = pts[pts[:, 5] == -1]
        expect = np.array(pts[:, 0:5], dtype=raw.dtype)
        expect[:, 3] = np.tanh(expect[:, 3])
        assert out.dtype == expect.dtype and out.tobytes() == expect.tobytes()


def test_vertical_only_keeps_every_point_of_the_same_rows():
    E = _extrinsic()
    xyz, true = _scan(E, valid_frac=0.9, seed=11)
    beam, col, _, inc_desc, _ = top_beam_ids(xyz, E, INC)
    for pv in (0.0, 0.42, 0.93):
        keep_v, kept_v = ring_pattern_mask(beam, col, inc_desc, 1.33, None, pv, 0.7)
        keep_b, kept_b = ring_pattern_mask(beam, col, inc_desc, 1.33, 0.332, pv, 0.7)
        assert np.array_equal(kept_v, kept_b)                    # the vertical lattice is untouched
        assert np.array_equal(keep_v, np.isin(beam, kept_v))     # every point of a kept row, nothing else
        assert np.all(keep_v[keep_b])                            # a superset of the binned selection
        assert keep_v.sum() > keep_b.sum()
        assert set(np.unique(true[keep_v])) == set(kept_v)       # whole TRUE rows (recovery is exact here)
    # phase_h is unused
    a, _ = ring_pattern_mask(beam, col, inc_desc, 1.33, None, 0.42, 0.0)
    b, _ = ring_pattern_mask(beam, col, inc_desc, 1.33, None, 0.42, 0.99)
    assert np.array_equal(a, b)


def test_vertical_only_points_nlz_side_lidars_and_vertical_phase_stream():
    raw, counts, calib, _ = _raw_frame(12)
    cfg_v = EasyDict(SPACING_DEG=1.33, AZ_RES_DEG=None, TOP_ONLY=True)
    cfg_b = EasyDict(SPACING_DEG=1.33, AZ_RES_DEG=0.332, TOP_ONLY=True)
    out_v = ring_pattern_points(raw, counts, calib, cfg_v, rng=np.random.default_rng(3))
    out_b = ring_pattern_points(raw, counts, calib, cfg_b, rng=np.random.default_rng(3))
    top = raw[:counts[0]]
    beam, _, _, inc_desc, _ = top_beam_ids(top[:, :3], calib['extrinsic'], calib['inclinations'])
    draw = np.random.default_rng(3)
    pv = draw.random()
    keep, _ = ring_pattern_mask(beam, None, inc_desc, 1.33, None, pv, None)
    pts = top[keep & (top[:, 5] == -1)]
    assert np.array_equal(out_v[:, :3], pts[:, :3])              # exactly the kept rows' non-NLZ TOP points, in order
    assert len(out_v) > 2 * len(out_b)                           # ~2.4 Waymo columns per 0.332-deg bin
    xyz_v = {tuple(p) for p in out_v[:, :3]}
    assert all(tuple(p) in xyz_v for p in out_b[:, :3])          # same rows: the binned cloud is a subset
    out_all = ring_pattern_points(raw, counts, calib, EasyDict(cfg_v, TOP_ONLY=False), rng=np.random.default_rng(3))
    assert len(out_all) == len(out_v) + counts[1]


def test_vertical_only_empty_top_block_and_absent_key_still_refused():
    E = _extrinsic()
    raw = np.zeros((5, 6), np.float32); raw[:, 5] = -1
    out = ring_pattern_points(raw, [0, 5, 0, 0, 0], {'extrinsic': E, 'inclinations': INC},
                              EasyDict(SPACING_DEG=1.33, AZ_RES_DEG=None, TOP_ONLY=True))
    assert out.shape == (0, 5)
    raw, counts, calib, _ = _raw_frame(13)
    try:                                           # absent is NOT "off": only an explicit null switches binning off
        ring_pattern_points(raw, counts, calib, EasyDict(SPACING_DEG=1.33, TOP_ONLY=True))
    except KeyError:
        return
    raise AssertionError('a RING_PATTERN block without AZ_RES_DEG must be refused')


def test_vertical_only_config_differs_from_27807_in_az_res_only():
    base = _ringpattern_cfg('centerpoint-ringpattern-waymo2nuscenes')
    vonly = _ringpattern_cfg('centerpoint-ringpattern-vonly-waymo2nuscenes')
    assert vonly.DATA_CONFIG.RING_PATTERN.AZ_RES_DEG is None
    vonly.DATA_CONFIG.RING_PATTERN.AZ_RES_DEG = base.DATA_CONFIG.RING_PATTERN.AZ_RES_DEG
    vonly.pop('_BASE_CONFIG_'); base.pop('_BASE_CONFIG_')          # the chain itself, one link longer
    assert vonly == base
