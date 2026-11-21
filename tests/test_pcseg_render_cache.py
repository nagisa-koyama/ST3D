"""The S3b render cache reader and its PandasetSeg hook (pcseg/render_cache.py; experiments_md 20261116_01 §8.9).

CPU only, synthetic caches and frames: labels and intensities follow (src_off, src_idx); a source label mapping to IGNORE
stays -1; every inconsistency between the cache and the source frames raises. The >= 40-frame identity check against
the real reader runs as a CPU job (tools/analysis/pcseg_render_cache_check.py).
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'analysis'))

from pcseg import pandaset_seg                                                                    # noqa: E402
from pcseg.render_cache import RenderCache                                                        # noqa: E402


def make_cache(tmp_path, recs, device=1):
    d = tmp_path / 'cache'
    d.mkdir()
    (d / 'manifest.json').write_text(json.dumps({'kind': 'test', 'source_device': device, 'target': 't', 'depth': 3,
                                                 'tolerance_deg': 0.1}))
    for (seq, f), rec in recs.items():
        (d / seq).mkdir(exist_ok=True)
        np.savez(d / seq / ('%02d.npz' % f), **rec)
    return str(d)


def source_frames(n=(5, 7, 6)):
    """Frames 9, 10, 11 of a fake sequence: intensity = frame + index / 100, raw label = 100 * frame + index."""
    out = {}
    for g, k in zip((9, 10, 11), n):
        pts = np.zeros((k, 4), np.float32)
        pts[:, :3] = np.random.default_rng(g).normal(size=(k, 3))
        pts[:, 3] = g + np.arange(k) / 100
        out[g] = (pts, 100 * g + np.arange(k))
    return out


def rec_for(src_off, src_idx, n_src=(5, 7, 6)):
    m = len(src_off)
    return {'src_off': np.asarray(src_off, np.int8), 'src_idx': np.asarray(src_idx, np.int32),
            'xyz': np.arange(3 * m, dtype=np.float32).reshape(m, 3), 'n_src': np.asarray(n_src, np.int32),
            'mixed': np.zeros(m, bool)}


def test_labels_and_intensity_follow_the_index(tmp_path):
    frames = source_frames()
    rec = rec_for([0, -1, 1, 0, 1], [6, 0, 5, 2, 0])
    cache = RenderCache({'CACHE': make_cache(tmp_path, {('001', 10): rec})}, device=1)
    pts, raw = cache.load(lambda g: frames[g], '001', 10)
    np.testing.assert_array_equal(raw, [1006, 900, 1105, 1002, 1100])
    np.testing.assert_allclose(pts[:, 3], [10.06, 9.00, 11.05, 10.02, 11.00], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(pts[:, :3], rec['xyz'])            # the cached coordinates, not the source's


def test_cache_holds_no_label_and_reads_only_used_frames(tmp_path):
    frames = source_frames()
    rec = rec_for([0, 0], [1, 2], n_src=(-1, 7, -1))                  # a sequence start / end: one frame only
    path = make_cache(tmp_path, {('001', 10): rec})
    with np.load(os.path.join(path, '001', '10.npz')) as z:
        assert not {'label', 'labels', 'segment', 'raw', 'class'} & set(z.files)
    read = []
    RenderCache({'CACHE': path}, 1).load(lambda g: read.append(g) or frames[g], '001', 10)
    assert read == [10]


@pytest.mark.parametrize('bad, msg', [
    (dict(src_off=[0, 0], src_idx=[1, 7]), 'out of range'),
    (dict(src_off=[0, 0], src_idx=[1, -1]), 'out of range'),
    (dict(src_off=[0, 2], src_idx=[1, 1]), 'offsets'),
    (dict(src_off=[0, 1], src_idx=[1, 1], n_src=(5, 7, -1)), 'records no source frame'),
    (dict(src_off=[0, 0], src_idx=[1, 1], n_src=(5, 8, 6)), 'was built on'),
])
def test_inconsistent_cache_raises(tmp_path, bad, msg):
    frames = source_frames()
    rec = rec_for(bad['src_off'], bad['src_idx'], bad.get('n_src', (5, 7, 6)))
    cache = RenderCache({'CACHE': make_cache(tmp_path, {('001', 10): rec})}, 1)
    with pytest.raises(ValueError, match=msg):
        cache.load(lambda g: frames[g], '001', 10)


def test_bad_spec_device_and_missing_frame_raise(tmp_path):
    path = make_cache(tmp_path, {('001', 10): rec_for([0], [0])})
    with pytest.raises(ValueError, match='renders device'):
        RenderCache({'CACHE': path}, device=0)
    with pytest.raises(ValueError, match='unknown keys'):
        RenderCache({'CACHE': path, 'KIND': 'x'}, 1)
    with pytest.raises(FileNotFoundError):
        RenderCache({'CACHE': str(tmp_path / 'none')}, 1)
    with pytest.raises(FileNotFoundError, match='not in the cache'):
        RenderCache({'CACHE': path}, 1).record('001', 11)


def test_non_finite_coordinates_raise(tmp_path):
    rec = rec_for([0, 0], [0, 1])
    rec['xyz'][1, 2] = np.nan
    cache = RenderCache({'CACHE': make_cache(tmp_path, {('001', 10): rec})}, 1)
    with pytest.raises(ValueError, match='non-finite'):
        cache.record('001', 10)


def fake_dataset(tmp_path, monkeypatch, frames, ops):
    root = tmp_path / 'root'
    (root / '001' / 'annotations' / 'semseg').mkdir(parents=True)
    calls = []

    def read_frame(r, seq, g, device, pose):
        calls.append(g)
        pts, raw = frames[g]
        return pts, raw, np.arange(len(pts))
    monkeypatch.setattr(pandaset_seg, 'read_frame', read_frame)
    monkeypatch.setattr(pandaset_seg.PandasetSeg, 'pose', lambda self, s, f: None)
    cfg = {'ROOT': str(root), 'SEQUENCES': ['001'], 'LIDAR_DEVICE': 1, 'SHIFT_COOR': [0.0, 0.0, 0.3],
           'CLASS_MAP': 'pandaset_scalr17', 'GRID_SIZE': 0.05, 'FACTOR_OPS': ops}
    return pandaset_seg.PandasetSeg(cfg, training=True), calls


def raw_ids():
    """(a raw PandaSet id the class map sends to IGNORE, two raw ids of different model classes)."""
    from pcseg.class_map import build_class_map
    cmap = build_class_map('pandaset_scalr17')
    ids = np.arange(0, 43)
    lab = cmap(ids)
    valid = ids[lab >= 0]
    other = valid[lab[valid] != lab[valid[0]]][0]
    return cmap, int(ids[lab == -1][0]), int(valid[0]), int(other)


def labelled_frames():
    cmap, ignore_raw, a, b = raw_ids()
    frames = source_frames()
    for g in frames:
        frames[g] = (frames[g][0], np.full(len(frames[g][0]), b if g == 9 else a))
    frames[10][1][3] = ignore_raw
    return frames, cmap, a, b


def test_dataset_hook_moves_labels_by_the_index_and_keeps_ignore(tmp_path, monkeypatch):
    frames, cmap, valid_raw, other_raw = labelled_frames()
    rec = rec_for([0, 0, -1], [3, 4, 0])
    ds, calls = fake_dataset(tmp_path, monkeypatch, frames, {'RENDER': {'CACHE': make_cache(tmp_path, {('001', 10): rec})}})
    assert ds.factor_ops is None and ds.render is not None
    coord, strength, segment, key = ds.load(10)
    assert key == '001/10'
    np.testing.assert_allclose(coord, rec['xyz'] + np.float32([0, 0, 0.3]), atol=1e-6)   # SHIFT_COOR after the render
    assert segment[0] == -1                                           # IGNORE at the source stays IGNORE
    assert segment[1] == cmap(np.array([valid_raw]))[0]
    assert segment[2] == cmap(np.array([other_raw]))[0] != segment[1]  # frame 9's label, through the map
    np.testing.assert_allclose(strength[:, 0], [10.03, 10.04, 9.00], atol=1e-6)
    assert sorted(set(calls)) == [9, 10]


def test_dataset_hook_runs_other_ops_on_the_rendered_cloud(tmp_path, monkeypatch):
    frames, _, _, _ = labelled_frames()
    rec = rec_for([0, 0, 0], [0, 1, 2])
    rec['xyz'] = np.float32([[10, 0, 0], [-10, 0, 0], [10, 1, 0]])    # the middle point is behind the sensor
    ds, _ = fake_dataset(tmp_path, monkeypatch, frames, {
        'RENDER': {'CACHE': make_cache(tmp_path, {('001', 10): rec})}, 'CONE': {'FOV_DEG': 60.0, 'HEADING_DEG': 0.0}})
    assert ds.factor_ops is not None
    coord, strength, segment, _ = ds.load(10)
    np.testing.assert_allclose(coord[:, 0], [10, 10])
    np.testing.assert_allclose(strength[:, 0], [10.00, 10.02], atol=1e-6)


def test_mixed_flag():
    from pcseg_render_cache import mixed_flag
    # cell 0: winner (range 10, class 1) and a class-2 point at 10.3 m -> mixed; cell 1: a class-2 point 5 m behind the
    # winner -> not mixed; cell 2: an IGNORE point near the winner -> not mixed; cell -1 never counts.
    cell = np.array([0, 0, 1, 1, 2, 2, -1])
    rng = np.array([10.0, 10.3, 20.0, 25.0, 30.0, 30.1, 30.0])
    lab = np.array([1, 2, 1, 2, 3, -1, 4])
    cells, win = np.array([0, 1, 2]), np.array([0, 2, 4])
    np.testing.assert_array_equal(mixed_flag(cell, rng, lab, cells, win), [True, False, False])


def write_track(tmp_path, times, pos, yaw_deg):
    from scipy.spatial.transform import Rotation
    d = tmp_path / 'seq' / 'lidar'
    d.mkdir(parents=True)
    q = Rotation.from_euler('z', yaw_deg, degrees=True).as_quat()
    (d / 'poses.json').write_text(json.dumps([{'position': dict(zip('xyz', p)), 'heading': dict(zip('xyzw', qq))}
                                              for p, qq in zip(np.asarray(pos).tolist(), q.tolist())]))
    (d / 'timestamps.json').write_text(json.dumps(list(times)))
    return str(tmp_path)


def test_pose_track_extrapolates_beyond_both_ends(tmp_path):
    """A scan before the first pose (Pandar64 frame 0) or after the last one is placed with the end segment's motion,
    not with the end pose held (the clip this replaces would put t0 - 0.05 s at the first pose)."""
    from pcseg.render import PoseTrack
    times = [10.0, 10.1, 10.2, 10.3]
    v, w = np.array([8.0, 1.0, 0.0]), 5.0                             # m/s, deg/s yaw
    pos = [v * (t - 10.0) for t in times]
    yaw = [w * (t - 10.0) for t in times]
    track = PoseTrack(write_track(tmp_path, times, pos, yaw), 'seq')
    tq = np.array([9.95, 9.9, 10.05, 10.25, 10.35])
    rot, p = track._at(tq)
    np.testing.assert_allclose(p, v[None] * (tq - 10.0)[:, None], atol=1e-9)
    np.testing.assert_allclose(rot.as_euler('xyz', degrees=True)[:, 2], w * (tq - 10.0), atol=1e-9)
    xyz = np.array([[5.0, 2.0, 1.0]] * len(tq))
    np.testing.assert_allclose(track.world_from_ego_at(track.ego_at(xyz, tq), tq), xyz, atol=1e-9)


def test_dedup_scans_keeps_one_copy_closest_to_the_frame():
    from pcseg_render_cache import dedup_scans

    def scan(t0, n=4):
        return (np.zeros((n, 3)), t0 + np.linspace(0, 0.096, n))
    # frames 0 and 1 hold the same scan (same absolute times), frame 2 its own
    src = {0: scan(0.0), 1: scan(0.0), 2: scan(0.1)}
    assert dedup_scans(0, {g: src[g] for g in (0, 1)}) == ([0], [1])
    assert dedup_scans(1, src) == ([1, 2], [0])
    assert dedup_scans(2, {g: src[g] for g in (1, 2)}) == ([1, 2], [])
    assert dedup_scans(5, {4: scan(0.4), 5: scan(0.5), 6: scan(0.6)}) == ([4, 5, 6], [])
