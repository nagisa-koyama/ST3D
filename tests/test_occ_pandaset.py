"""pcocc on PandaSet data (experiments_md 20261121_02 §5 checks 1, 2 and 8; bug_prevention_checklist rule 8).

- Check 8, pcseg's export surface as pcocc relies on it: the class order, IGNORE = -1, N_FRAMES, has_semseg, the cone,
  the frame convention and point identity (pcocc's world points moved to the ego frame equal read_frame's).
- Check 1, window identity: with the window cut to frame t, movers and visibility off and no range cut, each GT equals
  the plain voxelization of frame t's points (read_frame), voxel by voxel, on >= 40 frames.
- Check 2, round trip on a short window with everything on.
- The two pinned data files from the semseg session.
Skipped when the data is absent.
"""
import os

import numpy as np
import pytest

from pcseg import IGNORE
from pcseg.class_map import build_class_map
from pcseg.engine import cone_mask
from pcseg.pandaset_seg import N_FRAMES, has_semseg, load_poses, read_frame

from pcocc import EMPTY, IGNORE_VOXEL
from pcocc.build import SequenceGT
from pcocc.grid import PANDASET_GRID
from pcocc.poses import Poses
from pcocc.scans import (CENSUS, PANDAR64, PANDARGT, SENSOR_MODELS, SequenceScans, sensor_origins, stale_frames)

ROOT = '/home/koyama/data/pandaset/dataset'
P1_TRAIN = ['014', '050', '079', '048', '093', '091', '063', '104', '100', '092', '012', '047', '018', '006', '099',
            '085', '035', '041', '052', '105', '030', '113', '002', '084', '028', '119', '044', '005', '102', '034',
            '077', '064', '067', '058', '019', '015', '037', '095', '120', '066', '023', '071', '117', '098', '139',
            '038', '116', '046', '088', '089', '040', '033', '016', '024', '122', '039', '158', '069', '124', '123',
            '106']
P1_VAL = ['045', '059', '055', '051', '020', '097', '073', '043', '003', '101', '027', '056', '011', '078', '080',
          '109', '042', '021', '094', '057']
SCALR17 = ['road', 'traffic_sign', 'barrier', 'pedestrian', 'vegetation', 'road_marking', 'sidewalk', 'manmade',
           'traffic_cone', 'car', 'motorcycle', 'truck', 'bus', 'bicycle', 'other_vehicle', 'ground', 'driveway']

pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason='PandaSet absent')


def _frames():
    seqs = [s for s in P1_TRAIN[:8] + P1_VAL if has_semseg(ROOT, s)][:16]
    return [(s, f) for s in seqs for f in (0, 41, 79)]        # 48 frames, sequence ends included


# --- check 8: pcseg's export surface --------------------------------------------------------------------------------

def test_pcseg_exports_pinned():
    cm = build_class_map('pandaset_scalr17')
    assert cm.names == SCALR17 and cm.num_classes == 17
    assert IGNORE == -1 and N_FRAMES == 80
    assert cm(np.array([4, 1, 2, 3, 0])).tolist() == [IGNORE] * 5        # reflection, transients, stored 0
    assert cm(np.array([7, 13, 41])).tolist() == [0, 9, 7]
    assert has_semseg(ROOT, '097') and not has_semseg(ROOT, '045')
    xy = np.array([[10.0, 0.0], [10.0, 5.7], [10.0, 5.8], [-10.0, 0.0], [10.0, -5.7]])
    assert cone_mask(xy, 60.0).tolist() == [True, True, False, False, True]
    assert sum(has_semseg(ROOT, s) for s in P1_TRAIN) == 45 and sum(has_semseg(ROOT, s) for s in P1_VAL) == 15


@pytest.mark.parametrize('seq,frame', _frames())
def test_point_identity_with_read_frame(seq, frame):
    cm = build_class_map('pandaset_scalr17')
    s = SequenceScans(ROOT, seq, cm, frames=[frame])
    poses = Poses(ROOT, seq)
    pts, raw, keep = read_frame(ROOT, seq, frame, -1, load_poses(ROOT, seq)[frame])
    assert np.array_equal(keep, np.arange(len(s)))
    ego = poses.ego_from_world_static(s.xyz, frame)
    np.testing.assert_allclose(ego, pts[:, :3].astype(np.float64), rtol=0, atol=2e-5)
    assert np.array_equal(s.raw, raw) and np.array_equal(s.cls, cm(raw))
    for d in (PANDAR64, PANDARGT):
        p_d, raw_d, keep_d = read_frame(ROOT, seq, frame, d, load_poses(ROOT, seq)[frame])
        assert np.array_equal(np.flatnonzero(s.dev == d), keep_d)


def test_pose_track_round_trip_and_frame_times():
    poses = Poses(ROOT, '097')
    x = np.random.default_rng(0).normal(0, 20, (100, 3))
    for f in (0, 40, 79):
        w = poses.world_from_ego_at(poses.ego_from_world_static(x, f), poses.times[f])
        np.testing.assert_allclose(w, x, atol=1e-9)
    rot, pos = poses.at(poses.times)
    np.testing.assert_allclose(pos, poses.pos, atol=1e-9)
    # extrapolation before frame 0 continues the first segment
    _, p = poses.at([poses.times[0] - (poses.times[1] - poses.times[0])])
    np.testing.assert_allclose(p[0], 2 * poses.pos[0] - poses.pos[1], atol=1e-9)


def _reference_pose_at(poses, times):
    """pcocc's own pose interpolation as it stood at ST3D c233e3c, before pcocc.poses became an adapter over
    pcseg.render.PoseTrack: the reference the swap must reproduce."""
    from scipy.spatial.transform import Rotation, Slerp
    t = np.atleast_1d(np.asarray(times, dtype=np.float64))
    tc = np.clip(t, poses.times[0], poses.times[-1])
    rot = Slerp(poses.times, poses.rot)(tc)
    pos = np.stack([np.interp(tc, poses.times, poses.pos[:, k]) for k in range(3)], axis=1)
    quat = None
    for m, (i, j), e in ((t < poses.times[0], (0, 1), 0), (t > poses.times[-1], (-2, -1), -1)):
        if not m.any():
            continue
        s = (t[m] - poses.times[e]) / (poses.times[j] - poses.times[i])
        pos[m] = poses.pos[e] + s[:, None] * (poses.pos[j] - poses.pos[i])
        w = (poses.rot[i].inv() * poses.rot[j]).as_rotvec()
        quat = rot.as_quat() if quat is None else quat
        quat[m] = (poses.rot[e] * Rotation.from_rotvec(s[:, None] * w)).as_quat()
    if quat is not None:
        rot = Rotation.from_quat(quat)
    return rot, pos


@pytest.mark.parametrize('seq,frame', _frames())
def test_pose_track_swap_is_identity(seq, frame):
    """bug_prevention_checklist rule 8: the swap to PoseTrack, at every point time of the frame (both devices; frames
    0 and 79 reach beyond the pose ends), gives the c233e3c poses bit for bit, and so the same ray origins and ego
    coordinates."""
    from pcocc.poses import SWAP
    from pcseg.render import PoseTrack
    s = SequenceScans(ROOT, seq, build_class_map('pandaset_scalr17'), frames=[frame])
    poses = Poses(ROOT, seq)
    assert isinstance(poses, PoseTrack)
    rot, pos = poses.at(s.t)
    rr, rp = _reference_pose_at(poses, s.t)
    assert np.array_equal(pos, rp) and np.array_equal(rot.as_quat(), rr.as_quat())
    o = sensor_origins()[PANDAR64]
    ref_origin = rr.apply(np.broadcast_to(o[None], (len(s), 3)) @ SWAP) + rp
    assert np.array_equal(poses.world_from_ego_at(o[None], s.t), ref_origin)
    ref_ego = poses.rot[frame].inv().apply(s.xyz - poses.pos[frame]) @ SWAP.T
    assert np.array_equal(poses.ego_from_world_static(s.xyz, frame), ref_ego)


# --- the semseg session's data files ----------------------------------------------------------------------------------

def test_pinned_data_files():
    o = sensor_origins()
    np.testing.assert_allclose(o[PANDAR64], [0.782, 0.001, 1.860], atol=1e-3)
    np.testing.assert_allclose(o[PANDARGT], [1.220, -0.001, 1.573], atol=1e-3)
    st = stale_frames()
    assert len(st) == 103
    assert all(not v.get(PANDAR64) for v in st.values())
    gt = {s: v.get(PANDARGT, set()) for s, v in st.items()}
    assert sum(len(v) for v in gt.values()) == 86 and sum(bool(v) for v in gt.values()) == 52
    assert set().union(*gt.values()) == {0, 79}
    p1 = [s for s in P1_TRAIN + P1_VAL if has_semseg(ROOT, s)]
    assert len(p1) == 60 and sum(len(gt[s]) for s in p1) == 56
    assert os.path.isfile(CENSUS) and os.path.isfile(SENSOR_MODELS)


def test_stale_points_flagged():
    st = stale_frames()
    seq = next(s for s in P1_VAL if has_semseg(ROOT, s) and st[s].get(PANDARGT))
    f = sorted(st[seq][PANDARGT])[0]
    s = SequenceScans(ROOT, seq, build_class_map('pandaset_scalr17'), stale=st, frames=[f])
    assert np.array_equal(s.stale, s.dev == PANDARGT)


# --- check 1: window identity ---------------------------------------------------------------------------------------

def _plain_voxelization(seq, frame, devices, cm):
    """Independent path: read_frame points, class map, a dense count table, argmax (first maximum = lowest class)."""
    pts, raw, _ = read_frame(ROOT, seq, frame, -1, load_poses(ROOT, seq)[frame])
    d = np.concatenate([read_frame(ROOT, seq, frame, k, load_poses(ROOT, seq)[frame])[2] * 0 + k
                        for k in (0, 1)])                         # device per point in read_frame's -1 order
    order = np.concatenate([read_frame(ROOT, seq, frame, k, load_poses(ROOT, seq)[frame])[2] for k in (0, 1)])
    dev = np.empty(len(pts), dtype=np.int64)
    dev[order] = d
    m = np.isin(dev, devices) & (raw != 4)
    g = PANDASET_GRID
    ijk = np.floor((pts[m, :3].astype(np.float64) - g.lo) / g.voxel).astype(np.int64)
    ins = np.all((ijk >= 0) & (ijk < g.shape), axis=1)
    cls = cm(raw[m])[ins]
    # read_frame's points are float32 and pcocc's float64: a point within float32 rounding of a voxel face may land
    # on either side (seen on 2 of 42 frames, 5 points, all < 1 um from a face). Inside the grid (|coordinate| < 64 m)
    # float32 rounds by at most half an ulp, 1.9 um; the voxels points within 2 um of a face can reach are left out of
    # the comparison and counted.
    eps = 2e-6
    p = pts[m, :3].astype(np.float64)[ins]
    near = np.any(np.abs((p - g.lo) / g.voxel - np.round((p - g.lo) / g.voxel)) < eps / g.voxel, axis=1)
    loose = np.zeros(g.shape, dtype=bool)
    for sx in (-eps, eps):
        for sy in (-eps, eps):
            for sz in (-eps, eps):
                k = np.floor((p[near] + [sx, sy, sz] - g.lo) / g.voxel).astype(np.int64)
                k = k[np.all((k >= 0) & (k < g.shape), axis=1)]
                loose[k[:, 0], k[:, 1], k[:, 2]] = True
    ijk = ijk[ins]
    counts = np.zeros(g.shape + (18,), dtype=np.int16)        # < 32767 points per voxel and class in one frame
    np.add.at(counts, (ijk[:, 0], ijk[:, 1], ijk[:, 2], np.where(cls == IGNORE, 17, cls)), 1)
    label = np.zeros(g.shape, dtype=np.uint8)
    occ = counts.sum(-1) > 0
    label[occ] = IGNORE_VOXEL
    voted = counts[..., :17].sum(-1) > 0
    label[voted] = (np.argmax(counts[..., :17], axis=-1)[voted] + 1).astype(np.uint8)
    return label, loose


@pytest.mark.parametrize('seq,frame', _frames()[:42])
def test_window_identity(seq, frame):
    cm = build_class_map('pandaset_scalr17')
    sg = SequenceGT(ROOT, seq, PANDASET_GRID, cm, sensor_origins(), stale=None, range_cut=(0.0, np.inf),
                    movers=False, frames=[frame], log=lambda *a: None)
    out, _ = sg.frame(frame, window_frames=[frame], visibility=False)
    for g, devices in (('g64', (0,)), ('ggt', (1,)), ('gu', (0, 1))):
        ref, loose = _plain_voxelization(seq, frame, devices, cm)
        label, unknown = out[g]
        assert loose.sum() <= 64, (g, int(loose.sum()))
        assert np.array_equal(label[~loose], ref[~loose]), (g, int((label != ref)[~loose].sum()))
        assert np.array_equal(unknown[~loose], (ref == EMPTY)[~loose])


# --- check 2: round trip on a short window ------------------------------------------------------------------------------

def test_round_trip_short_window():
    cm = build_class_map('pandaset_scalr17')
    seq, t = '097', 40
    frames = list(range(t - 2, t + 3))
    sg = SequenceGT(ROOT, seq, PANDASET_GRID, cm, sensor_origins(), stale=stale_frames(), frames=frames,
                    log=lambda *a: None)
    out, stats = sg.frame(t)
    s = sg.scans
    for g, devices in (('g64', (0,)), ('ggt', (1,)), ('gu', (0, 1))):
        label, unknown = out[g]
        assert not np.any(unknown & (label != EMPTY))
        # every static frame-t point of the GT's devices that may occupy lands in an occupied, known voxel
        m = sg.base & np.isin(s.dev, devices) & (s.frame == t) & (sg.mover < 0)
        ijk, ins = PANDASET_GRID.index(sg.poses.ego_from_world_static(s.xyz[m], t))
        ijk = ijk[ins]
        lab = label[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        assert np.all(lab != EMPTY) and not np.any(unknown[ijk[:, 0], ijk[:, 1], ijk[:, 2]])
        # a voxel holding a voting frame-t point is never IGNORE_VOXEL
        voting = s.cls[m][ins] != IGNORE
        assert np.all(lab[voting] != IGNORE_VOXEL)
        assert stats[g]['free'] > 0 and stats[g]['unknown'] > 0
    # GU's traversal is the union of the two sensors', and GU holds every Pandar64 point where G64 does: so nothing
    # known in G64 and nothing FREE in GGT is unknown in GU. (A voxel OCCUPIED in GGT by a PandarGT mover point can be
    # unknown in GU, which places that point at the Pandar64 sibling's cuboid.)
    free_ggt = ~out['ggt'][1] & (out['ggt'][0] == EMPTY)
    assert not np.any(out['gu'][1] & ~out['g64'][1])
    assert not np.any(out['gu'][1] & free_ggt)
