"""pcseg.render: the index-returning z-buffer and its label rule (experiments_md episodes/20261116_01 §8.9, S3b).

- the winner of every cell is the nearest input point (ties: the lower index); a point off the lattice never wins;
- points generated on a lattice's rays render back to themselves (identity);
- a filled point carries src = -1 and label -1; a winner's label -1 stays -1 (pcseg.IGNORE passes through);
- PoseTrack's frame-time transform equals pcseg.pandaset_seg.read_frame on >= 40 TRAIN frames, and its per-time
  transform at the frame time equals it; a real PandarGT frame renders back onto its own rays, time-aware.
The renderer has no random draw, so there is no frame-keyed draw to test.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pcseg.render import (IGNORE, PoseTrack, RingLattice, TemplateLattice, angles_deg, assign_time_aware,  # noqa
                          carry_labels, nearest_time, render, render_cells, unit_from_angles, zbuffer)

ROOT = '/home/koyama/data/pandaset/dataset'


def test_zbuffer_winner_is_nearest():
    rs = np.random.default_rng(0)
    n = 5000
    cell = rs.integers(-1, 300, n)
    rng = rs.uniform(1, 50, n)
    rng[rs.choice(n, 200, replace=False)] = 7.0              # ties
    cells, win = zbuffer(cell, rng)
    assert np.all(np.diff(cells) > 0)
    for c, w in zip(cells, win):
        idx = np.flatnonzero(cell == c)
        best = idx[rng[idx] == rng[idx].min()].min()
        assert w == best
    assert set(cells) == set(cell[cell >= 0])
    assert np.all(cell[win] >= 0)


def test_off_lattice_points_never_win():
    cells, win = zbuffer(np.array([-1, -1, 3]), np.array([0.5, 0.1, 9.0]))
    assert cells.tolist() == [3] and win.tolist() == [2]
    cells, win = zbuffer(np.array([-1, -1]), np.array([1.0, 2.0]))
    assert len(cells) == 0 and len(win) == 0


def ring_points(lat, cells, rng):
    return lat.rays(cells) * np.asarray(rng)[:, None]


def test_identity_on_ring_lattice():
    el = np.array([-25.0, -15.0, -5.0, -1.0, 0.0, 1.0, 5.0, 15.0])
    lat = RingLattice(el, 0.2, az_phase_deg=0.07, row_daz_deg=np.linspace(-1, 1, len(el)))
    rs = np.random.default_rng(1)
    cells = rs.choice(lat.n_cells, 3000, replace=False)
    rng = rs.uniform(2, 80, len(cells))
    q = ring_points(lat, cells, rng)
    out = render(q, lat)
    assert len(out['src']) == len(cells)
    assert np.array_equal(cells[out['src']], out['cell'])
    np.testing.assert_allclose(out['q'], q[out['src']], atol=1e-9)
    # with a footprint, points 0.5 deg off their row are dropped
    lat_fp = RingLattice(el, 0.2, el_tol_deg=0.07)
    a, e, r = angles_deg(q)
    cell, _ = lat_fp.assign(unit_from_angles(a, e + 0.5) * r[:, None])
    assert np.all(cell == -1)


def test_template_lattice_tolerance():
    lat = TemplateLattice(np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.15]]), tol_deg=0.05)
    q = unit_from_angles(np.array([0.01, 0.09, 0.0, 0.3]), np.array([0.0, 0.01, 0.14, 0.0])) * 10
    cell, rng = lat.assign(q)
    assert cell.tolist() == [0, 1, 2, -1]
    np.testing.assert_allclose(rng, 10.0)


def test_fill_points_carry_minus_one():
    lat = RingLattice(np.array([-2.0, -1.0, 0.0, 1.0]), 1.0)
    col = 180                                              # azimuth bin [0, 1) deg
    cells = np.array([0 * lat.n_cols + col, 2 * lat.n_cols + col, 3 * lat.n_cols + col + 1])
    q = ring_points(lat, cells, [10.0, 10.2, 30.0])
    labels = np.array([4, 7, IGNORE])
    out = render(q, lat, fill=True, k_rows=4, dr_max=0.3)
    filled = out['src'] < 0
    assert filled.sum() == 1
    assert out['cell'][filled][0] == 1 * lat.n_cols + col
    np.testing.assert_allclose(out['rng'][filled], 10.1)
    lab = carry_labels(labels, out['src'])
    assert lab[filled].tolist() == [IGNORE]
    by_cell = dict(zip(out['cell'].tolist(), lab.tolist()))
    assert by_cell[0 * lat.n_cols + col] == 4 and by_cell[2 * lat.n_cols + col] == 7
    assert by_cell[3 * lat.n_cols + col + 1] == IGNORE      # a winner labelled -1 stays -1


def test_minus_one_label_passes_through_the_zbuffer():
    lat = TemplateLattice(np.array([[0.0, 0.0]]), tol_deg=0.05)
    q = unit_from_angles(np.zeros(3), np.zeros(3)) * np.array([[5.0], [9.0], [12.0]])
    out = render_cells(*lat.assign(q), lat)
    assert out['src'].tolist() == [0]
    assert carry_labels(np.array([IGNORE, 3, 3]), out['src']).tolist() == [IGNORE]
    assert carry_labels(np.array([2, IGNORE, 3]), out['src']).tolist() == [2]
    assert carry_labels(np.array([2, 5]), np.array([1, -1, 0])).tolist() == [5, IGNORE, 2]


def _train_frames(n_seq, frames):
    import yaml
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../tools/cfgs/pcseg/spunet-pandaset64.yaml')
    seqs = [s for s in yaml.safe_load(open(cfg))['DATA']['SPLITS']['train']
            if os.path.isdir(os.path.join(ROOT, s, 'annotations', 'semseg'))]
    return [(s, f) for s in seqs[:n_seq] for f in frames]


@pytest.mark.skipif(not os.path.isdir(ROOT), reason='PandaSet not on this machine')
def test_posetrack_matches_read_frame_on_40_frames():
    import pandas as pd
    from pcseg.pandaset_seg import load_poses, read_frame
    jobs = _train_frames(20, (5, 66))
    assert len(jobs) >= 40
    for seq, f in jobs:
        track = PoseTrack(ROOT, seq)
        pts, _, keep = read_frame(ROOT, seq, f, -1, load_poses(ROOT, seq)[f])
        lidar = pd.read_pickle(os.path.join(ROOT, seq, 'lidar', '%02d.pkl.gz' % f))
        xyz = lidar[['x', 'y', 'z']].to_numpy()[keep]
        ego = track.ego_static(xyz, f)
        np.testing.assert_allclose(ego, pts[:, :3], atol=2e-3, rtol=0)   # read_frame returns float32
        np.testing.assert_allclose(track.ego_at(xyz[:500], track.times[f]), ego[:500], atol=1e-6)
        back = track.world_from_ego_at(ego[:500], track.times[f])
        np.testing.assert_allclose(back, xyz[:500], atol=1e-6)


@pytest.mark.skipif(not os.path.isdir(ROOT), reason='PandaSet not on this machine')
def test_pandargt_frame_renders_onto_its_own_rays():
    import pandas as pd
    for seq, f in _train_frames(2, (40,)):
        track = PoseTrack(ROOT, seq)
        lidar = pd.read_pickle(os.path.join(ROOT, seq, 'lidar', '%02d.pkl.gz' % f))
        g = lidar[lidar['d'] == 1]
        xyz, t = g[['x', 'y', 'z']].to_numpy(), g['t'].to_numpy()
        O, R = np.array([1.16, 0.0, 1.58]), np.eye(3)
        q = (track.ego_at(xyz, t) - O) @ R
        a, e, r = angles_deg(q)
        lat = TemplateLattice(np.c_[a, e], 0.07)
        cell, q2, times = assign_time_aware(track, xyz, lat, O, R, nearest_time(a, e, t), float(np.median(t)))
        assert (cell == np.arange(len(xyz))).mean() > 0.99
        out = render_cells(cell, np.linalg.norm(q2, axis=1), lat)
        assert (out['src'] == out['cell']).mean() > 0.99
