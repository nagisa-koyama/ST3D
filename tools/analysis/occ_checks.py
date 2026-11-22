"""Checks 4-7, 9 and 10 of experiments_md 20261121_02 §5 / §3.1 on a built sequence (the step-2 probe).

  4  pose: frame t+10's static points against frame t's, both moved into frame t (point-to-plane distance, manmade
     and road), per device; doubled walls would read >= one voxel.
  5  mover traces: voxels of a vehicle or pedestrian class lying more than one voxel outside every frame-t cuboid.
  6  duplicates: in GU, vehicle voxels inside a PandarGT sibling cuboid but outside its Pandar64 sibling (where GU
     must NOT put the object), and connected vehicle blobs against frame-t vehicle cuboids.
  7  target dump: per-class voxel counts, unknown share per range bin in the cone, BEV images with the cone drawn.
  9  share of mover points captured > 50 ms from their frame time; the trace count of check 5 with and without the
     per-point interpolation (GU rebuilt with interpolate=False on the dump frames).
  10 road points below the grid floor, per range bin (a property of the scene, the same in every GT).
GT data, analysis only. Run as a CPU Slurm job:
  python3 tools/analysis/occ_checks.py --seq 097 --gt-dir <probe dir> --frames 10 30 50 70
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from pcseg.class_map import build_class_map                      # noqa: E402
from pcocc import EMPTY, IGNORE_VOXEL                           # noqa: E402
from pcocc.build import SequenceGT                               # noqa: E402
from pcocc.grid import PANDASET_GRID, cone_voxels               # noqa: E402
from pcocc.movers import inside, to_box                          # noqa: E402
from pcocc.scans import sensor_origins, stale_frames             # noqa: E402

ROOT = '/home/koyama/data/pandaset/dataset'
GTS = ('g64', 'ggt', 'gu')


def load_gt(gt_dir, seq, t):
    z = np.load(os.path.join(gt_dir, seq, '%02d.npz' % t))
    return {g: (z[g + '_label'], z[g + '_unknown']) for g in GTS}


def point_to_plane(ref, qry, k=8):
    from scipy.spatial import cKDTree
    if len(ref) < k or not len(qry):
        return None
    _, nn = cKDTree(ref).query(qry, k=k)
    nb = ref[nn]                                           # [M, k, 3]
    c = nb.mean(1)
    cov = np.einsum('mki,mkj->mij', nb - c[:, None], nb - c[:, None]) / k
    w, v = np.linalg.eigh(cov)
    n = v[:, :, 0]                                         # smallest eigenvector = plane normal
    d = np.abs(np.einsum('mi,mi->m', qry - c, n))
    return {'n': int(len(d)), 'median_m': float(np.median(d)), 'p90_m': float(np.percentile(d, 90))}


def check4(sg, t, cm):
    s = sg.scans
    out = {}
    groups = {'manmade': [cm.names.index('manmade')], 'road': [cm.names.index('road'), cm.names.index('road_marking')]}
    for dev in (0, 1):
        for name, cls in groups.items():
            pts = []
            for f in (t, t + 10):
                m = sg.base & (s.dev == dev) & (s.frame == f) & (sg.mover < 0) & np.isin(s.cls, cls)
                e = sg.poses.ego_from_world_static(s.xyz[m], t)
                keep = (np.hypot(e[:, 0], e[:, 1]) < 30) & (e[:, 0] > 0)
                pts.append(e[keep])
            out['dev%d_%s' % (dev, name)] = point_to_plane(pts[0], pts[1])
    return out


def voxel_centres_world(sg, t, mask):
    ijk = np.argwhere(mask)
    ego = sg.grid.lo + (ijk + 0.5) * sg.grid.voxel
    from pcocc.poses import SWAP
    w = sg.poses.rot[t].apply(ego @ SWAP) + sg.poses.pos[t]
    return ijk, w


def outside_all_cuboids(sg, t, world, pad):
    cub = sg.cub
    rows = np.flatnonzero(cub.frame == t)
    out = np.ones(len(world), dtype=bool)
    for r in rows:
        q = to_box(world, cub.centre[r], cub.yaw[r])
        out &= ~inside(q, cub.dims[r] + 2 * pad)
    return out


def check5(sg, t, gts, cm):
    veh = [cm.names.index(n) + 1 for n in ('car', 'truck', 'bus', 'other_vehicle', 'motorcycle', 'bicycle')]
    ped = [cm.names.index('pedestrian') + 1]
    res = {}
    for g, (label, unknown) in gts.items():
        r = {}
        for name, ids in (('vehicle', veh), ('pedestrian', ped)):
            m = np.isin(label, ids)
            ijk, w = voxel_centres_world(sg, t, m)
            o = outside_all_cuboids(sg, t, w, pad=sg.grid.voxel)
            r[name] = {'voxels': int(m.sum()), 'outside_every_cuboid_by_more_than_a_voxel': int(o.sum())}
        res[g] = r
    return res


def check6(sg, t, gts, cm):
    from scipy import ndimage
    cub = sg.cub
    veh = [cm.names.index(n) + 1 for n in ('car', 'truck', 'bus', 'other_vehicle', 'motorcycle', 'bicycle')]
    label = gts['gu'][0]
    m = np.isin(label, veh)
    ijk, w = voxel_centres_world(sg, t, m)
    doubled = 0
    pairs = 0
    for r in np.flatnonzero((cub.frame == t) & (cub.sensor == 1) & (cub.sibling >= 0)):
        a = cub.row_at.get((t, int(cub.sibling[r])))
        if a is None:
            continue
        pairs += 1
        in_b = inside(to_box(w, cub.centre[r], cub.yaw[r]), cub.dims[r])
        out_a = ~inside(to_box(w, cub.centre[a], cub.yaw[a]), cub.dims[a] + 2 * sg.grid.voxel)
        doubled += int((in_b & out_a).sum())
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    # frame-t vehicle cuboids (Pandar64 side: sensor -1 or 0) with their centre in the grid
    ego = sg.poses.ego_from_world_static(cub.centre[cub.frame == t], t)
    lbl = cub.label[cub.frame == t]
    sen = cub.sensor[cub.frame == t]
    vehicle_labels = {'Car', 'Pickup Truck', 'Medium-sized Truck', 'Semi-truck', 'Towed Object', 'Motorcycle',
                      'Other Vehicle - Construction Vehicle', 'Other Vehicle - Uncommon', 'Other Vehicle - Pedicab',
                      'Bus', 'Bicycle', 'Train', 'Trolley', 'Tram / Subway'}
    in_grid = np.all((ego >= sg.grid.lo) & (ego < sg.grid.hi), axis=1)
    n_cub = int(np.sum(in_grid & np.isin(sen, [-1, 0]) & np.isin(lbl, list(vehicle_labels))))
    return {'sibling_pairs': pairs, 'gu_vehicle_voxels_in_pandargt_sibling_only': doubled,
            'gu_vehicle_blobs_ge5_voxels': int((sizes >= 5).sum()), 'vehicle_cuboids_p64_side_in_grid': n_cub}


def check7(sg, t, gts, cm, fig_dir, seq):
    cone = cone_voxels(sg.grid)
    xs = sg.grid.lo[0] + sg.grid.voxel * (np.arange(sg.grid.shape[0]) + 0.5)
    ys = sg.grid.lo[1] + sg.grid.voxel * (np.arange(sg.grid.shape[1]) + 0.5)
    rr = np.hypot(xs[:, None], ys[None, :])
    bins = [0, 10, 20, 30, 40, 50, 60]
    out = {}
    for g, (label, unknown) in gts.items():
        counts = {cm.names[c]: int((label == c + 1).sum()) for c in range(cm.num_classes)}
        counts['ignore_voxel'] = int((label == IGNORE_VOXEL).sum())
        unk = {}
        for a, b in zip(bins[:-1], bins[1:]):
            col = cone & (rr >= a) & (rr < b)
            if col.any():
                unk['%d-%d' % (a, b)] = float(unknown[col].mean())
        out[g] = {'class_voxels': counts, 'unknown_share_in_cone_by_range_m': unk,
                  'free_in_cone': int(((label == EMPTY) & ~unknown & cone[:, :, None]).sum())}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
        cmap = plt.get_cmap('tab20', 20)
        for ax, g in zip(axes, GTS):
            label, unknown = gts[g]
            occ = (label != EMPTY)
            top = np.where(occ.any(-1), sg.grid.shape[2] - 1 - np.argmax(occ[:, :, ::-1], axis=-1), -1)
            lab = np.take_along_axis(label, np.clip(top, 0, None)[..., None], -1)[..., 0].astype(float)
            img = np.full(label.shape[:2] + (3,), 1.0)
            unk_col = unknown.all(-1)
            img[unk_col] = 0.85
            for c in range(1, cm.num_classes + 1):
                img[(top >= 0) & (lab == c)] = cmap(c - 1)[:3]
            img[(top >= 0) & (lab == IGNORE_VOXEL)] = 0.3
            ax.imshow(np.transpose(img, (1, 0, 2))[::-1, :], extent=[sg.grid.lo[0], sg.grid.hi[0], sg.grid.lo[1],
                                                                      sg.grid.hi[1]])
            ax.plot([0, 51.2], [0, 51.2 * np.tan(np.pi / 6)], 'k--', lw=0.8)
            ax.plot([0, 51.2], [0, -51.2 * np.tan(np.pi / 6)], 'k--', lw=0.8)
            ax.set_title('%s %s/%02d (grey: unknown column; dark: ignore)' % (g.upper(), seq, t), fontsize=9)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(c)[:3]) for c in range(cm.num_classes)]
        fig.legend(handles, cm.names, loc='lower center', ncol=9, fontsize=7)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        path = os.path.join(fig_dir, 'bev_%s_%02d.png' % (seq, t))
        fig.savefig(path, dpi=110)
        plt.close(fig)
        out['figure'] = path
    except ImportError as e:
        out['figure'] = 'matplotlib missing: %s' % e
    return out


def check10(sg, cm):
    s = sg.scans
    road = np.isin(s.cls, [cm.names.index('road'), cm.names.index('road_marking')])
    bins = [0, 10, 20, 30, 40, 51.2]
    res = {}
    for dev in (0, 1):
        tot = {}
        for t in range(80):
            m = road & (s.frame == t) & (s.dev == dev) & sg.in_range
            if not m.any():
                continue
            e = sg.poses.ego_from_world_static(s.xyz[m], t)
            ing = (np.abs(e[:, 1]) < 25.6)
            for a, b in zip(bins[:-1], bins[1:]):
                mm = ing & (e[:, 0] >= a) & (e[:, 0] < b)
                k = '%g-%g' % (a, b)
                n, below = tot.get(k, (0, 0))
                tot[k] = (n + int(mm.sum()), below + int((e[mm, 2] < sg.grid.lo[2]).sum()))
        res['dev%d' % dev] = {k: {'points': n, 'below_floor_share': (below / n if n else None)}
                              for k, (n, below) in tot.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--gt-dir', required=True)
    ap.add_argument('--frames', nargs='+', type=int, default=[10, 30, 50, 70])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.gt_dir, 'checks')
    os.makedirs(out_dir, exist_ok=True)
    cm = build_class_map('pandaset_scalr17')
    t0 = time.time()
    sg = SequenceGT(ROOT, args.seq, PANDASET_GRID, cm, sensor_origins(), stale_frames())
    res = {'seq': args.seq, 'frames': args.frames, 'gt_dir': args.gt_dir, 'sequence_stats': sg.stats}
    s = sg.scans
    mv = sg.mover >= 0
    dt = np.abs(s.t - sg.poses.times[s.frame])
    res['check9_mover_points'] = {'mover_points': int(mv.sum()),
                                  'share_captured_over_50ms_from_frame_time': float((dt[mv] > 0.05).mean()),
                                  'by_device': {d: float((dt[mv & (s.dev == d)] > 0.05).mean())
                                                for d in (0, 1) if (mv & (s.dev == d)).any()}}
    res['check10_road_below_floor'] = check10(sg, cm)
    per = {}
    for t in args.frames:
        gts = load_gt(args.gt_dir, args.seq, t)
        rebuilt, _ = sg.frame(t)
        same = {g: bool(np.array_equal(rebuilt[g][0], gts[g][0]) and np.array_equal(rebuilt[g][1], gts[g][1]))
                for g in GTS}
        per[t] = {'rebuilt_equals_stored': same,
                  'check4_pose': check4(sg, t, cm) if t + 10 < 80 else None,
                  'check5_traces': check5(sg, t, gts, cm),
                  'check6_duplicates': check6(sg, t, gts, cm),
                  'check7_dump': check7(sg, t, gts, cm, out_dir, args.seq)}
        print('frame %d done (%.0f s)' % (t, time.time() - t0), flush=True)
    # check 9, without interpolation: GU (and the singles) rebuilt with each frame's cuboid as annotated
    sg0 = SequenceGT(ROOT, args.seq, PANDASET_GRID, cm, sensor_origins(), stale_frames(), interpolate=False)
    for t in args.frames:
        g0, _ = sg0.frame(t)
        per[t]['check9_traces_without_interpolation'] = check5(sg0, t, g0, cm)
        per[t]['check9_mover_points_without_interpolation'] = int((sg0.mover >= 0).sum())
    res['frames_detail'] = per
    res['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, 'checks_%s.json' % args.seq), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({t: {'traces_gu': per[t]['check5_traces']['gu'],
                          'traces_gu_no_interp': per[t]['check9_traces_without_interpolation']['gu'],
                          'doubled': per[t]['check6_duplicates'], 'pose': per[t]['check4_pose'],
                          'same': per[t]['rebuilt_equals_stored']} for t in args.frames}, indent=1))


if __name__ == '__main__':
    main()
