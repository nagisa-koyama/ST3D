"""Where the check-5 traces the mover shell does not remove come from (experiments_md 20261121_02 §6.2).

For one sequence: every point of a movable raw class (PandaSet 13-33) that is in range, not stale, and NOT a mover
member falls into the static accumulation, so it can smear. Each such point is classed, per device, by its own frame's
cuboids (its device's set):
  in_stationary   inside a cuboid flagged stationary; with that uuid's centre travel over the sequence, to find
                  objects flagged stationary that move;
  near_mover      within 1 m of a mover's box at the point's time, but outside box + shell; with its time offset
                  from the cuboid time and the mover's sensor_id;
  no_cuboid       neither.
Reported by raw class and by range from the sensor.

--cuboid-time: which time a mover cuboid represents. For every in-track mover cuboid, its own devices' movable-class
points of that frame within 2 m are tested against the box posed by the uuid's track at each point's time, with the
cuboid times set to (a) the median time of the points inside it (pcocc's rule), (b) the frame time, (c) the frame time
minus 50 ms. The hypothesis that puts the most points inside the unpadded boxes fits the annotation best; reported per
cuboid sensor_id. GT data, analysis only. Run as a CPU Slurm job.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from pcseg.class_map import build_class_map                                  # noqa: E402
from pcocc.build import MOVER_SHELL, SequenceGT                              # noqa: E402
from pcocc.grid import PANDASET_GRID                                         # noqa: E402
from pcocc.movers import RAW_MOVABLE, Track, inside, to_box                  # noqa: E402
from pcocc.scans import sensor_origins, stale_frames                         # noqa: E402

ROOT = '/home/koyama/data/pandaset/dataset'


def cuboid_time_test(sg):
    s, cub = sg.scans, sg.cub
    movable = sg.base & np.isin(s.raw, RAW_MOVABLE)
    order = np.argsort(s.frame, kind='stable')
    bounds = np.searchsorted(s.frame[order], np.arange(81))
    rows = np.flatnonzero(~cub.stationary & cub.in_track)
    hyps = {'median_point_time': cub.time.copy(), 'frame_time': cub.frame_times[cub.frame].copy(),
            'frame_time_minus_50ms': cub.frame_times[cub.frame] - 0.05}
    saved = cub.time
    res = {}
    for name, times in hyps.items():
        cub.time = times
        tracks = {}
        cnt = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            u = int(cub.uid[r])
            if u not in tracks:
                tracks[u] = Track(cub, u)
            f = cub.frame[r]
            idx = order[bounds[f]:bounds[f + 1]]
            idx = idx[movable[idx] & np.isin(s.dev[idx], cub.devices(r))]
            c0 = cub.centre[r]
            r_xy = 0.5 * np.hypot(cub.dims[r][0], cub.dims[r][1]) + 2.0
            idx = idx[np.hypot(s.xyz[idx, 0] - c0[0], s.xyz[idx, 1] - c0[1]) <= r_xy]
            if not len(idx):
                continue
            c, yaw = tracks[u].at(s.t[idx])
            m = inside(to_box(s.xyz[idx], c, yaw), cub.dims[r])
            k = int(cub.sensor[r])
            cnt[k][0] += int(m.sum())
            cnt[k][1] += int(len(idx))
        res[name] = {str(k): {'inside': v[0], 'near': v[1]} for k, v in sorted(cnt.items())}
    cub.time = saved
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cuboid-time', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cm = build_class_map('pandaset_scalr17')
    sg = SequenceGT(ROOT, args.seq, PANDASET_GRID, cm, sensor_origins(), stale_frames())
    s, cub = sg.scans, sg.cub
    if args.cuboid_time:
        res = {'seq': args.seq, 'cuboid_time_test': cuboid_time_test(sg)}
        with open(os.path.join(args.out, 'cuboid_time_%s.json' % args.seq), 'w') as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))
        return
    cand = sg.base & np.isin(s.raw, RAW_MOVABLE) & (sg.mover < 0)
    rng = np.linalg.norm(s.xyz - sg.origin, axis=1)
    kind = np.full(len(s), '', dtype=object)
    toff = np.full(len(s), np.nan)
    mover_sensor = np.full(len(s), -9)
    stat_uid = np.full(len(s), -1)
    tracks = {}
    order = np.argsort(s.frame, kind='stable')
    bounds = np.searchsorted(s.frame[order], np.arange(81))
    for f in range(80):
        idx = order[bounds[f]:bounds[f + 1]]
        idx = idx[cand[idx]]
        for r in np.flatnonzero(cub.frame == f):
            sub = idx[np.isin(s.dev[idx], cub.devices(r)) & (kind[idx] == '')]
            if not len(sub):
                continue
            if cub.stationary[r]:
                m = inside(to_box(s.xyz[sub], cub.centre[r], cub.yaw[r]), cub.dims[r] + 2 * MOVER_SHELL)
                kind[sub[m]] = 'in_stationary'
                stat_uid[sub[m]] = cub.uid[r]
            else:
                u = int(cub.uid[r])
                if u not in tracks:
                    tracks[u] = Track(cub, u)
                if not len(tracks[u].t):
                    continue
                c, yaw = tracks[u].at(s.t[sub])
                m = inside(to_box(s.xyz[sub], c, yaw), cub.dims[r] + 2.0)
                kind[sub[m]] = 'near_mover'
                toff[sub[m]] = s.t[sub[m]] - cub.time[r]
                mover_sensor[sub[m]] = cub.sensor[r]
        rest = idx[kind[idx] == '']
        kind[rest] = 'no_cuboid'
    # travel of stationary-flagged uuids
    travel = {}
    for u in np.unique(cub.uid[cub.stationary]):
        c = cub.centre[cub.uid == u]
        travel[int(u)] = float(np.max(np.linalg.norm(c[:, :2] - c[0, :2], axis=1)))
    res = {'seq': args.seq, 'mover_shell_m': MOVER_SHELL, 'candidates': int(cand.sum()), 'by_device': {}}
    for d in (0, 1):
        md = cand & (s.dev == d)
        out = {}
        for k in ('in_stationary', 'near_mover', 'no_cuboid'):
            mk = md & (kind == k)
            byc = collections.Counter(int(r) for r in s.raw[mk])
            rb = np.histogram(rng[mk], bins=[0, 10, 20, 30, 50, 70])[0].tolist()
            e = {'points': int(mk.sum()), 'by_raw_class': dict(sorted(byc.items())), 'by_range_0_10_20_30_50_70': rb}
            if k == 'in_stationary' and mk.any():
                tr = np.array([travel[int(u)] for u in stat_uid[mk]])
                e['points_in_uuids_travelling_over_1m'] = int((tr > 1.0).sum())
                e['uuids_travelling_over_1m'] = sorted({cub.uuid_names[int(u)][:8] for u in stat_uid[mk] if travel[int(u)] > 1.0})
            if k == 'near_mover' and mk.any():
                e['time_offset_s_quantiles_5_50_95'] = np.nanpercentile(toff[mk], [5, 50, 95]).round(3).tolist()
                e['by_mover_sensor_id'] = dict(collections.Counter(int(x) for x in mover_sensor[mk]))
            out[k] = e
        res['by_device'][d] = out
    res['stationary_uuids'] = len(travel)
    res['stationary_uuids_travelling_over_1m'] = int(sum(v > 1.0 for v in travel.values()))
    with open(os.path.join(args.out, 'trace_sources_%s.json' % args.seq), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
