"""Swap values of the reflection rows R11 (threshold) / R11r (random twin), and what they do to M1's measurement
(experiments_md 20261003_05 §5 item 35g; design fixed 2026-11-07 13:11 JST). CPU, label-free: TRAIN clouds only, no label
is read.

For each target (kitti = S2, waymo = N->W TOP, pandar64 = PandaSet spin), on M1's 200 drawn TRAIN frames
(/home/koyama/data/reflection_m1/m1.json):

A. The loader's own input to the arm. The arm's keep_mask is wrapped to record what the loader hands it (sensor-frame
   xyz, intensity in the loader's scale, frame key), through the same reading functions __getitem__ calls: KITTI
   get_lidar + kitti_factor_swap.apply_points; Waymo get_lidar(return_top_mask=True) + the segment's TOP extrinsic +
   waymo_factor_swap.reflection_keep; PandaSet _get_lidar_points. Each return is classified (reflection_drop.classify);
   the frame's plane is compared with M1's (KITTI and Pandar64 read the same points, so it must be equal; Waymo's
   stored cloud is not M1's range image, so the difference is reported).
B. The swap values. e(b, c) from M1's readout, applied where e > 0 and both sensors have >= 2,000 eligible slots in
   (b, c) (else 0). T(b, c) = the e-quantile of the pooled loader intensities in (b, c): T = v[ceil(e n) - 1] of the
   sorted values, Q = (e - share(v < T)) / share(v == T), so share(v < T) + Q share(v == T) = e exactly.
   Expected dropped share overall, per bin, and the untouched share (ego, beyond 75 m, outside both height classes).
C. M1 re-measured on the swapped TRAIN clouds (achieved selectivity): M1's reader per frame (rings), the arm's mask
   (KITTI / Pandar64: the loader's mask on the same points, checked equal by cell and intensity; Waymo: the arm applied
   to M1's range-image TOP returns with the loader's intensity scale tanh), then M1's pass (plane ground, dup_frac 0.25,
   M1's d0). Arms: none (must reproduce M1's h on KITTI / Pandar64), threshold, random. Per (b, c): h, S against
   nuScenes' h, S; and the realised dropped share per cell against e.

Run as a Slurm CPU job from ST3D/tools (PandaSet needs --bind /home/koyama/code/ST3D:/root/ST3D):
    python3 analysis/reflection_r11_stats.py --out_dir /home/koyama/data/reflection_r11 --workers 8
"""
import argparse
import contextlib
import io
import json
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, REPO)
sys.path.insert(0, TOOLS)
sys.path.append(HERE)
os.chdir(TOOLS)
import _init_path  # noqa: E402,F401
import reflection_raydrop_m1 as m1  # noqa: E402
from easydict import EasyDict  # noqa: E402
from pcdet.datasets import reflection_drop as R  # noqa: E402

M1_JSON = '/home/koyama/data/reflection_m1/m1.json'
TOP_CALIB = '/home/koyama/data/waymo_top_calib/top_calib.pkl'
TARGETS = {   # M1 sensor: (row tag, oracle config)
    'kitti': ('s2', 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml'),
    'waymo': ('n2w', 'cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml'),
    'pandar64': ('pandaset', 'cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml'),
}
BINS = ['%g-%g' % (R.RANGE_EDGES[i], R.RANGE_EDGES[i + 1]) for i in range(R.N_BINS)]
DUP_FRAC = 0.25
MIN_SLOTS = 2000


def dummy(mode='random', waymo=False):
    c = {'MODE': mode, 'E': {'ground': [0.0] * R.N_BINS, 'above': [0.0] * R.N_BINS}}
    if waymo:
        c['TOP_CALIB'] = TOP_CALIB
    return EasyDict({'REFLECTION_DROP': c})


# --------------------------------------------------------------------------------------------- A. the loader's input
CAPTURE = []
_ORIG_KEEP = R.keep_mask


def _capture(xyz_sensor, inten, cfg, frame_key):
    CAPTURE.append((np.array(xyz_sensor, dtype=np.float64), np.asarray(inten, dtype=np.float64).copy(), frame_key))
    return _ORIG_KEEP(xyz_sensor, inten, cfg, frame_key)


def build_dataset(cfg_path):
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = cfg_from_yaml_file(cfg_path, EasyDict())
        ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                    workers=0, logger=common_utils.create_logger(), training=True,
                                    model_ontology=cfg.get('ONTOLOGY'))
    return ds


def loader_inputs(sensor, frames):
    """[(xyz_sensor, intensity, frame key)] per M1 frame, as the loader hands them to the arm."""
    R.keep_mask = _capture
    try:
        ds = build_dataset(TARGETS[sensor][1])
        out = []
        if sensor == 'kitti':
            from pcdet.datasets.kitti import kitti_factor_swap as kfs
            swap = dummy()
            for fr in frames:
                CAPTURE.clear()
                kfs.apply_points(ds.get_lidar(fr['id']), swap, fr['id'])
                out.append(CAPTURE[0])
        elif sensor == 'waymo':
            from pcdet.datasets.waymo import waymo_factor_swap as wfs
            swap = dummy(waymo=True)
            calib = pickle.load(open(TOP_CALIB, 'rb'))
            for fr in frames:
                seq = fr['id']
                infos = pickle.load(open(ds.data_path / seq / ('%s.pkl' % seq), 'rb'))
                assert len(infos) == fr['n_frames'], '%s: %d infos, M1 counted %d records' % (seq, len(infos),
                                                                                             fr['n_frames'])
                info = infos[fr['frame']]
                assert info['point_cloud']['sample_idx'] == fr['frame']
                pts, is_top = ds.get_lidar(seq, fr['frame'], info['num_points_of_each_lidar'], return_top_mask=True)
                CAPTURE.clear()
                wfs.reflection_keep(pts, is_top, calib[seq]['extrinsic'], swap, info['frame_id'])
                out.append(CAPTURE[0])
        else:
            ds.dataset_cfg.TRAIN_FACTOR_SWAP = dummy()
            idx = {(str(i['sequence']), int(i['frame_idx'])): k for k, i in enumerate(ds.pandaset_infos)}
            for fr in frames:
                info = ds.pandaset_infos[idx[(fr['seq'], int(fr['frame']))]]
                CAPTURE.clear()
                ds._get_lidar_points(info, ds._get_pose(info))
                out.append(CAPTURE[0])
        return out
    finally:
        R.keep_mask = _ORIG_KEEP


# --------------------------------------------------------------------------------------------- B. swap values
def e_table(res, sensor):
    src, tgt = res['sensors']['nuscenes']['tables']['all'], res['sensors'][sensor]['tables']['all']
    e_read = res['readout'][sensor]['e']
    E = np.zeros(R.N_BINS * 2)
    rows = []
    for b in range(R.N_BINS):
        for c, cname in enumerate(R.CLASSES):
            k = '%d,%d' % (b, c)
            e = e_read.get('%s %s' % (BINS[b], cname), None)
            ok = e is not None and e > 0 and src[k]['slots'] >= MIN_SLOTS and tgt[k]['slots'] >= MIN_SLOTS
            if ok:
                E[b * 2 + c] = e
            rows.append(dict(bin=BINS[b], cls=cname, e_m1=e, slots_nus=src[k]['slots'], slots_tgt=tgt[k]['slots'],
                             h_nus=src[k]['h'], h_tgt=tgt[k]['h'], S_nus=src[k]['S'], applied=bool(ok)))
    return E, rows


def threshold(v, e):
    """(T, Q) with share(v < T) + Q share(v == T) = e on the pooled values v."""
    v = np.sort(v)
    n = len(v)
    k = max(int(math.ceil(e * n)) - 1, 0)
    t = float(v[k])
    lt, eq = np.sum(v < t) / n, np.sum(v == t) / n
    q = (e - lt) / eq
    assert -1e-12 <= q <= 1 + 1e-12, (e, lt, eq)
    return t, float(min(max(q, 0.0), 1.0))


def as_cfg(mode, E, T=None, Q=None):
    def lists(a):
        return {name: [None if (a is None or not np.isfinite(a[b * 2 + c])) else float(a[b * 2 + c])
                       for b in range(R.N_BINS)] for c, name in enumerate(R.CLASSES)}
    c = {'MODE': mode, 'E': {name: [float(E[b * 2 + c]) for b in range(R.N_BINS)] for c, name in enumerate(R.CLASSES)}}
    if mode == 'threshold':
        c['T'], c['Q'] = lists(T), lists(Q)
    return c


# --------------------------------------------------------------------------------------------- C. M1 on the swapped clouds
def m1_frame(args):
    """M1's pass on one frame: unswapped, threshold, random. Returns per arm (slots, returns) as m1.pass2, plus the
    realised drop per cell and the alignment check."""
    sensor, fr, cell_a, inten_a, key, cfgs, d0 = args
    fr = dict(fr)
    if sensor == 'kitti':          # m1.json keeps the frame ids, not the paths: rebuilt as M1's draw built them
        fr['path'] = os.path.join(m1.DATA, 'kitti', 'training', 'velodyne', fr['id'] + '.bin')
    elif sensor == 'waymo':
        fr['path'] = os.path.join(m1.DATA, 'waymo', 'raw_data', fr['id'] + '.tfrecord')
    d = m1.READERS[sensor](fr)
    if sensor == 'waymo':
        xyz = np.c_[d['x'], d['y'], d['z']]
        cell, _ = R.classify(xyz)
        inten = np.tanh(np.asarray(d['inten'], dtype=np.float32)).astype(np.float64)
        aligned = None
    else:
        xyz = np.c_[d['x'], d['y'], d['z']]
        cell, _ = R.classify(xyz)
        inten = np.asarray(d['inten'], dtype=np.float64) / (255.0 if sensor == 'pandar64' else 1.0)
        aligned = bool(np.array_equal(cell, cell_a) and np.array_equal(inten, inten_a))
        cell, inten = cell_a, inten_a
    out = {'aligned': aligned}
    for arm, c in (('none', None),) + tuple(cfgs.items()):
        if c is None:
            keep = np.ones(len(cell), dtype=bool)
        else:
            keep = ~R.drop_mask(cell, inten, EasyDict(c), R.frame_rng(key, c['MODE']))
        dk = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in d.items()}
        dz, _ = m1.ground_height(dk, 'plane')
        arr = dict(ring=dk['ring'].astype(np.int32), az=dk['az'].astype(np.float64), r=dk['r'].astype(np.float32),
                   dz=dz.astype(np.float32), inten=np.asarray(dk['inten'], dtype=np.float32))
        sl, ret, _, _ = m1.pass2((arr, d0, DUP_FRAC))
        dropped = np.bincount(cell[(~keep) & (cell >= 0)], minlength=R.N_BINS * 2)
        out[arm] = dict(sl=sl, ret=ret, dropped=dropped)
    out['n_cell'] = np.bincount(cell[cell >= 0], minlength=R.N_BINS * 2)
    out['n'] = len(cell)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='/home/koyama/data/reflection_r11')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--sensors', nargs='+', default=list(TARGETS))
    ap.add_argument('--frames', type=int, default=0, help='first N of M1\'s frames only (smoke)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res = json.load(open(M1_JSON))
    nus = res['sensors']['nuscenes']['tables']['all']
    out = dict(design='experiments_md 20261003_05 §5 item 35g', m1=M1_JSON, min_slots=MIN_SLOTS, bins=BINS,
               classes=list(R.CLASSES), targets={})
    lines = []
    for sensor in a.sensors:
        t0 = time.time()
        frames = res['sensors'][sensor]['frames']
        if a.frames:
            frames = frames[:a.frames]
        # A
        inputs = loader_inputs(sensor, frames)
        cells, intens, keys, plane_dev = [], [], [], []
        for fr, (xyz, inten, key) in zip(frames, inputs):
            cell, plane = R.classify(xyz)
            cells.append(cell); intens.append(inten); keys.append(key)
            if plane is not None and fr.get('plane') is not None:
                plane_dev.append(float(np.max(np.abs(np.asarray(plane) - np.asarray(fr['plane'])))))
            else:
                plane_dev.append(None if (plane is None) == (fr.get('plane') is None) else float('inf'))
        cell_all, inten_all = np.concatenate(cells), np.concatenate(intens)
        n_all = len(cell_all)
        # B
        E, rows = e_table(res, sensor)
        T, Q = np.full(R.N_BINS * 2, np.nan), np.full(R.N_BINS * 2, np.nan)
        for k in range(R.N_BINS * 2):
            m = cell_all == k
            rows[k]['returns_pooled'] = int(m.sum())
            rows[k]['median_intensity'] = float(np.median(inten_all[m])) if m.any() else None
            if E[k] > 0:
                T[k], Q[k] = threshold(inten_all[m], E[k])
                rows[k]['T'], rows[k]['Q'] = T[k], Q[k]
                rows[k]['share_at_T'] = float(np.mean(inten_all[m] == T[k]))
        n_cell = np.bincount(cell_all[cell_all >= 0], minlength=R.N_BINS * 2)
        exp_drop = float((E * n_cell).sum() / n_all)
        cfgs = {'threshold': as_cfg('threshold', E, T, Q), 'random': as_cfg('random', E)}
        # C
        d0 = res['sensors'][sensor]['d0_deg']
        jobs = [(sensor, fr, cells[i], intens[i], keys[i], cfgs, d0) for i, fr in enumerate(frames)]
        with Pool(a.workers) as pool:
            fo = pool.map(m1_frame, jobs, chunksize=1)
        tabs, realised = {}, {}
        for arm in ('none', 'threshold', 'random'):
            sl = m1.concat([f[arm]['sl'] for f in fo], ('rb', 'cl', 'hole', 'count', 'inten'))
            ret = m1.concat([f[arm]['ret'] for f in fo], ('rb', 'cl', 'inten'))
            tabs[arm] = m1.cell_stats(sl, ret)
            realised[arm] = np.sum([f[arm]['dropped'] for f in fo], axis=0)
        n_cell_m1 = np.sum([f['n_cell'] for f in fo], axis=0)
        # shares over all returns of the frames (loader cloud), by 3D range bin of the return (classified or not)
        r_all = np.concatenate([np.linalg.norm(x[0], axis=1) for x in inputs])
        rb_all = R.range_bin(r_all)
        per_bin = []
        for b in range(R.N_BINS):
            nb = int(np.sum(rb_all == b))
            per_bin.append(dict(bin=BINS[b], returns=nb, expected_drop_share=float((E * n_cell)[b * 2:b * 2 + 2].sum() / max(nb, 1)),
                                untouched_share=float(np.sum((rb_all == b) & (cell_all < 0)) / max(nb, 1))))
        agg = dict(frames=len(frames), returns=n_all, classified=int((cell_all >= 0).sum()),
                   untouched_share=float(np.mean(cell_all < 0)), expected_drop_share=exp_drop,
                   beyond_75m_share=float(np.mean(r_all >= 75.0)), ego_share=float(np.mean(r_all < R.EGO_M)),
                   plane_dev_max=(None if not [p for p in plane_dev if p is not None] else
                                  max(p for p in plane_dev if p is not None)),
                   plane_exact_frames=int(sum(1 for p in plane_dev if p == 0.0)),
                   aligned_frames=(None if sensor == 'waymo' else int(sum(bool(f['aligned']) for f in fo))))
        cells_out = []
        for k in range(R.N_BINS * 2):
            key = '%d,%d' % (k // 2, k % 2)
            row = dict(rows[k])
            for arm in ('none', 'threshold', 'random'):
                v = tabs[arm][key]
                row['h_%s' % arm], row['S_%s' % arm] = v['h'], v['S']
            row['h_m1'], row['S_m1'] = res['sensors'][sensor]['tables']['all'][key]['h'], \
                res['sensors'][sensor]['tables']['all'][key]['S']
            for arm in ('threshold', 'random'):
                row['realised_drop_%s' % arm] = float(realised[arm][k] / max(n_cell_m1[k], 1))
            cells_out.append(row)
        out['targets'][sensor] = dict(row_tag=TARGETS[sensor][0], oracle_cfg=TARGETS[sensor][1], aggregate=agg,
                                      per_bin=per_bin, cells=cells_out, configs=cfgs, sec=round(time.time() - t0, 1))
        # text
        lines.append('\n## %s (%s rows; %d M1 frames; %.0f s)' % (sensor, TARGETS[sensor][0], len(frames), time.time() - t0))
        lines.append('returns %d, untouched %.3f (ego %.3f, beyond 75 m %.3f, outside both classes the rest); expected '
                     'dropped share of all returns %.4f; plane vs M1: exact on %d of %d frames, max |dev| %s; M1 reader '
                     'aligned with the loader on %s frames' % (
                         n_all, agg['untouched_share'], agg['ego_share'], agg['beyond_75m_share'], exp_drop,
                         agg['plane_exact_frames'], len(frames), agg['plane_dev_max'], agg['aligned_frames']))
        lines.append('per bin (all returns): ' + ' | '.join('%s drop %.3f untouched %.3f' % (
            p['bin'], p['expected_drop_share'], p['untouched_share']) for p in per_bin))
        lines.append('bin class | slots nuS / tgt | e (M1) | applied | T | Q | share at T | median I | h tgt none (M1) | '
                     'h thr | h rnd | h nuS | S nuS | S thr | S rnd | S none | realised thr / rnd')
        for row in cells_out:
            f = (lambda x, p='%.3f': '-' if x is None or (isinstance(x, float) and not np.isfinite(x)) else p % x)
            lines.append('%s %s | %d / %d | %s | %s | %s | %s | %s | %s | %s (%s) | %s | %s | %s | %s | %s | %s | %s | '
                         '%s / %s' % (
                             row['bin'], row['cls'], row['slots_nus'], row['slots_tgt'], f(row['e_m1']),
                             'yes' if row['applied'] else 'no', f(row.get('T'), '%.6g'), f(row.get('Q')),
                             f(row.get('share_at_T')), f(row['median_intensity'], '%.4g'), f(row['h_none'], '%.4f'),
                             f(row['h_m1'], '%.4f'), f(row['h_threshold'], '%.4f'), f(row['h_random'], '%.4f'),
                             f(row['h_nus'], '%.4f'), f(row['S_nus'], '%.2f'), f(row['S_threshold'], '%.2f'),
                             f(row['S_random'], '%.2f'), f(row['S_none'], '%.2f'), f(row['realised_drop_threshold']),
                             f(row['realised_drop_random'])))
        print('\n'.join(lines[-(4 + len(cells_out)):]), flush=True)
    json.dump(out, open(os.path.join(a.out_dir, 'r11_stats.json'), 'w'), indent=1, default=m1._json_default)
    open(os.path.join(a.out_dir, 'r11_stats.txt'), 'w').write('\n'.join(lines) + '\n')
    print('saved %s/r11_stats.json, r11_stats.txt' % a.out_dir)


if __name__ == '__main__':
    main()
