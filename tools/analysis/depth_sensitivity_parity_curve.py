"""Accumulation-depth sensitivity: the label-free parity curve of rule I2 against N (experiments_md 20261017_10).

ANALYSIS ONLY, CPU. Reads unlabelled TRAIN point clouds of source and target and the method rows' resolved W&B configs;
reads no label of any dataset. It never selects a depth (20261017_10 §1 rule 1).

For each sparse -> dense pair and each source platform, against N (frames including the anchor, = MAX_SWEEPS):
  * per-bin ratio mean(source) / mean(target) of the radial point histogram, 5 m bins, 13 bins over 5-70 m;
  * bins at ratio >= 1.00, minimum ratio, N* = the smallest N at which all 13 bins reach 1.00 (rule I2,
    20261017_03 §2), and N*_20 = the same over the 10 bins beyond 20 m (the rule S3's depth was actually chosen by,
    20260928_02 §1);
  * the voxel count of every anchor and the share of anchors above MAX_NUMBER_OF_VOXELS (train); N_cap (rule I3:
    largest N <= N* with at most 10% of anchors above the cap).

Variants, computed in one pass over the same anchors and frames:
  V0   sensor frame, the point-cloud range applied WITHOUT SHIFT_COOR (x, y, z), 360 deg: what
       accumulation_deep_nuscenes.py / accumulation_matrix.py measured;
  V1a  the voxel input: SHIFT_COOR, then x, y, z inside POINT_CLOUD_RANGE (x, y by mask_points_by_range, z by the
       voxeliser), 360 deg;
  V1   I2 AS WRITTEN: V1a inside the target's evaluated FOV on both sides (KITTI: the camera's horizontal wedge from
       its calibration; PandarGT: EVAL_FOV_DEGREE about EVAL_FOV_HEADING; Waymo: 360 deg);
  V2   diagnostic: occupied voxels (not points) per radial bin, inside the same FOV as V1.
Each nuScenes variant also has a 'distinct' twin that drops the padded scene-start copies of the 200-sweep infos (the
loader keeps them unless DEDUP_SWEEPS; 20261007_03 §15).

Sources are accumulated exactly as the training loaders do it, without motion compensation (ego-only; compensation
moves only labelled objects' points):
  nuScenes: info['sweeps'][:N-1] of the 200-sweep TRAIN infos (consecutive, most recent first), keyframe
            remove_ego_points(1.5), sweeps remove_ego_points(1.0), sweep transform_matrix into the anchor frame;
  PandaSet: the N-1 preceding annotated frames of the same train sequence, stopping at the first gap
            (motion_compensation.preceding_frames), all transformed with the ANCHOR's pose, then the normative axis swap.

Parts (one Slurm CPU job each; outputs are .npz under --out on NFS):
    python3 analysis/depth_sensitivity_parity_curve.py nuscenes --out DIR [--anchors 100] [--procs 8]
    python3 analysis/depth_sensitivity_parity_curve.py targets  --out DIR [--frames 200]
    python3 analysis/depth_sensitivity_parity_curve.py pandaset --out DIR [--anchors 100] [--frames 200] [--procs 8]
    python3 analysis/depth_sensitivity_parity_curve.py report   --out DIR          # tables + summary.json
Add --smoke for a 3-anchor / 3-frame / short-grid check.
"""
import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
DATA = TOOLS.parent / 'data'
WANDB = Path('/home/koyama/data/wandb')

# The method rows whose resolved configs define the voxel input (read at run time, never typed in here).
RUN_CFG = {
    'nuscenes2kitti': WANDB / 'run-20261006_004144-o3l96a3p/files/config.yaml',   # 27468, S2 method teacher
    'nuscenes2waymo': TOOLS / 'wandb/run-20261004_014034-0ru2bxor/files/config.yaml',  # 27185 (30/20), Waymo twin
    'spin2flash': WANDB / 'run-20260929_215545-vjutu18e/files/config.yaml',      # 26692, S3 N = 4
}

EDGES = np.linspace(0.0, 75.0, 16)          # 5 m bins, 0-75 m
BAND = slice(1, 14)                         # 5-70 m: 13 bins
BAND20 = slice(4, 14)                       # 20-70 m: 10 bins
NS_GRID = list(range(1, 21)) + list(range(22, 31, 2)) + list(range(35, 81, 5))
PD_GRID = list(range(1, 13)) + [14, 16, 18, 20, 25, 30]
VARIANTS = ('V0', 'V1a', 'V1', 'V2')


# ----------------------------------------------------------------------------------------------------------------
# configs
# ----------------------------------------------------------------------------------------------------------------
def run_cfg(name):
    import yaml
    c = yaml.safe_load(open(RUN_CFG[name]))
    return {k: (v['value'] if isinstance(v, dict) and 'value' in v else v) for k, v in c.items()}


def voxel_params(dcfg):
    vp = [p for p in dcfg['DATA_PROCESSOR'] if p['NAME'] == 'transform_points_to_voxels']
    assert len(vp) == 1, dcfg['DATA_PROCESSOR']
    return np.asarray(vp[0]['VOXEL_SIZE'], float), int(vp[0]['MAX_NUMBER_OF_VOXELS']['train'])


def geometry(dcfg):
    pcr = np.asarray(dcfg['POINT_CLOUD_RANGE'], float)
    shift = np.asarray(dcfg.get('SHIFT_COOR') or [0, 0, 0], float)
    vs, cap = voxel_params(dcfg)
    return dict(pcr=pcr, shift=shift, voxel_size=vs, cap=cap)


# ----------------------------------------------------------------------------------------------------------------
# the per-frame statistic
# ----------------------------------------------------------------------------------------------------------------
def in_box(p, pcr):
    return ((p[:, 0] >= pcr[0]) & (p[:, 0] <= pcr[3]) & (p[:, 1] >= pcr[1]) & (p[:, 1] <= pcr[4])
            & (p[:, 2] >= pcr[2]) & (p[:, 2] < pcr[5]))


def legacy_mask(p, pcr):
    """domain_gap_analysis.mask_range: x, y, z with a strict upper bound, sensor frame."""
    return np.all((p[:, :3] >= pcr[:3]) & (p[:, :3] < pcr[3:]), axis=1)


def rhist(xy):
    return np.histogram(np.hypot(xy[:, 0], xy[:, 1]), bins=EDGES)[0].astype(np.float64)


def fov_of(xy, fwd, left, lo_deg, hi_deg):
    """Azimuth about the sensor origin, measured from the vehicle's forward axis (fwd, left: unit 2-vectors)."""
    if lo_deg is None:
        return np.ones(len(xy), bool)
    az = np.degrees(np.arctan2(xy @ left, xy @ fwd))
    return (az >= lo_deg) & (az <= hi_deg)


def voxel_keys(p, pcr, vs):
    g = np.floor((p[:, :3] - pcr[:3]) / vs).astype(np.int64)
    n = np.round((pcr[3:] - pcr[:3]) / vs).astype(np.int64)
    ok = np.all((g >= 0) & (g < n), axis=1)
    g = g[ok]
    return (g[:, 0] * n[1] + g[:, 1]) * n[2] + g[:, 2], n


def key_xy(keys, n, pcr, vs):
    ix = keys // (n[1] * n[2]); iy = (keys // n[2]) % n[1]
    return np.column_stack([(ix + 0.5) * vs[0] + pcr[0], (iy + 0.5) * vs[1] + pcr[1]])


class Accumulator:
    """Running per-variant histograms of one anchor's accumulated cloud, plus its running set of occupied voxels.

    `fovs` maps a target name to its (lo, hi) azimuth wedge in degrees, (None, None) = 360 deg. V0 / V1a and the voxel
    set do not depend on the FOV, so one accumulator serves every target of a source."""

    def __init__(self, geo, fwd, left, fovs):
        self.g, self.fwd, self.left, self.fovs = geo, fwd, left, fovs
        z = lambda: np.zeros(len(EDGES) - 1)
        self.h = {'V0': z(), 'V1a': z()}; self.hd = {'V0': z(), 'V1a': z()}      # hd: distinct sweeps only
        for t in fovs:
            self.h['V1/' + t] = z(); self.hd['V1/' + t] = z()
        self.keys = np.zeros(0, np.int64); self.keys_d = np.zeros(0, np.int64); self.n = None

    def add(self, p, distinct):
        g = self.g
        hs = {'V0': rhist(p[legacy_mask(p, g['pcr'])][:, :2])}
        q = p[:, :3] + g['shift']
        q = q[in_box(q, g['pcr'])]
        hs['V1a'] = rhist(q[:, :2])
        for t, fov in self.fovs.items():
            hs['V1/' + t] = rhist(q[fov_of(q[:, :2], self.fwd, self.left, *fov)][:, :2])
        k, self.n = voxel_keys(q, g['pcr'], g['voxel_size'])
        k = np.unique(k)
        for v, hv in hs.items():
            self.h[v] += hv
            if distinct:
                self.hd[v] += hv
        self.keys = np.union1d(self.keys, k)
        if distinct:
            self.keys_d = np.union1d(self.keys_d, k)

    def snapshot(self):
        """{target: {V0, V1a, V1, V2 and their '_d' twins, nvox, nvox_d}}."""
        g = self.g
        out = {t: {} for t in self.fovs}
        for tag, h, keys in (('', self.h, self.keys), ('_d', self.hd, self.keys_d)):
            xy = key_xy(keys, self.n, g['pcr'], g['voxel_size']) if self.n is not None else np.zeros((0, 2))
            for t, fov in self.fovs.items():
                o = out[t]
                o['V0' + tag] = h['V0'].copy(); o['V1a' + tag] = h['V1a'].copy(); o['V1' + tag] = h['V1/' + t].copy()
                o['V2' + tag] = rhist(xy[fov_of(xy, self.fwd, self.left, *fov)])
                o['nvox' + tag] = float(len(keys))
        return out


# ----------------------------------------------------------------------------------------------------------------
# nuScenes sources
# ----------------------------------------------------------------------------------------------------------------
def kitti_camera_wedge(infos):
    """Median horizontal camera FOV of KITTI (degrees, about the lidar origin, + = left), from the calibration of the
    sampled TRAIN frames: the rays through the left and right image edges at the principal row."""
    lo, hi = [], []
    for info in infos:
        P2 = np.asarray(info['calib']['P2']); R0 = np.asarray(info['calib']['R0_rect'])
        Tr = np.asarray(info['calib']['Tr_velo_to_cam'])
        W = float(info['image']['image_shape'][1])
        fx, cx = P2[0, 0], P2[0, 2]
        M = (R0[:3, :3] @ Tr[:3, :3])                 # velo -> rect rotation
        az = []
        for u in (0.0, W):
            d = np.linalg.solve(M, np.array([(u - cx) / fx, 0.0, 1.0]))
            az.append(np.degrees(np.arctan2(d[1], d[0])))
        hi.append(az[0]); lo.append(az[1])            # u = 0 is the image's left edge = lidar +y
    return float(np.median(lo)), float(np.median(hi)), (float(np.min(lo)), float(np.max(lo)),
                                                        float(np.min(hi)), float(np.max(hi)))


def _ns_load(root, path, radius):
    p = np.fromfile(str(root / path), dtype=np.float32).reshape([-1, 5])[:, :4]
    return p[~((np.abs(p[:, 0]) < radius) & (np.abs(p[:, 1]) < radius))]


def ns_anchor(job):
    """One nuScenes anchor: snapshots at every grid N for the KITTI-wedge and the 360 deg (Waymo) FOVs."""
    root, a, grid, geo, wedge = job
    R = np.asarray(a['ref_from_car'])[:3, :3]           # car -> lidar
    fwd = (R @ np.array([1.0, 0, 0]))[:2]; left = (R @ np.array([0, 1.0, 0]))[:2]
    fwd /= np.linalg.norm(fwd); left /= np.linalg.norm(left)
    acc = Accumulator(geo, fwd, left, {'kitti': wedge, 'waymo': (None, None)})
    seen = {a['lidar_path']}
    acc.add(_ns_load(root, a['lidar_path'], 1.5), True)
    snaps = {'kitti': {}, 'waymo': {}}; n_used = {}; n_distinct = 1

    def take(n):
        s = acc.snapshot()
        for t in snaps:
            snaps[t][n] = s[t]
        n_used[n] = (n, n_distinct)

    if 1 in grid:
        take(1)
    sweeps = a['sweeps']
    for n in range(2, max(grid) + 1):
        if n - 2 >= len(sweeps):
            break
        s = sweeps[n - 2]
        distinct = s['transform_matrix'] is not None and s['lidar_path'] not in seen
        seen.add(s['lidar_path']); n_distinct += int(distinct)
        q = _ns_load(root, s['lidar_path'], 1.0)
        tm = s['transform_matrix']
        if tm is not None:
            tm = np.asarray(tm)
            q = np.column_stack([(tm[:3, :3] @ q[:, :3].T).T + tm[:3, 3], q[:, 3]])
        acc.add(q, distinct)
        if n in grid:
            take(n)
    return snaps, n_used


def part_nuscenes(args):
    import gc
    cfg_k, cfg_w = run_cfg('nuscenes2kitti'), run_cfg('nuscenes2waymo')
    root = DATA / 'nuscenes' / 'v1.0-trainval'
    out = {}
    kitti_infos = pickle.load(open(DATA / 'kitti/kitti_infos_train.pkl', 'rb'))
    step = max(1, len(kitti_infos) // args.frames)
    lo, hi, rng = kitti_camera_wedge(kitti_infos[::step][:args.frames])
    print('KITTI camera wedge (deg, + = left): [%.2f, %.2f]; per-frame range lo %.2f..%.2f hi %.2f..%.2f'
          % ((lo, hi) + rng), flush=True)
    meta = root / 'v1.0-trainval'
    logs = {l['token']: l for l in json.load(open(meta / 'log.json'))}
    scene_log = {s['token']: logs[s['log_token']] for s in json.load(open(meta / 'scene.json'))}
    log_of = {s['token']: scene_log[s['scene_token']] for s in json.load(open(meta / 'sample.json'))}
    t0 = time.time()
    infos = pickle.load(open(root / cfg_k['DATA_CONFIGS']['NUSCENES_N008']['INFO_PATH']['train'][0], 'rb'))
    print('loaded %d 200-sweep train infos in %.0f s' % (len(infos), time.time() - t0), flush=True)
    grid = [n for n in NS_GRID if n <= (6 if args.smoke else 80)]
    jobs = {}
    for plat, block in (('n008', 'NUSCENES_N008'), ('n015', 'NUSCENES_N015')):
        dk, dw = cfg_k['DATA_CONFIGS'][block], cfg_w['DATA_CONFIGS'][block]
        geo = geometry(dk)
        gw = geometry(dw)
        assert all(np.allclose(geo[k], gw[k]) for k in ('pcr', 'shift', 'voxel_size')) and geo['cap'] == gw['cap'], \
            'the KITTI and Waymo method rows must share the source voxel input'
        assert dk['VEHICLE'] == plat
        sub = [i for i in infos if log_of[i['token']]['vehicle'] == plat]
        step = max(1, len(sub) // args.anchors)
        anchors = sub[::step][:args.anchors]
        jobs[plat] = [(root, dict(lidar_path=a['lidar_path'], ref_from_car=np.asarray(a['ref_from_car']),
                                  sweeps=[dict(lidar_path=s['lidar_path'], transform_matrix=s['transform_matrix'])
                                          for s in a['sweeps'][:max(grid) - 1]]),
                       grid, geo, (lo, hi)) for a in anchors]
        out[plat + '_geo'] = geo
        print('%s: %d train frames, %d anchors, MAX_NUMBER_OF_VOXELS %d, SHIFT_COOR %s, PCR %s, voxel %s'
              % (plat, len(sub), len(anchors), geo['cap'], geo['shift'], geo['pcr'], geo['voxel_size']), flush=True)
    del infos, log_of
    gc.collect()
    res = {'grid': grid, 'wedge': (lo, hi)}
    res.update(out)
    with Pool(args.procs) as pool:
        for plat, js in jobs.items():
            t0 = time.time()
            rows = pool.map(ns_anchor, js, chunksize=1)
            print('%s: %d anchors in %.0f s' % (plat, len(rows), time.time() - t0), flush=True)
            res[plat] = rows
    with open(Path(args.out) / ('nuscenes%s.pkl' % ('_smoke' if args.smoke else '')), 'wb') as f:
        pickle.dump(res, f)
    print('wrote', Path(args.out) / 'nuscenes.pkl', flush=True)


# ----------------------------------------------------------------------------------------------------------------
# targets: KITTI and Waymo TRAIN
# ----------------------------------------------------------------------------------------------------------------
def target_hist(points, geo, fwd, left, fov):
    acc = Accumulator(geo, fwd, left, {'t': fov})
    acc.add(points, True)
    s = acc.snapshot()['t']
    return {k: s[k] for k in ('V0', 'V1a', 'V1', 'V2', 'nvox')}


def part_targets(args):
    cfg_k, cfg_w = run_cfg('nuscenes2kitti'), run_cfg('nuscenes2waymo')
    res = {}
    # KITTI: velodyne, x forward / y left; the wedge is applied per frame from that frame's calibration
    infos = pickle.load(open(DATA / 'kitti/kitti_infos_train.pkl', 'rb'))
    step = max(1, len(infos) // args.frames)
    sel = infos[::step][:args.frames]
    geo = geometry(cfg_k['DATA_CONFIG_TAR'])
    rows = []
    for info in sel:
        lo, hi, _ = kitti_camera_wedge([info])
        idx = info['point_cloud']['lidar_idx']
        p = np.fromfile(str(DATA / f'kitti/training/velodyne/{idx}.bin'), dtype=np.float32).reshape([-1, 4])
        rows.append(target_hist(p, geo, np.array([1.0, 0]), np.array([0, 1.0]), (lo, hi)))
    res['kitti'] = rows; res['kitti_geo'] = geo
    print('KITTI: %d of %d train frames, SHIFT_COOR %s' % (len(rows), len(infos), geo['shift']), flush=True)
    # Waymo: train sequences of ImageSets, the re-extracted processed cloud, all lidars, NLZ filter (get_lidar)
    W = DATA / 'waymo'
    seqs = [l.strip().replace('.tfrecord', '') for l in open(W / 'ImageSets/train.txt') if l.strip()]
    geo = geometry(cfg_w['DATA_CONFIG_TAR'])
    si = int(cfg_w['DATA_CONFIG_TAR']['SAMPLED_INTERVAL']['train'])
    rows, used = [], []
    pick = np.linspace(0, len(seqs) - 1, args.frames).round().astype(int)
    for j, s in enumerate(pick):
        d = W / cfg_w['DATA_CONFIG_TAR']['PROCESSED_DATA_TAG'] / seqs[s]
        files = sorted(d.glob('*.npy'))
        files = [f for f in files if int(f.stem) % si == 0]
        f = files[(j * 37) % len(files)]
        assert os.path.getmtime(f) > time.mktime((2026, 9, 27, 0, 0, 0, 0, 0, -1)), \
            '%s predates the 2026-09-27 re-extraction (corrupt TOP cloud)' % f
        a = np.load(f)
        p = a[a[:, 5] == -1][:, :4]
        rows.append(target_hist(p, geo, np.array([1.0, 0]), np.array([0, 1.0]), (None, None)))
        used.append(str(f))
    res['waymo'] = rows; res['waymo_geo'] = geo; res['waymo_files'] = used
    print('Waymo: %d frames from %d train sequences, SHIFT_COOR %s, SAMPLED_INTERVAL %d'
          % (len(rows), len(seqs), geo['shift'], si), flush=True)
    with open(Path(args.out) / ('targets%s.pkl' % ('_smoke' if args.smoke else '')), 'wb') as f:
        pickle.dump(res, f)


# ----------------------------------------------------------------------------------------------------------------
# PandaSet spin -> flash
# ----------------------------------------------------------------------------------------------------------------
PD_ROOT = Path('/home/koyama/data/pandaset/dataset')
_POSES = {}


def pd_points(seq, frame, pose, device):
    import pandas as pd
    import pandaset as ps
    df = pd.read_pickle(PD_ROOT / seq / 'lidar' / ('%02d.pkl.gz' % frame))
    df = df[df.d == device]
    w = df.to_numpy()
    ego = ps.geometry.lidar_points_to_ego(w[:, :3], pose)
    ego = ego[:, [1, 0, 2]]
    ego[:, 1] = -ego[:, 1]
    return np.column_stack([ego, w[:, 3] / 255.0]).astype(np.float32)


def pd_pose(seq, frame):
    if seq not in _POSES:
        _POSES[seq] = json.load(open(PD_ROOT / seq / 'lidar' / 'poses.json'))
    return _POSES[seq][frame]


def pd_anchor(job):
    seq, frame, have, grid, geo, cone = job
    pose = pd_pose(seq, frame)
    acc = Accumulator(geo, np.array([1.0, 0]), np.array([0, 1.0]), {'flash': cone})
    snaps, used = {}, {}
    n = 0
    for k in range(0, max(grid)):
        if (frame - k) not in have:          # preceding_frames stops at the first gap
            break
        acc.add(pd_points(seq, frame - k, pose, 0), True)
        n += 1
        if n in grid:
            snaps[n] = acc.snapshot()['flash']; used[n] = n
    return snaps, used


def pd_target(job):
    seq, frame, geo, cone = job
    p = pd_points(seq, frame, pd_pose(seq, frame), 1)
    return target_hist(p, geo, np.array([1.0, 0]), np.array([0, 1.0]), cone)


def part_pandaset(args):
    cfg = run_cfg('spin2flash')
    src, tar = cfg['DATA_CONFIG'], cfg['DATA_CONFIG_TAR']
    assert int(src['LIDAR_DEVICE']) == 0 and int(tar['LIDAR_DEVICE']) == 1
    train = list(src['SEQUENCES']['train'])
    assert train == list(tar['SEQUENCES']['train'])
    deg = float(tar.get('EVAL_FOV_DEGREE', 60.0)); head = float(tar.get('EVAL_FOV_HEADING', 0.0))
    # extract_fov_data: heading increases clockwise, 0 = +x; |angle| < half; the cone used to score PandarGT
    cone = (-head - deg / 2.0, -head + deg / 2.0)
    gs, gt = geometry(src), geometry(tar)
    frames = []
    for s in train:
        fs = sorted(int(p.name[:2]) for p in (PD_ROOT / s / 'lidar').glob('*.pkl.gz'))
        frames += [(s, f) for f in fs]
    have = {}
    for s, f in frames:
        have.setdefault(s, set()).add(f)
    grid = [n for n in PD_GRID if n <= (3 if args.smoke else 30)]
    step = max(1, len(frames) // args.anchors)
    anchors = frames[::step][:args.anchors]
    tstep = max(1, len(frames) // args.frames)
    tframes = frames[tstep // 2::tstep][:args.frames]
    print('PandaSet: %d train sequences, %d frames; %d anchors, %d target frames; cone [%.1f, %.1f] deg; '
          'source cap %d (target %d), SHIFT_COOR %s'
          % (len(train), len(frames), len(anchors), len(tframes), cone[0], cone[1], gs['cap'], gt['cap'],
             gs['shift']), flush=True)
    res = {'grid': grid, 'cone': cone, 'src_geo': gs, 'tar_geo': gt}
    with Pool(args.procs) as pool:
        t0 = time.time()
        res['spin'] = pool.map(pd_anchor, [(s, f, have[s], grid, gs, cone) for s, f in anchors], chunksize=1)
        print('spin anchors in %.0f s' % (time.time() - t0), flush=True)
        res['flash'] = pool.map(pd_target, [(s, f, gt, cone) for s, f in tframes], chunksize=1)
    with open(Path(args.out) / ('pandaset%s.pkl' % ('_smoke' if args.smoke else '')), 'wb') as f:
        pickle.dump(res, f)


# ----------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------
def curve(src_rows, tar_rows, var, grid, cap, tag=''):
    """src_rows: list of per-anchor {N: snapshot}; tar_rows: list of target snapshots."""
    th = np.mean([r[var] for r in tar_rows], axis=0)
    rows = []
    for n in grid:
        have = [r[n] for r in src_rows if n in r]
        if not have:
            continue
        sh = np.mean([h[var + tag] for h in have], axis=0)
        ratio = sh / np.maximum(th, 1e-9)
        nv = np.array([h['nvox' + tag] for h in have])
        rows.append(dict(N=n, anchors=len(have), ratio=ratio[BAND].round(3).tolist(),
                         bins_ok=int(np.sum(ratio[BAND] >= 1.0)), min_ratio=float(ratio[BAND].min()),
                         bins_ok20=int(np.sum(ratio[BAND20] >= 1.0)), min_ratio20=float(ratio[BAND20].min()),
                         cap_share=float(np.mean(nv > cap)), vox_p50=float(np.median(nv)),
                         vox_p90=float(np.percentile(nv, 90))))
    nstar = next((r['N'] for r in rows if r['bins_ok'] == 13), None)
    nstar20 = next((r['N'] for r in rows if r['bins_ok20'] == 10), None)
    lim = nstar if nstar is not None else max(r['N'] for r in rows)
    ok = [r['N'] for r in rows if r['N'] <= lim and r['cap_share'] <= 0.10]
    ncap = max(ok) if ok else None
    return dict(rows=rows, N_star=nstar, N_star20=nstar20, N_cap=ncap,
                N_cap_note=None if nstar is not None else 'N* not reached in the grid; N_cap over the whole grid')


def part_report(args):
    out = Path(args.out)
    sfx = '_smoke' if args.smoke else ''
    summary = {}
    ns = pickle.load(open(out / ('nuscenes%s.pkl' % sfx), 'rb'))
    tg = pickle.load(open(out / ('targets%s.pkl' % sfx), 'rb'))
    pdd = pickle.load(open(out / ('pandaset%s.pkl' % sfx), 'rb'))
    print('KITTI camera wedge used on the nuScenes side: [%.2f, %.2f] deg' % tuple(ns['wedge']))
    for pair, tname in (('nuscenes2kitti', 'kitti'), ('nuscenes2waymo', 'waymo')):
        for plat in ('n008', 'n015'):
            src = [r[0][tname] for r in ns[plat]]
            cap = ns[plat + '_geo']['cap']
            for var in VARIANTS:
                for tag in ('', '_d'):
                    if var == 'V2' and tag == '_d':
                        continue
                    key = '%s/%s/%s%s' % (pair, plat, var, tag)
                    summary[key] = curve(src, tg[tname], var, ns['grid'], cap, tag)
    src = [r[0] for r in pdd['spin']]
    for var in VARIANTS:
        summary['spin2flash/pandar64/%s' % var] = curve(src, pdd['flash'], var, pdd['grid'], pdd['src_geo']['cap'])
        summary['spin2flash/pandar64/%s/cap150k' % var] = curve(src, pdd['flash'], var, pdd['grid'], 150000)
    # depth delivered: anchors whose accumulation stopped short of N (scene / sequence start)
    for plat in ('n008', 'n015'):
        short = {n: int(sum(1 for r in ns[plat] if n not in r[1])) for n in ns['grid']}
        dist = {n: float(np.mean([r[1][n][1] for r in ns[plat] if n in r[1]])) for n in ns['grid']}
        summary['delivered/%s' % plat] = dict(short=short, mean_distinct=dist)
    summary['delivered/pandar64'] = {n: int(sum(1 for r in pdd['spin'] if n not in r[1])) for n in pdd['grid']}
    summary['meta'] = dict(wedge=ns['wedge'], cone=pdd['cone'], grid_ns=ns['grid'], grid_pd=pdd['grid'],
                           n_anchor_ns={p: len(ns[p]) for p in ('n008', 'n015')}, n_anchor_pd=len(pdd['spin']),
                           n_target={'kitti': len(tg['kitti']), 'waymo': len(tg['waymo']), 'flash': len(pdd['flash'])},
                           caps={'nuscenes': ns['n008_geo']['cap'], 'pandaset': pdd['src_geo']['cap']})
    with open(out / ('summary%s.json' % sfx), 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    for key, c in summary.items():
        if key.startswith(('delivered', 'meta')):
            print('\n==', key, json.dumps(c, default=float)[:600])
            continue
        print('\n== %s   N* %s   N*_20 %s   N_cap %s %s' % (key, c['N_star'], c['N_star20'], c['N_cap'],
                                                         c['N_cap_note'] or ''))
        print('   N  anch  ok13  min13  ok10  min10  cap%   vox_p50   ratios 5-70 m')
        for r in c['rows']:
            print('  %3d  %4d  %4d  %5.2f  %4d  %5.2f  %4.0f  %8.0f   %s'
                  % (r['N'], r['anchors'], r['bins_ok'], r['min_ratio'], r['bins_ok20'], r['min_ratio20'],
                     100 * r['cap_share'], r['vox_p50'], ' '.join('%.2f' % x for x in r['ratio'])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['nuscenes', 'targets', 'pandaset', 'report'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--anchors', type=int, default=100)
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.anchors, args.frames = 3, 3
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    {'nuscenes': part_nuscenes, 'targets': part_targets, 'pandaset': part_pandaset, 'report': part_report}[args.part](args)
    import resource
    print('part %s done in %.0f s; max RSS self %.1f GB, children %.1f GB'
          % (args.part, time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
             resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6), flush=True)


if __name__ == '__main__':
    main()
