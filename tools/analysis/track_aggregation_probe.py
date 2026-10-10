"""Whole-track object aggregation from SOURCE labels, rendered on the Waymo TOP lattice: does it give sparse far cars
Waymo's lattice-row coverage? (experiments_md 20261010_04, design study; CPU feasibility probe, no training.)

Per anchor of a nuScenes source config (the re-render gate's config and population, 20261009_01 §4 / §7):
- control: the loader's accumulated cloud (27468's recipe, consecutive Boston 15 / Singapore 10, box-compensated);
- track: control + the in-box points of every Car track from the OTHER keyframes of the scene (2 Hz, annotated - no
  box interpolation), each moved rigidly from that keyframe's box onto the anchor's box of the same track
  (nuScenes `instance_token`). Windows: keyframes within +-5 s, and the whole scene (<= ~20 s).
Both are z-buffered on the Waymo TOP median lattice at the source ego pose (re-render operator, no fill); RC is G1's
(sparse far cars: Car boxes at 20-75 m with 10-49 control-cloud points, moving ego; occlusion judged on the render).
RC_front counts only in-box returns within 1.0 m of the box's entry surface along the ray, so returns from a car's FAR
side seen through gaps of its near side (a render of an aggregated, not meshed, object) do not count.
Source labels only (UDA-legal); no target point or label is read. Waymo's own RC is quoted from 20261009_01 §9.1.

    python analysis/track_aggregation_probe.py measure <cfg> <DATA_CONFIGS key> <frames> <out.npz>
    python analysis/track_aggregation_probe.py report <out.npz> [<out.npz> ...]
    python analysis/track_aggregation_probe.py g3c <cfg> <DATA_CONFIGS key> <frames> <out.npz>   (leave-anchor-out G3c)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off
from pcdet.datasets import rerender_utils as RR
from pcdet.datasets.motion_compensation import DevkitSweepCompensator, find_devkit_meta_dir, _yaw_to_R
from pcdet.datasets.processor.point_sampler import lattice_pixels
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
from lattice_depth_ceiling import travel_direction, sector_of, SECTORS
from lattice_sampler_precheck import silhouette

OCCL_M = 0.5
FRONT_M = 1.0
WINDOWS = (5.0, 1e9)
SPAN = dict(fill='rule', dr_max=0.3, span_max_deg=1.5)   # the gate's deciding fill A, reported
# Waymo's own RC on sparse far cars (10-49 TOP points, own lattice) and the deep render's, 20261009_01 §9.1
WAYMO_RC = {(20, 0): 0.61, (20, 1): 0.50, (20, 2): 0.53, (40, 0): 0.62, (40, 1): 0.62, (40, 2): 0.67}
DEEP_RC = {'NUSCENES_N008': {(20, 0): 0.40, (20, 1): 0.50, (20, 2): 0.42, (40, 0): 0.38, (40, 1): 0.55, (40, 2): 0.43},
           'NUSCENES_N015': {(20, 0): 0.38, (20, 1): 0.40, (20, 2): 0.40, (40, 0): 0.33, (40, 1): 0.54, (40, 2): 0.48}}


def inside(xyz, box):
    d = xyz - box[:3]; R = _yaw_to_R(-box[6])
    lx = d[:, 0] * R[0, 0] + d[:, 1] * R[0, 1]; ly = d[:, 0] * R[1, 0] + d[:, 1] * R[1, 1]
    return (np.abs(lx) <= box[3] / 2) & (np.abs(ly) <= box[4] / 2) & (np.abs(d[:, 2]) <= box[5] / 2)


def move(xyz, then, now):
    d = xyz - then[:3]; Rf = _yaw_to_R(now[6] - then[6])     # 2 x 2, about z
    out = d.copy(); out[:, :2] = d[:, :2] @ Rf.T
    return out + now[:3]


class TrackAggregator:
    def __init__(self, ds):
        self.ds = ds
        self.comp = DevkitSweepCompensator(ds.infos, find_devkit_meta_dir(ds.root_path, ds.dataset_cfg.VERSION),
                                           classes={'car'})
        tok2info = {inf['token']: k for k, inf in enumerate(ds.infos)}
        self.f2i = {f: tok2info[t] for t, f in self.comp.tok2frame.items()}

    def points(self, i, window_s, per_track=False):
        """(aggregated in-box points of other keyframes moved onto the anchor's boxes, (N, 3), anchor lidar frame;
        per anchor track: closest range to the sensor over the scene's keyframes, keyframes contributing).
        per_track: the first item is {track: (N, 3)} instead."""
        c = self.comp; info = self.ds.infos[i]; ai = c.tok2frame[info['token']]
        S_a = c.frames[ai]['S']; now = c._boxes_in(ai, S_a); t0 = c.frames[ai]['t']
        closest = {k: np.hypot(*v[:2]) for k, v in now.items()}; nkf = {k: 0 for k in now}
        out = []; by = {}
        for j in c.scene_order[c.frames[ai]['scene']]:
            if j == ai or abs(c.frames[j]['t'] - t0) > window_s:
                continue
            then = c._boxes_in(j, S_a); common = [k for k in then if k in now]
            if not common:
                continue
            T = S_a @ np.linalg.inv(c.frames[j]['S'])
            jinfo = self.ds.infos[self.f2i[j]]
            p = np.fromfile(str(self.ds.root_path / jinfo['lidar_path']), dtype=np.float32).reshape(-1, 5)[:, :3]
            p = p[~((np.abs(p[:, 0]) < 1.5) & (np.abs(p[:, 1]) < 1.5))].astype(np.float64)
            sensor_j = T[:3, 3]
            xyz = p @ T[:3, :3].T + T[:3, 3]
            for k in common:
                m = inside(xyz, then[k])
                closest[k] = min(closest[k], np.hypot(*(then[k][:2] - sensor_j[:2])))
                if m.any():
                    q = move(xyz[m], then[k], now[k]); out.append(q); nkf[k] += 1
                    by.setdefault(k, []).append(q)
        if per_track:
            return {k: np.concatenate(v) for k, v in by.items()}, now, closest, nkf
        return (np.concatenate(out) if out else np.zeros((0, 3))), now, closest, nkf


def g3c(cfg, key, n, out):
    """Leave-anchor-out transport accuracy (G3c of 20261010_04 §5.1): per sparse far car, the OTHER keyframes' in-box
    points moved onto the anchor box vs the anchor KEYFRAME's own in-box returns, each z-buffered alone on the Waymo
    lattice; |dr| on cells where both have a return. Source labels only."""
    dc = cfg.DATA_CONFIGS[key]; dc.RERENDER = None
    spec = RR.lattice_spec('waymo_top'); th, nc, hs = spec['thetas'], spec['n_cols'], spec['height']
    shift = np.array(dc.get('SHIFT_COOR', [0, 0, 0]), np.float64)
    ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
    agg = TrackAggregator(ds); recs = []
    step = max(1, len(ds) // n)
    for fi, i in enumerate(list(range(0, len(ds), step))[:n]):
        info = ds.infos[i]
        tracks, now, closest, nkf = agg.points(i, 1e9, per_track=True)
        own = np.fromfile(str(ds.root_path / info['lidar_path']), dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
        own = own[~((np.abs(own[:, 0]) < 1.5) & (np.abs(own[:, 1]) < 1.5))]
        for k, box in now.items():
            r = np.hypot(*box[:2])
            if not 20 <= r < 75 or k not in tracks:
                continue
            o = own[inside(own, box)]
            if not 10 <= len(o) < 50:            # sparse in the anchor keyframe itself
                continue
            imgs = []
            for p in (o, tracks[k]):
                row, col, rng, v = RR.to_lattice(p + shift, hs, th, nc)
                imgs.append(RR.zbuffer(row, col, rng, v, len(th), nc)[0])
            both = np.isfinite(imgs[0]) & np.isfinite(imgs[1])
            if both.any():
                sd = imgs[1][both] - imgs[0][both]          # aggregated minus own: > 0 = behind the real surface
                recs.extend((fi, r, len(o), abs(dd), closest[k], dd) for dd in sd)
        if (fi + 1) % 20 == 0:
            print(f'  {key}: {fi + 1} frames, {len(recs)} cells', flush=True)
    R = np.array(recs, float); np.savez(out, key=key, cells=R)
    for lo, hi in ((20, 40), (40, 75), (20, 75)):
        s = (R[:, 1] >= lo) & (R[:, 1] < hi)
        print(f'{key} {lo}-{hi} m: {s.sum()} cells, {len(np.unique(R[s][:, [0, 1]], axis=0))} cars; median |dr| '
              f'{np.median(R[s, 3]):.3f} m, share > 0.5 m {np.mean(R[s, 3] > 0.5):.3f} (behind the own return '
              f'{np.mean(R[s, 5] > 0.5):.3f}, in front {np.mean(R[s, 5] < -0.5):.3f}; behind by > 1.2 m {np.mean(R[s, 5] > 1.2):.3f})')


def rc_records(Qv, b, cand, inc_desc, nc, hs):
    pix, _, rng = lattice_pixels(Qv, inc_desc, nc, hs)
    okp = np.nonzero(pix >= 0)[0]
    up, pinv = np.unique(pix[okp], return_inverse=True)
    o = np.lexsort((rng[okp], pinv)); nidx = okp[o[np.r_[0, np.nonzero(np.diff(pinv[o]))[0] + 1]]] if len(okp) else okp
    nrng = rng[nidx] if len(nidx) else np.zeros(0)
    inb = roiaware_pool3d_utils.points_in_boxes_cpu(Qv.astype(np.float32), b[cand, :7].astype(np.float32)) > 0
    res = []
    for jj, j in enumerate(cand):
        sp, srow, tin = silhouette(b[j], inc_desc, hs=hs, n_cols=nc)
        if not len(sp):
            res.append((np.nan, np.nan, 0)); continue
        if len(up):
            pos = np.clip(np.searchsorted(up, sp), 0, len(up) - 1); has = up[pos] == sp
            occl = has & (nrng[pos] < tin - OCCL_M) & ~inb[jj][nidx[pos]]
        else:
            occl = np.zeros(len(sp), bool)
        vp, vrow, vtin = sp[~occl], srow[~occl], tin[~occl]
        nv = len(np.unique(vrow))
        if nv < 1:
            res.append((np.nan, np.nan, 0)); continue
        sel = inb[jj] & (pix >= 0); pp = pix[sel]; pr = rng[sel]
        o2 = np.argsort(vp); vps = vp[o2]; k = np.clip(np.searchsorted(vps, pp), 0, len(vps) - 1); isv = vps[k] == pp
        front = isv & (pr <= vtin[o2][k] + FRONT_M)
        res.append((len(np.unique(pp[isv] // nc)) / nv, len(np.unique(pp[front] // nc)) / nv, int(sel.sum())))
    return res


def measure(cfg, key, n, out):
    dc = cfg.DATA_CONFIGS[key]; dc.RERENDER = None
    spec = RR.lattice_spec('waymo_top'); th, nc, hs = spec['thetas'], spec['n_cols'], spec['height']
    inc_desc = th[::-1]
    shift = np.array(dc.get('SHIFT_COOR', [0, 0, 0]), np.float64)
    ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
    agg = TrackAggregator(ds)
    rows = []
    with _augmentation_off(ds):
        step = max(1, len(ds) // n); frames = list(range(0, len(ds), step))[:n]
        dirs = [travel_direction(ds.infos[i]) for i in frames]
        mv = np.array([d for d in dirs if d is not None]); axis = mv.sum(0) / np.linalg.norm(mv.sum(0))
        for fi, i in enumerate(frames):
            fwd = dirs[fi] if dirs[fi] is not None else axis; moving = dirs[fi] is not None
            smp = ds[i]
            b = smp['gt_boxes']; b = b[b[:, 7] == 1] if b is not None and len(b) else np.zeros((0, 8))
            if not len(b):
                continue
            ns = (roiaware_pool3d_utils.points_in_boxes_cpu(smp['points'][:, :3], b[:, :7]) > 0).sum(1)
            rr = np.hypot(b[:, 0], b[:, 1]); cand = np.nonzero((rr >= 20) & (rr < 75) & (ns >= 10) & (ns < 50))[0]
            if not len(cand):
                continue
            P = ds.get_lidar_with_sweeps(i, max_sweeps=ds.dataset_cfg.MAX_SWEEPS)[:, :3].astype(np.float64) + shift
            clouds = {'control': P}
            meta = None
            for w in WINDOWS:
                A, now, closest, nkf = agg.points(i, w)
                clouds[w] = np.concatenate([P, A + shift])
                if meta is None or w == WINDOWS[-1]:
                    meta = (now, closest, nkf, len(A))
            now, closest, nkf, nA = meta
            # anchor track of each candidate box: nearest centre (boxes are the same annotations, shifted in z)
            tids = list(now); cen = np.array([now[k][:3] for k in tids]) if tids else np.zeros((0, 3))
            recs = {name: rc_records(RR.rerender(np.column_stack([Q, np.zeros(len(Q))]), spec, fill='none')[0][:, :3]
                                     .astype(np.float64), b, cand, inc_desc, nc, hs) for name, Q in clouds.items()}
            Qa = RR.rerender(np.column_stack([clouds[WINDOWS[-1]], np.zeros(len(clouds[WINDOWS[-1]]))]), spec, **SPAN)[0]
            recA = rc_records(Qa[:, :3].astype(np.float64), b, cand, inc_desc, nc, hs)
            for jj, j in enumerate(cand):
                k = None
                if len(cen):
                    dd = np.hypot(cen[:, 0] - b[j, 0], cen[:, 1] - b[j, 1]); k = tids[int(np.argmin(dd))] if dd.min() < 0.2 else None
                row = [fi, rr[j], sector_of(b[j:j + 1, :2], fwd)[0], moving, ns[j]]
                for name in ['control'] + list(WINDOWS):
                    row += list(recs[name][jj])
                row += list(recA[jj])
                row += [closest.get(k, np.nan) if k else np.nan, nkf.get(k, 0) if k else 0]
                rows.append(row)
            if (fi + 1) % 20 == 0:
                print(f'  {key}: {fi + 1} / {len(frames)} frames, {len(rows)} sparse far cars', flush=True)
    np.savez(out, key=key, rows=np.array(rows, float), windows=np.array(WINDOWS), n_frames=len(frames))
    print(f'saved {out}: {len(rows)} records')


def report(paths):
    for p in paths:
        G = np.load(p); key = str(G['key']); R = G['rows']
        # columns: 0 fi, 1 range, 2 sector, 3 moving, 4 ns, then (RC, RC_front, pts) x [control, w5, all, all+A], closest, nkf
        c = lambda block, k: 5 + 3 * block + k
        pop = R[(R[:, 3] > 0)]
        print(f'\n# {key}: {int(G["n_frames"])} frames, {len(pop)} sparse far cars in moving-ego frames')
        print('| ring | sector | cars | control RC | +-5 s RC / front | whole scene RC / front | whole + fill A RC | '
              'deep render (§8.2) | Waymo own (§9.1) | closest < 20 m | keyframes |')
        print('|---|---|---|---|---|---|---|---|---|---|---|')
        for lo, hi in ((20, 40), (40, 75)):
            for s in range(3):
                q = pop[(pop[:, 1] >= lo) & (pop[:, 1] < hi) & (pop[:, 2] == s)]
                if not len(q):
                    continue
                m = lambda col: np.nanmedian(q[:, col])
                print(f'| {lo}-{hi} | {SECTORS[s]} | {len(q)} | {m(c(0, 0)):.2f} | {m(c(1, 0)):.2f} / {m(c(1, 1)):.2f} | '
                      f'{m(c(2, 0)):.2f} / {m(c(2, 1)):.2f} | {m(c(3, 0)):.2f} | {DEEP_RC[key][(lo, s)]:.2f} | '
                      f'{WAYMO_RC[(lo, s)]:.2f} | {np.mean(q[:, -2] < 20):.2f} | {np.median(q[:, -1]):.0f} |')
        q = pop
        for lab, s in (('closest approach < 20 m', q[:, -2] < 20), ('closest approach >= 20 m', q[:, -2] >= 20)):
            if s.any():
                print(f'  {lab}: {s.sum()} cars, whole-scene RC {np.nanmedian(q[s, c(2, 0)]):.2f} / front '
                      f'{np.nanmedian(q[s, c(2, 1)]):.2f}, control {np.nanmedian(q[s, c(0, 0)]):.2f}')
        cells = [((lo, hi), s) for lo, hi in ((20, 40), (40, 75)) for s in range(3)]
        n_within = sum(1 for (lo, hi), s in cells
                       if len(q[(q[:, 1] >= lo) & (q[:, 1] < hi) & (q[:, 2] == s)]) and
                       WAYMO_RC[(lo, s)] <= np.nanmedian(q[(q[:, 1] >= lo) & (q[:, 1] < hi) & (q[:, 2] == s), c(2, 1)]) + 0.10)
        print(f'  cells where Waymo own RC <= whole-scene FRONT RC + 0.10: {n_within} of {len(cells)}')


if __name__ == '__main__':
    if sys.argv[1] == 'report':
        report(sys.argv[2:]); sys.exit(0)
    cfg = EasyDict(); cfg_from_yaml_file(sys.argv[2], cfg)
    (g3c if sys.argv[1] == 'g3c' else measure)(cfg, sys.argv[3], int(sys.argv[4]), sys.argv[5])
