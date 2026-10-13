"""The (c1) CPU gate of experiments_md 20261010_04 §5.1 (the 19:25 amendment and the operational definitions
pre-declared 2026-10-13 23:37 JST), on the aggregation cache (`analysis/build_track_agg_cache.py`). Source labels only,
except the G1' reference, which is read from a file (see the report for its provenance).

    python analysis/track_aggregation_gate.py g3c <cfg> <DATA_CONFIGS key> <cache_dir> <frames> <out.npz>
    python analysis/track_aggregation_gate.py g1 <cfg> <DATA_CONFIGS key> <cache_dir> <raydrop.npz> <frames> <out.npz>
    python analysis/track_aggregation_gate.py g4 <trackagg cfg> <DATA_CONFIGS key> <frames> <out.npz>
    python analysis/track_aggregation_gate.py report <reference: val_analysis | ref.npz> <g1 n008.npz> <g1 n015.npz> \
        <g3c n008.npz> <g3c n015.npz> [<g4 n008.npz> <g4 n015.npz>]

g3c - leave-anchor-out transport accuracy with the additions (item 7): per sparse far car of the anchor keyframe
      (20-75 m, 10-49 own in-box points), ICP fitted on its EVEN-ring own points and |dr| scored on its ODD-ring own
      points, then the reverse; the other keyframes' points are ICP-refined and back-face culled, z-buffered alone on
      the Waymo median lattice, against the scored own points z-buffered alone. Ray-drop off.
g1  - lattice-row coverage (item 8): Probe B's population and RC / RC_front (control = 27468's loader cloud, moving
      ego, Car boxes at 20-75 m with 10-49 control points), now with the cached ICP, culling and ray-drop (seed fixed).
g4  - the hook's time (cache read, transforms, culling, ray-drop) plus the z-buffer rule's time on the appended cloud,
      and the voxel count of the training sample against the 150k cap (item 9).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import pickle
import sys
import time
import zlib
from pathlib import Path

import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets import rerender_utils as RR
from pcdet.datasets.nuscenes import track_aggregation as TA
from pcdet.datasets.point_calibration import _augmentation_off
from pcdet.datasets.processor.point_sampler import lattice_pixels, zbuffer_keep
from pcdet.utils import common_utils
from lattice_depth_ceiling import travel_direction, sector_of, SECTORS
from track_aggregation_probe import rc_records, WAYMO_RC

SENSOR_Z = 2.184
CELLS = [((20, 40), s) for s in range(3)] + [((40, 75), s) for s in range(3)]


def dataset(cfg, key, training=True):
    dc = cfg.DATA_CONFIGS[key]
    return build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                            logger=common_utils.create_logger(), training=training, model_ontology=cfg.get('ONTOLOGY'))[0]


def load_index(cache):
    with open(Path(cache) / 'index.pkl', 'rb') as f:
        return pickle.load(f)


def sensor_lidar(dc):
    shift = np.array(dc.get('SHIFT_COOR', [0, 0, 0]), np.float64)
    return np.array([0.0, 0.0, SENSOR_Z - shift[2]]), shift


def g3c(cfg, key, cache, n, out):
    dc = cfg.DATA_CONFIGS[key]; dc.RERENDER = None; dc.TRACK_AGG = None
    ds = dataset(cfg, key); index = load_index(cache)
    spec = RR.lattice_spec('waymo_top'); th, nc, hs = spec['thetas'], spec['n_cols'], spec['height']
    sensor, shift = sensor_lidar(dc)
    step = max(1, len(ds.infos) // n); recs = []; fits = [0, 0]
    for fi, i in enumerate(list(range(0, len(ds.infos), step))[:n]):
        scene_tok, a = index[ds.infos[i]['token']]
        sc = TA.load_scene(Path(cache) / f'{scene_tok}.pkl'); fa = sc['frames'][a]
        for k, box in zip(fa['tids'], fa['boxes']):
            r = np.hypot(*box[:2])
            if not 20 <= r < 75 or k not in fa['points']:
                continue
            own_loc, own_attr = fa['points'][k]
            own_loc = np.asarray(own_loc, np.float64); ring = np.asarray(own_attr)[:, 1].astype(int)
            if not 10 <= len(own_loc) < 50:
                continue
            others = [(np.asarray(fj['points'][k][0], np.float64)) for j, fj in enumerate(sc['frames'])
                      if j != a and k in fj['points']]
            if not others:
                continue
            s_loc = TA.to_box_local(sensor[None], box)[0]
            for parity in (0, 1):
                fit, score = own_loc[ring % 2 == parity], own_loc[ring % 2 != parity]
                if not len(score):
                    continue
                agg = []
                for src in others:
                    yaw, t, ok, _ = TA.icp_4dof(src, fit)
                    fits[0] += 1; fits[1] += int(ok)
                    loc = TA.apply_4dof(src, yaw, t)
                    agg.append(loc[TA.visible_mask(loc, box[3:6], s_loc)])
                agg = np.concatenate(agg)
                if not len(agg):
                    continue
                imgs = []
                for p in (TA.from_box_local(score, box), TA.from_box_local(agg, box)):
                    row, col, rng, v = RR.to_lattice(p + shift, hs, th, nc)
                    imgs.append(RR.zbuffer(row, col, rng, v, len(th), nc)[0])
                both = np.isfinite(imgs[0]) & np.isfinite(imgs[1])
                sd = imgs[1][both] - imgs[0][both]                 # aggregated minus own: > 0 = behind the surface
                recs.extend((fi, r, len(own_loc), abs(dd), dd, parity, zlib.crc32(k.encode())) for dd in sd)
        if (fi + 1) % 20 == 0:
            print(f'  {key}: {fi + 1} frames, {len(recs)} cells, ICP accepted {fits[1]} / {fits[0]}', flush=True)
    R = np.array(recs, float)
    np.savez(out, key=key, cells=R, icp_fits=np.array(fits), n_frames=n)
    for lo, hi in ((20, 40), (40, 75), (20, 75)):
        s = (R[:, 1] >= lo) & (R[:, 1] < hi)
        print(f'{key} {lo}-{hi} m: {s.sum()} cells; median |dr| {np.median(R[s, 3]):.3f} m, share > 0.5 m '
              f'{np.mean(R[s, 3] > 0.5):.3f} (behind {np.mean(R[s, 4] > 0.5):.3f}, in front {np.mean(R[s, 4] < -0.5):.3f})')
    print(f'ICP accepted {fits[1]} of {fits[0]} fits')


def g1(cfg, key, cache, raydrop, n, out):
    dc = cfg.DATA_CONFIGS[key]; dc.RERENDER = None; dc.TRACK_AGG = None
    ds = dataset(cfg, key); index = load_index(cache); rd = TA.load_raydrop(raydrop)
    spec = RR.lattice_spec('waymo_top'); th, nc, hs = spec['thetas'], spec['n_cols'], spec['height']
    inc_desc = th[::-1]
    sensor, shift = sensor_lidar(dc)
    rows = []
    with _augmentation_off(ds):
        step = max(1, len(ds) // n); frames = list(range(0, len(ds), step))[:n]
        dirs = [travel_direction(ds.infos[i]) for i in frames]
        for fi, i in enumerate(frames):
            if dirs[fi] is None:
                continue                                           # the gate's population is moving-ego frames
            fwd = dirs[fi]
            smp = ds[i]
            b = smp['gt_boxes']; b = b[b[:, 7] == 1] if b is not None and len(b) else np.zeros((0, 8))
            if not len(b):
                continue
            from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
            ns = (roiaware_pool3d_utils.points_in_boxes_cpu(smp['points'][:, :3], b[:, :7]) > 0).sum(1)
            rr = np.hypot(b[:, 0], b[:, 1]); cand = np.nonzero((rr >= 20) & (rr < 75) & (ns >= 10) & (ns < 50))[0]
            if not len(cand):
                continue
            info = ds.infos[i]
            P = ds.get_lidar_with_sweeps(i, max_sweeps=ds.dataset_cfg.MAX_SWEEPS)[:, :3].astype(np.float64)
            scene_tok, a = index[info['token']]
            sc = TA.load_scene(Path(cache) / f'{scene_tok}.pkl')
            A = TA.aggregate_anchor(sc, a, sensor)[:, :3]
            rng = np.random.default_rng(zlib.crc32(info['token'].encode()))
            A_rd = A[TA.raydrop_keep(A, rd['rates'], rd['range_edges'], rd['elev_split_rad'], th, nc, sensor[2], rng)]
            clouds = {'control': P + shift, 'agg': np.concatenate([P, A]) + shift,
                      'agg_raydrop': np.concatenate([P, A_rd]) + shift}
            recs = {name: rc_records(RR.rerender(np.column_stack([Q, np.zeros(len(Q))]), spec, fill='none')[0][:, :3]
                                     .astype(np.float64), b, cand, inc_desc, nc, hs) for name, Q in clouds.items()}
            fa = sc['frames'][a]
            cen = np.asarray(fa['boxes'])[:, :3] + shift if len(fa['tids']) else np.zeros((0, 3))
            closest = {}
            for k in fa['tids']:
                closest[k] = min(np.hypot(*np.asarray(f['boxes'])[f['tids'].index(k)][:2])
                                 for f in sc['frames'] if k in f['tids'])
            for jj, j in enumerate(cand):
                k = None
                if len(cen):
                    dd = np.hypot(cen[:, 0] - b[j, 0], cen[:, 1] - b[j, 1])
                    k = fa['tids'][int(np.argmin(dd))] if dd.min() < 0.2 else None
                row = [fi, rr[j], sector_of(b[j:j + 1, :2], fwd)[0], 1, ns[j]]
                for name in ('control', 'agg', 'agg_raydrop'):
                    row += list(recs[name][jj])
                row += [closest.get(k, np.nan) if k else np.nan]
                rows.append(row)
            if (fi + 1) % 20 == 0:
                print(f'  {key}: {fi + 1} / {len(frames)} frames, {len(rows)} sparse far cars', flush=True)
    np.savez(out, key=key, rows=np.array(rows, float), n_frames=len(frames))
    print(f'saved {out}: {len(rows)} records')


def g4(cfg, key, n, out):
    ds = dataset(cfg, key)
    assert ds.dataset_cfg.get('TRACK_AGG', None) is not None, 'g4 needs the trackagg config'
    sampler = None
    for p in ds.dataset_cfg.DATA_PROCESSOR:
        if p.NAME == 'sample_points_learned':
            from pcdet.datasets.processor.point_sampler import LearnedPointSampler
            sampler = LearnedPointSampler.load(p.WEIGHTS)
    assert sampler is not None and sampler.rule_kind == 'zbuffer', "g4 times 27834's z-buffer rule"
    cap = [p for p in ds.dataset_cfg.DATA_PROCESSOR if p.NAME == 'transform_points_to_voxels'][0].MAX_NUMBER_OF_VOXELS['train']
    shift = np.array(ds.dataset_cfg.get('SHIFT_COOR', [0, 0, 0]), np.float32)
    step = max(1, len(ds.infos) // n); recs = []
    for fi, i in enumerate(list(range(0, len(ds.infos), step))[:n]):
        info = ds.infos[i]
        t0 = time.time(); A = ds.track_agg_points(info); t_hook = time.time() - t0
        P = ds.get_lidar_with_sweeps(i, max_sweeps=ds.dataset_cfg.MAX_SWEEPS)
        Q = np.concatenate([P, A])[:, :3] + shift
        t0 = time.time()
        pix, _, rng = lattice_pixels(Q, sampler.lattice_inc, sampler.lattice_cols, sampler.lattice_sensor_z)
        zbuffer_keep(pix, rng); t_z = time.time() - t0
        smp = ds[i]
        recs.append((fi, len(P), len(A), t_hook, t_z, len(smp['voxels']), int(len(smp['voxels']) >= cap)))
        if (fi + 1) % 20 == 0:
            print(f'  {key}: {fi + 1} frames', flush=True)
    R = np.array(recs, float)
    np.savez(out, key=key, recs=R, cap=cap)
    t = R[:, 3] + R[:, 4]
    print(f'{key}: {len(R)} anchors; aggregated points per anchor median {np.median(R[:, 2]):.0f} (control '
          f'{np.median(R[:, 1]):.0f}); hook + z-buffer time median {np.median(t):.3f} s, p99 {np.percentile(t, 99):.3f} s, '
          f'< 0.5 s in {np.mean(t < 0.5):.3f}; voxels median {np.median(R[:, 5]):.0f}, at the {cap} cap in {R[:, 6].sum():.0f}')


def g1_table(path, ref):
    G = np.load(path); R = G['rows']
    # columns: 0 fi, 1 range, 2 sector, 3 moving, 4 ns, then (RC, RC_front, pts) x [control, agg, agg_raydrop], closest
    c = lambda block, k: 5 + 3 * block + k
    out = {}
    for (lo, hi), s in CELLS:
        q = R[(R[:, 1] >= lo) & (R[:, 1] < hi) & (R[:, 2] == s)]
        out[(lo, s)] = (len(q), *[np.nanmedian(q[:, c(b, k)]) if len(q) else np.nan for b in range(3) for k in range(2)],
                        np.mean(q[:, -1] < 20) if len(q) else np.nan)
    return str(G['key']), int(G['n_frames']), out


def report(ref_arg, g1_paths, g3_paths, g4_paths):
    if ref_arg == 'val_analysis':
        ref = WAYMO_RC; ref_name = 'Waymo own RC, Waymo VAL labels (20261009_01 §9.1; ANALYSIS)'
    else:
        d = np.load(ref_arg); ref = {(int(a), int(b)): float(v) for a, b, v in d['cells']}; ref_name = str(d['name'])
    print(f'Reference: {ref_name}\n')
    print('| platform | ring | sector | cars | control RC / front | aggregated RC / front | + ray-drop RC / **front** | '
          'reference | two-sided | one-sided | track ever < 20 m |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    two = one = tot = 0
    for p in g1_paths:
        key, nf, cells = g1_table(p, ref)
        for (lo, hi), s in CELLS:
            n_c, c0, c0f, a0, a0f, d0, d0f, near = cells[(lo, s)]
            r = ref[(lo, s)]; tw = abs(d0f - r) <= 0.10; on = r <= d0f + 0.10
            two += tw; one += on; tot += 1
            print(f'| {key} | {lo}-{hi} | {SECTORS[s]} | {n_c} | {c0:.2f} / {c0f:.2f} | {a0:.2f} / {a0f:.2f} | '
                  f'{d0:.2f} / **{d0f:.2f}** | {r:.2f} | {"yes" if tw else "no"} | {"yes" if on else "no"} | {near:.2f} |')
    print(f'\nG1\' two-sided (DECIDING): {two} of {tot} cells within 0.10 (bar >= 10); one-sided (reported): {one} of {tot}')
    for p in g3_paths:
        G = np.load(p); R = G['cells']; f = G['icp_fits']
        s = (R[:, 1] >= 20) & (R[:, 1] < 75)
        med, share = np.median(R[s, 3]), np.mean(R[s, 3] > 0.5)
        print(f'G3c {str(G["key"])}: {s.sum()} cells, median |dr| {med:.3f} m (bar 0.15), share > 0.5 m {share:.3f} '
              f'(bar 0.10) -> {"PASS" if med <= 0.15 and share <= 0.10 else "FAIL"}; behind {np.mean(R[s, 4] > 0.5):.3f}, '
              f'in front {np.mean(R[s, 4] < -0.5):.3f}; 20-40 m {np.mean(R[(R[:, 1] < 40), 3] > 0.5):.3f}, 40-75 m '
              f'{np.mean(R[(R[:, 1] >= 40), 3] > 0.5):.3f}; ICP accepted {f[1]} / {f[0]}')
    for p in g4_paths:
        G = np.load(p); R = G['recs']; t = R[:, 3] + R[:, 4]
        ok = np.mean(t < 0.5) >= 0.99 and np.mean(R[:, 6] == 0) >= 0.99
        print(f'G4 {str(G["key"])}: {len(R)} anchors, time < 0.5 s in {np.mean(t < 0.5):.3f} (p99 {np.percentile(t, 99):.3f} s), '
              f'under the cap in {np.mean(R[:, 6] == 0):.3f} -> {"PASS" if ok else "FAIL"}')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'report':
        a = sys.argv[3:]
        report(sys.argv[2], a[0:2], a[2:4], a[4:6]); sys.exit(0)
    cfg = EasyDict(); cfg_from_yaml_file(sys.argv[2], cfg)
    if mode == 'g3c':
        g3c(cfg, sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
    elif mode == 'g1':
        g1(cfg, sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), sys.argv[7])
    elif mode == 'g4':
        g4(cfg, sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        raise SystemExit(f'unknown mode {mode}')
