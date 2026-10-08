"""Pre-check of the lattice sampler on deep sources (experiments_md 20261008_01 §9). CPU, analysis.

Decides between re-rendering (candidates insufficient), an object-aware objective with structured selection (v1
drops or mis-arranges candidates) and a plain retrain (only the scene fit / cap / features fail), from:
- the VISIBLE Waymo-lattice silhouette of each sparse far source car (virtual Waymo TOP rays - published
  inclinations, 2,650 columns, sensor 2.184 m above ground - cast from the anchor onto the source box; pixels whose
  raw-cloud nearest point lies outside the box and > 0.5 m in front of it are occluded and dropped);
- per arm (raw / v1 / z-buffer rule): row coverage RC, pixel coverage, points per occupied silhouette pixel (PPP),
  paired retention; arrangement at matched counts; voxels vs the 250k cap; scene T1 / T2 (trainer's formulas, realised
  keeps); feature shift against v1's training range; the share of scene voxels inside sparse far cars.
Target side reads only Waymo TRAIN clouds and the native range-image statistics; Waymo val boxes are diagnosis.

    python analysis/lattice_sampler_precheck.py waymo <cfg> <train frames, 60 = v1's target> <val frames> <out.npz>
    python analysis/lattice_sampler_precheck.py source <cfg> <key> <sources, e.g. 15@27,15@27F,30@50F> <control>
           <v1 weights.npz> <frames> <train-range frames, 60> <out.npz>
    python analysis/lattice_sampler_precheck.py report <waymo.npz> <source.npz> [<source.npz> ...]
"""
import sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off, calibration_target_config
from pcdet.datasets.processor.point_sampler import (LearnedPointSampler, lattice_pixels, point_features, cell_index,
                                                    FEATURE_NAMES, LATTICE_FEATURE_NAMES, DET_VOXEL, DET_PCR)
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils, box_utils
from lattice_depth_ceiling import parse_config, travel_direction, sector_of, ring_of, STATS, N_COLS, SENSOR_Z, CAP, \
    BOX_VOX, SECTORS

ARMS = ('raw', 'v1', 'zbuffer')
FEATS = tuple(FEATURE_NAMES) + tuple(LATTICE_FEATURE_NAMES)
OCCL_M = 0.5


def voxel_table(x):
    """Occupied detector voxels: (centres (M, 3), point count, voxel id per point (-1 outside))."""
    ins = np.all((x >= DET_PCR[:3]) & (x < DET_PCR[3:]), 1)
    ijk = np.floor((x - DET_PCR[:3]) / DET_VOXEL).astype(np.int64)
    key = np.where(ins, (ijk[:, 0] * 2000 + ijk[:, 1]) * 100 + ijk[:, 2], -1)
    uk, inv, cnt = np.unique(key[ins], return_inverse=True, return_counts=True)
    vid = np.full(len(x), -1, np.int64); vid[ins] = inv
    cen = DET_PCR[:3] + (np.stack([uk // 200000, (uk // 100) % 2000, uk % 100], 1) + 0.5) * DET_VOXEL
    return cen, cnt, vid


def silhouette(box, inc_desc, hs=SENSOR_Z, n_cols=N_COLS):
    """Virtual-lattice pixels whose ray from (0, 0, hs) hits the box: (pixel ids, rows, entry range)."""
    c = box_utils.boxes_to_corners_3d(box[None, :7].astype(np.float64))[0]
    el = np.arctan2(c[:, 2] - hs, np.hypot(c[:, 0], c[:, 1])); az0 = np.arctan2(box[1], box[0])
    daz = (np.arctan2(c[:, 1], c[:, 0]) - az0 + np.pi) % (2 * np.pi) - np.pi
    rows = np.nonzero((inc_desc >= el.min() - 0.01) & (inc_desc <= el.max() + 0.01))[0]
    c_lo = int(np.floor((az0 + daz.min() + np.pi) / (2 * np.pi) * n_cols)) - 1
    c_hi = int(np.floor((az0 + daz.max() + np.pi) / (2 * np.pi) * n_cols)) + 1
    cols = np.arange(c_lo, c_hi + 1)
    if not len(rows) or len(cols) > n_cols // 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    R, C = np.meshgrid(rows, cols, indexing='ij'); R, C = R.ravel(), C.ravel()
    phi = inc_desc[R]; th = ((C % n_cols) + 0.5) / n_cols * 2 * np.pi - np.pi
    d = np.stack([np.cos(phi) * np.cos(th), np.cos(phi) * np.sin(th), np.sin(phi)], 1)
    ch, sh = np.cos(-box[6]), np.sin(-box[6])
    rot = lambda v: np.stack([v[..., 0] * ch - v[..., 1] * sh, v[..., 0] * sh + v[..., 1] * ch, v[..., 2]], -1)
    o = rot(np.array([0.0, 0.0, hs]) - box[:3]); dd = rot(d); e = np.asarray(box[3:6]) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-e - o) / dd; t2 = (e - o) / dd
    t1 = np.where(np.isfinite(t1), t1, -np.inf); t2 = np.where(np.isfinite(t2), t2, np.inf)
    tmin = np.minimum(t1, t2).max(1); tmax = np.maximum(t1, t2).min(1)
    hit = tmax >= np.maximum(tmin, 0)
    return (R[hit] * n_cols + C[hit] % n_cols), R[hit], np.maximum(tmin[hit], 0)


def nn_median(p):
    return np.median(cKDTree(p).query(p, k=2)[0][:, 1]) if len(p) >= 2 else np.nan


def box_arrangement(p):
    v = np.floor(p / BOX_VOX).astype(np.int64)
    nv = len(np.unique(v, axis=0))
    return len(p), nv, len(np.unique(v[:, 2])), len(p) / nv, nn_median(p)


def realised_counts(x, keep, raw_vid, raw_vring, raw_pix, raw_pring_of):
    """Per ring 1-pt / multi voxels, per row and per ring occupied virtual pixels, total voxels (realised keeps)."""
    vid = raw_vid[keep]; vid = vid[vid >= 0]
    cnt = np.bincount(vid, minlength=len(raw_vring))
    occ = cnt > 0; g = raw_vring
    ok = (g >= 0) & (g < 6)
    one = np.bincount(g[ok & (cnt == 1)], minlength=6)[:6]; mul = np.bincount(g[ok & (cnt >= 2)], minlength=6)[:6]
    pk = np.unique(raw_pix[keep & (raw_pix >= 0)])
    rows = np.bincount(pk // N_COLS, minlength=64)[:64]
    pr = raw_pring_of(pk); ring = np.bincount(pr[(pr >= 0) & (pr < 6)], minlength=6)[:6]
    return one, mul, rows, ring, int(occ.sum())


def source_mode(cfg, key, sources, control, wpath, n, n_train, out):
    st = np.load(STATS); inc_desc = np.sort(st['inclinations'])[::-1]
    v1 = LearnedPointSampler.load(wpath)
    assert v1.rule_kind == 'zbuffer', 'a lattice sampler is expected'
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIGS[key], class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
    assert control in sources

    def load(i, c):
        cnt, sel = parse_config(c); ds.dataset_cfg.MAX_SWEEPS = cnt; ds.dataset_cfg.SWEEP_SELECTION = sel
        return ds[i]

    def feats(x):
        rule, fx = v1.lattice_parts(x)
        return rule, np.concatenate([point_features(x, v1.shift_z), fx], 1)

    def keep_probs(rule, F):
        corr = v1.mlp(F)
        if v1.obs_mask is not None:
            corr = corr * v1.obs_mask[cell_index(F[:, 0], np.arctan2(F[:, 7], F[:, 6]))]
        sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return sig(rule + corr), sig(rule)          # v1, and the z-buffer rule (= v1 with its last layer zero)

    with _augmentation_off(ds):
        # v1's training range: the control source, 60 strided frames (v1 was trained on such frames)
        step = max(1, len(ds) // n_train); Fs = []
        for i in list(range(0, len(ds), step))[:n_train]:
            x = load(i, control)['points'][:, :3].astype(np.float64)
            if len(x):
                Fs.append(feats(x)[1])
        Fs = np.concatenate(Fs); lo, hi = np.percentile(Fs, 0.5, 0), np.percentile(Fs, 99.5, 0); del Fs
        print(f'{key}: v1 training range from {n_train} control frames', flush=True)
        step = max(1, len(ds) // n); frames = list(range(0, len(ds), step))[:n]
        dirs = [travel_direction(ds.infos[i]) for i in frames]
        mv = np.array([d for d in dirs if d is not None]); axis = mv.sum(0) / np.linalg.norm(mv.sum(0))
        acc = {(s, a): dict(one=np.zeros(6), mul=np.zeros(6), rows=np.zeros(64), ring=np.zeros(6), nv=[], kept=[])
               for s in sources for a in ARMS}
        shift = {s: np.zeros(len(FEATS)) for s in sources}; npts = {s: 0 for s in sources}
        blind = {s: np.zeros((6, 2)) for s in sources}
        sparse_rows, arr_rows = [], []
        for fi, i in enumerate(frames):
            fwd = dirs[fi] if dirs[fi] is not None else axis; moving = dirs[fi] is not None
            smp = {s: load(i, s) for s in sources}
            b0 = smp[control]['gt_boxes']; b0 = b0[b0[:, 7] == 1] if b0 is not None and len(b0) else np.zeros((0, 8))
            x0 = smp[control]['points'][:, :3]
            m0 = roiaware_pool3d_utils.points_in_boxes_cpu(x0, b0[:, :7]) if len(b0) else np.zeros((0, len(x0)))
            n0 = (m0 > 0).sum(1) if len(b0) else np.zeros(0)
            r0 = np.hypot(b0[:, 0], b0[:, 1]) if len(b0) else np.zeros(0)
            sparse = np.nonzero((n0 >= 10) & (n0 < 200) & (r0 >= 20) & (r0 < 75))[0]
            for si, s in enumerate(sources):
                x = smp[s]['points'][:, :3].astype(np.float64)
                rule, F = feats(x); p_v1, p_zb = keep_probs(rule, F)
                u = np.random.default_rng(1000 * fi + si).random(len(x))
                keeps = {'raw': np.ones(len(x), bool), 'v1': u < p_v1, 'zbuffer': u < p_zb}
                shift[s] += ((F < lo) | (F > hi)).sum(0); npts[s] += len(x)
                cen, vcnt, vid = voxel_table(x); vring = ring_of(np.hypot(cen[:, 0], cen[:, 1]))
                pix, _, rng = lattice_pixels(x, inc_desc, N_COLS, SENSOR_Z)
                okp = np.nonzero(pix >= 0)[0]
                up, pinv = np.unique(pix[okp], return_inverse=True)
                o = np.lexsort((rng[okp], pinv)); first = o[np.r_[0, np.nonzero(np.diff(pinv[o]))[0] + 1]]
                near_idx = okp[first]; near_rng = rng[near_idx]
                up_ring = ring_of(np.hypot(x[near_idx, 0], x[near_idx, 1]))

                def pring_of(pk, up=up, up_ring=up_ring):
                    return up_ring[np.searchsorted(up, pk)]
                for a in ARMS:
                    one, mul, rows, ring, nv = realised_counts(x, keeps[a], vid, vring, pix, pring_of)
                    A = acc[(s, a)]; A['one'] += one; A['mul'] += mul; A['rows'] += rows; A['ring'] += ring
                    A['nv'].append(nv); A['kept'].append(keeps[a].mean())
                bs = smp[s]['gt_boxes']; bs = bs[bs[:, 7] == 1] if bs is not None and len(bs) else np.zeros((0, 8))
                inb = roiaware_pool3d_utils.points_in_boxes_cpu(x, bs[:, :7]) > 0 if len(bs) else np.zeros((0, len(x)), bool)
                # measure 7: scene voxels inside this frame's sparse far cars (control-defined, matched here)
                sp_here = []
                for j in sparse:
                    k = np.where(np.all(np.isclose(bs[:, :7], b0[j, :7]), 1))[0] if len(bs) else []
                    sp_here.append(k[0] if len(k) else -1)
                ks = [k for k in sp_here if k >= 0]
                if ks and len(cen):
                    vin = roiaware_pool3d_utils.points_in_boxes_cpu(cen, bs[ks, :7]).max(0) > 0
                    ok = vring >= 0
                    blind[s][:, 0] += np.bincount(vring[ok & vin], minlength=6)[:6]
                blind[s][:, 1] += np.bincount(vring[vring >= 0], minlength=6)[:6]
                # arrangement at matched counts, every Car box with >= 10 points after the arm
                for j in range(len(bs)):
                    rb = np.hypot(*bs[j, :2]); sec = sector_of(bs[j:j + 1, :2], fwd)[0]
                    for ai, a in enumerate(ARMS):
                        pts = x[inb[j] & keeps[a]]
                        if len(pts) >= 10:
                            arr_rows.append((si, ai, rb, sec, moving) + box_arrangement(pts))
                # silhouettes of the sparse far cars
                for jj, j in enumerate(sparse):
                    k = sp_here[jj]
                    if k < 0:
                        continue
                    sp, srow, tin = silhouette(bs[k], inc_desc)
                    if not len(sp):
                        continue
                    pos = np.clip(np.searchsorted(up, sp), 0, len(up) - 1); has = up[pos] == sp
                    occl = has & (near_rng[pos] < tin - OCCL_M) & ~inb[k][near_idx[pos]]
                    vis = ~occl
                    if not vis.any():
                        continue
                    vp, vrow = sp[vis], srow[vis]
                    rec = [fi, r0[j], sector_of(b0[j:j + 1, :2], fwd)[0], moving, n0[j], si, len(vp), len(np.unique(vrow))]
                    for a in ARMS:
                        sel = inb[k] & keeps[a] & (pix >= 0)
                        pp = pix[sel]; inv_ = np.isin(pp, vp)
                        occ_p = np.unique(pp[inv_])
                        rows_cov = np.unique(occ_p // N_COLS)
                        rec += [len(rows_cov) / len(np.unique(vrow)), len(occ_p) / len(vp),
                                inv_.sum() / max(len(occ_p), 1), (inb[k] & keeps[a]).sum()]
                    sparse_rows.append(rec)
            if (fi + 1) % 20 == 0:
                print(f'  {key}: {fi + 1} / {len(frames)} frames', flush=True)
    m = len(frames)
    np.savez(out, key=key, sources=np.array(sources), control=control, n_frames=m, n_moving=len(mv), lo=lo, hi=hi,
             sparse=np.array(sparse_rows, float), arr=np.array(arr_rows, float),
             **{f'{s}__shift': shift[s] / npts[s] for s in sources}, **{f'{s}__blind': blind[s] for s in sources},
             **{f'{s}__{a}__{k}': (np.array(v) if isinstance(v, list) else v / m) for (s, a), d in acc.items()
                for k, v in d.items()})
    print(f'saved {out}')


def waymo_mode(cfg, n_train, n_val, out):
    logger = common_utils.create_logger(); fwd = np.array([1.0, 0.0])
    calib_cfg, split = calibration_target_config(cfg.DATA_CONFIG_TAR)
    tr, _, _ = build_dataloader(dataset_cfg=calib_cfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    one, mul, n = np.zeros(6), np.zeros(6), 0
    step = max(1, len(tr) // n_train); idx = 0
    while n < n_train and idx < len(tr):          # the trainer's strided(): skip empty frames, same stride
        x = tr[idx]['points'][:, :3].astype(np.float64); idx += step
        if not len(x):
            continue
        cen, cnt, _ = voxel_table(x); g = ring_of(np.hypot(cen[:, 0], cen[:, 1])); ok = g >= 0
        one += np.bincount(g[ok & (cnt == 1)], minlength=6)[:6]; mul += np.bincount(g[ok & (cnt >= 2)], minlength=6)[:6]; n += 1
    va, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    rows = []
    for i in list(range(0, len(va), max(1, len(va) // n_val)))[:n_val]:
        d = va[i]; b = d['gt_boxes']; b = b[b[:, 7] == 1]
        if not len(b):
            continue
        x = d['points'][:, :3]; m = roiaware_pool3d_utils.points_in_boxes_cpu(x, b[:, :7]) > 0
        for j in range(len(b)):
            if m[j].sum() >= 10:
                rows.append((np.hypot(*b[j, :2]), sector_of(b[j:j + 1, :2], fwd)[0]) + box_arrangement(x[m[j]]))
    st = np.load(STATS)
    np.savez(out, T_one=one / n, T_mul=mul / n, T_row=st['valid_row'] * N_COLS, T_ring=st['ring_valid'], n_train=n,
             val=np.array(rows, float))
    print(f'saved {out}: Waymo {split} 1-pt / multi voxels per ring {(one / n).round(0)} / {(mul / n).round(0)}')


def t1_t2(W, one, mul, rows, ring):
    l = np.log1p
    ok = (W['T_one'] + W['T_mul']) > 0
    t1 = ((l(one) - l(W['T_one'])) ** 2 + (l(mul) - l(W['T_mul'])) ** 2)[ok].mean()
    t2 = ((l(rows) - l(W['T_row'])) ** 2).mean() + ((l(ring) - l(W['T_ring'])) ** 2).mean()
    return t1, t2


def report_mode(wpath, paths):
    W = np.load(wpath); wv = W['val']
    rings = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-75']
    for p in paths:
        G = np.load(p); key = str(G['key']); sources = list(G['sources']); ctl = str(G['control'])
        print(f'\n# {key}: {int(G["n_frames"])} frames ({int(G["n_moving"])} moving); sources {sources}; control {ctl}')
        print('\n| source | arm | mean keep | voxels / frame (k, median / p90) | frames > 250k | T1 | T2 |')
        print('|---|---|---|---|---|---|---|')
        T1 = {}
        for s in sources:
            for a in ARMS:
                g = lambda k: G[f'{s}__{a}__{k}']
                t1, t2 = t1_t2(W, g('one'), g('mul'), g('rows'), g('ring')); T1[(s, a)] = t1; nv = g('nv')
                print(f'| {s} | {a} | {g("kept").mean():.2f} | {np.median(nv) / 1e3:.0f} / {np.percentile(nv, 90) / 1e3:.0f} | '
                      f'{(nv > CAP).mean():.2f} | {t1:.3f} | {t2:.3f} |')
        print('\nFeature shift: share of raw points outside v1\'s training range (p0.5-p99.5 on the control; 0.01 by construction there)')
        print('| source | ' + ' | '.join(FEATS) + ' |'); print('|---|' + '---|' * len(FEATS))
        for s in sources:
            print(f'| {s} | ' + ' | '.join(f'{v:.3f}' for v in G[f'{s}__shift']) + ' |')
        print('\nShare of each ring\'s occupied voxels inside sparse far cars (control-defined, 10-199 points, 20-75 m), raw:')
        print('| source | ' + ' | '.join(rings) + ' |'); print('|---|' + '---|' * 6)
        for s in sources:
            b = G[f'{s}__blind']; print(f'| {s} | ' + ' | '.join(f'{v:.4f}' for v in b[:, 0] / np.maximum(b[:, 1], 1)) + ' |')
        S = G['sparse']   # fi, r, sector, moving, n0, si, n_vis_pix, n_vis_rows, then per arm (RC, C, PPP, points)
        print('\nSparse far cars (moving frames), medians: RC = rows covered / visible silhouette rows, C = pixels covered, '
              'PPP = points per occupied silhouette pixel; retention = arm points / raw points')
        print('| source | ring | sector | boxes | silhouette rows / pixels | raw RC / C / PPP | v1 RC / C / PPP | v1 RC/raw | v1 retention 10-49 / 50-199 | z-buffer RC / C / PPP | z-buffer retention |')
        print('|---|---|---|---|---|---|---|---|---|---|---|')
        dec = {}
        for si, s in enumerate(sources):
            for lo_, hi_ in ((20, 40), (40, 75)):
                for sec in range(3):
                    q = S[(S[:, 5] == si) & (S[:, 3] > 0) & (S[:, 1] >= lo_) & (S[:, 1] < hi_) & (S[:, 2] == sec)]
                    if not len(q):
                        continue
                    a = lambda ai, k: q[:, 8 + 4 * ai + k]
                    med = lambda v: np.nanmedian(v)
                    ret = lambda ai, b0, b1: med((a(ai, 3) / np.maximum(a(0, 3), 1))[(q[:, 4] >= b0) & (q[:, 4] < b1)]) \
                        if ((q[:, 4] >= b0) & (q[:, 4] < b1)).any() else np.nan
                    rc_ratio = med(a(1, 0) / np.maximum(a(0, 0), 1e-9))
                    print(f'| {s} | {lo_}-{hi_} | {SECTORS[sec]} | {len(q)} | {med(q[:, 7]):.0f} / {med(q[:, 6]):.0f} | '
                          f'{med(a(0, 0)):.2f} / {med(a(0, 1)):.2f} / {med(a(0, 2)):.2f} | '
                          f'{med(a(1, 0)):.2f} / {med(a(1, 1)):.2f} / {med(a(1, 2)):.2f} | {rc_ratio:.2f} | '
                          f'{ret(1, 10, 50):.2f} / {ret(1, 50, 200):.2f} | '
                          f'{med(a(2, 0)):.2f} / {med(a(2, 1)):.2f} / {med(a(2, 2)):.2f} | {med(a(2, 3) / np.maximum(a(0, 3), 1)):.2f} |')
                    dec[(s, lo_, sec)] = (len(q), med(a(0, 0)), rc_ratio, med(a(1, 2)))
        Aw = G['arr']   # si, ai, r, sector, moving, p, v, z, ppv, nn
        print('\nArrangement at matched counts (moving frames), medians z-layers / voxels / points per voxel / NN (m); '
              'Waymo val by vehicle +x (DIAGNOSIS)')
        print('| ring | points | sector | Waymo | ' + ' | '.join(f'{s} {a}' for s in sources for a in ARMS) + ' |')
        print('|---|---|---|---|' + '---|' * (len(sources) * 3))
        for lo_, hi_ in ((20, 40), (40, 75)):
            for b0, b1 in ((10, 50), (50, 200)):
                for sec in range(3):
                    w = wv[(wv[:, 0] >= lo_) & (wv[:, 0] < hi_) & (wv[:, 1] == sec) & (wv[:, 2] >= b0) & (wv[:, 2] < b1)]
                    fmt = lambda r: (f'{np.median(r[:, 2]):.0f} / {np.median(r[:, 1]):.0f} / {np.median(r[:, 3]):.2f} / '
                                     f'{np.nanmedian(r[:, 4]):.3f} ({len(r)})') if len(r) else '-'
                    cells = []
                    for si, s in enumerate(sources):
                        for ai in range(3):
                            r = Aw[(Aw[:, 0] == si) & (Aw[:, 1] == ai) & (Aw[:, 4] > 0) & (Aw[:, 2] >= lo_) & (Aw[:, 2] < hi_)
                                   & (Aw[:, 3] == sec) & (Aw[:, 5] >= b0) & (Aw[:, 5] < b1)]
                            cells.append(fmt(r[:, 5:]) if len(r) else '-')
                    print(f'| {lo_}-{hi_} | {b0}-{b1 - 1} | {SECTORS[sec]} | {fmt(w[:, 2:]) if len(w) else "-"} | ' + ' | '.join(cells) + ' |')
        # the pre-declared decision, on the LAST source (the rule's choice), cells with >= 15 boxes
        rs = sources[-1]
        print(f'\nDecision on {rs} (cells with >= 15 boxes):')
        A_cells, B_cells = [], []
        for (s, lo_, sec), (nb, rc, ratio, ppp) in dec.items():
            if s != rs or nb < 15:
                continue
            tag = f'{lo_}-{lo_ + 20 if lo_ == 20 else 75} m {SECTORS[sec]}'
            if rc < 0.8:
                A_cells.append(f'{tag} (raw RC {rc:.2f})')
            elif ratio < 0.9 or ppp > 1.5:
                B_cells.append(f'{tag} (v1 RC/raw {ratio:.2f}, PPP {ppp:.2f})')
        sh = G[f'{rs}__shift']; i_pc, i_vp = FEATS.index('log_pix_cands'), FEATS.index('log_vox_pts')
        nv = G[f'{rs}__v1__nv']
        C = dict(t1=T1[(rs, 'v1')] > 2 * T1[(ctl, 'v1')], cap=(nv > CAP).mean() > 0.05,
                 shift=max(sh[i_pc], sh[i_vp]) > 0.05)
        print(f'  (A) re-render in: {A_cells or "none"}')
        print(f'  (B) object-aware objective + structured selection in: {B_cells or "none"}')
        print(f'  (C) retrain (only if A and B hold nowhere): T1 {T1[(rs, "v1")]:.3f} vs control {T1[(ctl, "v1")]:.3f} -> {C["t1"]}; '
              f'cap -> {C["cap"]}; feature shift {max(sh[i_pc], sh[i_vp]):.3f} -> {C["shift"]}; '
              f'holds: {(not A_cells and not B_cells) and any(C.values())}')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'report':
        report_mode(sys.argv[2], sys.argv[3:]); sys.exit(0)
    cfg = EasyDict(); cfg_from_yaml_file(sys.argv[2], cfg)
    if mode == 'waymo':
        waymo_mode(cfg, int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        source_mode(cfg, sys.argv[3], sys.argv[4].split(','), sys.argv[5], sys.argv[6], int(sys.argv[7]), int(sys.argv[8]),
                    sys.argv[9])
