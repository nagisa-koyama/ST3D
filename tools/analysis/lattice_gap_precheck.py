"""Phase-free re-measure of the lattice pre-check (experiments_md 20261008_01 §11). CPU, analysis.

Per sparse far source car (control-defined, as in lattice_sampler_precheck.py): sort the elevations, from the virtual
Waymo TOP sensor (2.184 m), of the arm's in-box points that fall in VISIBLE silhouette pixels; take the gaps between
adjacent elevations plus the two ends to the topmost / bottommost visible silhouette rows; compare each gap with the
local Waymo inclination spacing at its midpoint. Excess-gap share = sum max(gap - s_W, 0) / visible extent (the
phase-averaged share of Waymo rows no candidate can fill); selectable = largest gap / s_W <= 1.0 (1.5). Variants:
occlusion margin 0.25 / 0.5 / 1.0 m, and the box bottom trimmed by 0.3 m (under-body proxy). Every car carries its
motion class relative to the ego (nuScenes box velocity; static / approached-receding / co-moving / unknown) and a
line-jitter diagnostic (median distance of its in-box points to the nearest box face). Source labels + the published
Waymo spec only.

    python analysis/lattice_gap_precheck.py source <cfg> <key> <sources, e.g. 15@27,15@27F,30@50F> <control>
           <v1 weights.npz> <frames> <out.npz>
    python analysis/lattice_gap_precheck.py report <source.npz> [<source.npz> ...]
"""
import sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off
from pcdet.datasets.processor.point_sampler import LearnedPointSampler, lattice_pixels, point_features, cell_index
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
from lattice_depth_ceiling import parse_config, travel_direction, sector_of, STATS, N_COLS, SENSOR_Z, SECTORS
from lattice_sampler_precheck import silhouette, ARMS

VARIANTS = (('m0.25', 0.25, 0.0), ('m0.5', 0.5, 0.0), ('m1.0', 1.0, 0.0), ('trim', 0.5, 0.3))  # name, margin, bottom trim
MAIN = 1
METRICS = ('excess', 'maxratio', 'maxgap_m', 'over015')
CLASSES = ('static', 'approached', 'co-moving', 'unknown')
V_STATIC, R_DOT = 0.5, 1.0
PREV_A = {'NUSCENES_N008': 5, 'NUSCENES_N015': 5}   # §10's (A) cells: all but 20-40 m ahead, both platforms


def ego_velocity(info):
    """Ego velocity in the anchor's lidar axes: minus the xy translation over the time lag of the stored past sweep
    nearest 5 m of displacement; zero if that sweep is < 3 m away."""
    best, bd = None, np.inf
    for sw in info['sweeps']:
        lag = float(sw.get('time_lag', 0.0))
        if lag <= 0 or sw.get('transform_matrix') is None:
            continue
        t = np.asarray(sw['transform_matrix'])[:2, 3]
        if abs(np.hypot(*t) - 5.0) < bd:
            best, bd = (t, lag), abs(np.hypot(*t) - 5.0)
    if best is None or np.hypot(*best[0]) < 3.0:
        return np.zeros(2)
    return -best[0] / best[1]


def motion_class(box, info, v_ego):
    ib = np.asarray(info['gt_boxes'])
    if ib.shape[1] < 9:
        return 3, np.nan
    cols = [0, 1, 3, 4, 5, 6]
    k = np.nonzero(np.all(np.isclose(ib[:, cols], box[cols], atol=1e-3), 1))[0]
    if not len(k):
        return 3, np.nan
    v = ib[k[0], 7:9]
    if not np.all(np.isfinite(v)):
        return 3, np.nan
    u = box[:2] / max(np.hypot(*box[:2]), 1e-6); r_dot = float(u @ (v - v_ego))
    if np.hypot(*v) < V_STATIC:
        return 0, r_dot
    return (1 if abs(r_dot) >= R_DOT else 2), r_dot


def gap_metrics(el, top, bot, rng3d, inc_asc, spacing):
    """excess share, largest gap / s_W, largest gap in metres, share of the extent in gaps > 0.15 m."""
    ext = top - bot
    vals = np.unique(np.r_[bot, np.clip(el, bot, top), top])
    g = np.diff(vals)
    if not len(g):
        return 1.0, np.inf, ext * rng3d, 1.0
    mid = (vals[:-1] + vals[1:]) / 2
    s = spacing[np.clip(np.searchsorted(inc_asc, mid) - 1, 0, len(spacing) - 1)]
    return (np.maximum(g - s, 0).sum() / ext, (g / s).max(), g.max() * rng3d, g[g * rng3d > 0.15].sum() / ext)


def face_distance(p, box):
    c, s = np.cos(-box[6]), np.sin(-box[6]); d = p - box[:3]
    q = np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c, d[:, 2]], 1)
    return np.median(np.min(np.asarray(box[3:6]) / 2 - np.abs(q), 1)) if len(p) else np.nan


def source_mode(cfg, key, sources, control, wpath, n, out):
    st = np.load(STATS); inc_asc = np.sort(st['inclinations']); inc_desc = inc_asc[::-1]; spacing = np.diff(inc_asc)
    v1 = LearnedPointSampler.load(wpath)
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIGS[key], class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))

    def load(i, c):
        cnt, sel = parse_config(c); ds.dataset_cfg.MAX_SWEEPS = cnt; ds.dataset_cfg.SWEEP_SELECTION = sel
        return ds[i]

    def keep_probs(x):
        rule, fx = v1.lattice_parts(x); F = np.concatenate([point_features(x, v1.shift_z), fx], 1)
        corr = v1.mlp(F)
        if v1.obs_mask is not None:
            corr = corr * v1.obs_mask[cell_index(F[:, 0], np.arctan2(F[:, 7], F[:, 6]))]
        sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return sig(rule + corr), sig(rule)

    rows = []
    with _augmentation_off(ds):
        step = max(1, len(ds) // n); frames = list(range(0, len(ds), step))[:n]
        dirs = [travel_direction(ds.infos[i]) for i in frames]
        mv = np.array([d for d in dirs if d is not None]); axis = mv.sum(0) / np.linalg.norm(mv.sum(0))
        for fi, i in enumerate(frames):
            fwd = dirs[fi] if dirs[fi] is not None else axis; v_ego = ego_velocity(ds.infos[i])
            smp = {s: load(i, s) for s in sources}
            b0 = smp[control]['gt_boxes']; b0 = b0[b0[:, 7] == 1] if b0 is not None and len(b0) else np.zeros((0, 8))
            x0 = smp[control]['points'][:, :3]
            n0 = (roiaware_pool3d_utils.points_in_boxes_cpu(x0, b0[:, :7]) > 0).sum(1) if len(b0) else np.zeros(0)
            r0 = np.hypot(b0[:, 0], b0[:, 1]) if len(b0) else np.zeros(0)
            sparse = np.nonzero((n0 >= 10) & (n0 < 200) & (r0 >= 20) & (r0 < 75))[0]
            if not len(sparse):
                continue
            cls = [motion_class(b0[j], ds.infos[i], v_ego) for j in sparse]
            for si, s in enumerate(sources):
                x = smp[s]['points'][:, :3].astype(np.float64)
                p_v1, p_zb = keep_probs(x)
                u = np.random.default_rng(1000 * fi + si).random(len(x))
                keeps = (np.ones(len(x), bool), u < p_v1, u < p_zb)
                pix, _, rng = lattice_pixels(x, inc_desc, N_COLS, SENSOR_Z)
                el = np.arctan2(x[:, 2] - SENSOR_Z, np.hypot(x[:, 0], x[:, 1]))
                okp = np.nonzero(pix >= 0)[0]
                up, pinv = np.unique(pix[okp], return_inverse=True)
                o = np.lexsort((rng[okp], pinv)); near_idx = okp[o[np.r_[0, np.nonzero(np.diff(pinv[o]))[0] + 1]]]
                near_rng = rng[near_idx]
                bs = smp[s]['gt_boxes']; bs = bs[bs[:, 7] == 1] if bs is not None and len(bs) else np.zeros((0, 8))
                inb_all = roiaware_pool3d_utils.points_in_boxes_cpu(x, bs[:, :7]) > 0 if len(bs) else None
                for jj, j in enumerate(sparse):
                    k = np.where(np.all(np.isclose(bs[:, :7], b0[j, :7]), 1))[0] if len(bs) else []
                    if not len(k):
                        continue
                    box = bs[k[0]]; inb = inb_all[k[0]]
                    rng3d = np.hypot(np.hypot(box[0], box[1]), box[2] - SENSOR_Z)
                    rec = [fi, r0[j], sector_of(b0[j:j + 1, :2], fwd)[0], dirs[fi] is not None, n0[j], si,
                           cls[jj][0], cls[jj][1], face_distance(x[inb], box)]
                    ok_all = True; ppp = []; w4 = []
                    for vn, margin, trim in VARIANTS:
                        bb = box.copy()
                        if trim:
                            bb[2] += trim / 2; bb[5] -= trim
                        sp, srow, tin = silhouette(bb, inc_desc)
                        if not len(sp):
                            ok_all = False; break
                        pos = np.clip(np.searchsorted(up, sp), 0, len(up) - 1); has = up[pos] == sp
                        vis = ~(has & (near_rng[pos] < tin - margin) & ~inb[near_idx[pos]])
                        vp, vrow = sp[vis], srow[vis]
                        if len(np.unique(vrow)) < 2:
                            ok_all = False; break
                        top, bot = inc_desc[vrow.min()], inc_desc[vrow.max()]
                        for ai, kp in enumerate(keeps):
                            sel = inb & kp & (pix >= 0)
                            sel[sel] = np.isin(pix[sel], vp)
                            rec += list(gap_metrics(el[sel], top, bot, rng3d, inc_asc, spacing))
                            if vn == 'm0.5':
                                occ = len(np.unique(pix[sel])); ppp.append(sel.sum() / max(occ, 1))
                                # w4 (amendment): per 4-column window, its own visible top / bottom rows; median over windows
                                wcol = (vp % N_COLS) // 4; pw = (pix[sel] % N_COLS) // 4; ex_w = []
                                for w in np.unique(wcol):
                                    wr = vrow[wcol == w]
                                    if len(np.unique(wr)) < 2:
                                        continue
                                    ex_w.append(gap_metrics(el[sel][pw == w], inc_desc[wr.min()], inc_desc[wr.max()],
                                                            rng3d, inc_asc, spacing)[0])
                                w4.append(np.median(ex_w) if ex_w else np.nan)
                    if ok_all:
                        rows.append(rec + ppp + w4)
            if (fi + 1) % 20 == 0:
                print(f'  {key}: {fi + 1} / {len(frames)} frames', flush=True)
    np.savez(out, key=key, sources=np.array(sources), control=control, n_frames=len(frames), rows=np.array(rows, float))
    print(f'saved {out}: {len(rows)} car-source records')


def report_mode(paths):
    NB = 9; NV = len(VARIANTS); NA = len(ARMS); NM = len(METRICS)
    tot = dict(prev=0, in_prev=0, extra=0, counts=np.zeros(NV, int))
    col = lambda v, a, m: NB + (v * NA + a) * NM + m
    ppp_col = lambda a: NB + NV * NA * NM + a
    w4_col = lambda a: NB + NV * NA * NM + NA + a
    for p in paths:
        G = np.load(p); key = str(G['key']); sources = list(G['sources']); R = G['rows']
        print(f'\n# {key}: sources {sources}, control {G["control"]}, {len(R)} car-source records '
              f'(fields: frame, range, sector, ego moving, control points, source, class, r_dot, face distance, ...)')
        mov = R[:, 3] > 0
        cell = lambda q, si, lo, hi, sec: q[(q[:, 5] == si) & (q[:, 1] >= lo) & (q[:, 1] < hi) & (q[:, 2] == sec)]
        print('\nMain variant (occlusion 0.5 m, full box), moving-ego frames, medians; selectable = share of cars with '
              'largest gap <= 1.0x / 1.5x s_W (raw)')
        print('| source | ring | sector | boxes | class shares static / appr / co-mov / unk | excess raw / v1 / zbuf | '
              'selectable raw 1.0x / 1.5x | largest gap raw (m) | extent in gaps > 0.15 m (raw) | v1 PPP | w4 excess raw / v1 / zbuf |')
        print('|---|---|---|---|---|---|---|---|---|---|---|')
        dec = {}
        for si, s in enumerate(sources):
            for lo, hi in ((20, 40), (40, 75)):
                for sec in range(3):
                    q = cell(R[mov], si, lo, hi, sec)
                    if not len(q):
                        continue
                    sh = [np.mean(q[:, 6] == c) for c in range(4)]
                    ex = [np.median(q[:, col(MAIN, a, 0)]) for a in range(NA)]
                    mr = q[:, col(MAIN, 0, 1)]
                    print(f'| {s} | {lo}-{hi} | {SECTORS[sec]} | {len(q)} | ' + ' / '.join(f'{v:.2f}' for v in sh) + ' | '
                          + ' / '.join(f'{v:.2f}' for v in ex) + f' | {np.mean(mr <= 1.0):.2f} / {np.mean(mr <= 1.5):.2f} | '
                          f'{np.median(q[:, col(MAIN, 0, 2)]):.2f} | {np.median(q[:, col(MAIN, 0, 3)]):.2f} | '
                          f'{np.median(q[:, ppp_col(1)]):.2f} | ' + ' / '.join(f'{np.nanmedian(q[:, w4_col(a)]):.2f}' for a in range(NA)) + ' |')
                    dec[(si, lo, sec)] = (len(q), ex[0], ex[1], np.median(q[:, ppp_col(1)]), np.mean(mr <= 1.5),
                                          [np.median(q[:, col(v, 0, 0)]) for v in range(NV)], np.nanmedian(q[:, w4_col(0)]))
        rs = len(sources) - 1
        print(f'\nDecision on {sources[rs]} (cells with >= 15 boxes), median raw excess-gap share per variant '
              f'{[v[0] for v in VARIANTS]}:')
        A, B, flags, counts = [], [], [], np.zeros(NV, int)
        for (si, lo, sec), (nb, exr, exv, ppp, sel15, exvar, w4r) in dec.items():
            if si != rs or nb < 15:
                continue
            tag = f'{lo}-{40 if lo == 20 else 75} m {SECTORS[sec]}'
            counts += np.array([e > 0.2 for e in exvar])
            line = f'  {tag}: boxes {nb}, excess ' + ' / '.join(f'{e:.2f}' for e in exvar) + f', w4 {w4r:.2f}'
            if exr > 0.2:
                A.append(tag); line += '  -> (A)'
                if sel15 >= 0.5:
                    flags.append(tag); line += ' [near the spacing]'
            else:
                if w4r > 0.2:
                    flags.append(tag + ' POOLING-SENSITIVE'); line += '  [not (A) pooled, (A) under w4: POOLING-SENSITIVE]'
                if (1 - exv) / max(1 - exr, 1e-9) < 0.9 or ppp > 1.5:
                    B.append(tag); line += f'  -> (B) (v1 coverage ratio {(1 - exv) / max(1 - exr, 1e-9):.2f}, PPP {ppp:.2f})'
                else:
                    line += '  -> none'
            print(line)
        prev = PREV_A[key]
        in_prev = [t for t in A if t != '20-40 m ahead']; extra = [t for t in A if t == '20-40 m ahead']
        tot['prev'] += prev; tot['in_prev'] += len(in_prev); tot['extra'] += len(extra); tot['counts'] += counts
        print(f'  (A) cells: {len(A)} ({len(in_prev)} of the {prev} §10 (A) cells on this platform); (B): {B or "none"}; '
              f'near the spacing: {flags or "none"}; (A) count per variant {dict(zip([v[0] for v in VARIANTS], counts.tolist()))}')
        print('\nBy object class (main variant, raw), cars ahead, moving-ego frames: median excess share (boxes)')
        print('| ring | class | ' + ' | '.join(sources) + ' |'); print('|---|---|' + '---|' * len(sources))
        for lo, hi in ((20, 40), (40, 75)):
            for c, cn in enumerate(CLASSES):
                cells = []
                for si in range(len(sources)):
                    q = cell(R[mov], si, lo, hi, 0); q = q[q[:, 6] == c]
                    cells.append(f'{np.median(q[:, col(MAIN, 0, 0)]):.2f} ({len(q)})' if len(q) else '-')
                print(f'| {lo}-{hi} | {cn} | ' + ' | '.join(cells) + ' |')
        if len(sources) >= 2:
            print('\nCo-moving test (ahead): drop in median raw excess share, control -> second source')
            for lo, hi in ((20, 40), (40, 75)):
                q0 = cell(R[mov], 0, lo, hi, 0); q1 = cell(R[mov], 1, lo, hi, 0)
                d = {}
                for name, cs in (('static+approached', (0, 1)), ('co-moving', (2,))):
                    a0 = q0[np.isin(q0[:, 6], cs)]; a1 = q1[np.isin(q1[:, 6], cs)]
                    d[name] = (np.median(a0[:, col(MAIN, 0, 0)]) - np.median(a1[:, col(MAIN, 0, 0)])) if len(a0) and len(a1) else np.nan
                print(f'  {lo}-{hi} m: static+approached {d["static+approached"]:.3f}, co-moving {d["co-moving"]:.3f}, '
                      f'ratio {d["static+approached"] / d["co-moving"] if d["co-moving"] else np.inf:.2f}')
        print('\nLine jitter: median distance of in-box points to the nearest box face (m), raw, moving-ego frames')
        print('| source | ring | static | moving (approached + co-moving) |'); print('|---|---|---|---|')
        for si, s in enumerate(sources[:2]):
            for lo, hi in ((20, 40), (40, 75)):
                q = R[mov & (R[:, 5] == si) & (R[:, 1] >= lo) & (R[:, 1] < hi)]
                st_, mo = q[q[:, 6] == 0, 8], q[np.isin(q[:, 6], (1, 2)), 8]
                print(f'| {s} | {lo}-{hi} | {np.nanmedian(st_):.3f} ({len(st_)}) | {np.nanmedian(mo):.3f} ({len(mo)}) |')
    if tot['prev']:
        verdict = ('STANDS' if tot['in_prev'] >= 0.9 * tot['prev'] and tot['extra'] <= 1 else
                   'FLIPS' if tot['in_prev'] <= 0.5 * tot['prev'] else 'PHASE-SENSITIVE')
        c = tot['counts']
        print(f'\nCOMBINED over {len(paths)} platform(s): (A) in {tot["in_prev"]} of the {tot["prev"]} §10 (A) cells and in '
              f'{tot["extra"]} other cell(s) -> §10 verdict {verdict}; (A) count per variant '
              f'{dict(zip([v[0] for v in VARIANTS], c.tolist()))} -> {"robust" if c.max() - c.min() <= 2 else "NOT robust"} to the margins')


if __name__ == '__main__':
    if sys.argv[1] == 'report':
        report_mode(sys.argv[2:]); sys.exit(0)
    cfg = EasyDict(); cfg_from_yaml_file(sys.argv[2], cfg)
    source_mode(cfg, sys.argv[3], sys.argv[4].split(','), sys.argv[5], sys.argv[6], int(sys.argv[7]), sys.argv[8])
