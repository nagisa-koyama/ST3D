"""CPU gate of the spec-conditioned re-rendering operator (experiments_md 20261009_01 §4, as amended in §7).

Per anchor of a nuScenes source config carrying DATA_CONFIGS.*.RERENDER (the operator is run on the pre-processor
accumulated cloud exactly as the loader hook runs it; every 10th frame the hook's own output is checked against it):
- G1 lattice-row coverage RC of sparse far cars - Car boxes at 20-75 m with 10-49 ACCUMULATED-SOURCE points (population
  before rendering), moving-ego frames, sectors ahead / sides / behind; occlusion judged on the rendered cloud. Also CC
  (silhouette-column coverage, for variant H's adoption rule) and, secondary, the population with 10-49 RENDERED points;
- G2 points per occupied silhouette pixel (1.00 by construction - a sanity check);
- G3 rendered points in no source box with no source point within 0.3 m, per ring, split z-buffered / filled (reported);
- G3' ring hold-out (deciding): drop the odd ring indices of every sweep, render + fill, compare each filled cell that
  holds a held-out return with that return's range;
- G4 render seconds per anchor on one core and frames over the 150k voxel cap;
- variants: the declared rule (K = 4 rows), A (angular span 1.5 deg), H (+ horizontal single-gap), none (z-buffer only).
Waymo mode (G5, ANALYSIS - reads Waymo val labels): z-layers and line spacing of Waymo val cars at matched counts.

    python analysis/rerender_gate.py source <cfg> <key> <frames> <out.npz>
    python analysis/rerender_gate.py waymo <cfg> <val frames> <out.npz>
    python analysis/rerender_gate.py report <waymo.npz> <source.npz> [<source.npz> ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'   # G4 is "one CPU core"
import contextlib
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off
from pcdet.datasets import rerender_utils as RR
from pcdet.datasets.processor.point_sampler import lattice_pixels, DET_VOXEL, DET_PCR
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
from lattice_depth_ceiling import travel_direction, sector_of, SECTORS
from lattice_sampler_precheck import silhouette

# 20261009_01 §7 amendment of 18:36: the angular span (A) DECIDES G1 / G3' / H; the row-count rule K = 4 is reported.
VARIANTS = (('A', dict(fill='rule', dr_max=0.3, span_max_deg=1.5)),
            ('K4', dict(fill='rule', k_rows=4, dr_max=0.3)),
            ('A+H', dict(fill='rule', dr_max=0.3, span_max_deg=1.5, h_fill=True)),
            ('none', dict(fill='none')))
DECIDE = 'A'
HOLDOUT = ('A', 'K4')
RINGS = (20, 30, 40, 50, 75)
OCCL_M = 0.5
CAP = 150000


@contextlib.contextmanager
def ring_in_intensity_column():
    """While active, nuScenes .bin reads return the RING index (5th column) in place of intensity (4th), so the
    loader's own accumulation (sweep selection, compensation, ego removal) carries ring ids."""
    orig = np.fromfile

    def patched(file, dtype=float, count=-1, sep='', offset=0, **kw):
        a = orig(file, dtype=dtype, count=count, sep=sep, offset=offset, **kw)
        if str(file).endswith('.bin') and a.size % 5 == 0:
            a = a.reshape(-1, 5).copy(); a[:, 3] = a[:, 4]; a = a.ravel()
        return a
    np.fromfile = patched
    try:
        yield
    finally:
        np.fromfile = orig


def in_pcr(x):
    return np.all((x[:, :3] >= DET_PCR[:3]) & (x[:, :3] < DET_PCR[3:]), 1)


def n_voxels(x):
    x = x[in_pcr(x)]
    ijk = np.floor((x[:, :3] - DET_PCR[:3]) / DET_VOXEL).astype(np.int64)
    return len(np.unique((ijk[:, 0] * 2000 + ijk[:, 1]) * 100 + ijk[:, 2]))


def ring_bin(r):
    g = np.digitize(r, RINGS) - 1
    return np.where((g >= 0) & (g < len(RINGS) - 1), g, -1)


def line_spacing(p, hs, n_cols, rng3d):
    """Median over 4-column windows of the median gap between adjacent scan LINES, in metres. Elevations closer than
    0.07 deg (half Waymo TOP's smallest row gap) are one line: Waymo's processed cloud jitters within a row (per-pixel
    pose), and those within-line gaps must not count as line spacing."""
    d = p[:, :3] - np.array([0, 0, hs]); el = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
    col = np.floor((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi) * n_cols).astype(np.int64) // 4
    gaps = []
    for w in np.unique(col):
        e = np.sort(el[col == w])
        if len(e) >= 2:
            d = np.diff(e); lines = e[np.r_[True, d > np.radians(0.07)]]
            if len(lines) >= 2:
                gaps.append(np.median(np.diff(lines)))
    return np.median(gaps) * rng3d if gaps else np.nan


def car_shape(p, hs, n_cols, rng3d):
    v = np.floor(p[:, :3] / np.array([0.1, 0.1, 0.15])).astype(np.int64)
    return len(p), len(np.unique(v[:, 2])), line_spacing(p, hs, n_cols, rng3d)


def source_mode(cfg, key, n, out):
    dc = cfg.DATA_CONFIGS[key]; rcfg = dc.RERENDER
    spec = RR.lattice_spec(rcfg.TARGET_LATTICE, rcfg.get('MOUNT_HEIGHT', None))
    th, nc, hs = spec['thetas'], spec['n_cols'], spec['height']; inc_desc = th[::-1]
    shift = np.array(dc.get('SHIFT_COOR', [0, 0, 0]), np.float32)
    ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
    rows, g3, g3h, g4, hook = [], [], [], [], []
    with _augmentation_off(ds):
        step = max(1, len(ds) // n); frames = list(range(0, len(ds), step))[:n]
        dirs = [travel_direction(ds.infos[i]) for i in frames]
        mv = np.array([d for d in dirs if d is not None]); axis = mv.sum(0) / np.linalg.norm(mv.sum(0))
        for fi, i in enumerate(frames):
            fwd = dirs[fi] if dirs[fi] is not None else axis; moving = dirs[fi] is not None
            ds.dataset_cfg.RERENDER = None
            smp = ds[i]                                            # raw source sample: boxes + accumulated count
            ds.dataset_cfg.RERENDER = rcfg
            with ring_in_intensity_column():
                P = ds.get_lidar_with_sweeps(i, max_sweeps=ds.dataset_cfg.MAX_SWEEPS)
            P[:, :3] += shift                                       # what the hook sees (no INTENSITY_SCALE / LEVEL_COOR here)
            ring_id = P[:, 3].astype(np.int64)
            # G4: the declared rule on one core
            t = time.perf_counter(); Q, fq = RR.rerender(P, spec, **dict(VARIANTS)[DECIDE]); sec = time.perf_counter() - t
            g4.append((sec, len(P), n_voxels(Q)))
            if fi % 10 == 0:                                       # the hook's own output vs the in-script render
                hk = ds[i]['points'][:, :3]
                qq = Q[(Q[:, 0] >= DET_PCR[0]) & (Q[:, 0] <= DET_PCR[3]) & (Q[:, 1] >= DET_PCR[1]) & (Q[:, 1] <= DET_PCR[4]), :3]
                hook.append((len(hk), len(qq), np.allclose(np.sort(hk[:, 0]), np.sort(qq[:, 0]), atol=1e-4)
                             if len(hk) == len(qq) else False))
            renders = {name: RR.rerender(P, spec, **kw) for name, kw in VARIANTS}
            renders = {k: (v[0][in_pcr(v[0])], v[1][in_pcr(v[0])]) for k, v in renders.items()}
            # G3 (reported): phantoms of the deciding render
            Qr, fr = renders[DECIDE]
            info = ds.infos[i]; allb = np.asarray(info['gt_boxes'])[:, :7].copy(); allb[:, 2] += shift[2]
            inb_any = (roiaware_pool3d_utils.points_in_boxes_cpu(Qr[:, :3], allb).max(0) > 0) if len(allb) else np.zeros(len(Qr), bool)
            near = cKDTree(P[:, :3]).query(Qr[:, :3], k=1, distance_upper_bound=0.3)[0] <= 0.3
            rb = ring_bin(np.hypot(Qr[:, 0], Qr[:, 1]))
            for g in range(len(RINGS) - 1):
                for filled in (False, True):
                    s = (rb == g) & (fr == filled)
                    g3.append((fi, g, filled, s.sum(), (s & ~inb_any & ~near).sum()))
            # G3' ring hold-out: drop the odd ring indices of every sweep, render + fill, compare with held-out returns
            keep = (ring_id % 2) == 0
            ho = P[~keep]; hr, hc, hrg, hv = RR.to_lattice(ho[:, :3], hs, th, nc)
            himg, _ = RR.zbuffer(hr, hc, hrg, hv, len(th), nc)
            for vi, vname in enumerate(HOLDOUT):
                Hq, hf = RR.rerender(P[keep], spec, **dict(VARIANTS)[vname])
                Hf = Hq[hf]
                if not len(Hf):
                    continue
                r_, c_, rg_, v_ = RR.to_lattice(Hf[:, :3], hs, th, nc)
                ref = himg[r_, c_]; okc = np.isfinite(ref)
                inb_h = (roiaware_pool3d_utils.points_in_boxes_cpu(Hf[:, :3], allb).max(0) > 0) if len(allb) else np.zeros(len(Hf), bool)
                rp = np.hypot(Hf[:, 0], Hf[:, 1])
                for j in np.nonzero(okc)[0]:
                    g3h.append((fi, rp[j], inb_h[j], abs(rg_[j] - ref[j]), vi))
            # G1 / G2 / CC on sparse far cars (population on the RAW accumulated sample)
            b = smp['gt_boxes']; b = b[b[:, 7] == 1] if b is not None and len(b) else np.zeros((0, 8))
            if not len(b):
                continue
            xs = smp['points'][:, :3]
            ns = (roiaware_pool3d_utils.points_in_boxes_cpu(xs, b[:, :7]) > 0).sum(1)
            rr = np.hypot(b[:, 0], b[:, 1])
            cand = np.nonzero((rr >= 20) & (rr < 75))[0]
            if not len(cand):
                continue
            per_var = {}
            for name in renders:
                Qv = renders[name][0][:, :3].astype(np.float64)
                pix, _, rng = lattice_pixels(Qv, th, nc, hs)
                okp = np.nonzero(pix >= 0)[0]
                up, pinv = np.unique(pix[okp], return_inverse=True)
                o = np.lexsort((rng[okp], pinv)); nidx = okp[o[np.r_[0, np.nonzero(np.diff(pinv[o]))[0] + 1]]] if len(okp) else okp
                inb = roiaware_pool3d_utils.points_in_boxes_cpu(Qv, b[cand, :7]) > 0 if len(Qv) else np.zeros((len(cand), 0), bool)
                per_var[name] = (Qv, pix, up, nidx, rng[nidx] if len(nidx) else np.zeros(0), inb)
            for jj, j in enumerate(cand):
                sp, srow, tin = silhouette(b[j], inc_desc, hs=hs, n_cols=nc)
                if not len(sp):
                    continue
                rec = [fi, rr[j], sector_of(b[j:j + 1, :2], fwd)[0], moving, ns[j]]
                good = True; shape = [0, 0, np.nan]
                for name in renders:
                    Qv, pix, up, nidx, nrng, inb = per_var[name]
                    if len(up):
                        pos = np.clip(np.searchsorted(up, sp), 0, len(up) - 1); has = up[pos] == sp
                        occl = has & (nrng[pos] < tin - OCCL_M) & ~inb[jj][nidx[pos]]
                    else:
                        occl = np.zeros(len(sp), bool)
                    vp, vrow = sp[~occl], srow[~occl]
                    if len(np.unique(vrow)) < 1:
                        good = False; break
                    sel = inb[jj] & (pix >= 0); pp = pix[sel]; isv = np.isin(pp, vp); occ = np.unique(pp[isv])
                    vcols = np.unique(vp % nc)
                    rec += [len(np.unique(occ // nc)) / len(np.unique(vrow)), len(np.unique(occ % nc)) / len(vcols),
                            isv.sum() / max(len(occ), 1), inb[jj].sum()]
                    if name == DECIDE and inb[jj].sum():
                        shape = list(car_shape(Qv[inb[jj]], hs, nc, np.hypot(rr[j], b[j, 2] - hs)))
                if good:
                    rows.append(rec + shape)
            if (fi + 1) % 20 == 0:
                print(f'  {key}: {fi + 1} / {len(frames)} frames', flush=True)
    np.savez(out, key=key, variants=np.array([v[0] for v in VARIANTS]), n_frames=len(frames), rows=np.array(rows, float),
             g3=np.array(g3, float), g3h=np.array(g3h, float), g4=np.array(g4, float), hook=np.array(hook, float))
    print(f'saved {out}: {len(rows)} sparse-car records, {len(g3h)} hold-out cells')


def waymo_mode(cfg, n_val, out):
    spec = RR.lattice_spec('waymo_top'); hs, nc = spec['height'], spec['n_cols']
    va, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=common_utils.create_logger(), training=False, model_ontology=cfg.get('ONTOLOGY'))
    rows = []
    for i in list(range(0, len(va), max(1, len(va) // n_val)))[:n_val]:
        d = va[i]; b = d['gt_boxes']; b = b[b[:, 7] == 1]
        if not len(b):
            continue
        x = d['points'][:, :3]; m = roiaware_pool3d_utils.points_in_boxes_cpu(x, b[:, :7]) > 0
        for j in range(len(b)):
            rr = np.hypot(*b[j, :2])
            if m[j].sum() >= 10 and 20 <= rr < 75:
                rows.append((rr,) + car_shape(x[m[j]], hs, nc, np.hypot(rr, b[j, 2] - hs)))
    np.savez(out, rows=np.array(rows, float)); print(f'saved {out}: {len(rows)} Waymo val cars')


def report_mode(wpath, paths):
    W = np.load(wpath)['rows']   # range, points, z-layers, line spacing
    names = [v[0] for v in VARIANTS]; NB = 5
    c = lambda v, k: NB + 4 * names.index(v) + k                  # k: RC, CC, PPP, points
    shape0 = NB + 4 * len(names)                                  # rule: points, z-layers, line spacing
    tot_g1 = {v: 0 for v in names}; tot_cells = 0; all_pass = dict(G2=True, G3h=True, G4=True)
    for p in paths:
        G = np.load(p); key = str(G['key']); R = G['rows']; mov = R[:, 3] > 0
        print(f'\n# {key}: {int(G["n_frames"])} frames; hook check (hook pts, script pts, equal): '
              f'{[tuple(int(a) for a in h) for h in G["hook"]]}')
        pop = mov & (R[:, 4] >= 10) & (R[:, 4] < 50)
        print('\nG1 / G2 / CC - sparse far cars (10-49 ACCUMULATED-SOURCE points, moving-ego frames), medians')
        print('| ring | sector | boxes | ' + ' | '.join(f'{v}: RC / CC / PPP / pts' for v in names) + ' |')
        print('|---|---|---|' + '---|' * len(names))
        for lo, hi in ((20, 40), (40, 75)):
            for sec in range(3):
                q = R[pop & (R[:, 1] >= lo) & (R[:, 1] < hi) & (R[:, 2] == sec)]
                tot_cells += 1
                cells = []
                for v in names:
                    if len(q):
                        rc = np.median(q[:, c(v, 0)]); tot_g1[v] += rc >= 0.8
                        cells.append(f'{rc:.2f} / {np.median(q[:, c(v, 1)]):.2f} / {np.median(q[:, c(v, 2)]):.2f} / '
                                     f'{np.median(q[:, c(v, 3)]):.0f}')
                        if v == DECIDE and not (1.0 <= np.median(q[:, c(v, 2)]) <= 1.1):
                            all_pass['G2'] = False
                    else:
                        cells.append('-')
                flag = ' (< 15 boxes)' if len(q) < 15 else ''
                print(f'| {lo}-{hi} | {SECTORS[sec]} | {len(q)}{flag} | ' + ' | '.join(cells) + ' |')
        pr = mov & (R[:, c(DECIDE, 3)] >= 10) & (R[:, c(DECIDE, 3)] < 50)
        print(f'secondary population (10-49 RENDERED points, {DECIDE}): {pr.sum()} cars, median RC {np.median(R[pr, c(DECIDE, 0)]):.2f}')
        g3 = G['g3']   # fi, ring, filled, n, phantom
        print('\nG3 (reported): phantom share = rendered points in no source box with no source point within 0.3 m')
        for g in range(len(RINGS) - 1):
            row = []
            for filled in (0, 1):
                s = (g3[:, 1] == g) & (g3[:, 2] == filled); n_ = g3[s, 3].sum(); ph = g3[s, 4].sum()
                row.append(f'{"filled" if filled else "z-buffered"} {ph / max(n_, 1):.3f} ({int(n_)})')
            s = g3[:, 1] == g
            print(f'  {RINGS[g]}-{RINGS[g + 1]} m: all {g3[s, 4].sum() / max(g3[s, 3].sum(), 1):.3f}; ' + '; '.join(row))
        H = G['g3h']   # fi, planar range, in box, |dr|, variant index into HOLDOUT
        for vi, vname in enumerate(HOLDOUT):
            Hv = H[H[:, 4] == vi] if len(H) else np.zeros((0, 5))
            tag = 'DECIDING' if vname == DECIDE else 'reported'
            print(f"\nG3' ({tag}) ring hold-out, {vname}: filled cells holding a held-out return, |dr| (m)")
            for g in range(len(RINGS) - 1):
                s_ = Hv[ring_bin(Hv[:, 1]) == g] if len(Hv) else np.zeros((0, 5))
                ib = s_[s_[:, 2] > 0]; ob = s_[s_[:, 2] == 0]
                f = lambda a: (f'median {np.median(a[:, 3]):.3f}, >0.5 m {np.mean(a[:, 3] > 0.5):.3f} ({len(a)})' if len(a) else '- (0)')
                print(f'  {RINGS[g]}-{RINGS[g + 1]} m: {f(s_)} | in boxes {f(ib)} | outside {f(ob)}'
                      + ('  [< 200 cells: UNRESOLVED]' if len(s_) < 200 else ''))
            pooled = Hv[(Hv[:, 1] >= 20) & (Hv[:, 1] < 75)] if len(Hv) else np.zeros((0, 5))
            enough = len(Hv) and all((ring_bin(Hv[:, 1]) == g).sum() >= 200 for g in range(len(RINGS) - 1))
            if enough:
                ok = np.median(pooled[:, 3]) <= 0.15 and np.mean(pooled[:, 3] > 0.5) <= 0.05
                print(f"  pooled 20-75 m: median {np.median(pooled[:, 3]):.3f}, share > 0.5 m {np.mean(pooled[:, 3] > 0.5):.3f} -> {'PASS' if ok else 'FAIL'}")
            else:
                ok = None; print('  UNRESOLVED (fewer than 200 comparable cells in some ring)')
            if vname == DECIDE:
                all_pass['G3h'] = ok if all_pass['G3h'] is not None and ok is not None else None
        t = G['g4']
        ok4 = np.median(t[:, 0]) < 0.5 and np.mean(t[:, 2] > CAP) <= 0.01
        print(f'\nG4: render s per anchor median {np.median(t[:, 0]):.3f} / p90 {np.percentile(t[:, 0], 90):.3f} '
              f'(input points median {np.median(t[:, 1]) / 1e3:.0f}k); frames over {CAP // 1000}k voxels {np.mean(t[:, 2] > CAP):.3f} '
              f'(median {np.median(t[:, 2]) / 1e3:.0f}k) -> {"PASS" if ok4 else "FAIL"}')
        all_pass['G4'] &= ok4
        # G5 (analysis)
        print(f'\nG5 (ANALYSIS, Waymo val labels): rendered ({DECIDE}) vs Waymo at matched counts, medians z-layers / line spacing (m)')
        for lo, hi in ((20, 40), (40, 75)):
            for a, z in ((10, 50), (50, 200)):
                rq = R[mov & (R[:, 1] >= lo) & (R[:, 1] < hi) & (R[:, shape0] >= a) & (R[:, shape0] < z)]
                wq = W[(W[:, 0] >= lo) & (W[:, 0] < hi) & (W[:, 1] >= a) & (W[:, 1] < z)]
                if len(rq) and len(wq):
                    zr, zw = np.median(rq[:, shape0 + 1]), np.median(wq[:, 2]); lr, lw = np.nanmedian(rq[:, shape0 + 2]), np.nanmedian(wq[:, 3])
                    print(f'  {lo}-{hi} m, {a}-{z - 1} pts: z-layers {zr:.0f} vs {zw:.0f} ({zr / zw:.2f}x), line spacing '
                          f'{lr:.3f} vs {lw:.3f} ({lr / lw:.2f}x)  [{len(rq)} / {len(wq)} cars]')
        print('\nRendered points per sparse far car vs Waymo (analysis; cannot choose H): ', end='')
        for lo, hi in ((20, 40), (40, 75)):
            rq = R[pop & (R[:, 1] >= lo) & (R[:, 1] < hi)]; wq = W[(W[:, 0] >= lo) & (W[:, 0] < hi)]
            print(f'{lo}-{hi} m {DECIDE} {np.median(rq[:, c(DECIDE, 3)]):.0f} / A+H {np.median(rq[:, c("A+H", 3)]):.0f} / Waymo (all >= 10 pts) {np.median(wq[:, 1]):.0f}; ', end='')
        print()
    print(f'\nCOMBINED: G1 cells with RC >= 0.8 (of {tot_cells}): ' + ', '.join(f'{v} {tot_g1[v]}' for v in names)
          + f' -> G1 ({DECIDE}) {"PASS" if tot_g1[DECIDE] >= 10 else "FAIL"}; G2 {"PASS" if all_pass["G2"] else "FAIL"}; '
          f"G3' {'UNRESOLVED' if all_pass['G3h'] is None else ('PASS' if all_pass['G3h'] else 'FAIL')}; G4 {'PASS' if all_pass['G4'] else 'FAIL'}")


if __name__ == '__main__':
    if sys.argv[1] == 'report':
        report_mode(sys.argv[2], sys.argv[3:]); sys.exit(0)
    cfg = EasyDict(); cfg_from_yaml_file(sys.argv[2], cfg)
    if sys.argv[1] == 'source':
        source_mode(cfg, sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        waymo_mode(cfg, int(sys.argv[3]), sys.argv[4])
