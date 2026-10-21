"""Label-free CPU gate of the relocate-and-render (RR) row (experiments_md 20261018_02 §3.1, pre-declared).

Per platform (DATA_CONFIGS key), on the training infos at k x floor(N / 200), k = 0..199, loaded with the row's config
(training mode, RR on, numpy seed per frame from (platform, index)), the RR step's input and output are captured inside
the real loader (the augmentor queue's first entry is replaced by a wrapper that calls the SAME
`relocate_render_scene` with the same config and the same global numpy RNG, only asking it to keep the moved donor
points). Per copy: G1 RC and G2 PPP on the RR output cloud with `source_coverage.lattice_coverage` (all the copy's
surviving rendered points), G3 phantoms against the relocated donor points, r', sector of the travel direction. Per
frame: donors, rejections, zero-point copies, originals emptied by the shadow, originals bit-identical, copy
collisions, the time-lag column. G4: full `dataset[i]` time with and without the RR entry (same process, same node).
Target dump on the first frames with copies: CenterHead targets, heatmap peaks at the copies, points per copy box.
SOURCE labels only; the lattice is the target's published calibration (TRAIN segments). No target cloud or label.

    python analysis/rr_gate.py run --cfg <cfg> --key NUSCENES_N008 --out <dir> [--n 200] [--g4 100] [--dump 3]
                                   [--images 2] [--max_frames m]
    python analysis/rr_gate.py report --dir <dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import copy  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _init_path  # noqa: F401,E402

PLAT_ID = {'NUSCENES_N008': 0, 'NUSCENES_N015': 1}
PLAT_NAME = {0: 'Boston', 1: 'Singapore'}
G1_RINGS = ((20.0, 40.0), (40.0, 75.2))
SECTORS = ('ahead', 'sides', 'behind')
RBINS = (0, 10, 20, 30, 40, 50, 60, 75.2)
G3_RINGS = (0, 20, 30, 40, 50, 75.2)
PHANTOM_M = 0.3


def build(cfg_file, key):
    import torch
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    torch.set_num_threads(1)
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    dcfg = cfg.DATA_CONFIGS[key]
    ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
    return cfg, dcfg, ds


def head_for(cfg, dcfg):
    import torch
    from pcdet.models.dense_heads.center_head import CenterHead
    head = CenterHead.__new__(CenterHead); torch.nn.Module.__init__(head)
    head.model_cfg = cfg.MODEL.DENSE_HEAD
    head.class_names = list(cfg.CLASS_NAMES); head.class_names_each_head = [list(cfg.CLASS_NAMES)]
    head.point_cloud_range = np.array(dcfg.POINT_CLOUD_RANGE, dtype=np.float32)
    head.voxel_size = [p for p in dcfg.DATA_PROCESSOR if p.NAME == 'transform_points_to_voxels'][0].VOXEL_SIZE
    stride = cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
    side = int(round((head.point_cloud_range[3] - head.point_cloud_range[0]) / head.voxel_size[0] / stride))
    return head, side, head.voxel_size[0] * stride


def in_box_counts(xyz, boxes):
    import torch
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    if not len(boxes):
        return np.zeros(0, np.int64)
    m = roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(np.ascontiguousarray(xyz[:, :3], np.float32)),
                                                  torch.from_numpy(np.ascontiguousarray(boxes[:, :7], np.float32)))
    return m.numpy().sum(1)


def ground_plane(xyz, seed=0):
    """Label-free descriptive ground plane z = a x + b y + c of the loader's cloud (3-40 m, |z| < 1 m, iterated
    least squares keeping residuals < 0.15 m). None if under 100 candidate points."""
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    p = xyz[(r > 3) & (r < 40) & (np.abs(xyz[:, 2]) < 1.0)].astype(np.float64)
    if len(p) < 100:
        return None
    if len(p) > 60000:
        p = p[np.random.default_rng(seed).choice(len(p), 60000, replace=False)]
    keep = np.abs(p[:, 2]) < 0.3
    coef = np.zeros(3)
    for _ in range(5):
        if keep.sum() < 50:
            return None
        A = np.c_[p[keep, 0], p[keep, 1], np.ones(int(keep.sum()))]
        coef = np.linalg.lstsq(A, p[keep, 2], rcond=None)[0]
        keep = np.abs(p[:, 2] - (p[:, 0] * coef[0] + p[:, 1] * coef[1] + coef[2])) < 0.15
    return coef


def blocked_share(c, scene_lat, lat):
    """Share of the copy's rendered columns (all rendered pixels, before occlusion) in which the scene (RR output
    without copy points) holds a return in the copy's row span nearer than the copy's nearest point in that column
    by > 0.5 m: a copy behind a surface (e.g. a facade) seen through the scene's sampling gaps scores high."""
    from pcdet.datasets.augmentor import relocate_render as rr
    _, rows, cols, _, rng = rr._render(c['moved'], lat['thetas'], lat['n_cols'], lat['height'])
    if not len(cols):
        return np.nan
    srow, scol, srng, sval = scene_lat
    ucol, inv = np.unique(cols, return_inverse=True)
    rmin = np.full(len(ucol), np.inf); np.minimum.at(rmin, inv, rng)
    sel = sval & np.isin(scol, ucol) & (srow >= rows.min()) & (srow <= rows.max())
    near = np.full(len(ucol), np.inf)
    if sel.any():
        np.minimum.at(near, np.searchsorted(ucol, scol[sel]), srng[sel])
    return float(np.mean(near < rmin - 0.5))


def image(path, title, before, after, rec, lat):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pcdet.datasets.augmentor import relocate_render as rr
    SURF, INK, INK2, MUTED, C_DON, C_COPY = '#fcfcfb', '#0b0b0b', '#52514e', '#b4b2ab', '#2a78d6', '#eb6834'
    fig, ax = plt.subplots(1, 3, figsize=(21, 7.4), facecolor=SURF, gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    copies = rec['copies']
    don = {c['donor'] for c in copies}
    for a, d, lab in ((ax[0], before, 'before RR (loader cloud, ground-origin frame)'),
                      (ax[1], after, f"after RR: {len(copies)} copies")):
        a.set_facecolor(SURF)
        p = d['points']
        a.scatter(p[:, 0], p[:, 1], s=0.02, c=MUTED, linewidths=0, rasterized=True)
        nb = len(before['gt_boxes'])
        for j, b in enumerate(d['gt_boxes']):
            cc = rr.bev_corners(b[None].astype(np.float64))[0]; cc = np.vstack([cc, cc[:1]])
            col = C_COPY if j >= nb else (C_DON if j in don else INK2)
            a.plot(cc[:, 0], cc[:, 1], color=col, lw=1.2 if j >= nb or j in don else 0.7)
        if d is after:
            for c in copies:
                q = p[c['point_index']]
                a.scatter(q[:, 0], q[:, 1], s=1.5, c=C_COPY, linewidths=0)
        a.set_xlim(-75.2, 75.2); a.set_ylim(-75.2, 75.2); a.set_aspect('equal')
        a.set_title(lab, color=INK, fontsize=11); a.tick_params(colors=INK2, labelsize=8)
        for s in a.spines.values():
            s.set_color(MUTED)
        a.set_xlabel('x (m)', color=INK2); a.set_ylabel('y (m)', color=INK2)
    a = ax[2]; a.set_facecolor(SURF)
    c = max(copies, key=lambda c: c['r_new'])
    h = lat['height']
    def ang(x):
        d = x[:, :3] - [0, 0, h]
        return np.degrees(np.arctan2(d[:, 1], d[:, 0])), np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
    az0, el0 = ang(c['moved']); az1, el1 = ang(after['points'][c['point_index']])
    ref = np.degrees(c['phi_new'])
    wrap = lambda v: (v - ref + 180) % 360 - 180
    a.scatter(wrap(az0), el0, s=2, c=C_DON, linewidths=0, label=f"moved donor points ({len(el0)})")
    a.scatter(wrap(az1), el1, s=9, c=C_COPY, linewidths=0, label=f"rendered copy points ({len(el1)})")
    th = np.degrees(lat['thetas']); lo, hi = min(el0.min(), el1.min()) - 0.3, max(el0.max(), el1.max()) + 0.3
    for t in th[(th >= lo) & (th <= hi)]:
        a.axhline(t, color=MUTED, lw=0.5, zorder=0)
    a.set_ylim(lo, hi)
    a.set_title(f"farthest copy: r' {c['r_new']:.1f} m (donor {c['r_donor']:.1f} m, {c['n_donor']} pts); "
                f"grey lines = TOP rows", color=INK, fontsize=10)
    a.set_xlabel('azimuth about the copy centre (deg)', color=INK2); a.set_ylabel('elevation about (0, 0, 2.184) (deg)',
                                                                                   color=INK2)
    a.tick_params(colors=INK2, labelsize=8); a.legend(loc='upper right', fontsize=9, frameon=False)
    for s in a.spines.values():
        s.set_color(MUTED)
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=110, facecolor=SURF); plt.close(fig)


def run(a):
    import torch
    from scipy.spatial import cKDTree
    from lattice_depth_ceiling import travel_direction, sector_of
    from source_coverage import lattice_coverage, nearest_per_pixel
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.utils import common_utils
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plat = PLAT_ID[a.key]
    cfg, dcfg, ds = build(a.cfg, a.key)
    aug = ds.data_augmentor
    q0 = aug.data_augmentor_queue[0]
    assert getattr(q0, 'func', None) is not None and q0.func.__name__ == 'relocate_render', 'RR is not first'
    assert [c.NAME for c in aug.augmentor_configs.AUG_CONFIG_LIST][0] == 'relocate_render'
    rr_cfg = q0.keywords['config']; lat = q0.keywords['lattice']
    assert list(rr_cfg.POINT_CLOUD_RANGE) == list(dcfg.POINT_CLOUD_RANGE), 'RR range != the dataset range'
    print(f"[rr_gate] {a.key}: {len(ds)} infos; lattice {lat['n_segments']} train segments, "
          f"{len(lat['thetas'])} rows, {lat['n_cols']} cols, h {lat['height']}; queue "
          f"{[getattr(f, 'func', f).__name__ for f in aug.data_augmentor_queue]}", flush=True)
    CAP = {}

    def rr_step(data_dict=None):
        CAP['before'] = {k: np.copy(data_dict[k]) for k in ('points', 'gt_boxes', 'gt_names', 'gt_boxes_mask')}
        o, rec = rr.relocate_render_scene(data_dict, rr_cfg, lat, rng=np.random, keep_points=True)
        CAP['rec'] = rec
        CAP['after'] = {k: np.copy(o[k]) for k in ('points', 'gt_boxes', 'gt_names', 'gt_boxes_mask')}
        return o
    fwd0 = aug.forward

    def fwd(data_dict):
        o = fwd0(data_dict)
        CAP['post'] = np.copy(o['gt_boxes'])
        return o
    queue_rr = list(aug.data_augmentor_queue)
    aug.data_augmentor_queue = [rr_step] + queue_rr[1:]
    aug.forward = fwd
    N = len(ds); step = N // a.n
    frames = [k * step for k in range(a.n)][:a.max_frames]
    th, nc, h = lat['thetas'], lat['n_cols'], lat['height']
    head, side, cell = head_for(cfg, dcfg)
    copies, frows, dumps, imgs = [], [], [], []
    t_start = time.time()
    for n_done, i in enumerate(frames):
        np.random.seed((plat * 1000003 + i) % (2 ** 31))
        CAP.clear()
        d = ds[i]
        rec, B, A = CAP['rec'], CAP['before'], CAP['after']
        info = ds.infos[i]
        fwdv = travel_direction(info); moving = fwdv is not None
        fv = fwdv if moving else np.array([1.0, 0.0])
        nb0 = len(B['gt_boxes']); k = rec['n_placed']
        P = A['points']
        # frame checks
        same = bool(np.array_equal(A['gt_boxes'][:nb0], B['gt_boxes']) and
                    np.array_equal(A['gt_names'][:nb0], B['gt_names']) and
                    np.array_equal(A['gt_boxes_mask'][:nb0], B['gt_boxes_mask']))
        cb = A['gt_boxes'][nb0:].astype(np.float64)
        coll = 0
        for j in range(k):
            others = np.concatenate([B['gt_boxes'][:, :7].astype(np.float64), cb[:j, :7]])
            coll += int(rr.rects_overlap(rr.bev_corners(cb[j:j + 1])[0], rr.bev_corners(others)).any())
        cnt_b = in_box_counts(B['points'], B['gt_boxes']); cnt_a = in_box_counts(P, A['gt_boxes'][:nb0])
        emptied = int(((cnt_b >= 1) & (cnt_a == 0)).sum())
        lost = int((cnt_a < cnt_b).sum())
        tcol = rr_cfg.TIME_COLUMN
        scene_t = float(B['points'][:, tcol].max()) if B['points'].shape[1] > tcol else np.nan
        copy_t = np.concatenate([P[c['point_index'], tcol] for c in rec['copies']]) if k else np.zeros(0)
        frows.append([plat, i, moving, rec['n_boxes'], rec['n_candidates'], rec['n_eligible'], k, rec['n_no_place'],
                      rec['n_zero'], rec['reject']['overlap'], rec['reject']['range'], rec['reject']['scene'],
                      same, coll, emptied, lost, len(B['points']), len(P), scene_t,
                      float(np.abs(copy_t).max()) if len(copy_t) else 0.0])
        plane = ground_plane(B['points'][:, :3], seed=i)
        pl = plane if plane is not None else np.full(3, np.nan)
        frows[-1] += [float(np.degrees(np.arctan(pl[0]))), float(np.degrees(np.arctan(pl[1]))), float(pl[2])]
        if k:
            npp = nearest_per_pixel(P[:, :3], th, nc, h)
            sec = sector_of(cb[:, :2], fv)
            iscopy = np.zeros(len(P), bool)
            for c in rec['copies']:
                iscopy[c['point_index']] = True
            from pcdet.datasets.rerender_utils import to_lattice
            scene_lat = to_lattice(P[~iscopy, :3].astype(np.float64), h, th, nc)
            for j, c in enumerate(rec['copies']):
                cov = lattice_coverage(P[:, :3], cb[j:j + 1, :7], [c['point_index']], th, nc, h, npp)[0]
                rp = P[c['point_index'], :3].astype(np.float64)
                dmin = cKDTree(c['moved']).query(rp, k=1)[0] if len(rp) else np.zeros(0)
                gz = lambda x, y: pl[0] * x + pl[1] * y + pl[2]
                bx = c['box']; dbx = c['donor_box']
                off_c = (bx[2] - bx[5] / 2) - gz(bx[0], bx[1]); off_d = (dbx[2] - dbx[5] / 2) - gz(dbx[0], dbx[1])
                n_foot = int(rr.points_in_rect(B['points'][:, :2], bx, rr_cfg.BEV_MARGIN).sum())
                copies.append([plat, i, moving, c['r_donor'], c['n_donor'], c['r_new'], c['phi_new'], sec[j],
                               c['n_rendered'], c['n_occluded'], c['n_final'], cov[0], cov[1], cov[2],
                               int((dmin > PHANTOM_M).sum()), len(rp), c['tries'], c['n_inbox_removed'],
                               c['n_shadow_removed'], off_c, off_d, n_foot, blocked_share(c, scene_lat, lat)])
        if k and len(dumps) < a.dump:
            dumps.append(dump_frame(d, CAP, rec, head, side, cell, cfg, i))
        if k >= 2 and len(imgs) < a.images:
            pth = out / f'bev_{a.key}_{i:05d}.png'
            image(pth, f'{a.key} ({PLAT_NAME[plat]}) train info {i}: relocate-and-render, before / after', B, A, rec, lat)
            imgs.append(str(pth))
        if n_done % 20 == 0:
            print(f'[rr_gate] {n_done + 1}/{len(frames)} frames, {len(copies)} copies, {time.time() - t_start:.0f} s',
                  flush=True)
    # G4: with and without the RR entry, same frames, same process
    aug.forward = fwd0
    g4 = []
    for i in frames[::2][:a.g4]:
        seed = (plat * 1000003 + i) % (2 ** 31)
        aug.data_augmentor_queue = queue_rr[1:]
        np.random.seed(seed); ds[i]                                     # warm the file cache, untimed
        np.random.seed(seed); t0 = time.perf_counter(); ds[i]; t_off = time.perf_counter() - t0
        aug.data_augmentor_queue = queue_rr
        np.random.seed(seed); t0 = time.perf_counter(); ds[i]; t_on = time.perf_counter() - t0
        g4.append([plat, i, t_off, t_on])
    res = dict(key=a.key, plat=plat, n_infos=N, frames=frames, copies=np.array(copies, np.float64).reshape(-1, 23),
               frames_rows=np.array(frows, np.float64).reshape(-1, 23), g4=np.array(g4, np.float64).reshape(-1, 4),
               dumps=dumps, images=imgs, lattice=dict(n_segments=lat['n_segments'], thetas=lat['thetas']),
               node=os.uname().nodename)
    pickle.dump(res, open(out / f'gate_{a.key}.pkl', 'wb'))
    print(f'[rr_gate] done {a.key}: {len(frames)} frames, {len(copies)} copies, images {imgs}', flush=True)


def dump_frame(d, CAP, rec, head, side, cell, cfg, i):
    import torch
    from pcdet.datasets.augmentor import relocate_render as rr
    gt = d['gt_boxes']; lab = gt[:, 7].astype(int)
    C = len(cfg.CLASS_NAMES)
    post = CAP['post']; k = rec['n_placed']
    copy_rows = post[len(post) - k:, :7]
    fin = [int(np.nonzero(np.all(gt[:, :7] == r, 1))[0][0]) if np.all(gt[:, :7] == r, 1).any() else -1
           for r in copy_rows]
    t = head.assign_targets(torch.from_numpy(gt[None]).float().clone(), feature_map_size=[side, side])
    hm = t['heatmaps'][0][0]
    pcr = head.point_cloud_range
    def pix(b):
        x = min(max((b[0] - pcr[0]) / cell, 0), side - 0.5); y = min(max((b[1] - pcr[1]) / cell, 0), side - 0.5)
        return int(y), int(x)
    peaks = [float(hm[0][pix(gt[f])]) if f >= 0 else np.nan for f in fin]
    cnt_final = in_box_counts(d['points'], gt[[f for f in fin if f >= 0]]) if any(f >= 0 for f in fin) else []
    pos = {cfg.CLASS_NAMES[c]: int((lab == c + 1).sum()) for c in range(C)}
    peaks_all = {cfg.CLASS_NAMES[c]: int(hm[c].eq(1).sum()) for c in range(C)}
    return dict(frame=i, n_copies=k, final_rows=fin, copy_labels=[int(lab[f]) if f >= 0 else None for f in fin],
                car_peak_at_copy=peaks, positives=pos, peaks=peaks_all,
                copies=[dict(r_donor=c['r_donor'], r_new=c['r_new'], n_donor=c['n_donor'], n_rendered=c['n_rendered'],
                             n_final_rr=c['n_final']) for c in rec['copies']] if rec.get('version', 1) == 1 else [],
                n_in_final_cloud=[int(x) for x in cnt_final])


def report(a):
    D = Path(a.dir)
    res = [pickle.load(open(p, 'rb')) for p in sorted(D.glob('gate_*.pkl'))]
    cp = np.concatenate([r['copies'] for r in res]); fr = np.concatenate([r['frames_rows'] for r in res])
    g4 = np.concatenate([r['g4'] for r in res])
    print('nodes:', {r['key']: r['node'] for r in res})
    for r in res:
        print(f"{r['key']}: {len(r['frames'])} frames of {r['n_infos']} (stride {r['n_infos'] // 200}), "
              f"{len(r['copies'])} copies; lattice from {r['lattice']['n_segments']} train segments")
    # columns
    PL, MOV, RD, ND, RN, SEC, NR, NOC, NF, RC, PPP, VR, NPH, NPT = 0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15
    print('\n== frames ==')
    PLATS = [p for p in (0, 1) if (fr[:, 0] == p).any()]
    for p in PLATS:
        f = fr[fr[:, 0] == p]
        print(f"{PLAT_NAME[p]}: {len(f)} frames ({int(f[:, 2].sum())} moving); per frame: boxes {f[:, 3].mean():.1f}, "
              f"Car candidates {f[:, 4].mean():.2f}, eligible donors {f[:, 5].mean():.2f}, copies {f[:, 6].mean():.2f} "
              f"(frames with >= 1 copy {int((f[:, 6] > 0).sum())}, with 8 {int((f[:, 6] == 8).sum())}); "
              f"no placement {int(f[:, 7].sum())}, zero-point copies {int(f[:, 8].sum())}; rejected tries "
              f"overlap {int(f[:, 9].sum())} / range {int(f[:, 10].sum())} / scene {int(f[:, 11].sum())}")
        print(f"   originals bit-identical in {int(f[:, 12].sum())} / {len(f)} frames; copy collisions "
              f"{int(f[:, 13].sum())}; originals emptied by RR {int(f[:, 14].sum())}, originals that lost points "
              f"{int(f[:, 15].sum())}; points per frame {f[:, 16].mean():.0f} -> {f[:, 17].mean():.0f}; scene "
              f"time-lag max {np.nanmax(f[:, 18]):.3f}, copy time-lag max {f[:, 19].max():.3f}")
    print(f'\n== copies: {len(cp)}; rendered {cp[:, NR].sum():.0f}, occluded {cp[:, NOC].sum():.0f}, final '
          f'{cp[:, NF].sum():.0f}; donor points median {np.median(cp[:, ND]):.0f}; r_donor median '
          f'{np.median(cp[:, RD]):.1f} m; r\' median {np.median(cp[:, RN]):.1f} m')
    ok = ~np.isnan(cp[:, RC])
    print(f'   copies with a visible silhouette (RC defined): {int(ok.sum())} of {len(cp)}')
    print('\n== G1 (deciding): median RC, moving-ego frames; cells platform x ring x sector ==')
    passed, cells = 0, 0
    print('| platform | ring | sector | copies | median RC | PPP median | flag |')
    print('|---|---|---|---|---|---|---|')
    for p in (0, 1):
        for lo, hi in G1_RINGS:
            for s in range(3):
                m = ok & (cp[:, PL] == p) & (cp[:, MOV] == 1) & (cp[:, RN] >= lo) & (cp[:, RN] < hi) & (cp[:, SEC] == s)
                cells += 1
                med = np.median(cp[m, RC]) if m.any() else np.nan
                ppp = np.median(cp[m, PPP]) if m.any() else np.nan
                passed += int(med >= 0.8)
                print(f'| {PLAT_NAME[p]} | {lo:g}-{hi:g} | {SECTORS[s]} | {int(m.sum())} | {med:.3f} | {ppp:.3f} | '
                      f'{"< 15" if m.sum() < 15 else ""} |')
    print(f'G1: {passed} of {cells} cells >= 0.8 -> {"PASS" if passed >= 10 else "FAIL"}')
    print('\n== RC per r\' bin (all frames), median [n] ==')
    for p in PLATS:
        row = []
        for lo, hi in zip(RBINS[:-1], RBINS[1:]):
            m = ok & (cp[:, PL] == p) & (cp[:, RN] >= lo) & (cp[:, RN] < hi)
            row.append(f'{lo:g}-{hi:g}: {np.median(cp[m, RC]) if m.any() else np.nan:.3f} [{int(m.sum())}]')
        print(f'{PLAT_NAME[p]}: ' + '; '.join(row))
    print('\n== secondary: copies with 10-49 final rendered points, median RC per G1 cell [n] ==')
    for p in PLATS:
        row = []
        for lo, hi in G1_RINGS:
            for s in range(3):
                m = ok & (cp[:, PL] == p) & (cp[:, MOV] == 1) & (cp[:, RN] >= lo) & (cp[:, RN] < hi) & \
                    (cp[:, SEC] == s) & (cp[:, NF] >= 10) & (cp[:, NF] < 50)
                row.append(f'{lo:g}-{hi:g} {SECTORS[s]}: {np.median(cp[m, RC]) if m.any() else np.nan:.3f} [{int(m.sum())}]')
        print(f'{PLAT_NAME[p]}: ' + '; '.join(row))
    print('\n== G2: PPP per G1 cell in [1.0, 1.1] ==')
    g2 = True
    for p in PLATS:
        for lo, hi in G1_RINGS:
            for s in range(3):
                m = ok & (cp[:, PL] == p) & (cp[:, MOV] == 1) & (cp[:, RN] >= lo) & (cp[:, RN] < hi) & (cp[:, SEC] == s)
                v = np.median(cp[m, PPP]) if m.any() else np.nan
                g2 &= bool(1.0 <= v <= 1.1)
    print(f'PPP overall: min {np.nanmin(cp[ok, PPP]):.3f}, median {np.nanmedian(cp[ok, PPP]):.3f}, max '
          f'{np.nanmax(cp[ok, PPP]):.3f} -> G2 {"PASS" if g2 else "FAIL"}')
    print('\n== G3: phantom share per r\' ring (pooled platforms), < 1% ==')
    g3 = True
    for lo, hi in zip(G3_RINGS[:-1], G3_RINGS[1:]):
        m = (cp[:, RN] >= lo) & (cp[:, RN] < hi)
        share = cp[m, NPH].sum() / max(cp[m, NPT].sum(), 1)
        g3 &= bool(share < 0.01)
        print(f'{lo:g}-{hi:g} m: {int(cp[m, NPH].sum())} of {int(cp[m, NPT].sum())} = {100 * share:.3f}%')
    print(f'G3 {"PASS" if g3 else "FAIL"}')
    print('\n== G4: dataset[i] time, same frames, same node ==')
    for p in PLATS:
        m = g4[:, 0] == p
        if m.any():
            print(f'{PLAT_NAME[p]}: {int(m.sum())} frames, off {g4[m, 2].sum():.1f} s, on {g4[m, 3].sum():.1f} s, '
                  f'ratio {g4[m, 3].sum() / g4[m, 2].sum():.3f} (median per frame {np.median(g4[m, 3] / g4[m, 2]):.3f})')
    ratio = g4[:, 3].sum() / g4[:, 2].sum()
    print(f'pooled ratio {ratio:.3f} -> G4 {"PASS" if ratio <= 1.20 else "FAIL"}')
    print('\n== target dump ==')
    for r in res:
        for dd in r['dumps']:
            print(f"{r['key']} frame {dd['frame']}: copies {dd['n_copies']}, final rows {dd['final_rows']}, labels "
                  f"{dd['copy_labels']}, Car heatmap at copy centres {dd['car_peak_at_copy']}; positives "
                  f"{dd['positives']}; peaks {dd['peaks']}; copy points in the final cloud {dd['n_in_final_cloud']}; "
                  + '; '.join(f"r {c['r_donor']:.1f}->{c['r_new']:.1f} donor {c['n_donor']} rendered "
                              f"{c['n_rendered']} kept {c['n_final_rr']}" for c in dd['copies']))
    print('\n== diagnostics (label-free, descriptive, NOT deciding; 20261018_02 §3.1 addendum) ==')
    for p in PLATS:
        f = fr[fr[:, 0] == p]; m = cp[:, PL] == p
        oc, od, nfoot, blk = cp[m, 19], cp[m, 20], cp[m, 21], cp[m, 22]
        print(f"{PLAT_NAME[p]}: fitted ground slope x {np.nanmedian(f[:, 20]):+.2f} deg / y {np.nanmedian(f[:, 21]):+.2f} "
              f"deg, z0 {np.nanmedian(f[:, 22]):+.3f} m (median over frames)")
        print(f"   copy bottom - local ground: median {np.nanmedian(oc):+.2f} m, |.| median {np.nanmedian(np.abs(oc)):.2f}, "
              f"> 0.5 m {np.nanmean(np.abs(oc) > 0.5):.1%}, > 1 m {np.nanmean(np.abs(oc) > 1.0):.1%}; donors: |.| median "
              f"{np.nanmedian(np.abs(od)):.2f}, > 0.5 m {np.nanmean(np.abs(od) > 0.5):.1%}")
        for lo, hi in ((0, 20), (20, 40), (40, 75.2)):
            mm = (cp[m, RN] >= lo) & (cp[m, RN] < hi)
            if mm.any():
                print(f"   r' {lo:g}-{hi:g}: copies {int(mm.sum())}, |ground offset| median {np.nanmedian(np.abs(oc[mm])):.2f} m, "
                      f"footprint with no scene point {np.mean(nfoot[mm] == 0):.1%}, behind a surface in >= half its "
                      f"columns {np.nanmean(blk[mm] >= 0.5):.1%} (median blocked share {np.nanmedian(blk[mm]):.2f})")
    print('\nimages:', [p for r in res for p in r['images']])



# ================================================================================================ gate v2 (§6.3)
def blocked_share_v2(placed, scene_lat, scene_hz, lat):
    """(v2 D2, v1 D2) shares of the copy's rendered columns blocked by a scene return in the copy's row span nearer by
    > 0.5 m than the copy's nearest point in that column; v2 also requires the return > 0.3 m above the gate's plane."""
    from pcdet.datasets.augmentor import relocate_render as rr
    _, rows, cols, _, rng = rr._render(placed, lat['thetas'], lat['n_cols'], lat['height'])
    if not len(cols):
        return np.nan, np.nan
    srow, scol, srng, sval = scene_lat
    ucol, inv = np.unique(cols, return_inverse=True)
    rmin = np.full(len(ucol), np.inf); np.minimum.at(rmin, inv, rng)
    base = sval & np.isin(scol, ucol) & (srow >= rows.min()) & (srow <= rows.max())
    out = []
    for sel in (base & (scene_hz > 0.3), base):
        near = np.full(len(ucol), np.inf)
        if sel.any():
            np.minimum.at(near, np.searchsorted(ucol, scol[sel]), srng[sel])
        out.append(float(np.mean(near < rmin - 0.5)))
    return out[0], out[1]


def image_v2(path, title, before, after, rec, lat):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pcdet.datasets.augmentor import relocate_render as rr
    SURF, INK, INK2, MUTED, C_DB, C_COPY = '#fcfcfb', '#0b0b0b', '#52514e', '#b4b2ab', '#2a78d6', '#eb6834'
    fig, ax = plt.subplots(1, 3, figsize=(21, 7.4), facecolor=SURF, gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    copies = rec['copies']
    nb = len(before['gt_boxes'])
    for a, d, lab in ((ax[0], before, 'before RR v2 (loader cloud, ground-origin frame)'),
                      (ax[1], after, f"after RR v2: {len(copies)} copies (sparsified)")):
        a.set_facecolor(SURF)
        p = d['points']
        a.scatter(p[:, 0], p[:, 1], s=0.02, c=MUTED, linewidths=0, rasterized=True)
        for j, b in enumerate(d['gt_boxes']):
            cc = rr.bev_corners(b[None].astype(np.float64))[0]; cc = np.vstack([cc, cc[:1]])
            a.plot(cc[:, 0], cc[:, 1], color=C_COPY if j >= nb else INK2, lw=1.2 if j >= nb else 0.7)
        if d is after:
            for c in copies:
                q = p[c['point_index']]
                a.scatter(q[:, 0], q[:, 1], s=1.5, c=C_COPY, linewidths=0)
        a.set_xlim(-75.2, 75.2); a.set_ylim(-75.2, 75.2); a.set_aspect('equal')
        a.set_title(lab, color=INK, fontsize=11); a.tick_params(colors=INK2, labelsize=8)
        for sp in a.spines.values():
            sp.set_color(MUTED)
        a.set_xlabel('x (m)', color=INK2); a.set_ylabel('y (m)', color=INK2)
    a = ax[2]; a.set_facecolor(SURF)
    c = max(copies, key=lambda c: c['r_new'])
    h = lat['height']

    def ang(x):
        d = x[:, :3] - [0, 0, h]
        return np.degrees(np.arctan2(d[:, 1], d[:, 0])), np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
    ref = np.degrees(c['phi_new'])
    wrap = lambda v: (v - ref + 180) % 360 - 180
    for pts_, col, sz, lab in ((c['placed'], C_DB, 1, f"placed database points ({len(c['placed'])})"),
                               (c['rendered_unsparsified'], MUTED, 9, f"render before the sparsifier ({len(c['rendered_unsparsified'])})"),
                               (after['points'][c['point_index']], C_COPY, 9, f"kept, p = {c['p_keep']:.2f} ({c['n_final']})")):
        az, el = ang(pts_)
        a.scatter(wrap(az), el, s=sz, c=col, linewidths=0, label=lab)
    az0, el0 = ang(c['placed'])
    th = np.degrees(lat['thetas']); lo, hi = el0.min() - 0.3, el0.max() + 0.3
    for t in th[(th >= lo) & (th <= hi)]:
        a.axhline(t, color=MUTED, lw=0.5, zorder=0)
    a.set_ylim(lo, hi)
    a.set_title(f"farthest copy: r' {c['r_new']:.1f} m, RC {c['rc_loader']:.2f}; grey lines = TOP rows", color=INK,
                fontsize=10)
    a.set_xlabel('azimuth about the copy centre (deg)', color=INK2)
    a.set_ylabel('elevation about (0, 0, 2.184) (deg)', color=INK2)
    a.tick_params(colors=INK2, labelsize=8); a.legend(loc='upper right', fontsize=9, frameon=False)
    for sp in a.spines.values():
        sp.set_color(MUTED)
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=110, facecolor=SURF); plt.close(fig)


V2_COPY_COLS = ('plat', 'frame', 'moving', 'r_new', 'phi', 'sector', 'tries', 'rc_loader', 'vis_loader', 'n_db',
                'n_rendered', 'n_visible', 'p_keep', 'n_kept', 'n_final', 'rc_g1', 'ppp_g1', 'visrows_g1', 'n_phantom',
                'n_final_g3', 'off_plane', 'd2_v2', 'd2_v1', 'n_pts_grid', 'n_vox', 'n_layers', 'ppv', 'instance')
V3_EXTRA_COLS = ('fallback', 'bottom', 'd1v3_ref', 'd1v3_n')
V4_EXTRA_COLS = V3_EXTRA_COLS + ('d1v4_ref',)
V5_EXTRA_COLS = V4_EXTRA_COLS + ('vis_frac', 'fill_k') + tuple(f'{n}{b}' for n in ('hit', 'vis', 'rows', 'lost') for b in range(4))
V7_EXTRA_COLS = V5_EXTRA_COLS + ('ground_kind',)
D1V5_COLS = ('plat', 'frame', 'r', 'gt_bottom', 'est_loo', 'kind_loo', 'est_frame', 'kind_frame')
EXTRA_COLS = {2: (), 3: V3_EXTRA_COLS, 4: V4_EXTRA_COLS, 5: V5_EXTRA_COLS, 6: V5_EXTRA_COLS, 7: V7_EXTRA_COLS}
FRAME_EXTRA = {4: ('rej_zrange',), 5: ('rej_zrange', 'rej_novis'), 6: ('rej_zrange', 'rej_novis'), 7: ('rej_zrange', 'rej_novis')}


def d1v5_rows(P, B, rr_cfg, plane40_frame, plat, frame, r_max=75.2):
    """§12.3 D1 v5: per source GT car of the frame (the §12.1 set, 0 < BEV range < r_max), the loader's E5a estimate on
    the leave-one-out cloud (the points within 0.5 m (BEV) of its rectangle removed, the 4-40 m plane refitted on it) with
    the frame's OTHER GT cars; and the same with the loader's per-frame plane on the full cloud. Rows as D1V5_COLS; and
    the number of cars dropped for a None plane."""
    from pcdet.datasets.augmentor import relocate_render as rr
    from rr_ground_explore import leave_one_out
    P = np.asarray(P, np.float64)[:, :3]
    boxes = np.asarray(B['gt_boxes'], np.float64)
    if not len(boxes):
        return [], 0
    car = rr.gt_car_mask(B, rr_cfg.DONOR_CLASS)
    r = np.hypot(boxes[:, 0], boxes[:, 1])
    ev = np.flatnonzero(car & (r > 0) & (r < r_max))
    kw = dict(radius=float(rr_cfg.E5A_RADIUS), fb_radius=float(rr_cfg.E5A_FB_RADIUS), d_lo=float(rr_cfg.NEAR_D_LO),
              fb_pct=float(rr_cfg.E5A_FB_PCT), fb_min_cells=int(rr_cfg.E5A_FB_MIN_CELLS), cell=float(rr_cfg.GROUND_CELL),
              cell_min_pts=int(rr_cfg.GROUND_CELL_MIN_PTS))
    gridF = rr._BevGrid(P) if len(ev) else None
    rows, dropped = [], 0
    for j in ev:
        b = boxes[j, :7]
        oth = boxes[car & (np.arange(len(boxes)) != j), :7]
        X = P[leave_one_out(P, b, gridF)]
        pl = rr.ground_plane(X, float(rr_cfg.E5A_PLANE_RMAX))
        if pl is None:
            dropped += 1
            continue
        e1, k1 = rr.e5a_bottom(X, rr._BevGrid(X), b, pl, oth, **kw)
        e2, k2 = rr.e5a_bottom(P, gridF, b, plane40_frame, oth, **kw) if plane40_frame is not None else (np.nan, -1)
        rows.append([plat, frame, r[j], b[2] - b[5] / 2, e1, k1, e2, k2])
    return rows, dropped
V2_FRAME_COLS = ('plat', 'frame', 'moving', 'n_boxes', 'has_plane', 'n_attempted', 'n_placed', 'n_no_place', 'n_zero',
                 'rej_range', 'rej_overlap', 'rej_scene', 'rej_empty', 'rej_vis', 'rej_rc', 'same', 'coll', 'emptied',
                 'lost', 'pts_before', 'pts_after', 'scene_t', 'copy_t', 'gp_sx', 'gp_sy', 'gp_z0', 'rr_sx', 'rr_sy')


# ------------------------------------------------------------------------------------------------- RR v8 (§13.3)
BG_ROW_COLS = ('plat', 'frame', 'r_new', 'n_rendered', 'n_visible', 'n_kept', 'n_final', 'bottom', 'low_share', 'ground_kind',
               'n1_viol', 'tries')
BG_FRAME_COLS = ('plat', 'frame', 'n_car', 'n_bg', 'bg_attempted', 'bg_no_place', 'bg_zero', 'same_car_boxes',
                 'same_final_gt', 'hm_equal', 'centres_in_bg', 'orig_emptied_on', 'orig_emptied_off', 'car_emptied_by_bg',
                 'car_points_lost', 'n_car_off', 'bg_rej_range', 'bg_rej_overlap', 'bg_rej_scene', 'bg_rej_empty',
                 'bg_rej_rc', 'bg_rej_zrange', 'bg_rej_novis')


def bg_overlap_violations(bg, gt, car, grow=0.5):
    """§13.3 N1: per background box, the number of GT boxes, car copies and OTHER background boxes its BEV rectangle grown by
    `grow` overlaps (the loader's free-space test, recomputed)."""
    from pcdet.datasets.augmentor import relocate_render as rr
    bg = np.asarray(bg, np.float64).reshape(-1, 7)
    others = [np.asarray(gt, np.float64).reshape(-1, 7)[:, :7], np.asarray(car, np.float64).reshape(-1, 7)]
    out = np.zeros(len(bg), np.int64)
    for j in range(len(bg)):
        g = rr.bev_corners(bg[j][None], grow)[0]
        rest = np.concatenate(others + [np.delete(bg, j, 0)])
        if len(rest):
            out[j] = int(rr.rects_overlap(g, rr.bev_corners(rest)).sum())
    return out


def centres_in_rects(xy, boxes):
    """How many of the (x, y) centres lie in any of the boxes' BEV rectangles (§13.3 N3: no peak on a background copy)."""
    from pcdet.datasets.augmentor import relocate_render as rr
    xy = np.asarray(xy, np.float64).reshape(-1, 2); b = np.asarray(boxes, np.float64).reshape(-1, 7)
    if not len(xy) or not len(b):
        return 0
    return int(np.any(np.stack([rr.points_in_rect(xy, bx, 0.0) for bx in b]), 0).sum())


def bg_rate_table(car_r, bg_r, edges=(0, 10, 20, 30, 40, 50, 60, 70, 75.2), tol=0.10):
    """§13.3 N2: [(lo, hi, n_car, n_bg, ratio, pass)] per bin and overall (lo = hi = None)."""
    car_r, bg_r = np.asarray(car_r, np.float64), np.asarray(bg_r, np.float64)
    rows = []
    for lo, hi in list(zip(edges[:-1], edges[1:])) + [(None, None)]:
        mc = np.ones(len(car_r), bool) if lo is None else (car_r >= lo) & (car_r < hi)
        mb = np.ones(len(bg_r), bool) if lo is None else (bg_r >= lo) & (bg_r < hi)
        nc, nb = int(mc.sum()), int(mb.sum())
        ratio = nb / nc if nc else np.nan
        rows.append((lo, hi, nc, nb, ratio, bool(nc > 0 and abs(ratio - 1.0) <= tol + 1e-12)))
    return rows


def bg_frame_checks(a, ds, i, seed_i, d, rec, B, A, P, CAP, BGM, rr_cfg, head, side, plat, bgrows, sheet, n_check_hm=False):
    """One gate frame of v8: the background copies' rows (N1 violations, points, the low share), then a paired car-only pass
    of the same frame and seed (N3: the car copies, the final gt_boxes and, on the dump frames, the heatmaps), and the
    emptied counts. Returns the BG_FRAME_COLS row."""
    import torch
    bgc = rec.get('bg_copies', [])
    car_boxes = np.array([c['box'] for c in rec['copies']], np.float64).reshape(-1, 7)
    bg_boxes = np.array([c['box'] for c in bgc], np.float64).reshape(-1, 7)
    viol = bg_overlap_violations(bg_boxes, B['gt_boxes'][:, :7], car_boxes, float(rr_cfg.BEV_MARGIN))
    for j, c in enumerate(bgc):
        fp = np.asarray(P[c['point_index'], :3], np.float64)
        low = float(np.mean(fp[:, 2] <= c['bottom'] + float(rr_cfg.CLEARANCE))) if len(fp) else np.nan
        bgrows.append([plat, i, c['r_new'], c['n_rendered'], c['n_visible'], c['n_kept'], c['n_final'], c['bottom'], low,
                       float(c['ground_kind']), int(viol[j]), c['tries']])
    cin = centres_in_rects(np.concatenate([B['gt_boxes'][:, :2], car_boxes[:, :2]]), bg_boxes)
    nb0 = len(B['gt_boxes'])
    cnt_b = in_box_counts(B['points'], B['gt_boxes']); cnt_a = in_box_counts(A['points'], A['gt_boxes'][:nb0])
    emp_on = int(((cnt_b >= 1) & (cnt_a == 0)).sum())
    if len(sheet) < 6 and len(bgc):
        sheet.append(dict(frame=i, points=np.asarray(P[:, :3], np.float32)[::4], car=car_boxes, bg=bg_boxes,
                          bg_points=[np.asarray(P[c['point_index'], :3], np.float32) for c in bgc]))
    # the paired car-only pass
    BGM['off'] = True
    try:
        CAP.clear(); np.random.seed(seed_i)
        d0 = ds[i]
        rec0, A0 = CAP['rec'], CAP['after']
    finally:
        BGM['off'] = False
    car0 = np.array([c['box'] for c in rec0['copies']], np.float64).reshape(-1, 7)
    same_car = bool(car0.shape == car_boxes.shape and np.array_equal(car0, car_boxes))
    same_gt = bool(np.array_equal(d0['gt_boxes'], d['gt_boxes']))
    hm_eq = np.nan
    if n_check_hm:
        t1 = head.assign_targets(torch.from_numpy(d['gt_boxes'][None]).float().clone(), feature_map_size=[side, side])
        t0 = head.assign_targets(torch.from_numpy(d0['gt_boxes'][None]).float().clone(), feature_map_size=[side, side])
        hm_eq = float(all(torch.equal(x, y) for x, y in zip(t1['heatmaps'], t0['heatmaps'])))
    cnt_a0 = in_box_counts(A0['points'], A0['gt_boxes'][:nb0])
    emp_off = int(((cnt_b >= 1) & (cnt_a0 == 0)).sum())
    fin_on = np.array([c['n_final'] for c in rec['copies']]); fin_off = np.array([c['n_final'] for c in rec0['copies']])
    if same_car and len(fin_on):
        emptied_by_bg = int(((fin_off > 0) & (fin_on == 0)).sum()); lost = int((fin_off - fin_on).sum())
    else:
        emptied_by_bg, lost = -1, -1
    b = rec.get('bg', {})
    rj = b.get('reject', {})
    return [plat, i, len(rec['copies']), len(bgc), b.get('n_attempted', 0), b.get('n_no_place', 0), b.get('n_zero', 0),
            float(same_car), float(same_gt), hm_eq, cin, emp_on, emp_off, emptied_by_bg, lost, len(rec0['copies'])] + \
        [rj.get(k, 0) for k in ('range', 'overlap', 'scene', 'empty', 'rc', 'zrange', 'novis')]


def bg_sheet(path, key, sheet, lat):
    """§13.3 N4: per frame, the BEV (scene grey, car copies blue, background copies orange) and an azimuth x inclination view
    of the background copies' points about the virtual sensor (the lattice rows)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pcdet.datasets.augmentor import relocate_render as rr
    n = max(len(sheet), 1)
    fig, ax = plt.subplots(n, 2, figsize=(14, 4.6 * n), squeeze=False)
    for r, f in enumerate(sheet):
        a0, a1 = ax[r]
        p = f['points']; m = np.hypot(p[:, 0], p[:, 1]) < 76
        a0.scatter(p[m, 0], p[m, 1], s=0.05, c='0.6', rasterized=True)
        for bx, col in [(b, 'tab:blue') for b in f['car']] + [(b, 'tab:orange') for b in f['bg']]:
            c = rr.bev_corners(bx[None])[0]; c = np.r_[c, c[:1]]
            a0.plot(c[:, 0], c[:, 1], c=col, lw=1.0)
        for q in f['bg_points']:
            a0.scatter(q[:, 0], q[:, 1], s=1.0, c='tab:orange')
        a0.set_aspect('equal'); a0.set_xlim(-76, 76); a0.set_ylim(-76, 76)
        a0.set_title(f'{key} info {f["frame"]}: car copies (blue boxes) and background copies (orange, no box in the labels)',
                     fontsize=8)
        for q in f['bg_points']:
            d = q.astype(np.float64) - np.array([0.0, 0.0, lat['height']])
            az = np.degrees(np.arctan2(d[:, 1], d[:, 0])); inc = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
            a1.scatter(az, inc, s=2.0)
        a1.set_xlabel('azimuth (deg)'); a1.set_ylabel('inclination about the virtual TOP (deg)')
        a1.set_title('background copies: rendered points (each colour one copy); horizontal stripes = lattice rows', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def bg_donor_sheet(path, key, bgdb, n=24, seed=0):
    """§13.3 (reported): 24 background donors in their box frame, side (x-z) and top (x-y) views."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids = np.sort(np.random.default_rng(seed).choice(bgdb.n, min(n, bgdb.n), replace=False))
    fig, ax = plt.subplots(len(ids) // 4 + (len(ids) % 4 > 0), 8, figsize=(24, 2.4 * (len(ids) // 4 + 1)), squeeze=False)
    for k, j in enumerate(ids):
        q = np.asarray(bgdb.points(int(j)), np.float64); s = bgdb.size[int(j)]
        r, c = divmod(k, 4)
        for v, (u, w, lab) in enumerate(((0, 2, 'x-z'), (0, 1, 'x-y'))):
            a_ = ax[r][2 * c + v]
            a_.scatter(q[:, u], q[:, w], s=0.3, c=q[:, 3] if q.shape[1] > 3 else 'k', cmap='viridis')
            a_.set_xlim(-s[0] / 2 - 0.2, s[0] / 2 + 0.2)
            a_.set_ylim((-s[2] / 2 - 0.1, s[2] / 2 + 0.1) if w == 2 else (-s[1] / 2 - 0.2, s[1] / 2 + 0.2))
            a_.set_title(f'donor {j} {lab}, {len(q)} pts', fontsize=7)
    fig.suptitle(f'{key}: background donors (box frame; colour = intensity)', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


def n2_extra_frames(n_infos, gate, n=1400):
    """§13 v8b (T05 §5): N2's extra frames, E = floor((k + 0.5) N / n) for k < n; a frame already in the gate's set or
    already taken moves to the next unused index (wrapping)."""
    used = set(int(g) for g in gate); out = []
    for k in range(n):
        f = int(np.floor((k + 0.5) * n_infos / n))
        while f in used:
            f = (f + 1) % n_infos
        used.add(f); out.append(f)
    return out


class _StopAfterRR(Exception):
    pass


def run_n2(a):
    """§13 v8b: N2's extra 1,400 frames per platform (sharded), recording only the car and background copies' ranges and
    counts; the pipeline stops after RR (the gate's per-frame seeds)."""
    from pcdet.datasets.augmentor import relocate_render as rr
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plat = PLAT_ID[a.key]
    cfg, dcfg, ds = build(a.cfg, a.key)
    aug = ds.data_augmentor
    q0 = aug.data_augmentor_queue[0]
    assert getattr(q0, 'func', None) is not None and q0.func.__name__ == 'relocate_render', 'RR is not first'
    rr_cfg = q0.keywords['config']; lat = q0.keywords['lattice']; db = q0.keywords['db']
    assert rr.bg_db_from_config(rr_cfg) is not None, 'the N2 mode is for background-copy configs'
    CAP = {}

    def rr_step(data_dict=None):
        o, rec = rr.relocate_render_scene_v2(data_dict, rr_cfg, lat, db, rng=np.random, keep_points=False)
        CAP['rec'] = rec
        raise _StopAfterRR
    aug.data_augmentor_queue = [rr_step] + list(aug.data_augmentor_queue[1:])
    N = len(ds)
    frames = n2_extra_frames(N, [k * (N // 200) for k in range(200)])[a.shard::a.nshards][:a.max_frames]
    car_r, bg_r, rows = [], [], []
    t0 = time.time()
    for n, i in enumerate(frames):
        np.random.seed((plat * 1000003 + i) % (2 ** 31))
        CAP.clear()
        try:
            ds[i]
        except _StopAfterRR:
            pass
        rec = CAP['rec']; b = rec.get('bg', {})
        cr = [c['r_new'] for c in rec['copies']]; br = [c['r_new'] for c in rec.get('bg_copies', [])]
        car_r += cr; bg_r += br
        rows.append([plat, i, len(cr), len(br), b.get('n_attempted', 0), b.get('n_no_place', 0), b.get('n_zero', 0)])
        if n % 50 == 0:
            print(f'[rr_gate n2] {n + 1}/{len(frames)} frames, car {len(car_r)}, background {len(bg_r)}, '
                  f'{time.time() - t0:.0f} s', flush=True)
    res = dict(key=a.key, plat=plat, frames=frames, car_r=np.array(car_r), bg_r=np.array(bg_r),
               rows=np.array(rows, np.float64).reshape(-1, 7), node=os.uname().nodename)
    pickle.dump(res, open(out / f'n2_v7_{a.key}_{a.shard:02d}.pkl', 'wb'))
    print(f'[rr_gate n2] done {a.key} shard {a.shard}: {len(frames)} frames, car {len(car_r)}, background {len(bg_r)}',
          flush=True)


def report_v8(res, cp, C, PL, verdict, D=None):
    """§13.3: N1-N4 (deciding), the reported items, and the launch decision for v8."""
    import glob
    rows = np.concatenate([r['bg']['rows'] for r in res]); fr = np.concatenate([r['bg']['frames'] for r in res])
    R = {k: j for j, k in enumerate(BG_ROW_COLS)}; F = {k: j for j, k in enumerate(BG_FRAME_COLS)}
    print('\n== v8 (§13.3): background copies ==')
    for p in PL:
        f = fr[fr[:, 0] == p]
        print(f"{PLAT_NAME[p]}: {len(f)} frames; car copies {int(f[:, F['n_car']].sum())}, background copies "
              f"{int(f[:, F['n_bg']].sum())} (attempted {int(f[:, F['bg_attempted']].sum())}, no placement "
              f"{int(f[:, F['bg_no_place']].sum())}, 0 points {int(f[:, F['bg_zero']].sum())}); background rejected tries "
              + ' / '.join(f"{k} {int(f[:, F['bg_rej_' + k]].sum())}" for k in ('range', 'overlap', 'scene', 'empty', 'rc',
                                                                               'zrange', 'novis')))
    n1 = int(rows[:, R['n1_viol']].sum()) if len(rows) else 0
    verdict_v8 = {'N1': n1 == 0}
    print(f"N1: BEV overlaps of background copies (grown 0.5 m) with GT boxes, car copies or other background copies: {n1} "
          f"-> {'PASS' if n1 == 0 else 'FAIL'}")
    car_all = [cp[:, C['r_new']]]; bg_all = [rows[:, R['r_new']] if len(rows) else np.zeros(0)]
    car_p = {p: [cp[cp[:, C['plat']] == p, C['r_new']]] for p in PL}
    bg_p = {p: [rows[rows[:, 0] == p, R['r_new']] if len(rows) else np.zeros(0)] for p in PL}
    n2 = [pickle.load(open(f, 'rb')) for f in sorted(Path(D).glob('n2_v7_*.pkl'))] if D is not None else []
    if n2:                                           # §13 v8b: N2 on the gate's 200 + 1,400 extra frames per platform
        for z in n2:
            car_all.append(z['car_r']); bg_all.append(z['bg_r'])
            car_p[z['plat']].append(z['car_r']); bg_p[z['plat']].append(z['bg_r'])
        nfr = {p: int(sum(len(z['frames']) for z in n2 if z['plat'] == p)) for p in PL}
        print(f"N2 sample: the gate's 200 frames + extra frames per platform {nfr} ({len(n2)} shard files)")
    car_all, bg_all = np.concatenate(car_all), np.concatenate(bg_all)
    print('N2: background / car copies per 10 m bin, pooled over both platforms (bar +-10%)')
    print('| bin | car copies | background copies | ratio | |')
    print('|---|---|---|---|---|')
    ok2 = True
    for lo, hi, nc, nb, ratio, good in bg_rate_table(car_all, bg_all):
        ok2 &= good
        print(f"| {'all' if lo is None else f'{lo:g}-{hi:g} m'} | {nc} | {nb} | {ratio:.3f} | {'pass' if good else 'FAIL'} |")
    for p in PL:
        t = bg_rate_table(np.concatenate(car_p[p]), np.concatenate(bg_p[p]))
        print(f"  {PLAT_NAME[p]} (reported): " + ', '.join(f"{'all' if lo is None else f'{lo:g}-{hi:g}'} {ratio:.3f}"
                                                            for lo, hi, nc, nb, ratio, good in t))
    verdict_v8['N2'] = ok2; print(f"N2 {'PASS' if ok2 else 'FAIL'}")
    hm = fr[np.isfinite(fr[:, F['hm_equal']]), F['hm_equal']]
    same_car = bool(fr[:, F['same_car_boxes']].all()); same_gt = bool(fr[:, F['same_final_gt']].all())
    cin = int(fr[:, F['centres_in_bg']].sum())
    ok3 = same_car and same_gt and len(hm) > 0 and bool(hm.all()) and cin == 0 and \
        all(int((fr[:, 0] == p).sum()) and int(np.isfinite(fr[fr[:, 0] == p, F['hm_equal']]).sum()) >= 1 for p in PL)
    per = ', '.join(f"{PLAT_NAME[p]} {int(np.isfinite(fr[fr[:, 0] == p, F['hm_equal']]).sum())}" for p in PL)
    print(f"N3: paired car-only pass of every gate frame (same seed): car copies identical in "
          f"{int(fr[:, F['same_car_boxes']].sum())} / {len(fr)} frames, final gt_boxes identical in "
          f"{int(fr[:, F['same_final_gt']].sum())} / {len(fr)}; CenterHead heatmaps identical in {int(hm.sum())} / {len(hm)} "
          f"dump frames ({per}); GT / car-copy centres inside a background rectangle {cin} -> {'PASS' if ok3 else 'FAIL'}")
    verdict_v8['N3'] = ok3
    V8 = Path('/home/koyama/data/relocate_render_20261018/v8')
    ok4 = True; lines = []
    for k in ('NUSCENES_N008', 'NUSCENES_N015'):
        fs = sorted(glob.glob(str(V8 / f'merge_{k}_*.txt')))
        txt = open(fs[-1]).read() if fs else ''
        san = [ln for ln in txt.splitlines() if 'SANITY' in ln]
        ok4 &= bool(san) and san[-1].rstrip().endswith('PASS')
        lines += [f'  {k}: ' + (san[-1] if san else 'no merge output')]
    print('N4 (database part, deciding): every stored donor point above its local ground + 0.3 m')
    for ln in lines:
        print(ln)
    sheets = [r['bg']['sheet'] for r in res]
    print(f"N4 (sheet, read by eye): {sheets}; donors (reported): {[r['bg']['donors'] for r in res]}")
    verdict_v8['N4_db'] = ok4
    print('\nreported: background vs car copies, points per copy by r\' bin (median rendered / visible / kept / final; n)')
    for lo, hi in ((0, 20), (20, 40), (40, 50), (50, 60), (60, 75.2)):
        mb = (rows[:, R['r_new']] >= lo) & (rows[:, R['r_new']] < hi) if len(rows) else np.zeros(0, bool)
        mc = (cp[:, C['r_new']] >= lo) & (cp[:, C['r_new']] < hi)
        med = lambda x: float(np.median(x)) if len(x) else np.nan
        print(f"  {lo:g}-{hi:g} m: background {med(rows[mb, R['n_rendered']]):.0f} / {med(rows[mb, R['n_visible']]):.0f} / "
              f"{med(rows[mb, R['n_kept']]):.0f} / {med(rows[mb, R['n_final']]):.0f} ({int(mb.sum())}); car "
              f"{med(cp[mc, C['n_rendered']]):.0f} / {med(cp[mc, C['n_visible']]):.0f} / {med(cp[mc, C['n_kept']]):.0f} / "
              f"{med(cp[mc, C['n_final']]):.0f} ({int(mc.sum())})")
    if len(rows):
        print(f"reported: background points within 0.3 m of their copy's bottom (ray snap): mean share "
              f"{np.nanmean(rows[:, R['low_share']]):.3%}, copies with any {np.mean(rows[:, R['low_share']] > 0):.1%}; ground "
              f"E5a / cells / plane {np.mean(rows[:, R['ground_kind']] == 0):.1%} / {np.mean(rows[:, R['ground_kind']] == 1):.1%}"
              f" / {np.mean(rows[:, R['ground_kind']] == 2):.1%}")
    for p in PL:
        f = fr[fr[:, 0] == p]
        print(f"reported, {PLAT_NAME[p]}: originals emptied by RR, v8 {int(f[:, F['orig_emptied_on']].sum())} against car copies "
              f"only {int(f[:, F['orig_emptied_off']].sum())}; car copies emptied by background copies "
              f"{int(f[f[:, F['car_emptied_by_bg']] >= 0, F['car_emptied_by_bg']].sum())}, car-copy points lost to them "
              f"{int(f[f[:, F['car_points_lost']] >= 0, F['car_points_lost']].sum())}")
    go = all(verdict_v8.values())
    print(f"\nLAUNCH DECISION v8 (N1, N2, N3, N4 database; the N4 sheet is read by eye): "
          f"{ {k: ('PASS' if v else 'FAIL') for k, v in verdict_v8.items()} } -> {'PASS' if go else 'FAIL'}")


def run_v2(a):
    import torch
    from scipy.spatial import cKDTree
    from lattice_depth_ceiling import travel_direction, sector_of
    from source_coverage import lattice_coverage, descriptors, crop
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.rerender_utils import to_lattice
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plat = PLAT_ID[a.key]
    cfg, dcfg, ds = build(a.cfg, a.key)
    aug = ds.data_augmentor
    q0 = aug.data_augmentor_queue[0]
    assert getattr(q0, 'func', None) is not None and q0.func.__name__ == 'relocate_render', 'RR is not first'
    rr_cfg = q0.keywords['config']; lat = q0.keywords['lattice']; db = q0.keywords['db']
    version = int(rr_cfg.VERSION)
    assert version in (2, 3, 4, 5, 6, 7) and db is not None
    assert list(rr_cfg.POINT_CLOUD_RANGE) == list(dcfg.POINT_CLOUD_RANGE), 'RR range != the dataset range'
    print(f"[rr_gate v2] {a.key}: {len(ds)} infos; database {db.n} instances ({db.points_path}); lattice "
          f"{lat['n_segments']} train segments; queue {[getattr(f, 'func', f).__name__ for f in aug.data_augmentor_queue]}",
          flush=True)
    CAP = {}

    bg_on = rr.bg_db_from_config(rr_cfg) is not None          # §13.3: v8, background copies
    rr_cfg_nobg = None
    if bg_on:
        from easydict import EasyDict as _ED
        rr_cfg_nobg = _ED(dict(rr_cfg))
        for _k in ('BG_DB_POINTS', 'BG_DB_INDEX', 'BG_DB_NORMALS'):
            rr_cfg_nobg[_k] = None
    BGM = {'off': False}

    def rr_step(data_dict=None):
        CAP['before'] = {k: np.copy(data_dict[k]) for k in ('points', 'gt_boxes', 'gt_names', 'gt_boxes_mask')}
        o, rec = rr.relocate_render_scene_v2(data_dict, rr_cfg_nobg if BGM['off'] else rr_cfg, lat, db, rng=np.random,
                                             keep_points=True)
        CAP['rec'] = rec
        CAP['after'] = {k: np.copy(o[k]) for k in ('points', 'gt_boxes', 'gt_names', 'gt_boxes_mask')}
        return o
    fwd0 = aug.forward

    def fwd(data_dict):
        o = fwd0(data_dict)
        CAP['post'] = np.copy(o['gt_boxes'])
        return o
    queue_rr = list(aug.data_augmentor_queue)
    aug.data_augmentor_queue = [rr_step] + queue_rr[1:]
    aug.forward = fwd
    N = len(ds); step = N // a.n
    frames = [k * step for k in range(a.n)][:a.max_frames]
    th, nc, h = lat['thetas'], lat['n_cols'], lat['height']
    head, side, cell = head_for(cfg, dcfg)
    copies, frows, dumps, imgs = [], [], [], []
    d1v5, d1v5_drop = [], 0
    bgrows, bgframes, sheet = [], [], []
    t_start = time.time()
    for n_done, i in enumerate(frames):
        seed_i = (plat * 1000003 + i) % (2 ** 31)
        np.random.seed(seed_i)
        CAP.clear()
        d = ds[i]
        rec, B, A = CAP['rec'], CAP['before'], CAP['after']
        fwdv = travel_direction(ds.infos[i]); moving = fwdv is not None
        fv = fwdv if moving else np.array([1.0, 0.0])
        nb0 = len(B['gt_boxes']); k = rec['n_placed']
        P = A['points']
        same = bool(np.array_equal(A['gt_boxes'][:nb0], B['gt_boxes']) and
                    np.array_equal(A['gt_names'][:nb0], B['gt_names']) and
                    np.array_equal(A['gt_boxes_mask'][:nb0], B['gt_boxes_mask']))
        cb = A['gt_boxes'][nb0:].astype(np.float64)
        coll = 0
        for j in range(k):
            others = np.concatenate([B['gt_boxes'][:, :7].astype(np.float64), cb[:j, :7]])
            coll += int(len(others) > 0 and rr.rects_overlap(rr.bev_corners(cb[j:j + 1])[0], rr.bev_corners(others)).any())
        cnt_b = in_box_counts(B['points'], B['gt_boxes']); cnt_a = in_box_counts(P, A['gt_boxes'][:nb0])
        tcol = rr_cfg.TIME_COLUMN
        copy_t = np.concatenate([P[c['point_index'], tcol] for c in rec['copies']]) if k else np.zeros(0)
        gp = ground_plane(B['points'][:, :3], seed=i)
        pl = gp if gp is not None else np.full(3, np.nan)
        rp = rec['plane'] if rec['plane'] is not None else np.full(3, np.nan)
        rj = rec['reject']
        frows.append([plat, i, moving, rec['n_boxes'], rec['plane'] is not None, rec['n_attempted'], k,
                      rec['n_no_place'], rec['n_zero'], rj['range'], rj['overlap'], rj['scene'], rj['empty'],
                      rj['visibility'], rj['rc'], same, coll, int(((cnt_b >= 1) & (cnt_a == 0)).sum()),
                      int((cnt_a < cnt_b).sum()), len(B['points']), len(P),
                      float(B['points'][:, tcol].max()) if B['points'].shape[1] > tcol else np.nan,
                      float(np.abs(copy_t).max()) if len(copy_t) else 0.0,
                      float(np.degrees(np.arctan(pl[0]))), float(np.degrees(np.arctan(pl[1]))), float(pl[2]),
                      float(np.degrees(np.arctan(rp[0]))), float(np.degrees(np.arctan(rp[1])))]
                     + ([rj['zrange']] if version >= 4 else []) + ([rj['novis']] if version >= 5 else []))
        if version >= 7:                             # §12.3 D1 v5 on the source GT cars of this frame (before RR)
            rw, dr = d1v5_rows(B['points'], B, rr_cfg, rec.get('plane40'), plat, i)
            d1v5 += rw; d1v5_drop += dr
        if k:
            sec = sector_of(cb[:, :2], fv)
            iscopy = np.zeros(len(P), bool)
            for c in rec['copies']:
                iscopy[c['point_index']] = True
            scene = P[~iscopy, :3].astype(np.float64)
            scene_lat = to_lattice(scene, h, th, nc)
            scene_hz = scene[:, 2] - (pl[0] * scene[:, 0] + pl[1] * scene[:, 1] + pl[2])
            stree = cKDTree(scene[:, :2]) if version >= 3 else None
            gcells = rr.GroundCells(B['points'][:, :3], rr_cfg.GROUND_CELL, int(rr_cfg.GROUND_CELL_MIN_PTS)) \
                if version >= 4 else None
            for j, c in enumerate(rec['copies']):
                ru = c['rendered_unsparsified']
                cloud = np.concatenate([scene, ru]); idx = [np.arange(len(scene), len(scene) + len(ru))]
                cov = lattice_coverage(cloud, cb[j:j + 1, :7], idx, th, nc, h)[0]
                fp = P[c['point_index'], :3].astype(np.float64)
                dmin = cKDTree(c['placed']).query(fp, k=1)[0] if len(fp) else np.zeros(0)
                bx = c['box']
                off = (bx[2] - bx[5] / 2) - (pl[0] * bx[0] + pl[1] * bx[1] + pl[2])
                d2v2, d2v1 = blocked_share_v2(c['placed'], scene_lat, scene_hz, lat)
                g = crop(fp); dsc = descriptors(g)
                row = [plat, i, moving, c['r_new'], c['phi_new'], sec[j], c['tries'], c['rc_loader'],
                       c['vis_share_loader'], c['n_db'], c['n_rendered'], c['n_visible'], c['p_keep'],
                       c['n_kept'], c['n_final'], cov[0], cov[1], cov[2], int((dmin > PHANTOM_M).sum()), len(fp),
                       off, d2v2, d2v1, dsc[0], dsc[1], dsc[2], dsc[3], c['instance']]
                if version >= 3:                     # D1 v3: the independent local reference (§7.2)
                    near = np.asarray(stree.query_ball_point(bx[:2], float(np.hypot(bx[3], bx[4]) / 2 + 3.1)), np.int64)
                    ref, nref = rr.ground_reference(scene[near], bx)
                    row += [float(c['ground_fallback']), c['bottom'], ref, nref]
                if version >= 4:                     # D1 v4: v3's ground-cell height as the reference (§8.2)
                    v = gcells.local(bx[0], bx[1], rr_cfg.GROUND_RADIUS)
                    row += [float(np.percentile(v, rr_cfg.GROUND_PCT)) if len(v) >= int(rr_cfg.GROUND_MIN_CELLS) else np.nan]
                if version >= 5:                     # §9.3: realised descriptors of the copy (kept points in the final cloud)
                    from rr_v5_fit import object_descriptors
                    from pcdet.datasets.rerender_utils import to_lattice as _tl
                    kr, kc, _, kv = _tl(fp, h, th, nc)
                    kpix = (kr * nc + kc)[kv]
                    vsp = c['sil_vis_pix']
                    hs, vs, rs, ls, fl = object_descriptors(vsp // nc, c['sil_vis_relh'], np.isin(vsp, kpix).astype(float)) \
                        if len(vsp) else (np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), np.nan)
                    row += [c['vis_frac'], fl] + list(hs) + list(vs) + list(rs) + list(ls)
                if version >= 7:                     # §12.1: 0 E5a, 1 cell fallback, 2 plane
                    row += [float(c['ground_kind'])]
                copies.append(row)
        if bg_on:                                    # §13.3 N1-N4, with a paired car-only pass of the same frame and seed
            bgframes.append(bg_frame_checks(a, ds, i, seed_i, d, rec, B, A, P, CAP, BGM, rr_cfg, head, side, plat, bgrows,
                                            sheet, n_check_hm=len(dumps) < a.dump and k > 0))
        if k and len(dumps) < a.dump:
            dumps.append(dump_frame_v2(d, CAP, rec, head, side, cell, cfg, i))
        if k >= 2 and len(imgs) < a.images:
            pth = out / f'bev_v{version}_{a.key}_{i:05d}.png'
            image_v2(pth, f'{a.key} ({PLAT_NAME[plat]}) train info {i}: relocate-and-render v{version}, before / after', B,
                     A, rec, lat)
            imgs.append(str(pth))
        if n_done % 20 == 0:
            print(f'[rr_gate v2] {n_done + 1}/{len(frames)} frames, {len(copies)} copies, {time.time() - t_start:.0f} s',
                  flush=True)
    aug.forward = fwd0
    g4 = []
    q_nobg = None
    if bg_on:                                        # §13.3: the car-only configuration (29690's) for G4
        import functools as _ft
        q_nobg = _ft.partial(q0.func, config=rr_cfg_nobg, lattice=lat, db=db)
    for i in frames[::2][:a.g4]:
        seed = (plat * 1000003 + i) % (2 ** 31)
        aug.data_augmentor_queue = queue_rr[1:]
        np.random.seed(seed); ds[i]
        np.random.seed(seed); t0 = time.perf_counter(); ds[i]; t_off = time.perf_counter() - t0
        aug.data_augmentor_queue = queue_rr
        np.random.seed(seed); t0 = time.perf_counter(); ds[i]; t_on = time.perf_counter() - t0
        row = [plat, i, t_off, t_on]
        if bg_on:
            aug.data_augmentor_queue = [q_nobg] + queue_rr[1:]
            np.random.seed(seed); t0 = time.perf_counter(); ds[i]; row.append(time.perf_counter() - t0)
            aug.data_augmentor_queue = queue_rr
        g4.append(row)
    ncol = len(V2_COPY_COLS) + len(EXTRA_COLS[version])
    res = dict(version=version, key=a.key, plat=plat, n_infos=N, frames=frames,
               copies=np.array(copies, np.float64).reshape(-1, ncol),
               frames_rows=np.array(frows, np.float64).reshape(-1, len(V2_FRAME_COLS) + len(FRAME_EXTRA.get(version, ()))),
               d1v5=np.array(d1v5, np.float64).reshape(-1, len(D1V5_COLS)), d1v5_dropped=d1v5_drop,
               g4=np.array(g4, np.float64).reshape(-1, 5 if bg_on else 4), dumps=dumps, images=imgs, db_n=db.n,
               db_points=db.points_path, node=os.uname().nodename)
    if bg_on:
        sp = out / f'bg_sheet_{a.key}.png'
        bg_sheet(sp, a.key, sheet, lat)
        dp = out / f'bg_donors_{a.key}.png'
        bg_donor_sheet(dp, a.key, rr.bg_db_from_config(rr_cfg))
        res['bg'] = dict(rows=np.array(bgrows, np.float64).reshape(-1, len(BG_ROW_COLS)),
                         frames=np.array(bgframes, np.float64).reshape(-1, len(BG_FRAME_COLS)), sheet=str(sp), donors=str(dp))
    pickle.dump(res, open(out / f'gate_v{version}_{a.key}.pkl', 'wb'))
    print(f'[rr_gate v{version}] done {a.key}: {len(frames)} frames, {len(copies)} copies, images {imgs}', flush=True)


def dump_frame_v2(d, CAP, rec, head, side, cell, cfg, i):
    out = dump_frame(d, CAP, rec, head, side, cell, cfg, i)
    out['copies'] = [dict(r_new=c['r_new'], n_db=c['n_db'], n_rendered=c['n_rendered'], n_visible=c['n_visible'],
                          p_keep=c['p_keep'], n_final_rr=c['n_final']) for c in rec['copies']]
    return out


def report_v2(a):
    D = Path(a.dir)
    version = max(v for v in (2, 3, 4, 5, 6, 7) if list(D.glob(f'gate_v{v}_*.pkl')))
    res = [pickle.load(open(p, 'rb')) for p in sorted(D.glob(f'gate_v{version}_*.pkl'))]
    cols = V2_COPY_COLS + EXTRA_COLS[version]
    C = {k: j for j, k in enumerate(cols)}
    F = {k: j for j, k in enumerate(V2_FRAME_COLS + FRAME_EXTRA.get(version, ()))}
    print(f'gate v{version}')
    cp = np.concatenate([r['copies'] for r in res]); fr = np.concatenate([r['frames_rows'] for r in res])
    g4 = np.concatenate([r['g4'] for r in res])
    PL = [p for p in (0, 1) if (fr[:, F['plat']] == p).any()]
    verdict = {}
    print('nodes:', {r['key']: r['node'] for r in res})
    for r in res:
        print(f"{r['key']}: {len(r['frames'])} frames of {r['n_infos']} (stride {r['n_infos'] // 200}), "
              f"{len(r['copies'])} copies; database {r['db_n']} instances ({r['db_points']})")
    print('\n== frames ==')
    for p in PL:
        f = fr[fr[:, F['plat']] == p]
        print(f"{PLAT_NAME[p]}: {len(f)} frames ({int(f[:, F['moving']].sum())} moving), with a plane "
              f"{int(f[:, F['has_plane']].sum())}; per frame: boxes {f[:, F['n_boxes']].mean():.1f}, copies "
              f"{f[:, F['n_placed']].mean():.2f} of {f[:, F['n_attempted']].mean():.2f} attempted (8 placed in "
              f"{int((f[:, F['n_placed']] == 8).sum())} frames); no placement {int(f[:, F['n_no_place']].sum())}, "
              f"0-point after the sparsifier {int(f[:, F['n_zero']].sum())}; rejected tries range "
              f"{int(f[:, F['rej_range']].sum())} / overlap {int(f[:, F['rej_overlap']].sum())} / scene "
              f"{int(f[:, F['rej_scene']].sum())} / empty {int(f[:, F['rej_empty']].sum())} / visibility "
              f"{int(f[:, F['rej_vis']].sum())} / RC {int(f[:, F['rej_rc']].sum())}")
        print(f"   originals bit-identical in {int(f[:, F['same']].sum())} / {len(f)}; copy collisions "
              f"{int(f[:, F['coll']].sum())}; originals emptied {int(f[:, F['emptied']].sum())}, that lost points "
              f"{int(f[:, F['lost']].sum())}; points per frame {f[:, F['pts_before']].mean():.0f} -> "
              f"{f[:, F['pts_after']].mean():.0f}; scene time-lag max {np.nanmax(f[:, F['scene_t']]):.3f}, copy "
              f"{f[:, F['copy_t']].max():.3f}; RANSAC plane slope x / y median {np.nanmedian(f[:, F['rr_sx']]):+.2f} / "
              f"{np.nanmedian(f[:, F['rr_sy']]):+.2f} deg (gate LSQ plane {np.nanmedian(f[:, F['gp_sx']]):+.2f} / "
              f"{np.nanmedian(f[:, F['gp_sy']]):+.2f})")
    ok = ~np.isnan(cp[:, C['rc_g1']])
    print(f"\n== copies: {len(cp)} (RC defined for {int(ok.sum())}); tries median {np.median(cp[:, C['tries']]):.0f}; "
          f"r' p10 / p50 / p90 {np.percentile(cp[:, C['r_new']], 10):.1f} / {np.median(cp[:, C['r_new']]):.1f} / "
          f"{np.percentile(cp[:, C['r_new']], 90):.1f} m; database points per copy median {np.median(cp[:, C['n_db']]):.0f}; "
          f"rendered {cp[:, C['n_rendered']].sum():.0f}, visible {cp[:, C['n_visible']].sum():.0f}, kept "
          f"{cp[:, C['n_kept']].sum():.0f}; p_keep min {cp[:, C['p_keep']].min():.4f} max {cp[:, C['p_keep']].max():.4f} "
          f"mean {cp[:, C['p_keep']].mean():.3f}; distinct instances {len(np.unique(cp[:, C['instance']] + 1e6 * cp[:, C['plat']]))}")
    print('\n== G1 (unsparsified renders): median RC, moving-ego frames ==')
    print('| platform | ring | sector | copies | median RC | PPP median |')
    print('|---|---|---|---|---|---|')
    passed = 0; g2 = True; g2_skip = {0: 0, 1: 0}
    for p in (0, 1):
        for lo, hi in G1_RINGS:
            for s_ in range(3):
                m = ok & (cp[:, C['plat']] == p) & (cp[:, C['moving']] == 1) & (cp[:, C['r_new']] >= lo) & \
                    (cp[:, C['r_new']] < hi) & (cp[:, C['sector']] == s_)
                med = np.median(cp[m, C['rc_g1']]) if m.any() else np.nan
                ppp, nsk = g2_cell_median(cp[m, C['ppp_g1']], skip_undefined=version >= 7)   # §12.3 (v7)
                g2_skip[p] += nsk
                passed += int(med >= 0.8); g2 &= bool(1.0 <= ppp <= 1.1)
                print(f"| {PLAT_NAME[p]} | {lo:g}-{hi:g} | {SECTORS[s_]} | {int(m.sum())} | {med:.3f} | {ppp:.3f} |"
                      + (' < 15' if m.sum() < 15 else ''))
    verdict['G1'] = passed >= 10; verdict['G2'] = g2
    print(f"G1: {passed} of 12 cells >= 0.8 -> {'PASS' if verdict['G1'] else 'FAIL'}; G2 -> {'PASS' if g2 else 'FAIL'}")
    if version >= 7:
        print(f"G2 (v7): copies skipped for an undefined PPP (no in-box point on a visible pixel), moving-ego G1 cells: "
              f"{', '.join(f'{PLAT_NAME[p]} {g2_skip[p]}' for p in PL)}")
    print('RC (G1) per r\' bin, all frames:')
    for p in PL:
        row = []
        for lo, hi in zip(RBINS[:-1], RBINS[1:]):
            m = ok & (cp[:, C['plat']] == p) & (cp[:, C['r_new']] >= lo) & (cp[:, C['r_new']] < hi)
            row.append(f"{lo:g}-{hi:g}: {np.median(cp[m, C['rc_g1']]) if m.any() else np.nan:.3f} [{int(m.sum())}]")
        print(f"  {PLAT_NAME[p]}: " + '; '.join(row))
    print(f"loader RC (acceptance) min {np.nanmin(cp[:, C['rc_loader']]):.3f}; G1 RC < 0.8 in {np.mean(cp[ok, C['rc_g1']] < 0.8):.1%} of copies")
    print('\n== G3: phantom share of kept points per r\' ring, < 1% ==')
    g3 = True
    for lo, hi in zip(G3_RINGS[:-1], G3_RINGS[1:]):
        m = (cp[:, C['r_new']] >= lo) & (cp[:, C['r_new']] < hi)
        sh = cp[m, C['n_phantom']].sum() / max(cp[m, C['n_final_g3']].sum(), 1)
        g3 &= bool(sh < 0.01)
        print(f"{lo:g}-{hi:g} m: {int(cp[m, C['n_phantom']].sum())} of {int(cp[m, C['n_final_g3']].sum())} = {100 * sh:.3f}%")
    verdict['G3'] = g3; print(f"G3 {'PASS' if g3 else 'FAIL'}")
    print('\n== G4: dataset[i] time, same frames, same node (database read from NFS) ==')
    for p in PL:
        m = g4[:, 0] == p
        print(f"{PLAT_NAME[p]}: {int(m.sum())} frames, off {g4[m, 2].sum():.1f} s, on {g4[m, 3].sum():.1f} s, ratio "
              f"{g4[m, 3].sum() / g4[m, 2].sum():.3f} (median per frame {np.median(g4[m, 3] / g4[m, 2]):.3f})")
    ratio = g4[:, 3].sum() / g4[:, 2].sum()
    if g4.shape[1] >= 5:                             # §13.3: v8 against the car-only configuration (29690's)
        for p in PL:
            m = g4[:, 0] == p
            print(f"{PLAT_NAME[p]}: car copies only {g4[m, 4].sum():.1f} s; v8 / car-only {g4[m, 3].sum() / g4[m, 4].sum():.3f}")
        print(f"pooled v8 / car-only {g4[:, 3].sum() / g4[:, 4].sum():.3f}; car-only / RR off {g4[:, 4].sum() / g4[:, 2].sum():.3f}")
    if version >= 3:
        print(f"pooled ratio {ratio:.3f} (v{version}: reported as cost, not deciding)")
    else:
        verdict['G4'] = ratio <= 1.20
        print(f"pooled ratio {ratio:.3f} -> G4 {'PASS' if verdict['G4'] else 'FAIL'}")
    print('\n== D1: |copy bottom - gate plane| per platform (median <= 0.25 m and > 1 m below 5%) ==')
    d1 = True
    for p in PL:
        o = np.abs(cp[cp[:, C['plat']] == p, C['off_plane']])
        med, big = np.nanmedian(o), np.nanmean(o > 1.0)
        d1 &= bool(med <= 0.25 and big < 0.05)
        print(f"{PLAT_NAME[p]}: median {med:.3f} m, > 0.5 m {np.nanmean(o > 0.5):.1%}, > 1 m {big:.2%}; signed median "
              f"{np.nanmedian(cp[cp[:, C['plat']] == p, C['off_plane']]):+.3f} m")
    if version >= 4:
        print(f"(v2's D1 against the gate's plane is descriptive in v{version}: it would read {'PASS' if d1 else 'FAIL'})")
        print(f"\n== D1 v4 ({'reported only in v7' if version >= 7 else 'deciding'}): |copy bottom - ground-cell reference| "
              f"among non-fallback copies with a reference ==")
        d1v4 = True
        for p in PL:
            m = cp[:, C['plat']] == p
            pop = m & (cp[:, C['fallback']] == 0) & np.isfinite(cp[:, C['d1v4_ref']])
            dev = np.abs(cp[pop, C['bottom']] - cp[pop, C['d1v4_ref']])
            med, big = (np.median(dev), np.mean(dev > 1.0)) if pop.any() else (np.nan, np.nan)
            d1v4 &= bool(med <= 0.25 and big < 0.05)
            zr = int(fr[fr[:, F['plat']] == p, F['rej_zrange']].sum())
            print(f"{PLAT_NAME[p]}: population {int(pop.sum())} of {int(m.sum())} copies; median {med:.3f} m, > 0.5 m "
                  f"{np.mean(dev > 0.5):.1%}, > 1 m {big:.2%}; signed median "
                  f"{np.median(cp[pop, C['bottom']] - cp[pop, C['d1v4_ref']]):+.3f} m; loader fallback "
                  f"{np.mean(cp[m, C['fallback']] > 0):.1%}, reference missing {np.mean(~np.isfinite(cp[m, C['d1v4_ref']])):.1%}; "
                  f"z-range rejected tries {zr}; bottom range {cp[m, C['bottom']].min():.2f} .. {cp[m, C['bottom']].max():.2f} m")
            print('| ring | copies | loader fallback | reference missing | |dev| median | > 0.5 m | > 1 m | D1 v3 (5th pct ring) |dev| median, > 1 m |')
            print('|---|---|---|---|---|---|---|---|')
            for lo, hi in ((0, 20), (20, 40), (40, 50), (50, 60), (60, 75.2)):
                mm = m & (cp[:, C['r_new']] >= lo) & (cp[:, C['r_new']] < hi)
                if not mm.any():
                    continue
                pp = mm & (cp[:, C['fallback']] == 0) & np.isfinite(cp[:, C['d1v4_ref']])
                dd = np.abs(cp[pp, C['bottom']] - cp[pp, C['d1v4_ref']])
                o3 = mm & np.isfinite(cp[:, C['d1v3_ref']])
                d3 = np.abs(cp[o3, C['bottom']] - cp[o3, C['d1v3_ref']])
                f = lambda a, fn: fn(a) if len(a) else np.nan
                print(f"| {lo:g}-{hi:g} m | {int(mm.sum())} | {np.mean(cp[mm, C['fallback']] > 0):.1%} | "
                      f"{np.mean(~np.isfinite(cp[mm, C['d1v4_ref']])):.1%} | {f(dd, np.median):.3f} m | "
                      f"{f(dd, lambda a: np.mean(a > 0.5)):.1%} | {f(dd, lambda a: np.mean(a > 1.0)):.1%} | "
                      f"{f(d3, np.median):.3f} m, {f(d3, lambda a: np.mean(a > 1.0)):.1%} |")
        if version >= 7:
            print(f"D1 v4 (v7: reported only) would read {'PASS' if d1v4 else 'FAIL'}")
            report_d1v5(res, cp, C, PL, verdict)
        else:
            verdict['D1v4'] = d1v4; print(f"D1 v4 {'PASS' if d1v4 else 'FAIL'}")
    elif version == 3:
        print(f"(v2's D1 against the gate's plane is descriptive in v3: it would read {'PASS' if d1 else 'FAIL'})")
        print('\n== D1 v3 (deciding): |copy bottom - local reference| among copies with an observed reference, per platform ==')
        d1v3 = True
        for p in PL:
            m = cp[:, C['plat']] == p
            obs = m & np.isfinite(cp[:, C['d1v3_ref']])
            dev = np.abs(cp[obs, C['bottom']] - cp[obs, C['d1v3_ref']])
            med, big = (np.median(dev), np.mean(dev > 1.0)) if obs.any() else (np.nan, np.nan)
            d1v3 &= bool(med <= 0.25 and big < 0.05)
            print(f"{PLAT_NAME[p]}: observed {int(obs.sum())} of {int(m.sum())} (ground unobserved {1 - obs.sum() / m.sum():.1%}); "
                  f"median {med:.3f} m, > 0.5 m {np.mean(dev > 0.5):.1%}, > 1 m {big:.2%}; signed median "
                  f"{np.median(cp[obs, C['bottom']] - cp[obs, C['d1v3_ref']]):+.3f} m; reference points per copy median "
                  f"{np.median(cp[m, C['d1v3_n']]):.0f}")
            for lo, hi in ((0, 20), (20, 40), (40, 50), (50, 60), (60, 75.2)):
                mm = m & (cp[:, C['r_new']] >= lo) & (cp[:, C['r_new']] < hi)
                if mm.any():
                    oo = mm & np.isfinite(cp[:, C['d1v3_ref']])
                    dd = np.abs(cp[oo, C['bottom']] - cp[oo, C['d1v3_ref']])
                    print(f"   r' {lo:g}-{hi:g}: copies {int(mm.sum())}, loader fallback {np.mean(cp[mm, C['fallback']] > 0):.1%}, "
                          f"ground unobserved {1 - oo.sum() / mm.sum():.1%}, |dev| median "
                          f"{np.median(dd) if len(dd) else np.nan:.3f} m, > 1 m {np.mean(dd > 1.0) if len(dd) else np.nan:.1%}")
            print(f"   loader fallback (plane) overall {np.mean(cp[m, C['fallback']] > 0):.1%}")
        verdict['D1v3'] = d1v3; print(f"D1 v3 {'PASS' if d1v3 else 'FAIL'}")
    else:
        verdict['D1'] = d1; print(f"D1 {'PASS' if d1 else 'FAIL'}")
    print('\n== D2: copies behind a surface in >= half their columns, per platform (< 10%) ==')
    d2 = True
    for p in PL:
        m = cp[:, C['plat']] == p
        v2s, v1s = np.nanmean(cp[m, C['d2_v2']] >= 0.5), np.nanmean(cp[m, C['d2_v1']] >= 0.5)
        d2 &= bool(v2s < 0.10)
        print(f"{PLAT_NAME[p]}: v2 definition {v2s:.2%} (median blocked share {np.nanmedian(cp[m, C['d2_v2']]):.3f}); "
              f"v1 definition, ground included (descriptive) {v1s:.2%}")
    if version >= 5:
        print(f"D2 (v{version}: reported only) would read {'PASS' if d2 else 'FAIL'}")
        report_v5_extra(cp, fr, C, F, PL, verdict, a)
    else:
        verdict['D2'] = d2; print(f"D2 {'PASS' if d2 else 'FAIL'}")
    print('\n== descriptive: copies on 27471\'s grid (kept points), p25 / p50 / p75 ==')
    for p in PL:
        for lo, hi in ((0, 20), (20, 40), (40, 50), (50, 60), (60, 75.2)):
            m = (cp[:, C['plat']] == p) & (cp[:, C['r_new']] >= lo) & (cp[:, C['r_new']] < hi)
            if not m.any():
                continue
            q = lambda k: '/'.join(f'{v:.1f}' if k == 'ppv' else f'{v:.0f}' for v in np.percentile(cp[m, C[k]], [25, 50, 75]))
            print(f"{PLAT_NAME[p]} {lo:g}-{hi:g} m [{int(m.sum())}]: n_pts {q('n_pts_grid')}, n_vox {q('n_vox')}, "
                  f"n_layers {q('n_layers')}, ppv {q('ppv')}")
    print('\n== target dump ==')
    dump_ok = True
    for r in res:
        for dd in r['dumps']:
            okd = all(x == 1 for x in dd['copy_labels']) and all(v == 1.0 for v in dd['car_peak_at_copy']) and \
                dd['positives'] == dd['peaks'] and -1 not in dd['final_rows']
            dump_ok &= okd
            print(f"{r['key']} frame {dd['frame']}: copies {dd['n_copies']}, labels {dd['copy_labels']}, Car heatmap at "
                  f"copy centres {dd['car_peak_at_copy']}; positives {dd['positives']} peaks {dd['peaks']}; copy points in "
                  f"the final cloud {dd['n_in_final_cloud']}; " + '; '.join(
                      f"r' {c['r_new']:.1f} db {c['n_db']} rendered {c['n_rendered']} visible {c['n_visible']} p "
                      f"{c['p_keep']:.2f} kept {c['n_final_rr']}" for c in dd['copies']))
    same_all = bool(fr[:, F['same']].all()) and int(fr[:, F['coll']].sum()) == 0
    verdict['dump'] = dump_ok and same_all
    print(f"target dump {'PASS' if verdict['dump'] else 'FAIL'} (originals bit-identical in all frames and no collision: {same_all})")
    print('\nimages:', [p for r in res for p in r['images']])
    print(f'\nVERDICT v{version} (deciding items; database sanity: see the merge outputs):',
          {k: ('PASS' if v else 'FAIL') for k, v in verdict.items()})
    if 'bg' in res[0]:                           # §13.3: v8 decides on N1-N4; the car path above is reported
        report_v8(res, cp, C, PL, verdict, Path(a.dir))
    elif version >= 7:                           # §12.3 amended: the launch decision from the deciding items only
        go = launch_decision_v7(verdict)
        print(f"LAUNCH DECISION v7 ({', '.join(V7_DECIDING)}; D1 v5 and D1 v4 reported only): {'PASS' if go else 'FAIL'}")


V7_DECIDING = ('G1', 'G2', 'G3', 'normals', 'dump', 'fit_check')


def launch_decision_v7(verdict):
    """§12.3 as amended 2026-10-19 21:27: the deciding items only (D1 v5 and D1 v4 are reported); a missing item fails."""
    return all(bool(verdict.get(k, False)) for k in V7_DECIDING)


def g2_cell_median(ppp, skip_undefined=False):
    """(median PPP of a G1 cell, copies skipped). v7 (§12.3) skips copies whose PPP is undefined (no in-box point on a
    visible silhouette pixel, 0 / 0); before v7 an undefined PPP made the median nan."""
    v = np.asarray(ppp, np.float64)
    if skip_undefined:
        ok = np.isfinite(v)
        return (float(np.median(v[ok])) if ok.any() else np.nan), int((~ok).sum())
    return (float(np.median(v)) if len(v) else np.nan), 0


def d1v5_summary(rw, lo=40.0, hi=75.2, est='est_loo', kind='kind_loo'):
    """(n, median |err|, signed median, share > 0.5 m, share > 1 m, kind shares (E5a, cells, plane)) over rows with
    lo <= r < hi."""
    c = {k: j for j, k in enumerate(D1V5_COLS)}
    m = (rw[:, c['r']] >= lo) & (rw[:, c['r']] < hi) & np.isfinite(rw[:, c[est]])
    e = rw[m, c[est]] - rw[m, c['gt_bottom']]; a = np.abs(e); k = rw[m, c[kind]]
    if not m.any():
        return 0, np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan, np.nan)
    return (int(m.sum()), float(np.median(a)), float(np.median(e)), float(np.mean(a > 0.5)), float(np.mean(a > 1.0)),
            tuple(float(np.mean(k == q)) for q in (0, 1, 2)))


def report_d1v5(res, cp, C, PL, verdict):
    """§12.3: D1 v5 (deciding) per platform over 40-75.2 m; per ring and the per-frame-plane variant reported; the
    copies' ground kinds."""
    rw = np.concatenate([r['d1v5'] for r in res]) if res else np.zeros((0, len(D1V5_COLS)))
    print('\n== D1 v5 (deciding): leave-one-out E5a on the gate frames\' source GT cars, |estimate - GT bottom|, 40-75.2 m, '
          'median <= 0.25 m and > 1 m < 5% in each platform ==')
    print(f"cars dropped for a None 4-40 m plane: {sum(int(r.get('d1v5_dropped', 0)) for r in res)}")
    ok = True
    for p in PL:
        q = rw[rw[:, 0] == p]
        n, med, sgn, g05, g1, ks = d1v5_summary(q)
        good = n > 0 and med <= 0.25 and g1 < 0.05
        ok &= bool(good)
        n2, med2, sgn2, g052, g12, ks2 = d1v5_summary(q, est='est_frame', kind='kind_frame')
        print(f"{PLAT_NAME[p]}: n {n}; median {med:.3f} m, signed {sgn:+.3f} m, > 0.5 m {g05:.1%}, > 1 m {g1:.2%}; kinds E5a / "
              f"cells / plane {ks[0]:.1%} / {ks[1]:.1%} / {ks[2]:.1%} -> {'PASS' if good else 'FAIL'}; with the loader's "
              f"per-frame plane (reported): median {med2:.3f} m, > 1 m {g12:.2%}")
        print('| ring | n | median abs | signed | > 0.5 m | > 1 m | E5a / cells / plane |')
        print('|---|---|---|---|---|---|---|')
        for lo, hi in ((0, 20), (20, 40), (40, 50), (50, 60), (60, 75.2)):
            n, med, sgn, g05, g1, ks = d1v5_summary(q, lo, hi)
            print(f"| {lo:g}-{hi:g} m | {n} | {med:.3f} | {sgn:+.3f} | {g05:.1%} | {g1:.1%} | "
                  f"{ks[0]:.1%} / {ks[1]:.1%} / {ks[2]:.1%} |")
        m = cp[:, C['plat']] == p
        kk = cp[m, C['ground_kind']]
        print(f"  copies' ground (reported): E5a {np.mean(kk == 0):.1%}, cell fallback {np.mean(kk == 1):.1%}, plane "
              f"{np.mean(kk == 2):.1%} of {int(m.sum())}")
    print(f"D1 v5 (reported only since the amendment of 2026-10-19 21:27, 20261018_02 §12.3; the user accepted the road-surface "
          f"error) would read {'PASS' if ok else 'FAIL'}")


def report_v5_extra(cp, fr, C, F, PL, verdict, a):
    """§9.3: the held-out fit check (deciding), normals (deciding), the copies' visible fraction, all-copies descriptors and
    the labelled comparison against Waymo val (reported)."""
    import rr_v5_fit as FT
    from box_pattern_visual import _band_of
    V5 = Path('/home/koyama/data/relocate_render_20261018/v5')
    z = np.load(Path(getattr(a, 'targets', None) or V5 / 'targets.npz'))
    T, tcols, n_fit = z['T'], [str(c) for c in z['cols']], int(z['n_fit_segments'])
    held = (T[:, 2] >= FT.UNOCC) & (T[:, 0] >= n_fit)
    th, tl, tf, tn = FT.target_stats(T, tcols, held)
    names = ['r'] + ['fill'] + [f'{n}{b}' for n in ('hit', 'vis', 'rows', 'lost') for b in range(4)]
    Tc = np.c_[cp[:, C['r_new']], cp[:, C['fill_k']]] if len(cp) else np.zeros((0, 2))
    Tc = np.c_[Tc, np.stack([cp[:, C[n]] for n in names[2:]], 1)] if len(cp) else np.zeros((0, len(names)))
    unocc = cp[:, C['vis_frac']] >= FT.UNOCC
    ch, cl, cf, cn = FT.target_stats(Tc, names, unocc)
    ah, al, af, an = FT.target_stats(Tc, names, np.ones(len(cp), bool))
    print('\n== v5 fit check (deciding): held-out targets vs the gate\'s unoccluded copies (realised), 40-75 m ==')
    print('| bin | band | targets n | copies n | hit target / copy / diff | row-lost target / copy / diff | all copies hit / lost |')
    print('|---|---|---|---|---|---|---|')
    ok = True
    for j in (2, 3, 4):
        lo, hi = FT.BINS[j]
        for k in range(4):
            dh, dl = ch[j, k] - th[j, k], cl[j, k] - tl[j, k]
            good = tn[j, k] >= 10 and cn[j, k] >= 10 and abs(dh) <= 0.05 and abs(dl) <= 0.08
            ok &= bool(good)
            print(f"| {lo:g}-{hi:g} | {k} | {int(tn[j, k])} | {int(cn[j, k])} | {th[j, k]:.3f} / {ch[j, k]:.3f} / {dh:+.3f} | "
                  f"{tl[j, k]:.3f} / {cl[j, k]:.3f} / {dl:+.3f} | {ah[j, k]:.3f} / {al[j, k]:.3f} |{'' if good else ' FAIL'}")
    verdict['fit_check'] = ok
    print(f"fit check {'PASS' if ok else 'FAIL'}")
    print('reported, all bins: hit / row-lost target (held-out) vs unoccluded copies, and fill p25/p50/p75')
    for j, (lo, hi) in enumerate(FT.BINS):
        print(f"  {lo:g}-{hi:g} m: targets n {int(tn[j].max())}, copies n {int(cn[j].max())}; hit "
              f"{np.round(th[j], 3).tolist()} vs {np.round(ch[j], 3).tolist()}; lost {np.round(tl[j], 3).tolist()} vs "
              f"{np.round(cl[j], 3).tolist()}; fill {np.round(tf[j], 3).tolist()} vs {np.round(cf[j], 3).tolist()}")
    print('\n== copies\' visible fraction (envelope) by r\' ring, p25 / p50 / p75 and share >= 0.9 ==')
    for p in PL:
        for lo, hi in ((0, 20), (20, 40), (40, 50), (50, 60), (60, 75.2)):
            m = (cp[:, C['plat']] == p) & (cp[:, C['r_new']] >= lo) & (cp[:, C['r_new']] < hi)
            if m.any():
                v = cp[m, C['vis_frac']]
                print(f"  {PLAT_NAME[p]} {lo:g}-{hi:g} m [{int(m.sum())}]: {np.round(np.percentile(v, (25, 50, 75)), 3).tolist()}, "
                      f">= 0.9 {np.mean(v >= 0.9):.1%}")
    print('\n== normals (deciding): unit length and fallback share, from the merge outputs ==')
    nok = True
    for k in ('NUSCENES_N008', 'NUSCENES_N015'):
        fs = sorted(V5.glob(f'normals_merge_{k}_*.txt'))
        txt = open(fs[-1]).read() if fs else ''
        lines = [ln for ln in txt.splitlines() if '[rr_normals]' in ln]
        nok &= bool(lines) and lines[-1].rstrip().endswith('PASS')
        for ln in lines:
            print('  ' + ln)
    verdict['normals'] = nok
    print(f"normals {'PASS' if nok else 'FAIL'}")
    uj = Path('/home/koyama/data/relocate_render_20261018/v6/unet_b.json')
    if getattr(a, 'unet_json', None) or (uj.exists() and 'gate_b' in str(a.dir) and '/v6/' in str(a.dir)):
        import json as _json
        u = _json.load(open(getattr(a, 'unet_json', None) or uj))
        print(f"\n== arm (b') U-Net (reported): train BCE per epoch {[round(x, 4) for x in u['curve']]}; held-out per-pixel "
              f"BCE {u['heldout_bce']:.4f}, AUC {u['heldout_auc']:.4f} over {u['heldout_pixels']} valid pixels (positive "
              f"{u['heldout_pos']:.4f}); patches train {u['n_train']}, held-out {u['n_heldout']}")
    print('\n== labelled comparison (analysis only): Waymo val C unoccluded / C all / D (20261018_03 rows_CD.npz) vs copies ==')
    w = np.load('/home/koyama/data/box_pattern_visual_20261018/rows_CD.npz')
    R_, B_ = w['rows'], w['boxes']
    rc = {k: j for j, k in enumerate([str(x) for x in w['row_cols']])}
    bc = {k: j for j, k in enumerate([str(x) for x in w['box_cols']])}
    gid_rows = R_[:, rc['gid']].astype(np.int64)
    band = _band_of(R_[:, rc['relh']])
    order = np.argsort(B_[:, bc['gid']])
    gids = B_[order, bc['gid']].astype(np.int64)
    pos = np.searchsorted(gids, gid_rows)
    nb_ = len(gids)
    agg = {}
    for nm, val in (('hit', R_[:, rc['n_hit']]), ('vis', R_[:, rc['n_vis']]), ('rows', np.ones(len(R_))),
                    ('lost', (R_[:, rc['n_hit']] == 0).astype(float))):
        A = np.zeros((nb_, 4))
        np.add.at(A, (pos, band), val)
        agg[nm] = A
    Bo = B_[order]
    Tw = np.c_[Bo[:, bc['r']], Bo[:, bc['fill']], agg['hit'], agg['vis'], agg['rows'], agg['lost']]
    unw = Bo[:, bc['n_vis']] / np.maximum(Bo[:, bc['n_sil']], 1) >= 0.9
    miss = np.asarray(w['miss'], bool)[order]
    groups = (('C unoccluded', unw), ('C all', np.ones(nb_, bool)), ('D all', miss))
    for gname, gm in groups:
        gh, gl, gf, gn = FT.target_stats(Tw, names, gm)
        for j in (2, 3, 4):
            lo, hi = FT.BINS[j]
            print(f"  {gname} {lo:g}-{hi:g} m: n {int(gn[j].max())}; hit {np.round(gh[j], 3).tolist()}; lost "
                  f"{np.round(gl[j], 3).tolist()}; fill {np.round(gf[j], 3).tolist()}")
    for gname, (gh, gl, gf, gn) in (('copies unoccluded', (ch, cl, cf, cn)), ('copies all', (ah, al, af, an))):
        for j in (2, 3, 4):
            lo, hi = FT.BINS[j]
            print(f"  {gname} {lo:g}-{hi:g} m: n {int(gn[j].max())}; hit {np.round(gh[j], 3).tolist()}; lost "
                  f"{np.round(gl[j], 3).tolist()}; fill {np.round(gf[j], 3).tolist()}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='mode', required=True)
    r = sp.add_parser('run')
    r.add_argument('--cfg', required=True); r.add_argument('--key', required=True); r.add_argument('--out', required=True)
    r.add_argument('--n', type=int, default=200); r.add_argument('--g4', type=int, default=100)
    r.add_argument('--dump', type=int, default=3); r.add_argument('--images', type=int, default=2)
    r.add_argument('--max_frames', type=int, default=None)
    r.add_argument('--n2_only', action='store_true', help="§13 v8b: N2's extra frames only (copy ranges and counts)")
    r.add_argument('--shard', type=int, default=0); r.add_argument('--nshards', type=int, default=1)
    q = sp.add_parser('report'); q.add_argument('--dir', required=True)
    a = ap.parse_args()
    if a.mode == 'run':
        from easydict import EasyDict
        from pcdet.config import cfg_from_yaml_file
        c = cfg_from_yaml_file(a.cfg, EasyDict())
        v = int(c.DATA_CONFIGS[a.key].DATA_AUGMENTOR.AUG_CONFIG_LIST[0].get('VERSION', 1))
        if a.n2_only:
            run_n2(a)
        else:
            run_v2(a) if v >= 2 else run(a)
    else:
        report_v2(a) if list(Path(a.dir).glob('gate_v[234567]_*.pkl')) else report(a)


if __name__ == '__main__':
    main()
