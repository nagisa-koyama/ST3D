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
V2_FRAME_COLS = ('plat', 'frame', 'moving', 'n_boxes', 'has_plane', 'n_attempted', 'n_placed', 'n_no_place', 'n_zero',
                 'rej_range', 'rej_overlap', 'rej_scene', 'rej_empty', 'rej_vis', 'rej_rc', 'same', 'coll', 'emptied',
                 'lost', 'pts_before', 'pts_after', 'scene_t', 'copy_t', 'gp_sx', 'gp_sy', 'gp_z0', 'rr_sx', 'rr_sy')


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
    assert version in (2, 3, 4) and db is not None
    assert list(rr_cfg.POINT_CLOUD_RANGE) == list(dcfg.POINT_CLOUD_RANGE), 'RR range != the dataset range'
    print(f"[rr_gate v2] {a.key}: {len(ds)} infos; database {db.n} instances ({db.points_path}); lattice "
          f"{lat['n_segments']} train segments; queue {[getattr(f, 'func', f).__name__ for f in aug.data_augmentor_queue]}",
          flush=True)
    CAP = {}

    def rr_step(data_dict=None):
        CAP['before'] = {k: np.copy(data_dict[k]) for k in ('points', 'gt_boxes', 'gt_names', 'gt_boxes_mask')}
        o, rec = rr.relocate_render_scene_v2(data_dict, rr_cfg, lat, db, rng=np.random, keep_points=True)
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
                     + ([rj['zrange']] if version == 4 else []))
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
                if version == 4 else None
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
                if version == 4:                     # D1 v4: v3's ground-cell height as the reference (§8.2)
                    v = gcells.local(bx[0], bx[1], rr_cfg.GROUND_RADIUS)
                    row += [float(np.percentile(v, rr_cfg.GROUND_PCT)) if len(v) >= int(rr_cfg.GROUND_MIN_CELLS) else np.nan]
                copies.append(row)
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
    for i in frames[::2][:a.g4]:
        seed = (plat * 1000003 + i) % (2 ** 31)
        aug.data_augmentor_queue = queue_rr[1:]
        np.random.seed(seed); ds[i]
        np.random.seed(seed); t0 = time.perf_counter(); ds[i]; t_off = time.perf_counter() - t0
        aug.data_augmentor_queue = queue_rr
        np.random.seed(seed); t0 = time.perf_counter(); ds[i]; t_on = time.perf_counter() - t0
        g4.append([plat, i, t_off, t_on])
    ncol = len(V2_COPY_COLS) + {2: 0, 3: len(V3_EXTRA_COLS), 4: len(V4_EXTRA_COLS)}[version]
    res = dict(version=version, key=a.key, plat=plat, n_infos=N, frames=frames,
               copies=np.array(copies, np.float64).reshape(-1, ncol),
               frames_rows=np.array(frows, np.float64).reshape(-1, len(V2_FRAME_COLS) + (1 if version == 4 else 0)),
               g4=np.array(g4, np.float64).reshape(-1, 4), dumps=dumps, images=imgs, db_n=db.n,
               db_points=db.points_path, node=os.uname().nodename)
    pickle.dump(res, open(out / f'gate_v{version}_{a.key}.pkl', 'wb'))
    print(f'[rr_gate v{version}] done {a.key}: {len(frames)} frames, {len(copies)} copies, images {imgs}', flush=True)


def dump_frame_v2(d, CAP, rec, head, side, cell, cfg, i):
    out = dump_frame(d, CAP, rec, head, side, cell, cfg, i)
    out['copies'] = [dict(r_new=c['r_new'], n_db=c['n_db'], n_rendered=c['n_rendered'], n_visible=c['n_visible'],
                          p_keep=c['p_keep'], n_final_rr=c['n_final']) for c in rec['copies']]
    return out


def report_v2(a):
    D = Path(a.dir)
    version = max(v for v in (2, 3, 4) if list(D.glob(f'gate_v{v}_*.pkl')))
    res = [pickle.load(open(p, 'rb')) for p in sorted(D.glob(f'gate_v{version}_*.pkl'))]
    cols = V2_COPY_COLS + {2: (), 3: V3_EXTRA_COLS, 4: V4_EXTRA_COLS}[version]
    C = {k: j for j, k in enumerate(cols)}
    F = {k: j for j, k in enumerate(V2_FRAME_COLS + (('rej_zrange',) if version == 4 else ()))}
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
    passed = 0; g2 = True
    for p in (0, 1):
        for lo, hi in G1_RINGS:
            for s_ in range(3):
                m = ok & (cp[:, C['plat']] == p) & (cp[:, C['moving']] == 1) & (cp[:, C['r_new']] >= lo) & \
                    (cp[:, C['r_new']] < hi) & (cp[:, C['sector']] == s_)
                med = np.median(cp[m, C['rc_g1']]) if m.any() else np.nan
                ppp = np.median(cp[m, C['ppp_g1']]) if m.any() else np.nan
                passed += int(med >= 0.8); g2 &= bool(1.0 <= ppp <= 1.1)
                print(f"| {PLAT_NAME[p]} | {lo:g}-{hi:g} | {SECTORS[s_]} | {int(m.sum())} | {med:.3f} | {ppp:.3f} |"
                      + (' < 15' if m.sum() < 15 else ''))
    verdict['G1'] = passed >= 10; verdict['G2'] = g2
    print(f"G1: {passed} of 12 cells >= 0.8 -> {'PASS' if verdict['G1'] else 'FAIL'}; G2 -> {'PASS' if g2 else 'FAIL'}")
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
    if version == 4:
        print(f"(v2's D1 against the gate's plane is descriptive in v4: it would read {'PASS' if d1 else 'FAIL'})")
        print('\n== D1 v4 (deciding): |copy bottom - ground-cell reference| among non-fallback copies with a reference ==')
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

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='mode', required=True)
    r = sp.add_parser('run')
    r.add_argument('--cfg', required=True); r.add_argument('--key', required=True); r.add_argument('--out', required=True)
    r.add_argument('--n', type=int, default=200); r.add_argument('--g4', type=int, default=100)
    r.add_argument('--dump', type=int, default=3); r.add_argument('--images', type=int, default=2)
    r.add_argument('--max_frames', type=int, default=None)
    q = sp.add_parser('report'); q.add_argument('--dir', required=True)
    a = ap.parse_args()
    if a.mode == 'run':
        from easydict import EasyDict
        from pcdet.config import cfg_from_yaml_file
        c = cfg_from_yaml_file(a.cfg, EasyDict())
        v = int(c.DATA_CONFIGS[a.key].DATA_AUGMENTOR.AUG_CONFIG_LIST[0].get('VERSION', 1))
        run_v2(a) if v >= 2 else run(a)
    else:
        report_v2(a) if list(Path(a.dir).glob('gate_v[234]_*.pkl')) else report(a)


if __name__ == '__main__':
    main()
