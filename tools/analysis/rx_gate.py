"""CPU gate of row extrusion (RX), X1-X4 and the reported items, for the arms H-G0, H-G1, L-G0, W-G0 (experiments_md
20261021_02; design, cells and bars fixed in T05 §5, commits 9a7c1f5 / b479ec7; readings in 20261021_02 §1.2). CPU only.

run (per platform): the nuScenes TRAIN infos at k x floor(N / 200), k = 0..199 (RR's gate frames), numpy seed per frame
(platform x 1000003 + index) mod 2^31, loaded with the RX row config. For every frame the full dataset[i] runs five times
with the same seed: RX off (the unmodified source = the 27471-equivalent loader) and the four arms; the ROW_EXTRUSION hook
is replaced by a wrapper that runs the SAME row_extrusion with the same generator (relocate_render.bg_generator of the global
state) and keeps the record. Measured on the slot output (the cloud the augmentor receives):
- X1 / X2: per source Car box (relocate_render.gt_car_mask) the full and visible silhouette on the virtual TOP
  (box_pattern_visual.silhouette_pixels / visible_mask with the slot cloud, conv 'rr', TRAIN-median lattice), row_records'
  rows with the box's points (roiaware test): RC, visible / full, per-band descriptors;
- X3: run-length histograms of the object pixels (return > 0.3 m above the measurement surface: H's S of the slot INPUT,
  the same for every arm) at 40-50 / 50-60 / 60-75.2 m, inside a Car box / outside every box grown 0.5 m in BEV;
- X4: new points below the operator's S, Car boxes emptied by the shadow, unique voxels of the final cloud, the final
  gt_boxes and (first `--dump` frames) CenterHead heatmaps and per-class positives against RX off.
Then G4-style loader time on every second frame (100): dataset[i] with the real hook against RX off, warm cache, one core.

report: X1-X4 per arm against the Waymo reference (rx_build_library.py reference: the held-out 20% segments), the
reported items and the LAUNCH DECISION (H-G0 only).

    python analysis/rx_gate.py run --cfg <rx cfg> --key NUSCENES_N008 --out <dir> [--n 200] [--g4 100] [--dump 3]
    python analysis/rx_gate.py report --dir <dir> --ref <reference.npz> --lib <library.npz> [--targets <RR targets.npz>]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
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
GATE_ARMS = ('off', 'H-G0', 'H-G1', 'L-G0', 'W-G0')
X_BINS = ((40.0, 50.0), (50.0, 60.0), (60.0, 75.0))           # X1 / X2 cells (car BEV centre range)
RING_BINS = ((0.0, 15.0), (15.0, 25.0), (25.0, 40.0), (40.0, 50.0), (50.0, 60.0), (60.0, 75.0))
VOX_CAP = 150000
BELOW_EPS = 1e-6                  # m: a point ON S (G1 beyond) is not below it
CAR_COLS = ('plat', 'frame', 'j', 'r', 'alpha', 'side', 'n_full', 'n_vis', 'rc', 'n_vrows', 'n_pts') + \
    tuple(f'{n}{b}' for n in ('hs', 'vs', 'rs', 'ls') for b in range(4))
X4_COLS = ('plat', 'frame', 'n_in', 'n_out', 'n_new', 'n_kept', 'n_below_new', 'n_below_kept', 'n_shadow', 'emptied_car',
           'emptied_any', 'car_pts_shadowed', 'nvox', 'g1_between', 'g1_beyond', 'g1_kept', 'kept_car', 'kept_other',
           'kept_bg', 'kept_g1', 'new_car', 'new_other', 'new_bg', 'new_g1', 't_rx', 'no_plane', 'gt_same', 'in_same',
           'no_meas_plane')


# --------------------------------------------------------------------------------------------- measurements
def rows_descriptors(rows):
    """Per-object (hs, vs, rs, ls) per relh band from row_records rows (columns row, n_vis, n_hit, relh): the same as
    rr_v5_fit.object_descriptors on the row's pixels (a row's band = that of its median relh)."""
    from box_pattern_visual import _band_of
    hs, vs, rs, ls = (np.zeros(4) for _ in range(4))
    if not len(rows):
        return hs, vs, rs, ls
    band = _band_of(rows[:, 3])
    for b in range(4):
        m = band == b
        hs[b] = rows[m, 2].sum(); vs[b] = rows[m, 1].sum(); rs[b] = m.sum(); ls[b] = (rows[m, 2] == 0).sum()
    return hs, vs, rs, ls


def source_pixels(xyz, lat):
    """Descending-row pixel ids (box_pattern_visual's convention, row 0 = highest) and 3D ranges of a loader cloud on the
    virtual TOP; -1 outside the field of view."""
    from pcdet.datasets.rerender_utils import to_lattice
    th, nc, h = lat['thetas'], lat['n_cols'], lat['height']
    row, col, rng, valid = to_lattice(xyz, h, th, nc)
    return np.where(valid, (len(th) - 1 - row) * nc + col, -1), rng


def source_car_rows(box, xyz, pix, npp, lat):
    """(n_full, n_vis, rows (k, 4)) of one source box on the cloud xyz (box_pattern_visual, conv 'rr')."""
    import box_pattern_visual as BV
    from pcdet.datasets.augmentor import relocate_render as rr
    nc = lat['n_cols']
    pose = BV.sensor_pose('rr', height=lat['height'])
    sil, tin, tout, ent = BV.silhouette_pixels(box, pose, np.asarray(lat['thetas'])[::-1], 'rr', nc, full=True)
    n_full = len(np.unique(sil))
    if not n_full:
        return 0, 0, np.zeros((0, 4)), 0
    rad = float(np.hypot(box[3], box[4]) / 2 + 0.1)
    cand = np.nonzero((np.abs(xyz[:, 0] - box[0]) <= rad) & (np.abs(xyz[:, 1] - box[1]) <= rad))[0]
    inb = cand[rr.points_in_box(xyz[cand], box)]
    inbox = np.zeros(len(xyz), bool); inbox[inb] = True
    vis = BV.visible_mask(sil, tin, npp, inbox)
    n_vis = len(np.unique(sil[vis]))
    op = pix[inb]; op = op[op >= 0]
    rows = BV.row_records(sil, tin, tout, ent, vis, op, np.zeros(len(op)), npp, box, nc)
    return n_full, n_vis, rows[:, [0, 1, 2, 3]] if len(rows) else np.zeros((0, 4)), len(inb)


def measure_cars(P, boxes, car, lat, plat, frame):
    import box_pattern_visual as BV
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.augmentor import row_extrusion as RX
    xyz = np.asarray(P[:, :3], np.float64)
    pix, rng = source_pixels(xyz, lat)
    npp = BV.nearest_per_pixel(pix, rng)
    out = []
    for j in np.nonzero(car)[0]:
        b = np.asarray(boxes[j, :7], np.float64)
        res = source_car_rows(b, xyz, pix, npp, lat)
        n_full, n_vis, rows, n_pts = res
        alpha = rr.relative_view_angle(b)
        rc = float(np.mean(rows[:, 2] >= 1)) if len(rows) else np.nan
        hs, vs, rs, ls = rows_descriptors(rows)
        out.append([plat, frame, j, float(np.hypot(b[0], b[1])), alpha, float(RX.aspect_side(alpha)), n_full, n_vis, rc,
                    len(rows), n_pts] + list(hs) + list(vs) + list(rs) + list(ls))
    return out


def measure_x3(P, boxes, car, S_meas, lat):
    from rx_build_library import x3_hist
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.augmentor import row_extrusion as RX
    xyz = np.asarray(P[:, :3], np.float64)
    idx, rim, _ = RX.range_image(xyz, lat)
    has = idx >= 0
    R, C = has.shape
    flat_i = idx.ravel(); hf = flat_i >= 0
    pts = xyz[flat_i[hf]]
    z = np.full(R * C, np.nan); z[hf] = pts[:, 2]
    s = np.full(R * C, np.nan); s[hf] = S_meas(pts[:, 0], pts[:, 1])
    with np.errstate(invalid='ignore'):
        obj = (z > s + RX.GROUND_TOL).reshape(R, C)
    rho = np.full(R * C, np.nan); rho[hf] = np.hypot(pts[:, 0], pts[:, 1])
    inside = np.zeros(R * C, bool); grown = np.zeros(R * C, bool)
    if car.any():
        inside[hf] = RX.points_in_boxes(pts, boxes[car, :7]).any(1)
    for b in np.asarray(boxes[:, :7], np.float64):
        grown[hf] |= rr.points_in_rect(pts[:, :2], b, 0.5)
    return x3_hist(has, rim, obj, rho.reshape(R, C), inside.reshape(R, C), has & ~grown.reshape(R, C))


def n_voxels(points, pcr, vs):
    p = np.asarray(points[:, :3], np.float64)
    pcr = np.asarray(pcr, np.float64); vs = np.asarray(vs, np.float64)
    grid = np.round((pcr[3:] - pcr[:3]) / vs).astype(np.int64)
    c = np.floor((p - pcr[:3]) / vs).astype(np.int64)
    m = np.all((c >= 0) & (c < grid), 1)
    c = c[m]
    return int(len(np.unique((c[:, 0] * grid[1] + c[:, 1]) * grid[2] + c[:, 2])))


def heatmap_summary(gt, head, side, names):
    import torch
    t = head.assign_targets(torch.from_numpy(np.asarray(gt)[None]).float().clone(), feature_map_size=[side, side])
    hms = [h[0].numpy() for h in t['heatmaps']]
    lab = np.asarray(gt)[:, 7].astype(int) if len(gt) else np.zeros(0, int)
    return dict(hash=[hashlib.sha1(np.ascontiguousarray(h).tobytes()).hexdigest() for h in hms],
                positives={n: int((lab == c + 1).sum()) for c, n in enumerate(names)},
                peaks={n: int((hms[0][c] == 1).sum()) for c, n in enumerate(names)})


# ---------------------------------------------------------------------------------------------------------- run
def run(a):
    import rr_gate
    from rr_gate import in_box_counts
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.augmentor import row_extrusion as RX
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plat = PLAT_ID[a.key]
    cfg, dcfg, ds = rr_gate.build(a.cfg, a.key)
    rx_cfg = dcfg.get('ROW_EXTRUSION', None)
    assert rx_cfg is not None and rx_cfg.ARM == 'H' and rx_cfg.GROUND == 'G0', 'the gate runs on the H-G0 row config'
    queue = [getattr(f, 'func', f).__name__ for f in ds.data_augmentor.data_augmentor_queue]
    assert 'relocate_render' not in queue, queue
    if a.library_override:                                    # smokes only: never the binding gate run
        rx_cfg.LIBRARY = a.library_override
    lat = RX.lattice(); lib = RX.load_library(str(rx_cfg.LIBRARY))
    head, side, _ = rr_gate.head_for(cfg, dcfg)
    vs = [p for p in dcfg.DATA_PROCESSOR if p.NAME == 'transform_points_to_voxels'][0].VOXEL_SIZE
    pcr = dcfg.POINT_CLOUD_RANGE
    print(f"[rx_gate] {a.key}: {len(ds)} infos; library {rx_cfg.LIBRARY} ({len(lib.cell)} cars, q "
          f"{np.round(lib.q, 4).tolist()}); lattice {lat['n_segments']} train segments; queue {queue}", flush=True)
    orig = RX.apply_in_loader
    MODE = {'arm': None, 'keep_dict': False}
    CAP = {}

    def hook(data_dict, cfg_):
        CAP['state'] = np.random.get_state()
        CAP['in'] = {k: np.copy(data_dict[k]) for k in ('points', 'gt_boxes', 'gt_names', 'gt_boxes_mask')}
        if MODE['keep_dict']:
            CAP['in_dict'] = copy.deepcopy(data_dict)
        if MODE['arm'] is None:
            CAP['rec'] = None; CAP['out'] = CAP['in']['points']
            return data_dict
        arm, ground = MODE['arm'].split('-')
        rng = rr.bg_generator(np.random)
        car = rr.gt_car_mask(data_dict)
        t0 = time.perf_counter()
        o, rec = RX.row_extrusion(data_dict['points'], data_dict['gt_boxes'], car, arm, ground, lat, lib, rng,
                                  keep_record=True)
        rec['t'] = time.perf_counter() - t0
        data_dict['points'] = o
        CAP['rec'] = rec; CAP['out'] = np.copy(o)
        return data_dict
    RX.apply_in_loader = hook
    N = len(ds); step = N // a.n
    frames = [k * step + a.offset for k in range(a.n)][:a.max_frames]          # offset 0 = the gate frames; smokes use 1
    cars = {m: [] for m in GATE_ARMS}; x3 = {m: np.zeros((3, 2, 65), np.int64) for m in GATE_ARMS}
    x4 = {m: [] for m in GATE_ARMS}; dumps = {m: [] for m in GATE_ARMS}
    selfcheck = None
    t_start = time.time()
    for n_done, i in enumerate(frames):
        seed_i = (plat * 1000003 + i) % (2 ** 31)
        ref_in = ref_gt = S_meas = cnt_in = car_in = None
        for m in GATE_ARMS:
            MODE['arm'] = None if m == 'off' else m
            MODE['keep_dict'] = m == 'H-G0' and selfcheck is None
            np.random.seed(seed_i); CAP.clear()
            d = ds[i]
            I, rec, Pout = CAP['in'], CAP['rec'], CAP['out']
            Pin = I['points']
            if m == 'off':
                ref_in, ref_gt = Pin, np.copy(d['gt_boxes'])
                car_in = rr.gt_car_mask(I)
                plane = rr.ground_plane(np.asarray(Pin[:, :3], np.float64), RX.PLANE_RMAX)
                S_meas = RX.Surface(plane, I['gt_boxes'][car_in, :7]) if plane is not None else None
                cnt_in = in_box_counts(Pin, I['gt_boxes'])
            in_same = bool(np.array_equal(Pin, ref_in))
            gt_same = bool(np.array_equal(d['gt_boxes'], ref_gt))
            cars[m] += measure_cars(Pout, I['gt_boxes'], car_in, lat, plat, i)
            if S_meas is not None:
                x3[m] += measure_x3(Pout, I['gt_boxes'], car_in, S_meas, lat)
            cnt_out = in_box_counts(Pout, I['gt_boxes'])
            emptied = (cnt_in >= 1) & (cnt_out == 0)
            if rec is not None and not rec.get('no_points', False) and not rec['no_plane']:
                sh = np.zeros(len(Pin), bool); sh[rec['shadow_idx']] = True
                car_sh = int(in_box_counts(Pin[sh], I['gt_boxes'][car_in]).sum()) if sh.any() and car_in.any() else 0
                S_op = RX.Surface(rec['plane'], np.zeros((0, 7)) if m.startswith('W') else I['gt_boxes'][car_in, :7])
                nx = rec['new_xyz']
                below = nx[:, 2] < S_op(nx[:, 0], nx[:, 1]) - BELOW_EPS if len(nx) else np.zeros(0, bool)
                kt, nt = rec['n_kept_type'], rec['n_new_type']
                row = [rec['n_in'], rec['n_out'], rec['n_new'], rec['n_kept'], int(below.sum()),
                       int((below & rec['new_kept']).sum()), rec['n_shadow'], int((emptied & car_in).sum()),
                       int(emptied.sum()), car_sh]
                g1 = [rec['n_g1_between'], rec['n_g1_beyond'], kt[3]]
                tail = list(kt) + list(nt) + [rec['t'], float(rec['no_plane'])]
            else:
                row = [len(Pin), len(Pout), 0, 0, 0, 0, 0, int((emptied & car_in).sum()), int(emptied.sum()), 0]
                g1 = [0, 0, 0]
                tail = [0] * 8 + [rec['t'] if rec is not None else 0.0, float(rec is not None and rec['no_plane'])]
            x4[m].append([plat, i] + row + [n_voxels(d['points'], pcr, vs)] + g1 + tail
                         + [float(gt_same), float(in_same), float(S_meas is None)])
            if n_done < a.dump:
                dumps[m].append(dict(frame=i, **heatmap_summary(d['gt_boxes'], head, side, list(cfg.CLASS_NAMES))))
            if MODE['keep_dict']:                             # the real hook on the same input and global state
                st = np.random.get_state()
                np.random.set_state(CAP['state'])
                o_real = orig(copy.deepcopy(CAP['in_dict']), rx_cfg)['points']
                adv = np.random.get_state()
                selfcheck = bool(np.array_equal(o_real, Pout)) and bool(np.array_equal(adv[1], CAP['state'][1])) \
                    and adv[2] == CAP['state'][2]
                np.random.set_state(st)
                print(f'[rx_gate] self-check, the real hook reproduces the gate wrapper and leaves the global RNG: '
                      f'{selfcheck}', flush=True)
        if n_done % 10 == 0:
            print(f'[rx_gate] {n_done + 1}/{len(frames)} frames, {time.time() - t_start:.0f} s', flush=True)
    RX.apply_in_loader = orig
    # loader time: the real hook against RX off, warm cache, one core
    g4 = []
    for i in frames[::2][:a.g4]:
        seed = (plat * 1000003 + i) % (2 ** 31)
        dcfg.ROW_EXTRUSION = None
        np.random.seed(seed); ds[i]
        np.random.seed(seed); t0 = time.perf_counter(); ds[i]; t_off = time.perf_counter() - t0
        dcfg.ROW_EXTRUSION = rx_cfg
        np.random.seed(seed); t0 = time.perf_counter(); ds[i]; t_on = time.perf_counter() - t0
        g4.append([plat, i, t_off, t_on])
    res = dict(key=a.key, plat=plat, n_infos=N, frames=frames, cfg=a.cfg, library=str(rx_cfg.LIBRARY), offset=a.offset,
               cars={m: np.array(v, np.float64).reshape(-1, len(CAR_COLS)) for m, v in cars.items()}, x3=x3,
               x4={m: np.array(v, np.float64).reshape(-1, len(X4_COLS)) for m, v in x4.items()}, dumps=dumps,
               g4=np.array(g4, np.float64).reshape(-1, 4), selfcheck=selfcheck, node=os.uname().nodename)
    pickle.dump(res, open(out / f'gate_{a.key}.pkl', 'wb'))
    print(f'[rx_gate] done {a.key}: {len(frames)} frames in {time.time() - t_start:.0f} s', flush=True)


# ------------------------------------------------------------------------------------------------------- report
def _bin(r, bins):
    r = np.asarray(r, np.float64)
    out = np.full(r.shape, -1, np.int64)
    for k, (lo, hi) in enumerate(bins):
        out[(r >= lo) & (r < hi)] = k
    return out


def ref_tables(ref):
    """Reference cars: (r, side, visfrac, rc, hs, vs, rs, ls) from rx_build_library.py reference."""
    from pcdet.datasets.augmentor import row_extrusion as RX
    C, Rw = ref['cars'], ref['rows']
    rc, D = [], []
    for c in C:
        rows = Rw[int(c[7]):int(c[7]) + int(c[8])].astype(np.float64)
        rc.append(float(np.mean(rows[:, 2] >= 1)) if len(rows) else np.nan)
        D.append(np.concatenate(rows_descriptors(rows)))
    D = np.array(D).reshape(-1, 16)
    return dict(r=C[:, 2], side=RX.aspect_side(np.radians(C[:, 3])), vis=C[:, 5] / C[:, 4], rc=np.array(rc), D=D,
                seg=C[:, 0])


def src_tables(cars):
    J = {k: i for i, k in enumerate(CAR_COLS)}
    D = cars[:, J['hs0']:J['ls3'] + 1]
    with np.errstate(invalid='ignore', divide='ignore'):
        vis = cars[:, J['n_vis']] / cars[:, J['n_full']]
    return dict(plat=cars[:, J['plat']], r=cars[:, J['r']], side=cars[:, J['side']].astype(bool), vis=vis,
                rc=cars[:, J['rc']], D=D)


def band_means(D, mask):
    """Per band: (n objects with >= 1 row in the band, mean hit rate, mean row-lost rate)."""
    hs, vs, rs, ls = D[:, 0:4], D[:, 4:8], D[:, 8:12], D[:, 12:16]
    out = []
    for b in range(4):
        m = mask & (rs[:, b] >= 1)
        n = int(m.sum())
        out.append((n, float(np.mean(hs[m, b] / vs[m, b])) if n else np.nan,
                    float(np.mean(ls[m, b] / rs[m, b])) if n else np.nan))
    return out


def report(a):
    from pcdet.datasets.augmentor import row_extrusion as RX
    res = {k: pickle.load(open(Path(a.dir) / f'gate_{k}.pkl', 'rb')) for k in a.keys}
    ref = np.load(a.ref, allow_pickle=True); lib = np.load(a.lib, allow_pickle=True)
    R = ref_tables(ref)
    L = []

    def P(s=''):
        L.append(s); print(s, flush=True)
    arms = [m for m in GATE_ARMS if m != 'off']
    P('RX gate report (experiments_md 20261021_02; T05 §5 RX, X1-X4 as declared)')
    P(f"inputs: gate {', '.join(str(Path(a.dir) / f'gate_{k}.pkl') for k in a.keys)}; reference {a.ref}; library {a.lib}")
    for k, r in res.items():
        P(f"  {k}: {len(r['frames'])} frames of {r['n_infos']} (stride {r['n_infos'] // 200}), node {r['node']}, "
          f"cfg {r['cfg']}; real-hook self-check {r['selfcheck']}")
    # library
    cell = lib['cell']
    P('')
    P('LIBRARY (fit-set teacher cars, visible / full >= 0.9) and q')
    P('  cells: ' + ', '.join(f"{RX.LIB_BINS[c // 2][0]:g}-{RX.LIB_BINS[c // 2][1]:g} {'side' if c % 2 else 'end'} "
                             f"{int((cell == c).sum())}" for c in range(10)))
    P('  q: ' + ', '.join(f'{lo:g}-{hi:g} m {lib["q"][j]:.4f} ({int(lib["q_num"][j])} / {int(lib["q_den"][j])})'
                          for j, (lo, hi) in enumerate(RX.Q_BINS)))
    P(f"  reference: {len(R['r'])} held-out teacher cars with a visible silhouette; frames without a plane "
      f"{int(ref['n_noplane'])} of {int(ref['n_frames'])}")
    # self-check of the reference against RR's targets (same selection and geometry)
    try:
        T = np.load(a.targets, allow_pickle=True)
        Tt = T['T']; nfit = int(T['n_fit_segments'])
        th = Tt[(Tt[:, 0] >= nfit)]
        mine = np.c_[R['seg'], R['r'], R['vis'], R['D']]
        theirs = np.c_[th[:, 0], th[:, 1], th[:, 2], th[:, 4:20]]
        same_n = len(mine) == len(theirs)
        diff = np.nan
        if same_n:
            a1 = mine[np.lexsort(mine[:, :3].T[::-1])]; a2 = theirs[np.lexsort(theirs[:, :3].T[::-1])]
            diff = float(np.max(np.abs(a1 - a2))) if len(a1) else 0.0
        P(f'  self-check against RR targets.npz held-out rows: {len(mine)} vs {len(theirs)} cars; max abs difference '
          f'{diff}')
    except Exception as e:  # noqa: BLE001
        P(f'  self-check against RR targets.npz not run: {e!r}')
    S = {m: src_tables(np.concatenate([res[k]['cars'][m] for k in a.keys])) for m in GATE_ARMS}
    X4 = {m: np.concatenate([res[k]['x4'][m] for k in a.keys]) for m in GATE_ARMS}
    J4 = {k: i for i, k in enumerate(X4_COLS)}
    verdict = {m: {} for m in arms}
    # ---------------- X1
    P('')
    P('X1 (deciding): median RC of source Car boxes after the operator vs the reference median; |diff| <= 0.10 in >= 10 of 12')
    rb, rs_ = _bin(R['r'], X_BINS), R['side']
    ref_med = {}
    for k in range(3):
        for sd in (0, 1):
            m = (rb == k) & (rs_ == bool(sd)) & np.isfinite(R['rc'])
            ref_med[k, sd] = (float(np.median(R['rc'][m])) if m.any() else np.nan, int(m.sum()))
    P('  reference: ' + '; '.join(f"{X_BINS[k][0]:g}-{X_BINS[k][1]:g} {'side' if sd else 'end'} {v[0]:.3f} (n {v[1]})"
                                  for (k, sd), v in ref_med.items()))
    for m in GATE_ARMS:
        T_ = S[m]; sb = _bin(T_['r'], X_BINS)
        n_ok = 0; cells = []
        for p in (0, 1):
            for k in range(3):
                for sd in (0, 1):
                    mm = (T_['plat'] == p) & (sb == k) & (T_['side'] == bool(sd)) & np.isfinite(T_['rc'])
                    med = float(np.median(T_['rc'][mm])) if mm.any() else np.nan
                    d = med - ref_med[k, sd][0]
                    ok = bool(np.isfinite(d) and abs(d) <= 0.10 + 1e-12)
                    n_ok += ok
                    cells.append(f"{PLAT_NAME[p][0]} {X_BINS[k][0]:g}-{X_BINS[k][1]:g} {'S' if sd else 'E'} "
                                 f"{med:.3f} (n {int(mm.sum())}, {d:+.3f}{'' if ok else ' X'})")
        tag = 'reported' if m == 'off' else ('PASS' if n_ok >= 10 else 'FAIL')
        P(f'  {m}: {n_ok} of 12 cells within 0.10 -> {tag}')
        P('     ' + '; '.join(cells))
        if m != 'off':
            verdict[m]['X1'] = n_ok >= 10
    # ---------------- X2
    P('')
    P('X2 (deciding): visible / full >= 0.9, pooled platforms; per bin x band |d hit| <= 0.05 and |d row-lost| <= 0.05 in all '
      '12 cells; a cell with < 10 objects on either side fails (reading)')
    ref_b = {k: band_means(R['D'], (rb == k) & (R['vis'] >= 0.9)) for k in range(3)}
    for m in GATE_ARMS:
        T_ = S[m]; sb = _bin(T_['r'], X_BINS)
        n_ok = 0; cells = []
        for k in range(3):
            sbm = band_means(T_['D'], (sb == k) & (T_['vis'] >= 0.9))
            for b in range(4):
                (ns, hs, ls), (nr, hr, lr) = sbm[b], ref_b[k][b]
                dh, dl = hs - hr, ls - lr
                ok = bool(ns >= 10 and nr >= 10 and np.isfinite(dh) and abs(dh) <= 0.05 + 1e-12 and abs(dl) <= 0.05 + 1e-12)
                n_ok += ok
                cells.append(f'{X_BINS[k][0]:g}-{X_BINS[k][1]:g} b{b} hit {hs:.3f}/{hr:.3f} ({dh:+.3f}) lost {ls:.3f}/{lr:.3f} '
                             f'({dl:+.3f}) n {ns}/{nr}{"" if ok else " X"}')
        tag = 'reported' if m == 'off' else ('PASS' if n_ok == 12 else 'FAIL')
        P(f'  {m}: {n_ok} of 12 cells -> {tag}')
        for c in cells:
            P('     ' + c)
        if m != 'off':
            verdict[m]['X2'] = n_ok == 12
    # ---------------- X3
    P('')
    P('X3 (deciding): S = median run inside Car boxes - median run outside every box grown 0.5 m, object pixels; '
      '|S_source - S_Waymo| <= 1 row in >= 5 of 6 cells')
    xb = ('40-50', '50-60', '60-75.2')
    SW = [RX.hist_median(ref['x3'][k, 0]) - RX.hist_median(ref['x3'][k, 1]) for k in range(3)]
    P('  Waymo: ' + '; '.join(f"{xb[k]} S {SW[k]:+.1f} (in {RX.hist_median(ref['x3'][k, 0]):.1f}, n {int(ref['x3'][k, 0].sum())}; "
                              f"out {RX.hist_median(ref['x3'][k, 1]):.1f}, n {int(ref['x3'][k, 1].sum())})" for k in range(3)))
    for m in GATE_ARMS:
        n_ok = 0; cells = []
        for k_, r in res.items():
            h = r['x3'][m]
            for b in range(3):
                si, so = RX.hist_median(h[b, 0]), RX.hist_median(h[b, 1])
                Sx = si - so; d = Sx - SW[b]
                ok = bool(np.isfinite(d) and abs(d) <= 1 + 1e-12)
                n_ok += ok
                cells.append(f'{PLAT_NAME[r["plat"]][0]} {xb[b]} S {Sx:+.1f} (in {si:.1f} n {int(h[b, 0].sum())}, out {so:.1f} '
                             f'n {int(h[b, 1].sum())}; {d:+.1f}{"" if ok else " X"})')
        tag = 'reported (unmodified source)' if m == 'off' else ('PASS' if n_ok >= 5 else 'FAIL')
        P(f'  {m}: {n_ok} of 6 -> {tag}')
        P('     ' + '; '.join(cells))
        if m != 'off':
            verdict[m]['X3'] = n_ok >= 5
    # ---------------- X4
    P('')
    P('X4 (deciding): new points below S < 1% (kept); Car boxes emptied by the shadow 0; frames within 150,000 voxels >= 99%; '
      'target dump identical with RX on and off')
    dump_off = {k: res[k]['dumps']['off'] for k in a.keys}
    for m in GATE_ARMS:
        x = X4[m]
        nk = x[:, J4['n_kept']].sum(); bk = x[:, J4['n_below_kept']].sum()
        share = bk / nk if nk else 0.0
        emp = int(x[:, J4['emptied_car']].sum())
        vox = float(np.mean(x[:, J4['nvox']] <= VOX_CAP))
        same = []
        for k in a.keys:
            for d1, d0 in zip(res[k]['dumps'][m], dump_off[k]):
                same.append(d1['hash'] == d0['hash'] and d1['positives'] == d0['positives'] and d1['peaks'] == d0['peaks'])
        dump_ok = bool(len(same) and all(same))
        gt_same = float(x[:, J4['gt_same']].mean()); in_same = float(x[:, J4['in_same']].mean())
        ok = bool(share < 0.01 and emp == 0 and vox >= 0.99 and dump_ok)
        if m == 'off':
            P(f'  off: frames within the cap {vox:.4f} (voxels p50 / p99 / max {np.percentile(x[:, J4["nvox"]], 50):.0f} / '
              f'{np.percentile(x[:, J4["nvox"]], 99):.0f} / {x[:, J4["nvox"]].max():.0f})')
            continue
        P(f'  {m}: below S {int(bk)} / {int(nk)} kept = {100 * share:.3f}% (generated {int(x[:, J4["n_below_new"]].sum())} / '
          f'{int(x[:, J4["n_new"]].sum())}); Car boxes emptied {emp} (any class {int(x[:, J4["emptied_any"]].sum())}); within the '
          f'cap {vox:.4f} (voxels p50 / p99 / max {np.percentile(x[:, J4["nvox"]], 50):.0f} / '
          f'{np.percentile(x[:, J4["nvox"]], 99):.0f} / {x[:, J4["nvox"]].max():.0f}); dump identical {sum(same)} / {len(same)}; '
          f'final gt_boxes identical on {gt_same:.4f} of frames, slot input identical {in_same:.4f} -> '
          f'{"PASS" if ok else "FAIL"}')
        verdict[m]['X4'] = ok
    # ---------------- reported
    P('')
    P('REPORTED')
    for k, r in res.items():
        g = r['g4']
        P(f"  loader time {k}: with RX (H-G0, the real hook) / without = {g[:, 3].sum() / g[:, 2].sum():.3f} over {len(g)} frames "
          f"(mean {g[:, 2].mean():.2f} s -> {g[:, 3].mean():.2f} s)")
    g = np.concatenate([r['g4'] for r in res.values()])
    P(f'  loader time pooled: {g[:, 3].sum() / g[:, 2].sum():.3f}')
    for m in arms:
        x = X4[m]
        P(f"  {m}: new points kept / output points {x[:, J4['n_kept']].sum() / x[:, J4['n_out']].sum():.4f} "
          f"(generated {x[:, J4['n_new']].sum() / x[:, J4['n_in']].sum():.4f} of the input); kept / generated per type car "
          f"{int(x[:, J4['kept_car']].sum())}/{int(x[:, J4['new_car']].sum())}, other {int(x[:, J4['kept_other']].sum())}/"
          f"{int(x[:, J4['new_other']].sum())}, background {int(x[:, J4['kept_bg']].sum())}/{int(x[:, J4['new_bg']].sum())}, "
          f"G1 {int(x[:, J4['kept_g1']].sum())}/{int(x[:, J4['new_g1']].sum())}; shadowed {int(x[:, J4['n_shadow']].sum())} "
          f"({x[:, J4['n_shadow']].mean():.1f} per frame; Car-box points {int(x[:, J4['car_pts_shadowed']].sum())}); RX time "
          f"{x[:, J4['t_rx']].mean():.2f} s per frame; frames without a plane {int(x[:, J4['no_plane']].sum())}")
    x = X4['H-G1']
    P(f"  H-G1 ground: between {x[:, J4['g1_between']].mean():.1f} and beyond {x[:, J4['g1_beyond']].mean():.1f} points per frame "
      f"generated, {x[:, J4['g1_kept']].mean():.1f} kept; share of the output {x[:, J4['g1_kept']].sum() / x[:, J4['n_out']].sum():.4f}")
    P(f"  frames without the measurement plane: {int(X4['off'][:, J4['no_meas_plane']].sum())}")
    P('  RC per ring (median; n), source Car boxes:')
    rring = _bin(R['r'], RING_BINS)
    P('    reference: ' + '; '.join(
        f'{lo:g}-{hi:g} {np.median(R["rc"][(rring == j) & np.isfinite(R["rc"])]) if ((rring == j) & np.isfinite(R["rc"])).any() else np.nan:.3f}'
        f' ({int(((rring == j) & np.isfinite(R["rc"])).sum())})' for j, (lo, hi) in enumerate(RING_BINS)))
    for m in GATE_ARMS:
        T_ = S[m]; rr_ = _bin(T_['r'], RING_BINS)
        for p in (0, 1):
            cells = []
            for j, (lo, hi) in enumerate(RING_BINS):
                mm = (T_['plat'] == p) & (rr_ == j) & np.isfinite(T_['rc'])
                cells.append(f'{lo:g}-{hi:g} {np.median(T_["rc"][mm]) if mm.any() else np.nan:.3f} ({int(mm.sum())})')
            P(f'    {m} {PLAT_NAME[p]}: ' + '; '.join(cells))
    # ---------------- decision
    P('')
    P('VERDICTS (deciding items X1-X4)')
    for m in arms:
        v = verdict[m]
        P(f"  {m}: " + ', '.join(f'{k} {"PASS" if v[k] else "FAIL"}' for k in ('X1', 'X2', 'X3', 'X4'))
          + f" -> {'PASSES' if all(v.values()) else 'FAILS'}")
    hg0 = all(verdict['H-G0'].values())
    P(f"LAUNCH DECISION (H-G0, the method row): {'PASS' if hg0 else 'FAIL'}")
    P('Arms passing every deciding item (T05 §5, user ~23:10: rows in the order H-G0, H-G1, L-G0, W-G0): '
      + (', '.join(m for m in arms if all(verdict[m].values())) or 'none'))
    fn = Path(a.dir) / f"report_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
    open(fn, 'w').write('\n'.join(L) + '\n')
    pickle.dump(verdict, open(Path(a.dir) / 'verdict.pkl', 'wb'))
    print(f'[rx_gate] report written to {fn}', flush=True)


# ------------------------------------------------------------------------------------ phase 1 analysis (v2 step 1)
# experiments_md 20261021_02 §6 (ANALYSIS; sets no value by itself): the X2 hit deficit of H-G0 and W-G0 decomposed per
# visible pixel, the source lines' horizontal fill, neighbouring returns along a lattice row, and an extent x drop sweep of
# the common rule already decided for v2 (T05 §5, 4fed06b) on NON-gate frames.
D_COLS = ('plat', 'frame', 'j', 'arm', 'r', 'side', 'n_full', 'n_vis') + \
    tuple(f'{n}{b}' for n in ('vis', 'hit', 'i', 'ii', 'iiiext', 'iiioth', 'iiiocc', 'rvis', 'rhit', 'kfhit', 'kfrvis',
                               'kfrhit', 'i_in') for b in range(4))
SW_E = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.665)
SW_P = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)
SW_CAR_E = (0.665, 0.45, 0.25)
SW_CAR_P = (1.0, 0.8, 0.6)
PAIR_G = 21                       # column gaps 0..20, last = >= 20
PAIR_DR = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf)
PAIR_RB = (0.0, 20.0, 40.0, 75.2)


def _capture_ds(a):
    """The H-G0 row dataset with its hook replaced by a capture of the slot input (RX off for the rest of the pipeline)."""
    import rr_gate
    from pcdet.datasets.augmentor import row_extrusion as RX
    cfg, dcfg, ds = rr_gate.build(a.cfg, a.key)
    rx_cfg = dcfg.ROW_EXTRUSION
    CAP = {}

    def hook(data_dict, cfg_):
        CAP['state'] = np.random.get_state()
        CAP['in'] = {k: np.copy(data_dict[k]) for k in ('points', 'gt_boxes', 'gt_names', 'gt_boxes_mask')}
        return data_dict
    RX.apply_in_loader = hook
    return cfg, dcfg, ds, rx_cfg, CAP


def _arm_out(Pin, I, car, arm, lat, lib, state):
    """v1's operator (arm H or W, G0) with v1's exact draws for that frame, and its record with candidate detail."""
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.augmentor import row_extrusion as RX
    st = np.random.get_state(); np.random.set_state(state)
    rng = rr.bg_generator(np.random)
    np.random.set_state(st)
    return RX.row_extrusion(Pin, I['gt_boxes'], car, arm, 'G0', lat, lib, rng, keep_record=True)


def _band_rows(rows_desc, relh):
    """Per visible pixel its row's band (row_records' rule: the band of the row's median relh)."""
    from box_pattern_visual import _band_of
    ur, inv = np.unique(rows_desc, return_inverse=True)
    med = np.array([np.median(relh[inv == k]) for k in range(len(ur))])
    return _band_of(med)[inv], inv, ur


def decomp_box(box, j, Pout, npp_out, pix_out, lat, in_obj_cols, kf_mask, gen, cand_map, cand, Pin, arm):
    """One source Car box on one arm's slot output: per relh band the visible pixels split into hit / (i) / (ii) /
    (iii: extent, other seed, occupied); row-conditional hits; keyframe hits (arm 'off'); (i) inside the seed-column span."""
    import box_pattern_visual as BV
    from pcdet.datasets.augmentor import relocate_render as rr
    nc = lat['n_cols']; R = len(lat['thetas'])
    pose = BV.sensor_pose('rr', height=lat['height'])
    sil, tin, tout, ent = BV.silhouette_pixels(box, pose, np.asarray(lat['thetas'])[::-1], 'rr', nc, full=True)
    n_full = len(np.unique(sil))
    if not n_full:
        return None
    xyz = Pout[:, :3]
    rad = float(np.hypot(box[3], box[4]) / 2 + 0.1)
    cnd = np.nonzero((np.abs(xyz[:, 0] - box[0]) <= rad) & (np.abs(xyz[:, 1] - box[1]) <= rad))[0]
    inb = cnd[rr.points_in_box(xyz[cnd], box)]
    inbox = np.zeros(len(xyz), bool); inbox[inb] = True
    vis = BV.visible_mask(sil, tin, npp_out, inbox)
    vp, first = np.unique(sil[vis], return_index=True)
    if not len(vp):
        return None
    relh = np.clip((ent[vis][first][:, 2] - (box[2] - box[5] / 2)) / box[5], 0, 1)
    band, rinv, ur = _band_rows(vp // nc, relh)
    op = pix_out[inb]; hit = np.isin(vp, op[op >= 0])
    kf = np.zeros(len(vp), bool)
    if kf_mask is not None:
        k_ = inb[kf_mask[inb]]; ok_ = pix_out[k_]; kf = np.isin(vp, ok_[ok_ >= 0])
    col = vp % nc
    asc = (R - 1 - vp // nc) * nc + col                        # ascending-row pixel ids (the operator's)
    cat = np.zeros(len(vp), np.int64)                          # 0 hit, 1 (i), 2 (ii), 3 iii-ext, 4 iii-other, 5 iii-occ
    if arm != 'off':
        gpix, gin, gkept = gen
        drop_pix = gpix[gin & ~gkept]
        has_seed_col = np.isin(col, in_obj_cols)
        for k in np.nonzero(~hit)[0]:
            if asc[k] in drop_pix:
                cat[k] = 2
            elif not has_seed_col[k]:
                cat[k] = 1
            else:
                c = cand_map.get(int(asc[k]), -1)
                if c < 0:
                    cat[k] = 4
                else:
                    s_in_b = rr.points_in_box(Pin[cand['seed'][c]:cand['seed'][c] + 1, :3], box)[0]
                    if not s_in_b:
                        cat[k] = 4
                    elif cand['held'][c] >= 0 and not (cand['rim'][c] > cand['rng_new'][c] + 0.5):
                        cat[k] = 5
                    else:
                        cat[k] = 3
    else:
        cat[~hit] = 1                                         # unmodified: every miss lacks a point (not split)
    # (i) inside the box's seed-column span (between its leftmost and rightmost seed columns), columns unwrapped
    i_in = np.zeros(len(vp), bool)
    if arm != 'off' and len(in_obj_cols):
        c0 = int(np.median(col))
        uw = lambda c: (np.asarray(c) - c0 + nc // 2) % nc
        lo, hi = uw(in_obj_cols).min(), uw(in_obj_cols).max()
        i_in = (cat == 1) & (uw(col) >= lo) & (uw(col) <= hi)
    out = {}
    for b in range(4):
        m = band == b
        rows_b = np.unique(rinv[m])
        rh = np.array([hit[rinv == q].any() for q in rows_b], bool) if len(rows_b) else np.zeros(0, bool)
        rk = np.array([kf[rinv == q].any() for q in rows_b], bool) if len(rows_b) else np.zeros(0, bool)
        mrh = m & np.isin(rinv, rows_b[rh]); mrk = m & np.isin(rinv, rows_b[rk])
        out[b] = [m.sum(), (hit & m).sum(), ((cat == 1) & m).sum(), ((cat == 2) & m).sum(), ((cat == 3) & m).sum(),
                  ((cat == 4) & m).sum(), ((cat == 5) & m).sum(), mrh.sum(), (hit & mrh).sum(), (kf & m).sum(),
                  mrk.sum(), (kf & mrk).sum(), (i_in & m).sum()]
    flat = [out[b][q] for q in range(13) for b in range(4)]
    return n_full, len(vp), flat


def row_pairs(idx, rim, xyz, boxes, lat):
    """Neighbouring returns along each lattice row of the INPUT image (consecutive occupied pixels, only empty pixels
    between): histogram (range bin of the mean horizontal range, class 0 same box / 1 both outside / 2 one in a box or
    different boxes, column gap 0..20+, |d range| bin)."""
    from pcdet.datasets.augmentor import row_extrusion as RX
    R, C = idx.shape
    H = np.zeros((len(PAIR_RB) - 1, 3, PAIR_G, len(PAIR_DR) - 1), np.int64)
    occ = idx >= 0
    pts = xyz[np.maximum(idx, 0)]
    bid = np.full(idx.shape, -1, np.int64)
    if len(boxes):
        m = RX.points_in_boxes(pts[occ], boxes)
        f = np.where(m.any(1), np.argmax(m, 1), -1); bid[occ] = f
    for r in range(R):
        cs = np.nonzero(occ[r])[0]
        if len(cs) < 2:
            continue
        c1, c2 = cs[:-1], cs[1:]
        g = np.minimum(c2 - c1 - 1, PAIR_G - 1)
        dr = np.abs(rim[r, c1] - rim[r, c2])
        rho = (np.hypot(pts[r, c1, 0], pts[r, c1, 1]) + np.hypot(pts[r, c2, 0], pts[r, c2, 1])) / 2
        b1, b2 = bid[r, c1], bid[r, c2]
        cls = np.where((b1 >= 0) & (b1 == b2), 0, np.where((b1 < 0) & (b2 < 0), 1, 2))
        rb = np.searchsorted(PAIR_RB, rho, side='right') - 1
        db = np.minimum(np.searchsorted(PAIR_DR, dr, side='right') - 1, len(PAIR_DR) - 2)
        ok = (rb >= 0) & (rb < len(PAIR_RB) - 1)
        np.add.at(H, (rb[ok], cls[ok], g[ok], db[ok]), 1)
    return H


def decomp(a):
    import box_pattern_visual as BV
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.augmentor import row_extrusion as RX
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plat = PLAT_ID[a.key]
    cfg, dcfg, ds, rx_cfg, CAP = _capture_ds(a)
    lat = RX.lattice(); lib = RX.load_library(str(rx_cfg.LIBRARY)); nc = lat['n_cols']
    tcol = dcfg.POINT_FEATURE_ENCODING.src_feature_list.index('timestamp')
    N = len(ds); step = N // a.n
    frames = [k * step + a.offset for k in range(a.n)][:a.max_frames]
    rows, pairs = [], np.zeros((len(PAIR_RB) - 1, 3, PAIR_G, len(PAIR_DR) - 1), np.int64)
    t0 = time.time()
    for n_done, i in enumerate(frames):
        np.random.seed((plat * 1000003 + i) % (2 ** 31)); CAP.clear()
        ds[i]
        I = CAP['in']; Pin = I['points']; boxes = I['gt_boxes'][:, :7].astype(np.float64)
        car = rr.gt_car_mask(I)
        xin = np.asarray(Pin[:, :3], np.float64)
        idx, rim, _ = RX.range_image(xin, lat)
        pairs += row_pairs(idx, rim, xin, boxes, lat)
        kf_in = Pin[:, tcol] == 0
        sel = [j for j in np.nonzero(car)[0] if 40.0 <= np.hypot(*boxes[j, :2]) < 75.0]
        if not sel:
            continue
        for arm in ('off', 'H', 'W'):
            if arm == 'off':
                Pout, rec, kf = Pin, None, kf_in
            else:
                Pout, rec = _arm_out(Pin, I, car, arm, lat, lib, CAP['state']); kf = None
            pix, rng = source_pixels(np.asarray(Pout[:, :3], np.float64), lat)
            npp = BV.nearest_per_pixel(pix, rng)
            cand, cand_map, gen, S = None, {}, None, None
            if rec is not None and 'cand' in rec:
                cand = rec['cand']; cand_map = {int(p_): k for k, p_ in enumerate(cand['pix'])}
                S = RX.Surface(rec['plane'], np.zeros((0, 7)) if arm == 'W' else boxes[car])
                occ = idx >= 0
                hp = xin[idx[occ]]
                obj_held = np.full(idx.shape, -1, np.int64)
                isobj = hp[:, 2] > S(hp[:, 0], hp[:, 1]) + RX.GROUND_TOL
                obj_held[occ] = np.where(isobj, idx[occ], -1)
            for j in sel:
                b = boxes[j]
                in_obj_cols = np.zeros(0, np.int64)
                if cand is not None:
                    oc = np.nonzero(obj_held >= 0)
                    hold = obj_held[oc]
                    near = (np.abs(xin[hold, 0] - b[0]) <= 4) & (np.abs(xin[hold, 1] - b[1]) <= 4)
                    inb = rr.points_in_box(xin[hold[near]], b)
                    in_obj_cols = np.unique(oc[1][near][inb])
                    gp = rec['new_pix']; gx = rec['new_xyz']
                    gin = rr.points_in_box(gx, b) if len(gx) else np.zeros(0, bool)
                    gen = (gp, gin, rec['new_kept'])
                res = decomp_box(b, j, Pout, npp, pix, lat, in_obj_cols, kf, gen, cand_map, cand, Pin, arm)
                if res is None:
                    continue
                n_full, n_vis, flat = res
                rows.append([plat, i, j, {'off': 0, 'H': 1, 'W': 2}[arm], float(np.hypot(*b[:2])),
                             float(RX.aspect_side(rr.relative_view_angle(b))), n_full, n_vis] + flat)
        if n_done % 10 == 0:
            print(f'[rx decomp] {a.key} {n_done + 1}/{len(frames)} frames, {len(rows)} box rows, {time.time() - t0:.0f} s',
                  flush=True)
    pickle.dump(dict(key=a.key, plat=plat, frames=frames, rows=np.array(rows, np.float64).reshape(-1, len(D_COLS)),
                     cols=D_COLS, pairs=pairs, offset=a.offset), open(out / f'decomp_{a.key}.pkl', 'wb'))
    print(f'[rx decomp] done {a.key}: {len(frames)} frames, {time.time() - t0:.0f} s', flush=True)


def sweep(a):
    """Extent E (deg about the seed's elevation, every seed) x independent keep p (every new point) under the common rule
    already decided for v2 (boxes only clip: an in-box seed's point stays in its box, an outside seed's enters no box;
    z >= S; the shadow never removes a return inside any box), from v1 H's candidates. Run-length histograms (X3) for the
    whole grid; RC and band hits (X1 / X2) of the Car boxes at 40-75 m for a 3 x 3 subset."""
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.augmentor import row_extrusion as RX
    from rx_build_library import x3_hist
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plat = PLAT_ID[a.key]
    cfg, dcfg, ds, rx_cfg, CAP = _capture_ds(a)
    lat = RX.lattice(); lib = RX.load_library(str(rx_cfg.LIBRARY)); nc = lat['n_cols']
    N = len(ds); step = N // a.n
    frames = [k * step + a.offset for k in range(a.n)][:a.max_frames]
    X3 = np.zeros((len(SW_E), len(SW_P), 3, 2, 65), np.int64)
    NEW = np.zeros((len(SW_E), len(SW_P), 2))          # kept new points, input points
    CARS = {(e, p_): [] for e in SW_CAR_E for p_ in SW_CAR_P}
    t0 = time.time()
    for n_done, i in enumerate(frames):
        np.random.seed((plat * 1000003 + i) % (2 ** 31)); CAP.clear()
        ds[i]
        I = CAP['in']; Pin = I['points']; boxes = I['gt_boxes'][:, :7].astype(np.float64)
        car = rr.gt_car_mask(I)
        xin = np.asarray(Pin[:, :3], np.float64)
        _, rec = _arm_out(Pin, I, car, 'H', lat, lib, CAP['state'])
        if rec.get('no_plane') or 'cand' not in rec:
            continue
        c = rec['cand']
        S = RX.Surface(rec['plane'], boxes[car])
        idx, rim, _ = RX.range_image(xin, lat)
        R, C = idx.shape
        occ = idx >= 0
        hp = xin[np.maximum(idx, 0)]
        rho_in = np.hypot(hp[..., 0], hp[..., 1])
        s_in = np.full(idx.shape, np.nan); s_in[occ] = S(hp[occ][:, 0], hp[occ][:, 1])
        with np.errstate(invalid='ignore'):
            obj_in = occ & (hp[..., 2] > s_in + RX.GROUND_TOL)
        incar_in = np.zeros(idx.shape, bool); grown_in = np.zeros(idx.shape, bool); inany_in = np.zeros(idx.shape, bool)
        if car.any():
            incar_in[occ] = RX.points_in_boxes(hp[occ], boxes[car]).any(1)
        if len(boxes):
            inany_in[occ] = RX.points_in_boxes(hp[occ], boxes).any(1)
        for b in boxes:
            grown_in[occ] |= rr.points_in_rect(hp[occ][:, :2], b, 0.5)
        sx = xin[c['seed']]
        held_inbox = np.zeros(len(c['pix']), bool)
        h_ok = c['held'] >= 0
        held_inbox[h_ok] = inany_in.ravel()[c['pix'][h_ok]]
        new_incar = RX.points_in_boxes(np.c_[sx[:, :2], c['z_new']], boxes[car]).any(1) if car.any() \
            else np.zeros(len(sx), bool)
        new_grown = np.zeros(len(sx), bool)
        for b in boxes:
            new_grown |= rr.points_in_rect(sx[:, :2], b, 0.5)
        clip = np.where(c['seed_in'], c['new_in_seed_box'], ~c['new_in_any_box'])
        base_ok = clip & (c['z_new'] >= c['s_at']) & ((c['held'] < 0) | ((c['rim'] > c['rng_new'] + 0.5) & ~held_inbox))
        obj_new = c['z_new'] > c['s_at'] + RX.GROUND_TOL
        rho_new = np.hypot(sx[:, 0], sx[:, 1])
        u = np.random.default_rng([plat, i, 7]).random(len(c['pix']))
        pr, pc = c['pix'] // nc, c['pix'] % nc
        for ie, E in enumerate(SW_E):
            okE = base_ok & (np.abs(c['dth']) <= np.radians(E) + 1e-12)
            for ip, p_ in enumerate(SW_P):
                k = okE & (u < p_)
                has = occ.copy(); rng = rim.copy(); obj = obj_in.copy(); inc = incar_in.copy(); gr = grown_in.copy()
                rho = rho_in.copy()
                has[pr[k], pc[k]] = True; rng[pr[k], pc[k]] = c['rng_new'][k]; obj[pr[k], pc[k]] = obj_new[k]
                inc[pr[k], pc[k]] = new_incar[k]; gr[pr[k], pc[k]] = new_grown[k]; rho[pr[k], pc[k]] = rho_new[k]
                X3[ie, ip] += x3_hist(has, rng, obj, rho, inc, has & ~gr)
                NEW[ie, ip] += [k.sum(), len(Pin)]
                if E in SW_CAR_E and p_ in SW_CAR_P:
                    keep_src = np.ones(len(Pin), bool)
                    sh = c['held'][k]; keep_src[sh[sh >= 0]] = False
                    newp = Pin[c['seed'][k]].copy(); newp[:, 2] = c['z_new'][k]
                    Pout = np.concatenate([Pin[keep_src], newp.astype(Pin.dtype)])
                    CARS[(E, p_)] += [r_ for r_ in measure_cars(Pout, I['gt_boxes'], car, lat, plat, i)
                                      if 40.0 <= r_[3] < 75.0]
        if n_done % 10 == 0:
            print(f'[rx sweep] {a.key} {n_done + 1}/{len(frames)} frames, {time.time() - t0:.0f} s', flush=True)
    pickle.dump(dict(key=a.key, plat=plat, frames=frames, E=SW_E, P=SW_P, x3=X3, new=NEW, offset=a.offset,
                     cars={k: np.array(v, np.float64).reshape(-1, len(CAR_COLS)) for k, v in CARS.items()}),
                open(out / f'sweep_{a.key}.pkl', 'wb'))
    print(f'[rx sweep] done {a.key}: {len(frames)} frames, {time.time() - t0:.0f} s', flush=True)


def _hshare(h, edges=(1, 2, 3, 4, 5, 6, 7, 8, 11)):
    """Shares of an integer histogram in [1], [2], ..., [8, 10], [11, ...) (run lengths)."""
    h = np.asarray(h, np.float64); n = h.sum()
    if not n:
        return [np.nan] * len(edges)
    out, lo = [], edges[0]
    for k, e in enumerate(edges[1:]):
        out.append(h[lo:e].sum() / n); lo = e
    out.append(h[lo:].sum() / n)
    return out


def _hq(h, q):
    """The q-quantile (0-1) of an integer sample given by its histogram (the smallest v with cumulative share >= q)."""
    c = np.cumsum(h) / max(np.sum(h), 1)
    return int(np.searchsorted(c, q - 1e-12))


def _ref_rowcond(ref):
    """Per held-out / fit reference car (vis >= 0.9): per band (n visible pixels, hits) over ALL rows and over rows with
    >= 1 hit."""
    from box_pattern_visual import _band_of
    C, Rw = ref['cars'], ref['rows']
    out = []
    for c in C:
        rows = Rw[int(c[7]):int(c[7]) + int(c[8])].astype(np.float64)
        b = _band_of(rows[:, 3]); h = rows[:, 2] >= 1
        rec = [c[2], c[5] / c[4]]
        for k in range(4):
            m = b == k
            rec += [rows[m, 1].sum(), rows[m, 2].sum(), rows[m & h, 1].sum(), rows[m & h, 2].sum()]
        out.append(rec)
    return np.array(out).reshape(-1, 18)


def phase1_report(a):
    from pcdet.datasets.augmentor import row_extrusion as RX
    L = []

    def P(t=''):
        L.append(t); print(t, flush=True)
    D = np.concatenate([pickle.load(open(Path(a.dir) / f'decomp_{k}.pkl', 'rb'))['rows'] for k in a.keys])
    pairs = sum(pickle.load(open(Path(a.dir) / f'decomp_{k}.pkl', 'rb'))['pairs'] for k in a.keys)
    SW = [pickle.load(open(Path(a.dir) / f'sweep_{k}.pkl', 'rb')) for k in a.keys]
    G = {k: pickle.load(open(Path(a.gate) / f'gate_{k}.pkl', 'rb')) for k in a.keys}
    ref = np.load(a.ref, allow_pickle=True); reff = np.load(a.ref_fit, allow_pickle=True)
    J = {c: i for i, c in enumerate(D_COLS)}
    col = lambda n, b: D[:, J[f'{n}{b}']]
    P('RX v2 step 1 (ANALYSIS; experiments_md 20261021_02 §6): v1 gate frames (decomposition) and non-gate frames (sweep)')
    P(f'  decomposition rows: {len(D)} (box x arm); frames per key: ' + ', '.join(
        f"{k} {len(pickle.load(open(Path(a.dir) / f'decomp_{k}.pkl', 'rb'))['frames'])}" for k in a.keys))
    # ---------------------------------------------------------------- 1. X2 decomposition
    RC = _ref_rowcond(ref); RCF = _ref_rowcond(reff)
    rbin = _bin(D[:, J['r']], X_BINS); vis = D[:, J['n_vis']] / D[:, J['n_full']]
    P('')
    P('1. X2 hit deficit decomposition (X2 population: Car boxes, visible / full >= 0.9; mean over objects of per-band pixel '
      'shares; Waymo = held-out reference, the gate\'s)')
    P('   cols: ref hit | arm hit | deficit | arm misses split: (i) no in-box seed in the column [of which inside the seed-column '
      'span] | (ii) generated, dropped | (iii) extent | (iii) other seed | (iii) occupied, nearer | n arm / ref')
    for arm, name in ((1, 'H-G0'), (2, 'W-G0')):
        P(f'  {name}:')
        for k in range(3):
            for b in range(4):
                m = (D[:, J['arm']] == arm) & (rbin == k) & (vis >= 0.9) & (col('vis', b) >= 1)
                mr = (_bin(RC[:, 0], X_BINS) == k) & (RC[:, 1] >= 0.9) & (RC[:, 2 + 4 * b] >= 1)
                v = col('vis', b)[m]
                f = {n: float(np.mean(col(n, b)[m] / v)) if m.any() else np.nan
                     for n in ('hit', 'i', 'ii', 'iiiext', 'iiioth', 'iiiocc', 'i_in')}
                rh = float(np.mean(RC[mr, 3 + 4 * b] / RC[mr, 2 + 4 * b])) if mr.any() else np.nan
                P(f"    {X_BINS[k][0]:g}-{X_BINS[k][1]:g} b{b}: ref {rh:.3f} | {f['hit']:.3f} | {rh - f['hit']:+.3f} | "
                  f"(i) {f['i']:.3f} [{f['i_in']:.3f}] | (ii) {f['ii']:.3f} | ext {f['iiiext']:.3f} | oth {f['iiioth']:.3f} | "
                  f"occ {f['iiiocc']:.3f} | n {int(m.sum())} / {int(mr.sum())}")
    # ---------------------------------------------------------------- 2. row-conditional hit
    P('')
    P('2. Hit rate among rows holding >= 1 point (per object: hits / visible pixels over those rows; mean over objects), and '
      'the keyframe-only source')
    P('   cols: Waymo held-out | Waymo fit | source accumulated | source keyframe | H-G0 | W-G0 || unconditional: source acc / '
      'keyframe hit')
    for k in range(3):
        for b in range(4):
            def rcond(arr, kk=k):
                mm = (_bin(arr[:, 0], X_BINS) == kk) & (arr[:, 1] >= 0.9) & (arr[:, 4 + 4 * b] >= 1)
                return float(np.mean(arr[mm, 5 + 4 * b] / arr[mm, 4 + 4 * b])) if mm.any() else np.nan

            def scond(arm, n_v='rvis', n_h='rhit'):
                mm = (D[:, J['arm']] == arm) & (rbin == k) & (vis >= 0.9) & (col(n_v, b) >= 1)
                return float(np.mean(col(n_h, b)[mm] / col(n_v, b)[mm])) if mm.any() else np.nan, int(mm.sum())
            m0 = (D[:, J['arm']] == 0) & (rbin == k) & (vis >= 0.9) & (col('vis', b) >= 1)
            acc_u = float(np.mean(col('hit', b)[m0] / col('vis', b)[m0])) if m0.any() else np.nan
            kf_u = float(np.mean(col('kfhit', b)[m0] / col('vis', b)[m0])) if m0.any() else np.nan
            sa, na = scond(0); sk, nk = scond(0, 'kfrvis', 'kfrhit'); sh, nh = scond(1); sw, nw = scond(2)
            P(f'    {X_BINS[k][0]:g}-{X_BINS[k][1]:g} b{b}: {rcond(RC):.3f} | {rcond(RCF):.3f} | {sa:.3f} (n {na}) | '
              f'{sk:.3f} (n {nk}) | {sh:.3f} | {sw:.3f} || {acc_u:.3f} / {kf_u:.3f}')
    # ---------------------------------------------------------------- 3. neighbouring returns along a row
    P('')
    P('3. Neighbouring returns along a lattice row of the INPUT image (consecutive occupied pixels; gap = empty columns between)')
    cls_name = ('same box', 'both outside boxes', 'box edge (one in a box / two boxes)')
    for rb in range(len(PAIR_RB) - 1):
        P(f'  mean horizontal range {PAIR_RB[rb]:g}-{PAIR_RB[rb + 1]:g} m:')
        for cl in range(3):
            H = pairs[rb, cl]; n = H.sum()
            g = H.sum(1); dr = H.sum(0)
            gs = [g[0], g[1], g[2], g[3], g[4:6].sum(), g[6:11].sum(), g[11:].sum()]
            P(f'    {cls_name[cl]}: n {int(n)}; gap 0 / 1 / 2 / 3 / 4-5 / 6-10 / >10: '
              + ' / '.join(f'{x / max(n, 1):.3f}' for x in gs)
              + '; |d range| <= 0.1 / 0.2 / 0.3 / 0.5 / 1 / 2 m: '
              + ' / '.join(f'{dr[:j].sum() / max(n, 1):.3f}' for j in range(1, 7)))
            for gg in (1, 2, 3, 4):
                h = H[gg] if gg < 4 else H[4:6].sum(0)
                P(f'      gap {gg if gg < 4 else "4-5"}: |d range| <= 0.3 m {h[:3].sum() / max(h.sum(), 1):.3f} of {int(h.sum())}')
    # ---------------------------------------------------------------- 4. X3 distributions
    P('')
    P('4. X3 run-length distributions (object pixels, pixel-weighted): shares of runs 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8-10 / >10; '
      'p25 / p50 / p75')
    xb = ('40-50', '50-60', '60-75.2')
    srcs = [('Waymo held-out', ref['x3']), ('Waymo fit set', reff['x3'])]
    for m in ('off', 'H-G0', 'W-G0'):
        for k, g in G.items():
            srcs.append((f'{m} {PLAT_NAME[g["plat"]]}', g['x3'][m]))
    for name, h in srcs:
        for b in range(3):
            for io, lab in ((0, 'in '), (1, 'out')):
                hh = h[b, io]
                P(f'  {name:22s} {xb[b]:8s} {lab}: ' + ' / '.join(f'{x:.3f}' for x in _hshare(hh))
                  + f'; {_hq(hh, .25)} / {_hq(hh, .5)} / {_hq(hh, .75)} (n {int(hh.sum())})')
    # ---------------------------------------------------------------- 5. sweep
    P('')
    P('5. Extent x drop sweep (common rule, boxes only clip, z >= S, shadow spares boxes; NON-gate frames k x floor(N / 200) + 1; '
      'pooled platforms): median run in / out per bin; Waymo fit set and held-out beside')
    SWF = [RX.hist_median(reff['x3'][b, 0]) for b in range(3)], [RX.hist_median(reff['x3'][b, 1]) for b in range(3)]
    SWH = [RX.hist_median(ref['x3'][b, 0]) for b in range(3)], [RX.hist_median(ref['x3'][b, 1]) for b in range(3)]
    P(f'  Waymo fit (in / out): ' + '; '.join(f'{xb[b]} {SWF[0][b]:g} / {SWF[1][b]:g}' for b in range(3))
      + '   held-out: ' + '; '.join(f'{xb[b]} {SWH[0][b]:g} / {SWH[1][b]:g}' for b in range(3)))
    X3s = sum(s_['x3'] for s_ in SW); NEW = sum(s_['new'] for s_ in SW)
    E_, P_ = SW[0]['E'], SW[0]['P']
    hits = []
    for ie, E in enumerate(E_):
        for ip, p_ in enumerate(P_):
            h = X3s[ie, ip]
            med = [(RX.hist_median(h[b, 0]), RX.hist_median(h[b, 1])) for b in range(3)]
            Sv = [mi - mo for mi, mo in med]
            per_plat = []
            for s_ in SW:
                hp_ = s_['x3'][ie, ip]
                per_plat.append(sum(abs((RX.hist_median(hp_[b, 0]) - RX.hist_median(hp_[b, 1]))
                                        - (RX.hist_median(ref['x3'][b, 0]) - RX.hist_median(ref['x3'][b, 1]))) <= 1
                                    for b in range(3)))
            match = all(med[b] == (SWF[0][b], SWF[1][b]) for b in range(3))
            if match:
                hits.append((E, p_))
            P(f'  E {E:5.3f} deg, p {p_:.2f}: ' + '; '.join(f'{xb[b]} {med[b][0]:g} / {med[b][1]:g} (S {Sv[b]:+g})'
                                                            for b in range(3))
              + f' | X3-like cells vs held-out {sum(per_plat)} / 6 | new points / input {NEW[ie, ip, 0] / max(NEW[ie, ip, 1], 1):.3f}'
              + (' | = Waymo fit medians' if match else ''))
    P('  grid points reproducing the Waymo fit-set medians (in / out) in all three bins: ' + (', '.join(
        f'E {e:g} p {q:g}' for e, q in hits) or 'none'))
    P('  car-level subset (Car boxes at 40-75 m on the non-gate frames): X1-like cells within 0.10 of the held-out reference '
      'median, and body-band (b1) hit per bin vs the reference')
    Rr = ref_tables(ref)
    for E in SW_CAR_E:
        for p_ in SW_CAR_P:
            cars = np.concatenate([s_['cars'][(E, p_)] for s_ in SW])
            T_ = src_tables(cars); sb = _bin(T_['r'], X_BINS); rb_ = _bin(Rr['r'], X_BINS)
            n_ok = 0
            for p0 in (0, 1):
                for k in range(3):
                    for sd in (0, 1):
                        mm = (T_['plat'] == p0) & (sb == k) & (T_['side'] == bool(sd)) & np.isfinite(T_['rc'])
                        mr = (rb_ == k) & (Rr['side'] == bool(sd)) & np.isfinite(Rr['rc'])
                        if mm.any() and abs(np.median(T_['rc'][mm]) - np.median(Rr['rc'][mr])) <= 0.10 + 1e-12:
                            n_ok += 1
            b1 = []
            for k in range(3):
                sm = band_means(T_['D'], (sb == k) & (T_['vis'] >= 0.9))[1]
                rm = band_means(Rr['D'], (rb_ == k) & (Rr['vis'] >= 0.9))[1]
                b1.append(f'{sm[1]:.3f}/{rm[1]:.3f}')
            P(f'    E {E:g} p {p_:g}: X1-like {n_ok} / 12; b1 hit ' + ', '.join(f'{xb[k]} {b1[k]}' for k in range(3)))
    fn = Path(a.dir) / f"phase1_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
    open(fn, 'w').write('\n'.join(L) + '\n')
    print(f'[rx phase1] written to {fn}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--cfg', required=True); r.add_argument('--key', required=True); r.add_argument('--out', required=True)
    r.add_argument('--n', type=int, default=200); r.add_argument('--g4', type=int, default=100)
    r.add_argument('--dump', type=int, default=3); r.add_argument('--max_frames', type=int, default=None)
    r.add_argument('--offset', type=int, default=0, help='0 = the gate frames; smokes use 1 (never a gate frame)')
    r.add_argument('--library_override', default=None, help='smokes only')
    p = sp.add_parser('report')
    p.add_argument('--dir', required=True); p.add_argument('--ref', required=True); p.add_argument('--lib', required=True)
    p.add_argument('--targets', default='/home/koyama/data/relocate_render_20261018/v5/targets.npz')
    p.add_argument('--keys', nargs='*', default=['NUSCENES_N008', 'NUSCENES_N015'])
    for name in ('decomp', 'sweep'):
        q = sp.add_parser(name)
        q.add_argument('--cfg', required=True); q.add_argument('--key', required=True); q.add_argument('--out', required=True)
        q.add_argument('--n', type=int, default=200); q.add_argument('--max_frames', type=int, default=None)
        q.add_argument('--offset', type=int, default=0 if name == 'decomp' else 1)
    q = sp.add_parser('phase1')
    q.add_argument('--dir', required=True); q.add_argument('--gate', required=True); q.add_argument('--ref', required=True)
    q.add_argument('--ref_fit', required=True); q.add_argument('--keys', nargs='*', default=['NUSCENES_N008', 'NUSCENES_N015'])
    a = ap.parse_args()
    if a.cmd == 'run':
        run(a)
    elif a.cmd == 'report':
        report(a)
    elif a.cmd == 'decomp':
        decomp(a)
    elif a.cmd == 'sweep':
        sweep(a)
    else:
        phase1_report(a)


if __name__ == '__main__':
    main()
