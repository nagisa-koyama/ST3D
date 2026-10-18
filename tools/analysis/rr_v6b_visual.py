"""Sampling results of RR v6 arm (b') (the per-box U-Net), box by box (experiments_md 20261018_03 §10). CPU, ANALYSIS.

Columns per range bin (rows = the boxes nearest the p25 / p50 / p75 of point count in a fixed-seed pool, as §1):
- B6: an RR v6 (b') copy, re-rendered on the gate v6 frames with the gate's config, seeds and capture (rr_gate.run_v2;
  checked against the gate's rows), in the sensor's-eye view of §2 (the virtual TOP): visible rendered pixels grey, kept
  points coloured by the U-Net's p at their pixel, envelope-hidden silhouette pixels as faint x.
- B6-p: the same copy's p map over its U-Net input patch (rows highest first, columns decreasing azimuth, as fed).
- W: a held-out Waymo label patch (20261018_02 §10.3; held-out = the 20% segments; far frames >= 45 m only): the U-Net's p
  over the patch, the real returns (label 1) as dots.
- C: the 20261018_03 §4 Waymo val car of that bin and quantile (extract_CD.pkl), drawn as before.
The U-Net input of a copy is captured by wrapping relocate_render.cnn_keep_prob: the wrapper rebuilds the patch and the
p map from the same inputs with the module's own build_patch / _load_cnn and checks that its p equals the loader's.

    python analysis/rr_v6b_visual.py extract_b6 --key NUSCENES_N008 --out <dir>
    python analysis/rr_v6b_visual.py w_patches --out <dir>
    python analysis/rr_v6b_visual.py plot --out <dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS.parent, TOOLS, TOOLS / 'analysis'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _init_path  # noqa: F401,E402
import box_pattern_visual as V  # noqa: E402

RR6B_CFG = 'cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-rr6b-nuscenes2waymo.yaml'
V6 = Path('/home/koyama/data/relocate_render_20261018/v6')
GATE = V6 / 'gate_b'
PATCHES = V6 / 'patches' / 'patches_00.npz'
WEIGHTS = V6 / 'unet_b.pt'
BPV_OUT = Path('/home/koyama/data/box_pattern_visual_20261018')
COLS = ('B6', 'W')                  # pools selected here; C reuses §4's picks
P_LO = 0.25                         # the p colour ramp starts at this point of the Blues map so p = 0 stays visible


# ------------------------------------------------------------------------------------------------- pure (tested)
def patch_band_means(X, P, edges=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Mean p and count over the patch's COVERED valid pixels (channel 1 > 0 and channel 4 > 0) by the h_rel band of
    channel 2. Returns (sum p per band, count per band)."""
    X = np.asarray(X, np.float32); P = np.asarray(P, np.float32)
    m = (X[1] > 0) & (X[4] > 0)
    b = np.minimum((np.clip(X[2][m], 0, 1) * 4).astype(np.int64), 3)
    return np.bincount(b, weights=P[m], minlength=4), np.bincount(b, minlength=4)


def bin_of(r):
    return V._bin_of(np.atleast_1d(r))


# ------------------------------------------------------------------------------------------------ extract B6
def extract_b6(a):
    import torch
    import pcdet
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.rerender_utils import to_lattice
    from rr_gate import build, PLAT_ID, V2_COPY_COLS
    from source_coverage import crop, descriptors
    torch.set_num_threads(1)
    print(f'[extract_b6] pcdet {pcdet.__file__}; RR {rr.__file__}', flush=True)
    C = {k: j for j, k in enumerate(V2_COPY_COLS)}
    key = a.key; plat = PLAT_ID[key]
    g = pickle.load(open(GATE / f'gate_v6_{key}.pkl', 'rb'))
    gcp = g['copies']
    cfg, dcfg, ds = build(RR6B_CFG, key)
    aug = ds.data_augmentor; q0 = aug.data_augmentor_queue[0]
    rr_cfg = q0.keywords['config']; lat = q0.keywords['lattice']; db = q0.keywords['db']
    assert int(rr_cfg.VERSION) == 6 and rr_cfg.RETURN_MODEL == 'cnn', (rr_cfg.VERSION, rr_cfg.RETURN_MODEL)
    REC = []
    orig = rr.cnn_keep_prob

    def recorder(local, kept, rxyz, rrow, rcol, box, sil, env, thetas, n_cols, height, weights):
        p = orig(local, kept, rxyz, rrow, rcol, box, sil, env, thetas, n_cols, height, weights)
        loc = np.asarray(local, np.float64); b = np.asarray(box, np.float64)
        o = np.array([0.0, 0.0, height])
        depth = np.linalg.norm(np.asarray(rxyz, np.float64) - o, axis=1) - float(np.linalg.norm(b[:3] - o))
        irank = rr.intensity_rank(loc[:, 3])[kept] if loc.shape[1] > 3 else np.ones(len(kept))
        hrel = np.clip((loc[kept, 2] + b[5] / 2) / b[5], 0.0, 1.0)
        sp, srow, _ = sil
        if not len(sp):
            return p
        X, (ri, rj), (si, sj) = rr.build_patch(srow, sp % n_cols, env['sil_vis'], rrow, rcol, depth, irank, hrel,
                                               np.log(np.hypot(b[0], b[1]) / rr.R_REF_M), n_cols, flip_rows=True,
                                               flip_cols=True, fixed=False)
        with torch.no_grad():
            lg = rr._load_cnn(str(weights))(torch.from_numpy(X[None]))[0].numpy()
        pr = 1.0 / (1.0 + np.exp(-lg))
        m = ri >= 0
        assert np.allclose(pr[ri[m], rj[m]], p[m], atol=1e-6), 'rebuilt p differs from the loader p'
        REC.append(dict(box=b.copy(), X=X.astype(np.float16), pr=pr.astype(np.float16),
                        rend_pix=np.asarray(rrow, np.int64) * n_cols + np.asarray(rcol, np.int64), ri=ri, rj=rj,
                        sil_pix=np.asarray(sp, np.int64), sil_rows=np.asarray(srow, np.int64),
                        sil_vis=np.asarray(env['sil_vis'], bool), si=si, sj=sj,
                        H=int(si.max()) + 1, W=int(sj.max()) + 1))
        return p
    rr.cnn_keep_prob = recorder
    CAP = {}

    def rr_step(data_dict=None):
        o, rec = rr.relocate_render_scene_v2(data_dict, rr_cfg, lat, db, rng=np.random, keep_points=True)
        CAP['rec'] = rec; CAP['points'] = np.copy(o['points'])
        return o
    aug.data_augmentor_queue = [rr_step] + list(aug.data_augmentor_queue)[1:]
    th, nc, h = lat['thetas'], lat['n_cols'], lat['height']
    out, n_bad, t0 = [], 0, time.time()
    for n_done, i in enumerate(g['frames'][:a.max_frames or None]):
        REC.clear(); CAP.clear()
        np.random.seed((plat * 1000003 + i) % (2 ** 31))
        ds[i]
        rec, P = CAP['rec'], CAP['points']
        grow = gcp[gcp[:, C['frame']] == i]
        if len(grow) != len(rec['copies']):
            n_bad += 1
        for j, c in enumerate(rec['copies']):
            same = j < len(grow) and abs(grow[j, C['r_new']] - c['r_new']) < 1e-6 and \
                grow[j, C['n_kept']] == c['n_kept'] and grow[j, C['n_final']] == c['n_final'] and \
                abs(grow[j, C['p_keep']] - c['p_keep']) < 1e-9 and grow[j, C['instance']] == c['instance']
            n_bad += int(not same)
            m = [e for e in REC if np.array_equal(e['box'], np.asarray(c['box'], np.float64))]
            if len(m) != 1:
                n_bad += 1
                continue
            e = m[0]
            kp = np.asarray(P[c['point_index'], :3], np.float64)
            kr, kc, _, kv = to_lattice(kp, h, th, nc)
            kpix = (kr * nc + kc)[kv]
            vp = np.asarray(c['vis_pix'], np.int64); vpp = np.asarray(c['vis_p'], np.float64)
            o = np.argsort(vp)
            pos = np.clip(np.searchsorted(vp[o], kpix), 0, max(len(vp) - 1, 0))
            k_p = np.where(len(vp) and (vp[o][pos] == kpix), vpp[o][pos], np.nan) if len(vp) else np.full(len(kpix), np.nan)
            rp = {int(px): (int(a_), int(b_)) for px, a_, b_ in zip(e['rend_pix'], e['ri'], e['rj']) if a_ >= 0}
            kij = np.array([rp.get(int(px), (-1, -1)) for px in kpix], np.int64).reshape(-1, 2)
            dsc = descriptors(crop(kp))
            vis_rows = np.unique((e['sil_pix'][e['sil_vis']]) // nc)
            out.append(dict(plat=plat, key=key, frame=i, j=j, instance=int(c['instance']), box=np.asarray(c['box']),
                            length=float(db.size[int(c['instance'])][0]), r=float(c['r_new']), p_keep=float(c['p_keep']),
                            n_visible=int(c['n_visible']), n_kept=int(c['n_kept']), n_final=int(c['n_final']),
                            desc=list(dsc), kept_xyz=kp.astype(np.float32), kept_pix=kpix, kept_p=k_p, kept_ij=kij,
                            vis_xyz=np.asarray(c['rendered_unsparsified'], np.float32), vis_pix=vp, vis_p=vpp,
                            sil_pix=e['sil_pix'], sil_vis=e['sil_vis'], X=e['X'], pr=e['pr'], H=e['H'], W=e['W'],
                            rows_vis=int(len(vis_rows)), rows_kept=int(len(np.unique(kpix // nc))),
                            thetas_deg=np.degrees(th), reproduced=bool(same)))
        if n_done % 25 == 0:
            print(f'[extract_b6] {key} {n_done + 1}/{len(g["frames"])} frames, {len(out)} copies, '
                  f'{time.time() - t0:.0f} s', flush=True)
    pickle.dump(out, open(Path(a.out) / f'b6_{key}.pkl', 'wb'))
    print(f'[extract_b6] {key}: {len(out)} copies (gate {len(gcp)}), not reproduced or unmatched: {n_bad}', flush=True)


# --------------------------------------------------------------------------------------------------- W patches
def w_patches(a):
    import torch
    from pcdet.datasets.augmentor import relocate_render as rr
    torch.set_num_threads(a.threads)
    z = np.load(PATCHES)
    M = z['meta']; ho = np.nonzero(M[:, 6] == 0)[0]
    X = z['X'][ho]; Y = z['Y'][ho]
    net = rr._load_cnn(str(WEIGHTS))
    pr = np.zeros((len(ho), X.shape[2], X.shape[3]), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(ho), 128):
            lg = net(torch.from_numpy(X[s:s + 128].astype(np.float32))).numpy()
            pr[s:s + 128] = (1.0 / (1.0 + np.exp(-lg))).astype(np.float16)
    v = X[:, 4] > 0
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(Y[v].astype(np.int64), pr[v].astype(np.float32))
    np.savez(Path(a.out) / 'w_heldout.npz', idx=ho, meta=M[ho], pr=pr, Y=Y, X=X)
    print(f'[w_patches] {len(ho)} held-out patches, AUC on valid pixels {auc:.4f} (unet_b.json: 0.9047), '
          f'{time.time() - t0:.0f} s', flush=True)


# --------------------------------------------------------------------------------------------------------- plot
def _select(r, npts, ok, ci, b):
    elig = np.nonzero(ok & (bin_of(r) == b))[0]
    pick, qv, pool_n = V.pick_quantile_boxes(npts[elig], V.SEED + 1000 * ci + b)
    return elig[pick], qv, pool_n, len(elig)


def _p_rgba(p):
    import matplotlib.cm as cm
    return cm.Blues(P_LO + (1 - P_LO) * np.clip(np.asarray(p, np.float64), 0, 1))


def _patch_img(X, P, show='valid'):
    """RGBA image of a patch: p (Blues ramp) on valid pixels, hidden pixels light grey, elsewhere the surface."""
    X = np.asarray(X, np.float32); P = np.asarray(P, np.float32)
    img = np.ones(P.shape + (4,)); img[..., :3] = [0.988, 0.988, 0.984]
    hid = X[3] > 0; val = X[4] > 0
    img[hid] = [0.85, 0.85, 0.83, 1.0]
    img[val] = _p_rgba(P[val])
    return img


def _b6_panel(ax, e, W_deg, H_deg, s):
    """Sensor's-eye view of a copy about the virtual TOP: visible render grey, kept coloured by p, hidden sil x."""
    pose = V.sensor_pose('rr', height=2.184)
    box = e['box']
    az0 = V.view_angles(np.asarray(box[:3])[None], pose)[0][0]
    _, cel = V.view_angles(V.box_corners(box), pose)
    el0 = 0.5 * (cel.min() + cel.max()); lo, hi = el0 - H_deg / 2, el0 + H_deg / 2

    def X(az):
        return -V.wrap_deg(az - az0)
    for rr_ in e['thetas_deg']:
        if lo <= rr_ <= hi:
            ax.axhline(rr_, color=V.C_ROW, lw=0.8, zorder=0)
    th = np.radians(e['thetas_deg']); nc = 2650
    hp = e['sil_pix'][~e['sil_vis']]
    if len(hp):
        haz = np.degrees((hp % nc + 0.5) / nc * 2 * np.pi - np.pi); hel = np.degrees(th[hp // nc])
        ax.scatter(X(haz), hel, s=s * 0.9, marker='x', c='#c9c7c0', linewidths=0.6, zorder=1, rasterized=True)
    vaz, vel = V.view_angles(e['vis_xyz'], pose)
    ax.scatter(X(vaz), vel, s=s * 1.3, c=V.C_UNSP, lw=0, zorder=2, rasterized=True)
    kaz, kel = V.view_angles(e['kept_xyz'], pose)
    ax.scatter(X(kaz), kel, s=s, c=_p_rgba(np.nan_to_num(e['kept_p'])), lw=0, zorder=3, rasterized=True)
    for pl in V.box_edge_polylines(box):
        paz, pel = V.view_angles(pl, pose)
        ax.plot(X(paz), pel, color=V.INK2, lw=0.8, alpha=0.85, zorder=4)
    ax.set_xlim(-W_deg / 2, W_deg / 2); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor(V.SURF); ax.tick_params(colors=V.INK2, labelsize=9, length=2)
    for sp in ax.spines.values():
        sp.set_color(V.MUTED)


def _patch_panel(ax, img, frame_hw, dots=None, kept=None):
    ax.imshow(img, interpolation='nearest', origin='upper', extent=(-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5))
    if dots is not None and len(dots[0]):
        ax.scatter(dots[1], dots[0], s=6, c=V.INK, lw=0, zorder=3)
    if kept is not None and len(kept):
        ax.scatter(kept[:, 1], kept[:, 0], s=10, facecolors='none', edgecolors=V.INK, linewidths=0.6, zorder=3)
    ax.set_xlim(-0.5, frame_hw[1] - 0.5); ax.set_ylim(frame_hw[0] - 0.5, -0.5); ax.set_aspect('equal')
    ax.set_facecolor(V.SURF); ax.tick_params(colors=V.INK2, labelsize=8, length=2)
    for sp in ax.spines.values():
        sp.set_color(V.MUTED)


def plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.cm as cm
    from pcdet.datasets.augmentor import relocate_render as rr
    out = Path(a.out)
    B6 = []
    for k in V.NUS_KEYS:
        B6 += pickle.load(open(out / f'b6_{k}.pkl', 'rb'))
    w = np.load(out / 'w_heldout.npz')
    WM, WX, WP, WY = w['meta'], w['X'], w['pr'], w['Y']
    ents_C = [e for e in pickle.load(open(BPV_OUT / 'extract_CD.pkl', 'rb')) if e['col'] == 'C']
    train_rows = np.sort(np.degrees(rr.waymo_top_train_lattice()['thetas']))
    for e in ents_C:
        e['_train_rows'] = train_rows; e['metrics'] = V._metrics(e)
    rB = np.array([e['r'] for e in B6]); nB = np.array([e['desc'][0] for e in B6], np.float64)
    okB = (nB >= 1) & (np.array([e['length'] for e in B6]) < V.LONG_M)
    rW = WM[:, 2]; nW = WM[:, 4]; okW = nW >= 1
    summary = dict(n_b6=len(B6), b6_reproduced=int(sum(e['reproduced'] for e in B6)), n_w=len(WM), bins={})
    cmap = LinearSegmentedColormap.from_list('pBlues', cm.Blues(np.linspace(P_LO, 1, 256)))
    FW, AXW = 20.0, 4.3
    for b, (lo_r, hi_r) in enumerate(V.BINS):
        name = f'{lo_r:g}-{hi_r:g}'
        sB, qB, poolB, eB = _select(rB, nB, okB, 0, b)
        sW, qW, poolW, eW = _select(rW, nW, okW, 1, b)
        # pools' mean p by band (covered valid pixels)
        fb = np.zeros((2, 2, 4))
        for i in np.nonzero(okB & (bin_of(rB) == b))[0]:
            s_, n_ = patch_band_means(B6[i]['X'], B6[i]['pr']); fb[0, 0] += s_; fb[0, 1] += n_
        for i in np.nonzero(okW & (bin_of(rW) == b))[0]:
            s_, n_ = patch_band_means(WX[i], WP[i]); fb[1, 0] += s_; fb[1, 1] += n_
        with np.errstate(invalid='ignore', divide='ignore'):
            mB, mW = fb[0, 0] / fb[0, 1], fb[1, 0] / fb[1, 1]
        summary['bins'][name] = dict(b6_pool=int(poolB), b6_eligible=int(eB), b6_q=np.round(qB, 1).tolist(),
                                     w_pool=int(poolW), w_eligible=int(eW), w_q=np.round(qW, 1).tolist(),
                                     b6_mean_p_by_band=np.round(mB, 3).tolist(), w_mean_p_by_band=np.round(mW, 3).tolist(),
                                     b6_pixels_by_band=fb[0, 1].astype(int).tolist(), w_pixels_by_band=fb[1, 1].astype(int).tolist(),
                                     b6_picks=[[B6[i]['key'], B6[i]['frame'], B6[i]['j'], round(B6[i]['r'], 1),
                                                int(B6[i]['desc'][0])] for i in sB],
                                     w_picks=[[int(w['idx'][i]), round(float(rW[i]), 1), int(nW[i])] for i in sW])
        Wd, Hd = V.sheet_window(lo_r)
        axh = AXW * Hd / Wd
        hw = [(B6[i]['H'], B6[i]['W']) for i in sB]
        for i in sW:
            vv = (WX[i, 3] > 0) | (WX[i, 4] > 0); rr_, cc_ = np.nonzero(vv)
            hw.append((rr_.max() + 1, cc_.max() + 1))
        frame_hw = (max([x[0] for x in hw] + [8]) + 1, max([x[1] for x in hw] + [16]) + 1)
        pax_h = AXW * frame_hw[0] / frame_hw[1]
        row_h = max(axh, pax_h)
        top_in, title_in, xl_in, leg_in = 2.2, 0.95, 0.45, 1.25
        FH = top_in + 3 * (title_in + row_h + xl_in) + leg_in
        fig = plt.figure(figsize=(FW, FH), facecolor=V.SURF)
        fig.text(0.5, 1 - 0.22 / FH, f'{name} m: RR v6 arm (b\') copies, their U-Net p, a held-out Waymo patch, a Waymo val car',
                 ha='center', va='top', fontsize=17, color=V.INK, weight='bold')
        fig.text(0.5, 1 - 0.62 / FH,
                 f'B6 and C: sensor\'s-eye view, {Wd:.1f} x {Hd:.1f} deg, {Wd / AXW:.2f} deg per inch. B6-p and W: U-Net patch '
                 f'pixels (rows highest first, columns decreasing azimuth), {frame_hw[0]} x {frame_hw[1]} frame, one scale in '
                 f'this sheet.\nRows: the boxes nearest the p25 / p50 / p75 of point count in a fixed-seed pool (B6: kept points '
                 f'on 27471\'s grid; W: real returns in the patch; C: §4\'s picks). p colour: one scale for all sheets (bar below).\n'
                 f'B6: grey = visible rendered pixels, coloured = kept points by p, x = silhouette pixels hidden by the envelope. '
                 f'B6-p / W: p on visible silhouette pixels, grey = hidden; rings (B6-p) = kept pixels, dots (W) = real returns.',
                 ha='center', va='top', fontsize=10.5, color=V.INK2, linespacing=1.4)
        heads = ('B6. RR v6 (b\') copy\nkept points coloured by the U-Net\'s p', 'B6-p. The same copy\'s p map\nover its U-Net patch',
                 'W. Held-out Waymo label patch\nthe U-Net\'s p, real returns as dots', 'C. Waymo val car (§4)\nfor reference')
        x0_in, colw_in = 0.75, 4.85
        for ci in range(4):
            fig.text((x0_in + ci * colw_in + AXW / 2) / FW, 1 - (top_in - 0.12) / FH, heads[ci], ha='center', va='bottom',
                     fontsize=12, color=V.INK, weight='bold', linespacing=1.3)
        s = V.MARK[b]
        for qi, q in enumerate(V.QS):
            ytop = FH - top_in - qi * (title_in + row_h + xl_in) - title_in
            for ci in range(4):
                hh = axh if ci in (0, 3) else pax_h
                ax = fig.add_axes([(x0_in + ci * colw_in) / FW, (ytop - hh) / FH, AXW / FW, hh / FH])
                if ci in (0, 1):
                    if qi >= len(sB):
                        ax.text(0.5, 0.5, 'no copy in this bin', ha='center', va='center', transform=ax.transAxes); ax.axis('off')
                        continue
                    e = B6[sB[qi]]
                    if ci == 0:
                        _b6_panel(ax, e, Wd, Hd, s)
                        ttl = (f"p{q} · {V.PLAT_NAME[e['plat']]} · train info {e['frame']} · inst {e['instance']}\n"
                               f"r {e['r']:.1f} m · n_pts {int(e['desc'][0])} · n_vox {int(e['desc'][1])} · n_layers {int(e['desc'][2])}\n"
                               f"p mean {np.mean(e['vis_p']):.2f} · kept / visible {e['n_kept']} / {e['n_visible']} · rows kept "
                               f"{e['rows_kept']} / {e['rows_vis']}" + ('' if e['reproduced'] else ' · NOT REPRODUCED'))
                        if qi == 2:
                            ax.set_xlabel('azimuth from the box centre (deg; image left = sensor\'s left)', fontsize=9.5, color=V.INK2)
                        ax.set_ylabel('elevation (deg)', fontsize=9.5, color=V.INK2)
                    else:
                        Xp = np.asarray(e['X'], np.float32); Pp = np.asarray(e['pr'], np.float32)
                        _patch_panel(ax, _patch_img(Xp, Pp), frame_hw, kept=e['kept_ij'][e['kept_ij'][:, 0] >= 0])
                        cv = (Xp[1] > 0) & (Xp[4] > 0)
                        ttl = (f"patch {e['H']} x {e['W']} · valid {int((Xp[4] > 0).sum())} · covered {int(cv.sum())}\n"
                               f"p mean on covered {Pp[cv].mean():.2f} · on valid {Pp[Xp[4] > 0].mean():.2f}\n"
                               f"kept {e['n_kept']} of {int(cv.sum())} covered")
                        if qi == 2:
                            ax.set_xlabel('patch column', fontsize=9.5, color=V.INK2)
                elif ci == 2:
                    if qi >= len(sW):
                        ax.text(0.5, 0.5, 'no held-out patch in this bin\n(labels exist at far frames >= 45 m only)',
                                ha='center', va='center', transform=ax.transAxes, color=V.INK2, fontsize=10)
                        ax.set_xticks([]); ax.set_yticks([])
                        for sp in ax.spines.values():
                            sp.set_color(V.MUTED)
                        continue
                    i = sW[qi]
                    Xp = WX[i].astype(np.float32); Pp = WP[i].astype(np.float32); Yp = WY[i]
                    val = Xp[4] > 0; pos_ = np.nonzero(val & (Yp > 0))
                    _patch_panel(ax, _patch_img(Xp, Pp), frame_hw, dots=pos_)
                    cv = (Xp[1] > 0) & val
                    ttl = (f"p{q} · held-out patch {int(w['idx'][i])} · r {rW[i]:.1f} m\n"
                           f"real returns {int(nW[i])} of {int(val.sum())} valid ({nW[i] / max(val.sum(), 1):.2f})\n"
                           f"p mean on valid {Pp[val].mean():.2f} · on covered {Pp[cv].mean() if cv.any() else np.nan:.2f}")
                    if qi == 2:
                        ax.set_xlabel('patch column', fontsize=9.5, color=V.INK2)
                else:
                    ee = [e for e in ents_C if e['bin'] == b and e['q'] == q]
                    if not ee:
                        ax.axis('off'); continue
                    e = ee[0]
                    V._panel(ax, e, Wd, Hd, b)
                    ttl = V._title(e, e['metrics'])
                    if qi == 2:
                        ax.set_xlabel('azimuth from the box centre (deg)', fontsize=9.5, color=V.INK2)
                ax.set_title(ttl, fontsize=9.4, color=V.INK, loc='left', pad=4, linespacing=1.25)
        # footer: pools' mean p by band, colour bar, legends
        fy = (leg_in - 0.15) / FH
        fig.text(0.75 / FW, fy, 'Pools\' mean p on covered visible pixels, by h_rel band 0-0.25 / 0.25-0.5 / 0.5-0.75 / 0.75-1 '
                 f'(pixels):\nB6 ({eB} copies): ' + ' / '.join(f'{x:.2f}' for x in mB) + '   (' +
                 ' / '.join(f'{int(x):,}' for x in fb[0, 1]) + ')\nW (' + (f'{eW} held-out patches): ' + ' / '.join(
                     f'{x:.2f}' for x in mW) + '   (' + ' / '.join(f'{int(x):,}' for x in fb[1, 1]) + ')' if eW else
                     '0 held-out patches): none at this range'),
                 ha='left', va='top', fontsize=11, color=V.INK, linespacing=1.45)
        cax = fig.add_axes([(x0_in + 2 * colw_in) / FW, 0.62 / FH, 3.0 / FW, 0.16 / FH])
        cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1)), cax=cax, orientation='horizontal')
        cb.set_label('U-Net p (return probability)', fontsize=10, color=V.INK2); cb.ax.tick_params(labelsize=9, colors=V.INK2)
        lax = fig.add_axes([(x0_in + 3 * colw_in) / FW, 0.1 / FH, AXW / FW, (leg_in - 0.2) / FH]); lax.axis('off')
        lax.legend(handles=[Line2D([], [], ls='', marker='o', color=V.C_UNSP, label='B6: visible rendered pixel'),
                            Line2D([], [], ls='', marker='x', color='#c9c7c0', label='B6: hidden silhouette pixel'),
                            Line2D([], [], ls='', marker='o', color=V.C_TOP, label='C: TOP lidar'),
                            Line2D([], [], color=V.C_ROW, lw=1.5, label='Waymo TOP row')],
                   loc='upper left', fontsize=9.5, frameon=False, labelcolor=V.INK2)
        fig.savefig(out / f'sheet_{name}m.png', dpi=100, facecolor=V.SURF)
        plt.close(fig)
        print(f'[plot] sheet_{name}m.png; B6 picks {summary["bins"][name]["b6_picks"]}; W picks {summary["bins"][name]["w_picks"]}',
              flush=True)
    summary['diag'] = diag(B6, okB, rB, WX, WP, WM)
    json.dump(summary, open(out / 'v6b_summary.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != 'bins'}, indent=1, default=float)[:6000], flush=True)
    diag_figure(summary['diag'], out)


def diag(B6, okB, rB, WX, WP, WM):
    """Input-channel distributions over covered visible pixels (rank intensity, depth minus centre range, h_rel) and the
    mean p with matched range bin and h_rel band, B6 copies against W held-out patches."""
    res = {}
    rW = WM[:, 2]
    for b, (lo, hi) in enumerate(V.BINS):
        name = f'{lo:g}-{hi:g}'
        pools = {'B6': [(np.asarray(B6[i]['X'], np.float32), np.asarray(B6[i]['pr'], np.float32))
                        for i in np.nonzero(okB & (bin_of(rB) == b))[0]],
                 'W': [(WX[i].astype(np.float32), WP[i].astype(np.float32)) for i in np.nonzero(bin_of(rW) == b)[0]]}
        d = {}
        for lab, L in pools.items():
            if not L:
                d[lab] = None; continue
            ch = {0: [], 1: [], 2: []}; ps, bands = [], []
            for X, P in L:
                m = (X[1] > 0) & (X[4] > 0)
                for c in ch:
                    ch[c].append(X[c][m])
                ps.append(P[m]); bands.append(np.minimum((np.clip(X[2][m], 0, 1) * 4).astype(int), 3))
            ch = {c: np.concatenate(v) for c, v in ch.items()}; ps = np.concatenate(ps); bands = np.concatenate(bands)
            d[lab] = dict(n_objects=len(L), n_pixels=int(len(ps)),
                          depth_q10_25_50_75_90=np.round(np.percentile(ch[0], (10, 25, 50, 75, 90)), 3).tolist(),
                          irank_q10_25_50_75_90=np.round(np.percentile(ch[1], (10, 25, 50, 75, 90)), 3).tolist(),
                          hrel_q10_25_50_75_90=np.round(np.percentile(ch[2], (10, 25, 50, 75, 90)), 3).tolist(),
                          band_share=np.round(np.bincount(bands, minlength=4) / max(len(bands), 1), 3).tolist(),
                          mean_p=round(float(ps.mean()), 3),
                          mean_p_by_band=[round(float(ps[bands == k].mean()), 3) if (bands == k).any() else np.nan for k in range(4)],
                          valid_per_object=int(np.median([int((X[4] > 0).sum()) for X, _ in L])),
                          covered_share_of_valid=round(float(np.mean([((X[1] > 0) & (X[4] > 0)).sum() / max((X[4] > 0).sum(), 1)
                                                                       for X, _ in L])), 3))
            hist = {}
            for c, edges in ((0, np.linspace(-3, 3, 31)), (1, np.linspace(0, 1, 21)), (2, np.linspace(0, 1, 21))):
                hist[str(c)] = (np.histogram(np.clip(ch[c], edges[0], edges[-1]), edges)[0] / max(len(ch[c]), 1)).round(4).tolist()
            d[lab]['hist'] = hist
        res[name] = d
    return res


def diag_figure(dg, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [n for n in dg if dg[n].get('B6') and dg[n].get('W')]
    if not names:
        return
    fig, axs = plt.subplots(len(names), 3, figsize=(20, 3.4 * len(names)), facecolor=V.SURF, squeeze=False)
    labs = ('depth minus centre range (m)', 'intensity rank', 'h_rel')
    edges = {0: np.linspace(-3, 3, 31), 1: np.linspace(0, 1, 21), 2: np.linspace(0, 1, 21)}
    for r_, n in enumerate(names):
        for c in range(3):
            ax = axs[r_, c]; ax.set_facecolor(V.SURF)
            e = edges[c]; ctr = 0.5 * (e[1:] + e[:-1]); wdt = (e[1] - e[0]) * 0.42
            ax.bar(ctr - wdt / 2, dg[n]['B6']['hist'][str(c)], width=wdt, color='#2a78d6', label=f"B6 copies ({dg[n]['B6']['n_pixels']:,} px)")
            ax.bar(ctr + wdt / 2, dg[n]['W']['hist'][str(c)], width=wdt, color='#eb6834', label=f"W held-out ({dg[n]['W']['n_pixels']:,} px)")
            ax.set_title(f'{n} m: {labs[c]}', fontsize=11, color=V.INK, loc='left')
            ax.tick_params(colors=V.INK2, labelsize=9)
            for sp in ax.spines.values():
                sp.set_color(V.MUTED)
            if c == 0:
                ax.set_ylabel('share of covered visible pixels', fontsize=9.5, color=V.INK2)
            ax.legend(fontsize=9, frameon=False, labelcolor=V.INK2)
    fig.suptitle('U-Net input channels on covered visible pixels: RR v6 (b\') copies (B6) vs held-out Waymo patches (W)',
                 fontsize=14, color=V.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / 'diag_channels.png', dpi=100, facecolor=V.SURF); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['extract_b6', 'w_patches', 'plot'])
    ap.add_argument('--out', required=True); ap.add_argument('--key', default='NUSCENES_N008')
    ap.add_argument('--max_frames', type=int, default=0); ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    {'extract_b6': extract_b6, 'w_patches': w_patches, 'plot': plot}[a.mode](a)
    print(f'[{a.mode}] done in {time.time() - t0:.0f} s', flush=True)
