"""RR v5 return-model fit (experiments_md 20261018_02 §9.2): label-free targets, fitting copies, the fit. CPU.

targets: teacher 27471's stored Waymo TRAIN pass (no target label is read): Car, score >= 0.18 (27704's Car SCORE_THRESH),
    length < 6 m, BEV range in [15, 75) m; per object the visible silhouette, rows and fill by 20261018_03 §7-8's code
    (box_pattern_visual.py, imported); unoccluded = visible / full >= 0.9; fit / held-out split 80 / 20 by segment.
copies: RR v5 copies (return model off) on 400 nuScenes TRAIN frames per platform, disjoint from the gate's 200; per
    unoccluded copy (envelope visible fraction >= 0.9), per visible silhouette pixel its row and the features of the
    visible rendered pixel there.
fit: arm a (physics, 8 parameters) or b (MLP 4-32-32-1 + sigma), expected descriptors over 16 fixed latent draws per copy,
    loss = bin x band squared differences of mean hit and mean row-lost rate + per-bin fill p25 / p50 / p75; torch Adam,
    seed 0, 2,000 steps, full batch.

    python analysis/rr_v5_fit.py targets --out <dir> [--workers 16]
    python analysis/rr_v5_fit.py copies --cfg <v5 cfg> --key NUSCENES_N008 --out <dir>
    python analysis/rr_v5_fit.py fit --arm a|b --dir <dir> [--steps 2000]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _init_path  # noqa: F401,E402

TOOLS = Path(__file__).resolve().parents[1]
PASS = Path('/home/koyama/data/pseudo_labels/jguwly5u_ep20_waymo_train_int10_thr0.0001/ps_label_e0.pkl')
CAR_THRESH = 0.18                       # 27704's Car SCORE_THRESH (rule B 0.17 raised to rule A's NEG_THRESH 0.180)
BINS = ((15.0, 25.0), (25.0, 40.0), (40.0, 50.0), (50.0, 60.0), (60.0, 75.0))
NB = 4
UNOCC = 0.9
N_DRAWS = 16
PLAT_ID = {'NUSCENES_N008': 0, 'NUSCENES_N015': 1}


def bin_of(r):
    r = np.asarray(r, np.float64)
    out = np.full(r.shape, -1, np.int64)
    for k, (lo, hi) in enumerate(BINS):
        out[(r >= lo) & (r < hi)] = k
    return out


def object_descriptors(rows_pix, rows_relh, hit):
    """Per object from its visible silhouette pixels: (per-band hit sums, per-band visible pixels, per-band rows, per-band
    lost rows, fill). rows_pix: the pixels' row ids; rows_relh: per pixel its entry relh; hit: per pixel 0 / 1 (or a
    probability). A row's band is that of the median relh of its pixels (box_pattern_visual.row_records)."""
    from box_pattern_visual import _band_of
    ur, inv = np.unique(rows_pix, return_inverse=True)
    k = len(ur)
    relh = np.array([np.median(rows_relh[inv == j]) for j in range(k)]) if k else np.zeros(0)
    band = _band_of(relh)
    nv = np.bincount(inv, minlength=k).astype(np.float64)
    nh = np.bincount(inv, weights=hit, minlength=k)
    hs, vs, rs, ls = (np.zeros(NB) for _ in range(4))
    for b in range(NB):
        m = band == b
        hs[b] = nh[m].sum(); vs[b] = nv[m].sum(); rs[b] = m.sum(); ls[b] = (nh[m] == 0).sum()
    return hs, vs, rs, ls, (nh.sum() / nv.sum()) if nv.sum() else np.nan


# ----------------------------------------------------------------------------------------------------- targets
G = {}


def _target_frame(task):
    import torch
    from scipy.spatial import cKDTree
    from pcdet.datasets.waymo.waymo_rings import top_beam_ids
    from source_coverage import in_box_indices
    import box_pattern_visual as BV
    torch.set_num_threads(1)
    fid, seq, sidx, n_top, boxes, seg_id = task
    raw = np.load(G['data_path'] / seq / ('%04d.npy' % sidx))
    cal = G['calib'][seq]
    beam, colf, _, inc_desc, _ = top_beam_ids(raw[:n_top, :3], cal['extrinsic'], cal['inclinations'])
    W = BV.W_TOP
    pix_all = beam * W + (np.floor(colf).astype(np.int64) % W)
    pose = BV.sensor_pose('waymo', extrinsic=cal['extrinsic'])
    top = raw[:n_top, :3].astype(np.float64)
    rng_all = np.linalg.norm(top - pose['origin'], axis=1)
    keep = raw[:n_top, 5] == -1
    npp = BV.nearest_per_pixel(pix_all, rng_all)
    P = np.ascontiguousarray(top[keep], np.float32); pk = pix_all[keep]; kidx = np.nonzero(keep)[0]
    tree = cKDTree(P[:, :2]) if len(P) else None
    idx = in_box_indices(P, boxes, tree)
    out = []
    for k in range(len(boxes)):
        bx = boxes[k]
        inbox = np.zeros(n_top, bool); inbox[kidx[idx[k]]] = True
        sil, tin, tout, ent = BV.silhouette_pixels(bx, pose, inc_desc, 'waymo', W, full=True)
        if not len(sil):
            continue
        vis = BV.visible_mask(sil, tin, npp, inbox)
        vsp, vent = sil[vis], ent[vis]
        n_full = len(np.unique(sil)); n_vis = len(np.unique(vsp))
        if not n_vis:
            continue
        relh = np.clip((vent[:, 2] - (bx[2] - bx[5] / 2)) / bx[5], 0.0, 1.0)
        hit = np.isin(vsp, pk[idx[k]]).astype(np.float64)
        hs, vs, rs, ls, fill = object_descriptors(vsp // W, relh, hit)
        out.append(np.r_[seg_id, float(np.hypot(bx[0], bx[1])), n_vis / n_full, fill, hs, vs, rs, ls, len(idx[k])])
    return out


def targets(a):
    from multiprocessing import Pool
    t0 = time.time()
    ps = pickle.load(open(PASS, 'rb'))
    fids = sorted(ps)
    seqs = sorted({f.rsplit('_', 1)[0] for f in fids})
    perm = np.random.default_rng(0).permutation(len(seqs))
    n_fit = int(round(0.8 * len(seqs)))
    seg_id = {seqs[j]: i for i, j in enumerate(perm)}            # seg_id < n_fit: fit set
    infos = pickle.load(open(TOOLS.parent / 'data/waymo/waymo_infos_train.pkl', 'rb'))
    want = set(fids); meta = {}
    for inf in infos:
        pc = inf['point_cloud']; fid = pc['lidar_sequence'] + '_%03d' % pc['sample_idx']
        if fid in want:
            meta[fid] = (pc['lidar_sequence'], int(pc['sample_idx']), int(inf['num_points_of_each_lidar'][0]))
    del infos
    tasks = []; n_sel = 0
    for f in fids:
        d = ps[f]
        gb = np.asarray(d['gt_boxes'], np.float64).reshape(-1, 9)
        m = (np.asarray(d['teacher_classes']) == 'Car') & (np.asarray(d['cls_scores']) >= CAR_THRESH) & (gb[:, 3] < 6.0)
        r = np.hypot(gb[:, 0], gb[:, 1]); m &= (r >= BINS[0][0]) & (r < BINS[-1][1])
        if m.any() and f in meta:
            seq, sidx, n_top = meta[f]
            tasks.append((f, seq, sidx, n_top, gb[m, :7], seg_id[seq])); n_sel += int(m.sum())
    del ps
    import box_pattern_visual as BV
    G.update(data_path=TOOLS.parent / 'data/waymo/waymo_processed_data', calib=pickle.load(open(BV.TOP_CALIB, 'rb')))
    print(f'[targets] {len(fids)} frames, {len(seqs)} segments ({n_fit} fit), {len(tasks)} frames with {n_sel} selected '
          f'boxes; {len(meta)} frames found in the infos; {time.time() - t0:.0f} s', flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for k, out in enumerate(pool.imap_unordered(_target_frame, tasks[:a.max_frames or None], chunksize=8)):
            rows += out
            if k % 2000 == 0:
                print(f'[targets] {k + 1}/{len(tasks)} frames, {len(rows)} boxes, {time.time() - t0:.0f} s', flush=True)
    T = np.array(rows, np.float64)
    cols = ['seg', 'r', 'visfrac', 'fill'] + [f'{n}{b}' for n in ('hit', 'vis', 'rows', 'lost') for b in range(NB)] + ['n_pts']
    np.savez(Path(a.out) / 'targets.npz', T=T, cols=np.array(cols), n_fit_segments=n_fit, n_segments=len(seqs),
             car_thresh=CAR_THRESH)
    un = T[:, 2] >= UNOCC
    print(f'[targets] {len(T)} boxes, unoccluded {int(un.sum())} (fit {int((un & (T[:, 0] < n_fit)).sum())}, held-out '
          f'{int((un & (T[:, 0] >= n_fit)).sum())}); per bin unoccluded: '
          + ', '.join(f'{lo:g}-{hi:g} {int((un & (bin_of(T[:, 1]) == j)).sum())}' for j, (lo, hi) in enumerate(BINS))
          + f'; {time.time() - t0:.0f} s', flush=True)


# ------------------------------------------------------------------------------------------------------ copies
def copies(a):
    import rr_gate
    from pcdet.datasets.augmentor import relocate_render as rr
    plat = PLAT_ID[a.key]
    cfg, dcfg, ds = rr_gate.build(a.cfg, a.key)
    aug = ds.data_augmentor; q0 = aug.data_augmentor_queue[0]
    rr_cfg = q0.keywords['config']; lat = q0.keywords['lattice']; db = q0.keywords['db']
    assert int(rr_cfg.VERSION) >= 5
    rr_cfg['RETURN_OFF'] = True                               # §9.2: fitting copies with the return model off
    nc = lat['n_cols']
    CAP = {}

    def step(data_dict=None):
        o, rec = rr.relocate_render_scene_v2(data_dict, rr_cfg, lat, db, rng=np.random, keep_points=True)
        CAP['rec'] = rec
        return o
    aug.data_augmentor_queue = [step] + list(aug.data_augmentor_queue[1:])
    N = len(ds)
    gate = {k * (N // 200) for k in range(200)}
    frames = [int(np.floor((k + 0.5) * N / 400)) for k in range(400)]
    dropped = [i for i in frames if i in gate]
    frames = [i for i in frames if i not in gate][:a.max_frames]
    print(f'[copies] {a.key}: {len(frames)} frames (dropped {len(dropped)} coinciding with the gate sample)', flush=True)
    C, S_copy, S_row, S_relh, S_feat, F = [], [], [], [], [], []
    n_all = 0; t0 = time.time(); fo = 0; ro = 0
    for n, i in enumerate(frames):
        np.random.seed(((plat + 2) * 1000003 + i) % (2 ** 31))
        CAP.clear(); ds[i]
        for c in CAP['rec']['copies']:
            n_all += 1
            if not (c['vis_frac'] >= UNOCC):
                continue
            cid = len(C)
            vsp = c['sil_vis_pix']; vrel = c['sil_vis_relh']
            vp, vf = c['vis_pix'], c['vis_feats']
            pos = {int(px): j for j, px in enumerate(vp)}
            fi = np.array([pos.get(int(px), -1) for px in vsp], np.int64)
            C.append([plat, i, c['r_new'], c['vis_frac'], c['instance']])
            rows = vsp // nc
            ur, inv = np.unique(rows, return_inverse=True)
            S_copy.append(np.full(len(vsp), cid)); S_row.append(ro + inv); S_relh.append(vrel)
            S_feat.append(np.where(fi >= 0, fo + fi, -1)); F.append(vf)
            fo += len(vf); ro += len(ur)
        if n % 50 == 0:
            print(f'[copies] {n + 1}/{len(frames)} frames, {len(C)} unoccluded of {n_all} copies, {time.time() - t0:.0f} s',
                  flush=True)
    np.savez(Path(a.out) / f'copies_{a.key}.npz', C=np.array(C, np.float64).reshape(-1, 5),
             s_copy=np.concatenate(S_copy), s_row=np.concatenate(S_row), s_relh=np.concatenate(S_relh),
             s_feat=np.concatenate(S_feat), F=np.concatenate(F) if F else np.zeros((0, 4)), n_all=n_all,
             frames=np.array(frames), dropped=np.array(dropped))
    print(f'[copies] done {a.key}: {len(C)} unoccluded copies of {n_all}, {fo} visible rendered pixels, {ro} rows, '
          f'{time.time() - t0:.0f} s', flush=True)


# --------------------------------------------------------------------------------------------------------- fit
def target_stats(T, cols, sel):
    """Bin x band mean hit and mean row-lost rate (objects with >= 1 row in the band), and per-bin fill quantiles."""
    c = {k: j for j, k in enumerate(cols)}
    b = bin_of(T[:, c['r']])
    hit = np.full((len(BINS), NB), np.nan); lost = np.full((len(BINS), NB), np.nan); n = np.zeros((len(BINS), NB))
    fq = np.full((len(BINS), 3), np.nan)
    for j in range(len(BINS)):
        m = sel & (b == j)
        if m.sum():
            fq[j] = np.percentile(T[m, c['fill']], (25, 50, 75))
        for k in range(NB):
            mm = m & (T[:, c[f'rows{k}']] > 0)
            n[j, k] = mm.sum()
            if mm.any():
                hit[j, k] = np.mean(T[mm, c[f'hit{k}']] / T[mm, c[f'vis{k}']])
                lost[j, k] = np.mean(T[mm, c[f'lost{k}']] / T[mm, c[f'rows{k}']])
    return hit, lost, fq, n


class CopyData:
    """Tensors of the fitting copies for the expected descriptors."""

    def __init__(self, paths, eps_seed=0):
        import torch
        from box_pattern_visual import _band_of
        Cs, sc, sr, srel, sf, Fs = [], [], [], [], [], []
        co = ro = fo = 0
        for p in paths:
            z = np.load(p)
            Cs.append(z['C']); sc.append(z['s_copy'] + co); sr.append(z['s_row'] + ro); srel.append(z['s_relh'])
            sf.append(np.where(z['s_feat'] >= 0, z['s_feat'] + fo, -1)); Fs.append(z['F'])
            co += len(z['C']); ro += int(z['s_row'].max()) + 1 if len(z['s_row']) else 0; fo += len(z['F'])
        C = np.concatenate(Cs); s_copy = np.concatenate(sc); s_row = np.concatenate(sr); s_relh = np.concatenate(srel)
        s_feat = np.concatenate(sf); F = np.concatenate(Fs)
        n_rows = int(s_row.max()) + 1
        relh_row = np.array([np.median(v) for v in np.split(s_relh[np.argsort(s_row, kind='stable')],
                                                             np.cumsum(np.bincount(s_row, minlength=n_rows))[:-1])])
        band_row = _band_of(relh_row)
        row_copy = np.zeros(n_rows, np.int64); row_copy[s_row] = s_copy
        self.n_copies, self.n_rows = len(C), n_rows
        self.bin = bin_of(C[:, 2])
        self.C = C
        T = lambda x, dt=torch.float64: torch.as_tensor(x, dtype=dt)
        self.F = T(F); self.s_copy = T(s_copy, torch.int64); self.s_row = T(s_row, torch.int64)
        self.s_feat = T(np.maximum(s_feat, 0), torch.int64); self.s_has = T(s_feat >= 0)
        self.band_row = T(band_row, torch.int64); self.row_copy = T(row_copy, torch.int64)
        self.npix_copy = T(np.bincount(s_copy, minlength=len(C)))
        cb = row_copy * NB + band_row                              # copy x band of each row
        self.rows_cb = T(np.bincount(cb, minlength=len(C) * NB).reshape(len(C), NB))
        npix_row = np.bincount(s_row, minlength=n_rows)
        self.vis_cb = T(np.bincount(cb, weights=npix_row, minlength=len(C) * NB).reshape(len(C), NB))
        self.cb_row = T(cb, torch.int64)
        self.feat_copy = T(np.zeros(len(F), np.int64), torch.int64)
        fc = np.zeros(len(F), np.int64); fc[s_feat[s_feat >= 0]] = s_copy[s_feat >= 0]; self.feat_copy = T(fc, torch.int64)
        self.eps = T(np.random.default_rng(eps_seed).standard_normal((len(C), N_DRAWS)))
        self.bin_t = T(self.bin, torch.int64)


def descriptors_torch(cd, logit, log_sigma):
    """Expected (C x 4 x D hit, C x 4 x D lost, C x D fill) of the copies under the model (D latent draws)."""
    import torch
    z = torch.exp(log_sigma) * cd.eps                                   # C x D
    p_feat = torch.sigmoid(logit[:, None] + z[cd.feat_copy])           # F x D
    p_pix = p_feat[cd.s_feat] * cd.s_has[:, None]                      # S x D
    D = z.shape[1]; C = cd.n_copies
    row_hit = torch.zeros(cd.n_rows, D, dtype=p_pix.dtype).index_add_(0, cd.s_row, p_pix)
    row_lost = torch.exp(torch.zeros(cd.n_rows, D, dtype=p_pix.dtype).index_add_(0, cd.s_row, torch.log1p(-p_pix.clamp(max=1 - 1e-9))))
    hit_cb = torch.zeros(C * NB, D, dtype=p_pix.dtype).index_add_(0, cd.cb_row, row_hit).view(C, NB, D)
    lost_cb = torch.zeros(C * NB, D, dtype=p_pix.dtype).index_add_(0, cd.cb_row, row_lost).view(C, NB, D)
    hit_rate = hit_cb / cd.vis_cb.clamp(min=1)[:, :, None]
    lost_rate = lost_cb / cd.rows_cb.clamp(min=1)[:, :, None]
    fill = hit_cb.sum(1) / cd.npix_copy[:, None]
    return hit_rate, lost_rate, fill


class Model:
    def __init__(self, arm, seed=0):
        import torch
        torch.manual_seed(seed)
        self.arm = arm
        dt = torch.float64
        if arm == 'a':
            self.theta = torch.zeros(7, dtype=dt, requires_grad=True)     # c0, c_band0, c_band2, c_band3, beta, gamma, delta
            self.params = [self.theta]
        else:
            self.net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU(),
                                           torch.nn.Linear(32, 1)).double()
            self.params = list(self.net.parameters())
        self.log_sigma = torch.zeros((), dtype=dt, requires_grad=True)
        self.params.append(self.log_sigma)

    def logit(self, F):
        import torch
        i_rel, h_rel, r, cos = F[:, 0], F[:, 1], F[:, 2], F[:, 3]
        if self.arm == 'a':
            t = self.theta
            cb = torch.stack([t[1], torch.zeros((), dtype=t.dtype), t[2], t[3]])
            band = torch.clamp((h_rel * 4).floor().long(), 0, 3)
            return t[0] + cb[band] + t[4] * torch.log(i_rel) + t[5] * torch.log(r / 50.0) + t[6] * torch.log(cos)
        x = torch.stack([torch.log(i_rel), h_rel, torch.log(r / 50.0), cos], 1)
        return self.net(x)[:, 0]

    def export(self):
        if self.arm == 'a':
            t = self.theta.detach().numpy()
            return dict(c0=t[0], c_band=np.array([t[1], 0.0, t[2], t[3]]), beta=t[4], gamma=t[5], delta=t[6],
                        log_sigma=float(self.log_sigma.detach()))
        L = [m for m in self.net if hasattr(m, 'weight')]
        return dict(W1=L[0].weight.detach().numpy().T, b1=L[0].bias.detach().numpy(), W2=L[1].weight.detach().numpy().T,
                    b2=L[1].bias.detach().numpy(), W3=L[2].weight.detach().numpy().T, b3=L[2].bias.detach().numpy(),
                    log_sigma=float(self.log_sigma.detach()))


def fit_model(cd, tgt, arm, steps=2000, seed=0, log_every=200, converge=None, verbose=False):
    """Fit one arm (§9.2). tgt = (hit (B x 4), lost (B x 4), fill quantiles (B x 3)) of the fit targets.
    converge = (window, rel): stop at the first step t >= window with L(t - window) - L(t) < rel * L(t - window) (§10.2),
    `steps` being the cap. Returns (model, [(step, loss)] every log_every and at the end); model.stop_step is set."""
    import torch
    torch.manual_seed(seed)
    m = Model(arm, seed)
    opt = torch.optim.Adam(m.params)
    th, tl, tf = (torch.as_tensor(x, dtype=torch.float64) for x in tgt)
    qs = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    hist, L = [], []
    t0 = time.time()
    for it in range(steps + 1):
        opt.zero_grad()
        loss = fit_loss(cd, m, th, tl, tf, qs)
        L.append(float(loss))
        stop = it == steps
        if converge is not None and it >= converge[0] and L[it - converge[0]] - L[it] < converge[1] * L[it - converge[0]]:
            stop = True
        if it % log_every == 0 or stop:
            hist.append((it, L[-1]))
            if verbose:
                print(f'[fit] step {it}: loss {L[-1]:.5f} ({time.time() - t0:.0f} s)', flush=True)
        if stop:
            m.stop_step = it
            break
        loss.backward(); opt.step()
    m.losses = L
    return m, hist


def fit_loss(cd, m, th, tl, tf, qs):
    import torch
    hit, lost, fill = descriptors_torch(cd, m.logit(cd.F), m.log_sigma)
    loss = torch.zeros((), dtype=torch.float64)
    for j in range(len(BINS)):
        mj = cd.bin_t == j
        if not bool(mj.any()):
            continue
        for k in range(NB):
            mk = mj & (cd.rows_cb[:, k] > 0)
            if not bool(mk.any()) or not np.isfinite(float(th[j, k])):
                continue
            loss = loss + (hit[mk, k].mean() - th[j, k]) ** 2 + (lost[mk, k].mean() - tl[j, k]) ** 2
        if np.isfinite(float(tf[j, 0])):
            loss = loss + ((torch.quantile(fill[mj].reshape(-1), qs) - tf[j]) ** 2).sum()
    return loss


def fit(a):
    D = Path(a.dir)
    z = np.load(Path(a.targets) if a.targets else D / 'targets.npz')
    T, cols, n_fit = z['T'], [str(c) for c in z['cols']], int(z['n_fit_segments'])
    sel = (T[:, 2] >= UNOCC) & (T[:, 0] < n_fit)
    th, tl, tf, tn = target_stats(T, cols, sel)
    cd = CopyData([D / f'copies_{k}.npz' for k in PLAT_ID])
    print(f'[fit] arm {a.arm}: {int(sel.sum())} fit targets, {cd.n_copies} fitting copies, {int(cd.s_has.sum())} visible '
          f'rendered pixels in {len(cd.s_row)} visible silhouette pixels; targets per bin x band\n{tn}', flush=True)
    t0 = time.time()
    conv = (500, 0.01) if a.converge else None
    m, hist = fit_model(cd, (th, tl, tf), a.arm, a.steps, converge=conv, verbose=True)
    par = m.export()
    np.savez(D / f'fit_{a.arm}.npz', **{k: np.asarray(v, np.float64) for k, v in par.items()})
    np.save(D / f'fit_{a.arm}_loss.npy', np.array(m.losses))
    print(f'[fit] arm {a.arm}: loss {hist[0][1]:.5f} -> {hist[-1][1]:.5f}, stopped at step {m.stop_step} '
          f'({"converged: < 1% over the last 500 steps" if a.converge and m.stop_step < a.steps else "step cap"}; '
          f'{time.time() - t0:.0f} s); trajectory {[(i, round(l, 5)) for i, l in hist]}', flush=True)
    if a.arm == 'a':
        print('[fit] arm a parameters: ' + ', '.join(f'{k} {np.round(v, 4).tolist()}' for k, v in par.items()), flush=True)
    else:
        print(f'[fit] arm b: sigma {np.exp(par["log_sigma"]):.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='mode', required=True)
    t = sp.add_parser('targets'); t.add_argument('--out', required=True); t.add_argument('--workers', type=int, default=16)
    t.add_argument('--max_frames', type=int, default=None)
    c = sp.add_parser('copies'); c.add_argument('--cfg', required=True); c.add_argument('--key', required=True)
    c.add_argument('--out', required=True); c.add_argument('--max_frames', type=int, default=None)
    f = sp.add_parser('fit'); f.add_argument('--arm', required=True, choices=('a', 'b')); f.add_argument('--dir', required=True)
    f.add_argument('--steps', type=int, default=2000)
    f.add_argument('--targets', default=None, help='targets.npz (default: <dir>/targets.npz)')
    f.add_argument('--converge', action='store_true', help='§10.2: stop when < 1% better over 500 steps (cap --steps)')
    a = ap.parse_args()
    {'targets': targets, 'copies': copies, 'fit': fit}[a.mode](a)


if __name__ == '__main__':
    main()
