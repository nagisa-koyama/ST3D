"""Arm (a) against arm (b') on (b')'s held-out Waymo label patches: pixel AUC and BCE (experiments_md 20261018_03 §11).
CPU, label-free (the labels are teacher-tracked TOP returns on Waymo TRAIN, 20261018_02 §10.3).

features: every held-out track (the 20% segments of rr_v6_unet.split_of_segments) is re-processed along
    rr_v6_unet.track_patches' steps (aggregate, far frames >= 45 m, render, envelope, patch, label); each regenerated
    patch must equal the stored one (X and Y bit-equal). At the patch positions of the rendered pixels it adds arm (a)'s
    inputs as relocate_render.return_features defines them: I_rel = the rendered aggregate point's stored intensity / the
    aggregate's median (clipped to [0.05, 20]); h_rel; r = the point's 3D range from the TOP origin; cos theta =
    |normal . ray| with the PCA normal of the aggregate (rr_build_db.instance_normals: 0.2 m, >= 5 other points; else 1;
    floor 0.01), the normal turned from the box frame by the box heading.
report: per range bin, AUC pooled and within patch, and BCE, for arm (a) (full, beta = 0, beta = delta = 0) and (b').

    python analysis/rr_ab_auc.py features --out <dir> [--workers 8]
    python analysis/rr_ab_auc.py report --out <dir>
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

PATCHES = Path('/home/koyama/data/relocate_render_20261018/v6/patches/patches_00.npz')
W_PASS = Path('/home/koyama/data/box_pattern_visual_20261018/v6b/w_heldout.npz')     # (b')'s p, job 29540
# arm (a), v6, converged (fit_a_29523.txt; 20261018_02 §10.6)
PAR_A = dict(c0=3.989, c_band=[-1.6292, 0.0, -4.3506, -7.3037], beta=0.3681, gamma=-5.9964, delta=0.285,
             log_sigma=1.1252)
I_CLIP, COS_MIN, R_REF = (0.05, 20.0), 0.01, 50.0
BINS = ((45.0, 50.0), (50.0, 60.0), (60.0, 80.0))
G = {}


# ------------------------------------------------------------------------------------------------ pure (tested)
def logit_a(i_rel, h_rel, r, cos, par=PAR_A, beta=None, delta=None):
    """Arm (a)'s logit without the latent (relocate_render.return_logit, 'physics')."""
    b = par['beta'] if beta is None else beta
    d = par['delta'] if delta is None else delta
    band = np.clip(np.floor(np.asarray(h_rel, np.float64) * 4).astype(np.int64), 0, 3)
    return (par['c0'] + np.asarray(par['c_band'])[band] + b * np.log(i_rel) + par['gamma'] * np.log(np.asarray(r) / R_REF)
            + d * np.log(cos))


def marginal_p(logit, log_sigma, n=16):
    """E_z[sigmoid(logit + sigma z)], z ~ N(0, 1), by n-point Gauss-Hermite."""
    x, w = np.polynomial.hermite.hermgauss(n)
    s = np.exp(log_sigma)
    L = np.asarray(logit, np.float64)[..., None] + np.sqrt(2.0) * s * x
    return (1.0 / (1.0 + np.exp(-L)) * w).sum(-1) / np.sqrt(np.pi)


def bce(p, y, eps=1e-6):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps); y = np.asarray(y, np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def within_auc(scores, labels, groups):
    """Mean over groups holding both classes of the per-group ROC AUC; (mean, number of groups)."""
    from sklearn.metrics import roc_auc_score
    g = np.asarray(groups); out = []
    o = np.argsort(g, kind='stable'); gs = g[o]
    starts = np.r_[0, np.nonzero(gs[1:] != gs[:-1])[0] + 1, len(gs)]
    for a_, b_ in zip(starts[:-1], starts[1:]):
        ii = o[a_:b_]; y = labels[ii]
        if 0 < y.sum() < len(y):
            out.append(roc_auc_score(y, scores[ii]))
    return (float(np.mean(out)) if out else np.nan), len(out)


# --------------------------------------------------------------------------------------------------- features
def track_features(tr):
    """rr_v6_unet.track_patches' steps for one held-out track, plus arm (a)'s inputs at the rendered patch positions."""
    import torch
    import rr_track_prereq as TP
    import rr_v6_unet as U
    from rr_build_db import instance_normals
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.waymo.waymo_rings import top_beam_ids
    from box_pattern_visual import sensor_pose, silhouette_pixels
    torch.set_num_threads(1)
    seq = tr['seq']; infos = TP.seg_infos(seq); cal = G['calib'][seq]
    E = np.asarray(cal['extrinsic'], np.float64); inc_desc = np.sort(np.asarray(cal['inclinations'], np.float64))[::-1]
    pose = sensor_pose('waymo', extrinsic=E)
    acc = rr.VoxelAccumulator(0.02, 4)
    for idx, vb in zip(tr['idx'], tr['vboxes']):
        top, keep = TP.top_block(seq, idx, infos)
        xyz = top[keep, :3].astype(np.float64)
        inb = rr.points_in_box(xyz, np.asarray(vb, np.float64))
        if inb.any():
            acc.add(np.c_[rr.to_box_frame(xyz[inb], vb), top[keep][inb, 3]])
    local = acc.points()
    if len(local) < 50:
        return []
    irank_all = rr.intensity_rank(local[:, 3])
    med = float(np.median(local[:, 3]))
    irel_all = np.clip(local[:, 3] / med, *I_CLIP) if med > 0 else np.ones(len(local))
    nrm_all, _ = instance_normals(local[:, :3])
    W = 2650
    res = []
    for idx, vb, r_ in zip(tr['idx'], tr['vboxes'], tr['rng']):
        if r_ < U.FAR_M:
            continue
        fb = np.asarray(vb, np.float64)
        top, keep = TP.top_block(seq, idx, infos)
        beam, colf, _, _, _ = top_beam_ids(top[:, :3], E, cal['inclinations'])
        pix_all = beam * W + (np.floor(colf).astype(np.int64) % W)
        xyz = top[:, :3].astype(np.float64)
        real_pix = np.unique(pix_all[keep & rr.points_in_box(xyz, fb)])
        sil, tin, tout, ent = silhouette_pixels(fb, pose, inc_desc, 'waymo', W, full=True)
        if not len(sil):
            continue
        srow, scol = sil // W, sil % W
        agg = rr.from_box_frame(local, fb)
        apix = TP.pixels_waymo(agg, E, inc_desc)
        arng = np.linalg.norm(agg - E[:3, 3], axis=1)
        o = np.lexsort((arng, apix)); first = np.r_[True, apix[o][1:] != apix[o][:-1]]; k = o[first]
        rpix, rr3 = apix[k], arng[k]
        ps = (agg[k] - E[:3, 3]) @ E[:3, :3]
        rhr = np.hypot(ps[:, 0], ps[:, 1])
        occ, plane = U.waymo_frame_occluders(xyz[keep], E, W)
        if occ is None:
            continue
        cols = np.unique(np.r_[scol, rpix % W])
        rref = np.full(len(cols), np.inf)
        np.minimum.at(rref, np.searchsorted(cols, scol), tin * np.cos(inc_desc[srow]))
        rrc = np.full(len(cols), np.inf); np.minimum.at(rrc, np.searchsorted(cols, rpix % W), rhr)
        rref = np.where(np.isfinite(rrc), rrc, rref)
        e = U.envelope_e(cols, rref, occ, W)
        sil_vis = inc_desc[srow] > e[np.searchsorted(cols, scol)]
        depth = rr3 - float(np.linalg.norm(fb[:3] - E[:3, 3]))
        hrel = np.clip((local[k, 2] + fb[5] / 2) / fb[5], 0.0, 1.0)
        out = rr.build_patch(srow, scol, sil_vis, rpix // W, rpix % W, depth, irank_all[k], hrel,
                             np.log(np.hypot(fb[0], fb[1]) / 50.0), W, fixed=True)
        if out is None:
            continue
        X, (ri, rj), (si, sj) = out
        Y, _ = U.patch_labels(sil, sil_vis, real_pix, si, sj)
        # arm (a)'s inputs at the rendered pixels (return_features' definitions, Waymo TOP as the sensor)
        d = agg[k] - E[:3, 3]; d = d / np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
        n = nrm_all[k].astype(np.float64); c, s_ = np.cos(fb[6]), np.sin(fb[6])
        nw = np.stack([c * n[:, 0] - s_ * n[:, 1], s_ * n[:, 0] + c * n[:, 1], n[:, 2]], 1)
        has = np.linalg.norm(n, axis=1) > 0
        cos = np.maximum(np.where(has, np.abs((nw * d).sum(1)), 1.0), COS_MIN)
        F = np.zeros((5, X.shape[1], X.shape[2]), np.float32)
        m = ri >= 0
        F[0, ri[m], rj[m]] = irel_all[k][m]; F[1, ri[m], rj[m]] = hrel[m]; F[2, ri[m], rj[m]] = rr3[m]
        F[3, ri[m], rj[m]] = cos[m]; F[4, ri[m], rj[m]] = has[m]
        res.append(dict(key=(int(tr['ti']), int(idx), round(float(np.hypot(fb[0], fb[1])), 4)), seq=seq,
                        X=X.astype(np.float16), Y=Y, F=F))
    return res


def features(a):
    from multiprocessing import Pool
    import rr_v6_unet as U
    from box_pattern_visual import TOP_CALIB
    tracks = pickle.load(open(U.TRACKS, 'rb'))
    split, _ = U.split_of_segments()
    ho_tracks = [t for t in tracks if not split[t['seq']]]
    G.update(calib=pickle.load(open(TOP_CALIB, 'rb')))
    z = np.load(PATCHES)
    M = z['meta']; hidx = np.nonzero(M[:, 6] == 0)[0]
    ZX, ZY = z['X'], z['Y']                  # read once: indexing the NpzFile re-reads the whole array every time
    keys = {}
    for i in hidx:
        keys.setdefault((int(M[i, 0]), int(M[i, 1]), round(float(M[i, 2]), 4)), []).append(int(i))
    print(f'[features] {len(ho_tracks)} held-out tracks, {len(hidx)} stored held-out patches', flush=True)
    F = np.zeros((len(hidx), 5, 48, 96), np.float16)
    pos_of = {int(i): n for n, i in enumerate(hidx)}
    done = np.zeros(len(hidx), bool); n_eq = n_neq = n_unmatched = n_ambig = 0
    t0 = time.time()
    with Pool(a.workers) as pool:
        for n, res in enumerate(pool.imap_unordered(track_features, ho_tracks, chunksize=2)):
            for r in res:
                cand = [i for i in keys.get(r['key'], []) if not done[pos_of[i]]]
                match = [i for i in cand if np.array_equal(ZX[i], r['X']) and np.array_equal(ZY[i], r['Y'])]
                if not cand:
                    n_unmatched += 1; continue
                if not match:
                    n_neq += 1; continue
                n_ambig += int(len(match) > 1)
                p = pos_of[match[0]]; done[p] = True; n_eq += 1
                F[p] = r['F'].astype(np.float16)
            if n % 200 == 0:
                print(f'[features] {n + 1}/{len(ho_tracks)} tracks, {n_eq} patches matched, {time.time() - t0:.0f} s', flush=True)
    np.savez(Path(a.out) / 'ab_features.npz', idx=hidx, F=F, done=done,
             F_cols=np.array(['i_rel', 'h_rel', 'r3d', 'cos', 'has_normal']))
    print(f'[features] matched bit-equal {n_eq} of {len(hidx)} stored held-out patches; regenerated but different '
          f'{n_neq}; without a stored key {n_unmatched}; duplicate keys {n_ambig}; {time.time() - t0:.0f} s', flush=True)


# ------------------------------------------------------------------------------------------------------ report
def report(a):
    from sklearn.metrics import roc_auc_score
    f = np.load(Path(a.out) / 'ab_features.npz'); w = np.load(W_PASS)
    assert np.array_equal(f['idx'], w['idx'])
    done = f['done']
    X = w['X'][done].astype(np.float32); Y = w['Y'][done].astype(np.int64); Pb = w['pr'][done].astype(np.float64)
    F = f['F'][done].astype(np.float64); r_box = w['meta'][done, 2]
    valid = X[:, 4] > 0
    covered = valid & (X[:, 1] > 0)
    assert np.all(F[:, 2][covered] > 0), 'a covered pixel without arm (a) features'
    pid = np.broadcast_to(np.arange(len(X))[:, None, None], valid.shape)
    sig = PAR_A['log_sigma']
    variants = {'a': {}, 'a, beta = 0': dict(beta=0.0), 'a, beta = delta = 0': dict(beta=0.0, delta=0.0)}
    res = dict(patches=int(done.sum()), of=int(len(done)), bins={})
    for lab_bin, (lo, hi) in [('all', (0, 1e9))] + [(f'{lo:g}-{hi:g}' if hi < 80 else f'{lo:g}+', (lo, hi)) for lo, hi in BINS]:
        pm = (r_box >= lo) & (r_box < hi)
        d = dict(patches=int(pm.sum()))
        for set_name, S in (('covered visible pixels', covered), ('all visible pixels', valid)):
            m = S & pm[:, None, None]
            y = Y[m]; g = pid[m]
            row = dict(pixels=int(m.sum()), positive_share=round(float(y.mean()), 4))
            for vn, kw in variants.items():
                L = np.full(Y.shape, -1e9)
                L[covered] = logit_a(F[:, 0][covered], F[:, 1][covered], F[:, 2][covered], F[:, 3][covered], **kw)
                s = L[m]
                pa = np.where(s > -1e8, marginal_p(np.maximum(s, -60.0), sig), 0.0)
                wa, ng = within_auc(s, y, g)
                row[vn] = dict(auc_pooled=round(float(roc_auc_score(y, s)), 4), auc_within=round(wa, 4), n_patches_both=ng,
                               bce_marginal=round(bce(pa, y), 4), mean_p=round(float(pa.mean()), 4))
            s = Pb[m]
            wa, ng = within_auc(s, y, g)
            row["b'"] = dict(auc_pooled=round(float(roc_auc_score(y, s)), 4), auc_within=round(wa, 4), n_patches_both=ng,
                             bce=round(bce(s, y), 4), mean_p=round(float(s.mean()), 4))
            d[set_name] = row
        res['bins'][lab_bin] = d
    nrm = F[:, 4][covered]
    res['normal_fallback_share_covered'] = round(float(1 - nrm.mean()), 4)
    res['i_rel_q10_50_90_covered'] = np.round(np.percentile(F[:, 0][covered], (10, 50, 90)), 3).tolist()
    res['cos_q10_50_90_covered'] = np.round(np.percentile(F[:, 3][covered], (10, 50, 90)), 3).tolist()
    json.dump(res, open(Path(a.out) / 'ab_auc.json', 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['features', 'report'])
    ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    {'features': features, 'report': report}[a.mode](a)
    print(f'[{a.mode}] done in {time.time() - t0:.0f} s', flush=True)
