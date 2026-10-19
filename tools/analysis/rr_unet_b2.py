"""RR arm (b''): the (b') U-Net retrained with input augmentation and a label-free validation carve-out
(experiments_md T05 §5, "Arm (b''), FIXED here at 12:33", 43279da; 20261018_03 §12). The architecture, channels and label
patches are (b')'s (rr_v6_unet.py, relocate_render._unet_class); nothing of the RR agent's code is edited.

prep (CPU): every TRAIN-segment track re-processed along rr_v6_unet.track_patches' steps (via rr_ab_auc.track_features,
    each patch bit-equal to the stored one), recording for each covered patch pixel the aggregate point behind it and each
    track's raw intensities, so the intensity augmentation can act BEFORE the within-instance rank transform. The
    validation carve-out: the sorted TRAIN segments, numpy default_rng(0) permutation, the first round(0.1 n) segments.
train (1 GPU): per patch, independently with probability 0.5 each,
    - intensity: each covered pixel's raw intensity x exp(eps), eps ~ N(0, 0.5^2); the covered pixels' ranks recomputed
      within the instance (rank among all its aggregate points, ties averaged, / n), as relocate_render.intensity_rank;
    - depth: + N(0, 0.15^2) m per covered pixel, then with probability 0.5 a 3 x 3 mean over covered pixels;
    per-sample generators default_rng([0, epoch, patch]). Adam lr 1e-3, batch 64, seed 0 (torch, the shuffling generator),
    at most 60 epochs; validation BCE (valid pixels, no augmentation) after every epoch; the lowest-validation epoch is kept,
    patience 5.
report (CPU): §11's table on the 20% held-out patches, (b'') beside (b') and (a).

    python analysis/rr_unet_b2.py prep --out <dir> [--workers 16]
    python analysis/rr_unet_b2.py train --out <dir> --weights <unet_b2.pt>
    python analysis/rr_unet_b2.py report --out <dir> --weights <unet_b2.pt>
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
VAL_SHARE, P_INT, P_DEPTH, P_FILTER = 0.10, 0.5, 0.5, 0.5
SIG_INT, SIG_DEPTH = 0.5, 0.15
MAX_EPOCHS, PATIENCE, LR, BATCH = 60, 5, 1e-3, 64


# ------------------------------------------------------------------------------------------------ pure (tested)
def val_segments(train_segments, share=VAL_SHARE, seed=0):
    """The validation carve-out: sorted TRAIN segments, default_rng(seed) permutation, the first round(share n)."""
    s = sorted(train_segments)
    perm = np.random.default_rng(seed).permutation(len(s))
    return {s[j] for j in perm[:int(round(share * len(s)))]}


def rank_within(vals):
    from scipy.stats import rankdata
    v = np.asarray(vals, np.float64)
    return rankdata(v, method='average') / max(len(v), 1)


def augment(X, cov, agg_int, rng):
    """One patch's augmented copy (float32). cov: (k, 3) int (row, col, aggregate index) of the covered pixels."""
    X = np.array(X, np.float32, copy=True)
    if not len(cov):
        return X
    i, j, k = cov[:, 0], cov[:, 1], cov[:, 2]
    if rng.random() < P_INT:
        v = np.array(agg_int, np.float64, copy=True)
        v[k] = v[k] * np.exp(rng.normal(0.0, SIG_INT, len(k)))
        X[1, i, j] = rank_within(v)[k]
    if rng.random() < P_DEPTH:
        d = X[0, i, j] + rng.normal(0.0, SIG_DEPTH, len(k)).astype(np.float32)
        if rng.random() < P_FILTER:
            from scipy.ndimage import convolve
            D = np.zeros(X.shape[1:], np.float64); Mk = np.zeros(X.shape[1:], np.float64)
            D[i, j] = d; Mk[i, j] = 1.0
            ker = np.ones((3, 3))
            s, c = convolve(D, ker, mode='constant'), convolve(Mk, ker, mode='constant')
            d = (s[i, j] / c[i, j]).astype(np.float32)
        X[0, i, j] = d
    return X


# ------------------------------------------------------------------------------------------------------- prep
def prep(a):
    from multiprocessing import Pool
    import rr_ab_auc as A
    import rr_v6_unet as U
    from box_pattern_visual import TOP_CALIB
    tracks = pickle.load(open(U.TRACKS, 'rb'))
    split, _ = U.split_of_segments()
    tr_tracks = [t for t in tracks if split[t['seq']]]
    vseg = val_segments([s for s, v in split.items() if v])
    A.G.update(calib=pickle.load(open(TOP_CALIB, 'rb')))
    z = np.load(PATCHES)
    M = z['meta']; ZX, ZY = z['X'], z['Y']
    tidx = np.nonzero(M[:, 6] == 1)[0]
    keys = {}
    for i in tidx:
        keys.setdefault((int(M[i, 0]), int(M[i, 1]), round(float(M[i, 2]), 4)), []).append(int(i))
    print(f'[prep] {len(tr_tracks)} TRAIN-segment tracks, {len(tidx)} stored train patches; validation segments '
          f'{len(vseg)} of {sum(split.values())}', flush=True)
    done = {}; covs = []; t_int = []; t_off = [0]; seg_of = {}
    n_eq = n_neq = n_unm = n_rank_bad = 0
    t0 = time.time()
    with Pool(a.workers) as pool:
        for n, res in enumerate(pool.imap_unordered(A.track_features, tr_tracks, chunksize=2)):
            if not res:
                continue
            slot = len(t_int)
            t_int.append(np.asarray(res[0]['agg_int'], np.float32)); t_off.append(t_off[-1] + len(res[0]['agg_int']))
            for r in res:
                cand = [i for i in keys.get(r['key'], []) if i not in done]
                match = [i for i in cand if np.array_equal(ZX[i], r['X']) and np.array_equal(ZY[i], r['Y'])]
                if not cand:
                    n_unm += 1; continue
                if not match:
                    n_neq += 1; continue
                i = match[0]; done[i] = slot; seg_of[i] = r['seq']; n_eq += 1
                cv = r['cov']
                rk = rank_within(r['agg_int'])[cv[:, 2]].astype(np.float16)
                n_rank_bad += int(not np.array_equal(rk, ZX[i][1, cv[:, 0], cv[:, 1]]))
                covs.append(np.c_[np.full(len(cv), i, np.int32), cv])
            if n % 500 == 0:
                print(f'[prep] {n + 1}/{len(tr_tracks)} tracks, {n_eq} patches matched, {time.time() - t0:.0f} s', flush=True)
    pidx = np.array(sorted(done), np.int64)
    np.savez(Path(a.out) / 'b2_prep.npz', patch=pidx, slot=np.array([done[i] for i in pidx], np.int64),
             val=np.array([seg_of[i] in vseg for i in pidx]), cov=np.concatenate(covs).astype(np.int32),
             t_off=np.array(t_off, np.int64), t_int=np.concatenate(t_int).astype(np.float32),
             val_segments=np.array(sorted(vseg)))
    print(f'[prep] matched bit-equal {n_eq} of {len(tidx)} stored train patches; different {n_neq}; without a key {n_unm}; '
          f'stored rank != recomputed rank in {n_rank_bad} patches; validation patches '
          f'{sum(seg_of[i] in vseg for i in pidx)}; {time.time() - t0:.0f} s', flush=True)


# ------------------------------------------------------------------------------------------------------ train
class AugSet:
    """Patches of one split with the augmentation; the epoch lives on the object (set before each loader pass)."""

    def __init__(self, X, Y, cov_of, int_of, augment_on):
        self.X, self.Y, self.cov_of, self.int_of, self.aug = X, Y, cov_of, int_of, augment_on
        self.epoch = 0; self.ids = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, n):
        x = self.X[n]
        if self.aug:
            x = augment(x, self.cov_of[n], self.int_of[n], np.random.default_rng([0, self.epoch, int(self.ids[n])]))
        else:
            x = x.astype(np.float32)
        return x, self.Y[n].astype(np.float32)


def train(a):
    import torch
    from pcdet.datasets.augmentor import relocate_render as rr
    torch.manual_seed(0); np.random.seed(0)
    p = np.load(Path(a.out) / 'b2_prep.npz')
    z = np.load(PATCHES)
    X = z['X']; Y = z['Y']
    pidx, slot, val = p['patch'], p['slot'], p['val']
    cov, t_off, t_int = p['cov'], p['t_off'], p['t_int']
    o = np.argsort(cov[:, 0], kind='stable'); cov = cov[o]
    starts = np.searchsorted(cov[:, 0], pidx); ends = np.searchsorted(cov[:, 0], pidx, side='right')
    cov_of = [cov[s:e, 1:] for s, e in zip(starts, ends)]
    int_of = [t_int[t_off[s]:t_off[s + 1]] for s in slot]
    fit, va = ~val, val
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert dev.type == 'cuda' or a.allow_cpu, 'no GPU visible (run the container with --nv)'
    ds = AugSet(X[pidx[fit]], Y[pidx[fit]], [c for c, f in zip(cov_of, fit) if f], [t for t, f in zip(int_of, fit) if f], True)
    ds.ids = pidx[fit]
    Xv = torch.from_numpy(X[pidx[va]].astype(np.float32)); Yv = torch.from_numpy(Y[pidx[va]].astype(np.float32))
    print(f'[train] device {dev}; fit patches {int(fit.sum())}, validation {int(va.sum())} '
          f'({len(p["val_segments"])} segments); held-out (20%) not loaded', flush=True)
    net = rr._unet_class()().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    g = torch.Generator().manual_seed(0)
    n = len(ds); curve, vcurve = [], []
    best, best_ep, best_state, bad = np.inf, 0, None, 0
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        ds.epoch = ep
        perm = torch.randperm(n, generator=g).numpy()
        batches = [perm[s:s + BATCH].tolist() for s in range(0, n, BATCH)]
        loader = torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=a.workers, persistent_workers=False)
        net.train(); tot = cnt = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            v = xb[:, 4] > 0
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xb)[v], yb[v])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * int(v.sum()); cnt += int(v.sum())
        curve.append(tot / max(cnt, 1))
        net.eval(); vt = vc = 0.0
        with torch.no_grad():
            for s in range(0, len(Xv), 256):
                xb, yb = Xv[s:s + 256].to(dev), Yv[s:s + 256].to(dev)
                v = xb[:, 4] > 0
                vt += float(torch.nn.functional.binary_cross_entropy_with_logits(net(xb)[v], yb[v], reduction='sum'))
                vc += int(v.sum())
        vcurve.append(vt / max(vc, 1))
        if vcurve[-1] < best:
            best, best_ep, bad = vcurve[-1], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        print(f'[train] epoch {ep}: train BCE {curve[-1]:.5f}, validation BCE {vcurve[-1]:.5f} (best {best:.5f} at {best_ep}; '
              f'{time.time() - t0:.0f} s)', flush=True)
        if bad >= PATIENCE:
            break
    torch.save(best_state, a.weights)
    res = dict(curve=curve, val_curve=vcurve, chosen_epoch=best_ep, val_bce=best, epochs_run=len(curve),
               max_epochs=MAX_EPOCHS, patience=PATIENCE, n_fit=int(fit.sum()), n_val=int(va.sum()),
               val_segments=int(len(p['val_segments'])), batch=BATCH, lr=LR,
               augmentation=dict(p_intensity=P_INT, sigma_log_intensity=SIG_INT, p_depth=P_DEPTH, sigma_depth_m=SIG_DEPTH,
                                 p_mean_filter=P_FILTER), device=str(dev), wall_s=time.time() - t0)
    json.dump(res, open(Path(a.weights).with_suffix('.json'), 'w'), indent=1)
    print(f'[train] saved {a.weights}: epoch {best_ep} of {len(curve)}, validation BCE {best:.5f}', flush=True)


# ----------------------------------------------------------------------------------------------------- report
def report(a):
    import torch
    from sklearn.metrics import roc_auc_score
    import rr_ab_auc as A
    from pcdet.datasets.augmentor import relocate_render as rr
    torch.set_num_threads(a.workers)
    f = np.load(A.W_PASS.parent.parent / 'ab_auc' / 'ab_features.npz'); w = np.load(A.W_PASS)
    assert np.array_equal(f['idx'], w['idx']) and f['done'].all()
    X = w['X'].astype(np.float32); Y = w['Y'].astype(np.int64); Pb = w['pr'].astype(np.float64)
    F = f['F'].astype(np.float64); r_box = w['meta'][:, 2]
    net = rr._unet_class()(); net.load_state_dict(torch.load(a.weights, map_location='cpu')); net.eval()
    P2 = np.zeros(Y.shape, np.float64)
    with torch.no_grad():
        for s in range(0, len(X), 128):
            P2[s:s + 128] = torch.sigmoid(net(torch.from_numpy(X[s:s + 128]))).numpy()
    valid = X[:, 4] > 0; covered = valid & (X[:, 1] > 0)
    pid = np.broadcast_to(np.arange(len(X))[:, None, None], valid.shape)
    La = np.full(Y.shape, -1e9)
    La[covered] = A.logit_a(F[:, 0][covered], F[:, 1][covered], F[:, 2][covered], F[:, 3][covered])
    res = dict(patches=int(len(X)), bins={})
    for lab, (lo, hi) in [('all', (0, 1e9))] + [(f'{lo:g}-{hi:g}' if hi < 80 else f'{lo:g}+', (lo, hi)) for lo, hi in A.BINS]:
        pm = (r_box >= lo) & (r_box < hi)
        d = dict(patches=int(pm.sum()))
        for set_name, S in (('covered visible pixels', covered), ('all visible pixels', valid)):
            m = S & pm[:, None, None]; y = Y[m]; gg = pid[m]
            row = dict(pixels=int(m.sum()), positive_share=round(float(y.mean()), 4))
            s = La[m]; pa = np.where(s > -1e8, A.marginal_p(np.maximum(s, -60.0), A.PAR_A['log_sigma']), 0.0)
            wa, ng = A.within_auc(s, y, gg)
            row['a'] = dict(auc_pooled=round(float(roc_auc_score(y, s)), 4), auc_within=round(wa, 4), n_patches_both=ng,
                            bce=round(A.bce(pa, y), 4), mean_p=round(float(pa.mean()), 4))
            for name, Pm in (("b'", Pb), ("b''", P2)):
                s = Pm[m]; wa, ng = A.within_auc(s, y, gg)
                row[name] = dict(auc_pooled=round(float(roc_auc_score(y, s)), 4), auc_within=round(wa, 4), n_patches_both=ng,
                                 bce=round(A.bce(s, y), 4), mean_p=round(float(s.mean()), 4))
            d[set_name] = row
        res['bins'][lab] = d
    json.dump(res, open(Path(a.out) / 'b2_auc.json', 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prep', 'train', 'report'])
    ap.add_argument('--out', required=True); ap.add_argument('--weights', default=None)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--allow_cpu', action='store_true')
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    {'prep': prep, 'train': train, 'report': report}[a.mode](a)
    print(f'[{a.mode}] done in {time.time() - t0:.0f} s', flush=True)
