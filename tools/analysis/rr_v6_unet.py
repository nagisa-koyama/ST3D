"""RR v6 arm (b'): label-free per-box patches from teacher tracks on Waymo TRAIN, and the U-Net (experiments_md
20261018_02 §10.3).

labels: per near-and-far teacher track (20261018_03 §9, read only), the aggregate of its in-box TOP first returns over all
    observations (box frame, 0.02 m centroid voxels, stored intensity); at each DETECTED far observation (>= 45 m) the
    aggregate rendered through that frame's own TOP lattice, the v5 occlusion envelope on the frame's TOP points (plane
    over 4-75.2 m), the label per visible silhouette pixel (a TOP first return inside the teacher box in that pixel), the
    6-channel 48 x 96 patch. No Waymo label is read: the boxes are teacher 27471's detections.
train: U-Net 32-64-128 on the train segments (the 80 / 20 split of targets.npz), Adam lr 1e-3, batch 64, 30 epochs,
    seed 0, 1 GPU; the epoch-30 weights; held-out per-pixel BCE and AUC.

    python analysis/rr_v6_unet.py labels --out <dir> --shard k --nshards N [--workers 16]
    python analysis/rr_v6_unet.py train --dir <dir> --out <weights.pt>
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _init_path  # noqa: F401,E402

TRACKS = Path('/home/koyama/data/box_pattern_visual_20261018/prereq/nearfar_tracks_train.pkl')
TARGETS = Path('/home/koyama/data/relocate_render_20261018/v5/targets.npz')
FAR_M = 45.0
PLANE_RMAX = 75.2
OCC_LO, OCC_HI, OCC_M = 0.3, 3.0, 0.5
G = {}


def split_of_segments():
    """{segment: True if train}: v5's split (sorted segments, default_rng(0) permutation, first round(0.8 n) train)."""
    import rr_track_prereq as TP
    ps = pickle.load(open(TP.PS_TRAIN, 'rb'))
    seqs = sorted({f.rsplit('_', 1)[0] for f in ps})
    perm = np.random.default_rng(0).permutation(len(seqs))
    n_fit = int(round(0.8 * len(seqs)))
    return {seqs[j]: (i < n_fit) for i, j in enumerate(perm)}, n_fit


def waymo_frame_occluders(top_xyz, E, n_cols=2650):
    """Occluder arrays (col, horizontal range, elevation) in the TOP sensor's axes, from the frame's TOP points whose height
    above the frame's RANSAC plane (4-75.2 m) lies in (0.3, 3.0] m (§10.3)."""
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.datasets.waymo.waymo_rings import top_columns_and_elevations
    plane = rr.ground_plane(top_xyz, PLANE_RMAX)
    if plane is None:
        return None, None
    hz = top_xyz[:, 2] - rr.plane_z(plane, top_xyz[:, 0], top_xyz[:, 1])
    q = top_xyz[(hz > OCC_LO) & (hz <= OCC_HI)]
    colf, el = top_columns_and_elevations(q, E)
    ps = (q - E[:3, 3]) @ E[:3, :3]
    occ = (np.floor(colf).astype(np.int64) % n_cols, np.hypot(ps[:, 0], ps[:, 1]), np.asarray(el, np.float64))
    return occ, plane


def envelope_e(cols, r_ref, occ, n_cols=2650, margin=OCC_M, nb=1):
    """e_occ per column from occluder arrays (the v5 rule, §9.1)."""
    oc, ohr, oel = occ
    o = np.argsort(oc, kind='stable'); oc, ohr, oel = oc[o], ohr[o], oel[o]
    start = np.searchsorted(oc, np.arange(n_cols + 1))
    e = np.full(len(cols), -np.inf)
    for k, (c, rr_) in enumerate(zip(cols, r_ref)):
        for d in range(-nb, nb + 1):
            cc = (int(c) + d) % n_cols
            s0, s1 = start[cc], start[cc + 1]
            if s1 > s0:
                m = ohr[s0:s1] < rr_ - margin
                if m.any():
                    e[k] = max(e[k], float(oel[s0:s1][m].max()))
    return e


def patch_labels(sil, sil_vis, real_pix, si, sj):
    """§10.3: label 1 at visible silhouette pixels holding a TOP first return inside the box, else 0 (48 x 96)."""
    from pcdet.datasets.augmentor import relocate_render as rr
    Y = np.zeros((rr.PATCH_H, rr.PATCH_W), np.uint8)
    lab = np.isin(sil, real_pix)
    m = np.asarray(sil_vis, bool) & lab
    Y[si[m], sj[m]] = 1
    return Y, lab


def track_patches(tr):
    """Patches and labels of one track at its detected far observations."""
    import torch
    import rr_track_prereq as TP
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
        return dict(dropped=1, X=[], Y=[], meta=[])
    irank_all = rr.intensity_rank(local[:, 3])
    W = 2650
    Xs, Ys, meta = [], [], []
    n_skip = 0
    for idx, vb, r_ in zip(tr['idx'], tr['vboxes'], tr['rng']):
        if r_ < FAR_M:
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
        occ, plane = waymo_frame_occluders(xyz[keep], E, W)
        if occ is None:
            continue
        cols = np.unique(np.r_[scol, rpix % W])
        rref = np.full(len(cols), np.inf)
        np.minimum.at(rref, np.searchsorted(cols, scol), tin * np.cos(inc_desc[srow]))
        rrc = np.full(len(cols), np.inf); np.minimum.at(rrc, np.searchsorted(cols, rpix % W), rhr)
        rref = np.where(np.isfinite(rrc), rrc, rref)
        e = envelope_e(cols, rref, occ, W)
        sil_vis = inc_desc[srow] > e[np.searchsorted(cols, scol)]
        depth = rr3 - float(np.linalg.norm(fb[:3] - E[:3, 3]))
        hrel = np.clip((local[k, 2] + fb[5] / 2) / fb[5], 0.0, 1.0)
        out = rr.build_patch(srow, scol, sil_vis, rpix // W, rpix % W, depth, irank_all[k], hrel,
                             np.log(np.hypot(fb[0], fb[1]) / 50.0), W, fixed=True)
        if out is None:
            n_skip += 1; continue
        X, _, (si, sj) = out
        Y, lab = patch_labels(sil, sil_vis, real_pix, si, sj)
        Xs.append(X.astype(np.float16)); Ys.append(Y)
        meta.append([tr['ti'], idx, float(np.hypot(fb[0], fb[1])), int(sil_vis.sum()), int((sil_vis & lab).sum()),
                     int(np.isin(sil[sil_vis], rpix).sum())])
    return dict(dropped=0, X=Xs, Y=Ys, meta=meta, skipped=n_skip, seq=seq)


def labels(a):
    from multiprocessing import Pool
    from box_pattern_visual import TOP_CALIB
    tracks = pickle.load(open(TRACKS, 'rb'))
    split, n_fit = split_of_segments()
    sel = [tracks[i] for i in range(a.shard, len(tracks), a.nshards)][:a.max_tracks or None]
    G.update(calib=pickle.load(open(TOP_CALIB, 'rb')))
    print(f'[labels] shard {a.shard}/{a.nshards}: {len(sel)} of {len(tracks)} tracks', flush=True)
    t0 = time.time(); X, Y, M = [], [], []; dropped = skipped = 0
    with Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(track_patches, sel, chunksize=2)):
            dropped += r['dropped']; skipped += r.get('skipped', 0)
            for x, y, m in zip(r['X'], r['Y'], r['meta']):
                X.append(x); Y.append(y); M.append(m + [float(split[r['seq']])])
            if n % 200 == 0:
                print(f'[labels] {n + 1}/{len(sel)} tracks, {len(X)} patches, {time.time() - t0:.0f} s', flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'patches_{a.shard:02d}.npz', X=np.stack(X) if X else np.zeros((0, 6, 48, 96), np.float16),
             Y=np.stack(Y) if Y else np.zeros((0, 48, 96), np.uint8), meta=np.array(M, np.float64).reshape(-1, 7),
             meta_cols=np.array(['track', 'frame', 'r', 'n_valid', 'n_pos', 'n_covered', 'train']),
             dropped=dropped, skipped=skipped)
    print(f'[labels] done shard {a.shard}: {len(X)} patches, {dropped} tracks dropped (< 50 points), {skipped} far frames '
          f'skipped (> 48 x 96), {time.time() - t0:.0f} s', flush=True)


def train(a):
    import torch
    from sklearn.metrics import roc_auc_score
    from pcdet.datasets.augmentor import relocate_render as rr
    torch.manual_seed(0); np.random.seed(0)
    fs = sorted(Path(a.dir).glob('patches_*.npz'))
    X = np.concatenate([np.load(f)['X'] for f in fs]); Y = np.concatenate([np.load(f)['Y'] for f in fs])
    M = np.concatenate([np.load(f)['meta'] for f in fs])
    tr = M[:, 6] == 1
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xt, Yt = torch.from_numpy(X[tr].astype(np.float32)), torch.from_numpy(Y[tr].astype(np.float32))
    print(f'[train] {len(fs)} shards, {len(X)} patches: train {int(tr.sum())}, held-out {int((~tr).sum())}; device {dev}; '
          f'positive share of valid train pixels {Yt[Xt[:, 4] > 0].mean():.4f}', flush=True)
    net = rr._unet_class()().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    n = len(Xt); curve = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train(); perm = torch.randperm(n, generator=g)
        tot, cnt = 0.0, 0.0
        for s in range(0, n, a.batch):
            idx = perm[s:s + a.batch]
            xb, yb = Xt[idx].to(dev), Yt[idx].to(dev)
            v = xb[:, 4] > 0
            lg = net(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lg[v], yb[v])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * int(v.sum()); cnt += int(v.sum())
        curve.append(tot / max(cnt, 1))
        print(f'[train] epoch {ep}: train BCE {curve[-1]:.5f} ({time.time() - t0:.0f} s)', flush=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, a.out)
    net.eval(); ho = ~tr
    ps, ys = [], []
    with torch.no_grad():
        for s in range(0, int(ho.sum()), 256):
            xb = torch.from_numpy(X[ho][s:s + 256].astype(np.float32)).to(dev)
            pr = torch.sigmoid(net(xb)).cpu().numpy(); v = X[ho][s:s + 256][:, 4] > 0
            ps.append(pr[v]); ys.append(Y[ho][s:s + 256][v])
    p = np.concatenate(ps).astype(np.float64); y = np.concatenate(ys).astype(np.float64)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    bce = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()); auc = float(roc_auc_score(y, p))
    res = dict(curve=curve, heldout_bce=bce, heldout_auc=auc, heldout_pixels=int(len(y)), heldout_pos=float(y.mean()),
               n_train=int(tr.sum()), n_heldout=int(ho.sum()), epochs=a.epochs, batch=a.batch)
    json.dump(res, open(Path(a.out).with_suffix('.json'), 'w'), indent=1)
    print(f'[train] saved {a.out}; held-out BCE {bce:.5f}, AUC {auc:.4f} over {len(y)} valid pixels (positive {y.mean():.4f})',
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='mode', required=True)
    l = sp.add_parser('labels'); l.add_argument('--out', required=True); l.add_argument('--shard', type=int, default=0)
    l.add_argument('--nshards', type=int, default=1); l.add_argument('--workers', type=int, default=16)
    l.add_argument('--max_tracks', type=int, default=None)
    t = sp.add_parser('train'); t.add_argument('--dir', required=True); t.add_argument('--out', required=True)
    t.add_argument('--epochs', type=int, default=30); t.add_argument('--batch', type=int, default=64)
    a = ap.parse_args()
    {'labels': labels, 'train': train}[a.mode](a)


if __name__ == '__main__':
    main()
