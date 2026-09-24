"""How deep can nuScenes accumulate, and does that reach KITTI and Lyft?

`MAX_SWEEPS` caps at 10 because a nuScenes info carries 9 stored sweeps. Chaining KEYFRAMES lifts
that: each keyframe carries its own 9 sweeps covering the 0.45 s before it and keyframes are 0.5 s
apart, so consecutive keyframes' sweeps very nearly tile the scene. Points map back with
S(i) @ inv(S(j)) where S = ref_from_car @ car_from_global, a composition that reproduces nuScenes'
own stored sweep transform to 1e-13. The chain stops at a scene boundary, which is the real ceiling.

Companion to accumulation_matrix.py, which measures every source/target pair but is bounded by
MAX_SWEEPS. This script answers only the cells that came back ">10" there.

Contributions are additive - a union's radial histogram is the sum of the per-frame histograms, and
so are points-in-box counts against the anchor's boxes - so one backward pass evaluates every depth.

No motion compensation: at these windows an uncompensated moving object gains almost nothing
(20260922_05 Ablation 3, x1.01 over 30 frames), so deep foreground gain is carried by parked cars.
"""
import json, pickle, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from domain_gap_analysis import (build_platforms, mask_range, radial_hist, points_in_boxes,
                                 MAX_DIST, DATA)

NS = DATA / 'nuscenes/v1.0-trainval'
_meta = NS / 'v1.0-trainval'
_tok2scene = {s['token']: s['scene_token'] for s in json.load(open(_meta / 'sample.json'))}
INFOS = pickle.load(open(NS / 'nuscenes_infos_10sweeps_train.pkl', 'rb'))
SCENE = [_tok2scene[i['token']] for i in INFOS]

_ld = lambda p: np.fromfile(str(NS / p), dtype=np.float32).reshape([-1, 5])[:, :3]
_rm = lambda p, c: p[~((np.abs(p[:, 0]) < c) & (np.abs(p[:, 1]) < c))]
_S = lambda i: np.asarray(i['ref_from_car']) @ np.asarray(i['car_from_global'])

EDGES = np.linspace(0, MAX_DIST, 16)
BAND = slice(1, 14)                       # 5-70 m
GRID = [1, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
NMAX, ANCHORS = max(GRID), 12
TARGETS = ['KITTI', 'Lyft (all)', 'Lyft 40-beam', 'Lyft 64-beam']


def frames_back(a, nmax):
    """Per-frame point sets going back from anchor a, in the anchor keyframe's coordinates."""
    Sa = _S(INFOS[a]); k, n = a, 0
    while n < nmax and k >= 0 and SCENE[k] == SCENE[a]:
        info = INFOS[k]
        T = Sa @ np.linalg.inv(_S(info)); R, t = T[:3, :3], T[:3, 3]
        p = _rm(_ld(info['lidar_path']), 1.5)
        yield p if k == a else (R @ p.T).T + t
        n += 1
        for s in info['sweeps']:
            if n >= nmax:
                break
            q = _rm(_ld(s['lidar_path']), 1.0)
            tm = s['transform_matrix']
            if tm is not None:
                q = (tm[:3, :3] @ q.T).T + tm[:3, 3]
            yield (R @ q.T).T + t
            n += 1
        k -= 1


def main():
    P = build_platforms()
    TH, TB = {}, {}
    for t in TARGETS + ['nuScenes n008 Boston', 'nuScenes n015 Singapore']:
        plat = P[t]; hs, bc = [], []
        for info in plat.sample(40):
            fr = plat.frame(info); p = mask_range(fr.points)
            hs.append(radial_hist(p, EDGES))
            m = fr.names == plat.car_class
            if m.sum():
                b = fr.boxes[m]; r = np.linalg.norm(b[:, :2], axis=1)
                b = b[(r >= 5.0) & (r < 70.0)]
                if len(b):
                    bc += list(points_in_boxes(p, b))
        TH[t], TB[t] = np.mean(hs, axis=0), (float(np.median(bc)) if bc else float('nan'))

    print('%-24s %5s %10s %8s  %s' % ('platform', 'N', 'pts/frame', 'pts/box', 'bins met, 5-70 m'))
    OUT = {}
    for label in ['nuScenes n008 Boston', 'nuScenes n015 Singapore']:
        toks = {i['token'] for i in P[label].infos}
        idx = [k for k in range(len(INFOS)) if INFOS[k]['token'] in toks]
        anchors = idx[::max(1, len(idx) // ANCHORS)][:ANCHORS]
        H = {n: [] for n in GRID}; Bx = {n: [] for n in GRID}; reach = []
        for a in anchors:
            info = INFOS[a]
            g = np.asarray(info['gt_boxes']); m = np.asarray(info['gt_names']) == 'car'
            boxes = None
            if m.sum():
                b = g[m][:, :7]; r = np.linalg.norm(b[:, :2], axis=1)
                b = b[(r >= 5.0) & (r < 70.0)]
                boxes = b if len(b) else None
            ch = np.zeros(15); cb = None if boxes is None else np.zeros(len(boxes)); n = 0
            for p in frames_back(a, NMAX):
                q = mask_range(np.column_stack([p, np.zeros(len(p))]))[:, :3]
                ch = ch + radial_hist(q, EDGES)
                if boxes is not None:
                    cb = cb + points_in_boxes(q, boxes)
                n += 1
                if n in H:
                    H[n].append(ch.copy())
                    if boxes is not None:
                        Bx[n].append(cb.copy())
            reach.append(n)
        res = {}
        for n in GRID:
            if not H[n]:
                continue
            h = np.mean(H[n], axis=0)
            bb = np.concatenate(Bx[n]) if Bx[n] else None
            res[n] = (h, float(np.median(bb)) if bb is not None else float('nan'))
            print('%-24s %5d %10.0f %8.0f  %s' % (label, n, h.sum(), res[n][1],
                  '  '.join('%s %d/13' % (t[:9], int(np.sum(h[BAND] >= TH[t][BAND]))) for t in TARGETS)))
        OUT[label] = res
        print('   median depth reached in-scene: %d frames\n' % int(np.median(reach)))

    print('\nSMALLEST N REACHING PARITY (keyframe chaining, NOT MAX_SWEEPS)\n')
    print('%-24s %-14s %9s %9s' % ('source', 'target', 'global', 'object'))
    for label, res in OUT.items():
        for t in TARGETS:
            gg = next((n for n in sorted(res) if np.all(res[n][0][BAND] >= TH[t][BAND])), None)
            oo = next((n for n in sorted(res) if res[n][1] >= TB[t]), None)
            print('%-24s %-14s %9s %9s' % (label, t, gg or '>%d' % NMAX, oo or '>%d' % NMAX))
    print("""
The grid is coarse, so a value of 15 means "between 11 and 15".
Rows deeper than a scene allows include only the anchors whose chains reached that far, so the box
population changes with N - which is why points/box goes non-monotonic past N=50. Nothing at or
below N=30 is affected. The object criterion is also undersampled: nuScenes Boston's points/box
swings 2.5-3x between independent samples (16.30 Car/frame, many distant), while Singapore's is
stable. Treat Boston object figures as indicative; the global figures are solid.""")


if __name__ == '__main__':
    main()
