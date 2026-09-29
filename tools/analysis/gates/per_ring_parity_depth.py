"""Per-RING accumulation depth from GLOBAL radial parity - the target-free rule behind
ACCUMULATION_DEPTH_BY_RANGE (20260929_06 item 4).

accumulation_deep_nuscenes.py asks for the smallest N at which EVERY 5-70 m bin of the source's
mean radial histogram meets the target's. That single N over-accumulates the near field to buy
parity in the far one. This asks the question per bin instead: for each 5 m bin, the smallest N at
which that bin alone meets the target's, using the same keyframe chaining, the same 40-frame
target histogram and the same anchors. Only target POINT CLOUDS are read - UDA-legal.

    python per_ring_parity_depth.py [anchors=12] [nmax=30]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import accumulation_deep_nuscenes as D
from domain_gap_analysis import build_platforms, mask_range, radial_hist

ANCHORS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 30
GRID = [n for n in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30] if n <= NMAX]
EDGES = D.EDGES
BINS = [(EDGES[i], EDGES[i + 1]) for i in range(len(EDGES) - 1)]

P = build_platforms()
kitti = P['KITTI']
th = np.mean([radial_hist(mask_range(kitti.frame(i).points), EDGES) for i in kitti.sample(40)], axis=0)

print('per-bin ratio source(N) / KITTI, global radial profile, 5 m bins; anchors=%d nmax=%d' % (ANCHORS, NMAX))
for label in ['nuScenes n008 Boston', 'nuScenes n015 Singapore']:
    toks = {i['token'] for i in P[label].infos}
    idx = [k for k in range(len(D.INFOS)) if D.INFOS[k]['token'] in toks]
    anchors = idx[::max(1, len(idx) // ANCHORS)][:ANCHORS]
    H = {n: [] for n in GRID}
    for a in anchors:
        ch = np.zeros(len(BINS)); n = 0
        for p in D.frames_back(a, NMAX):
            q = mask_range(np.column_stack([p, np.zeros(len(p))]))[:, :3]
            ch = ch + radial_hist(q, EDGES); n += 1
            if n in H:
                H[n].append(ch.copy())
    print('\n== %s' % label)
    print('%-5s ' % 'N' + ' '.join('%6s' % ('%d-%d' % b) for b in BINS[:14]))
    first = {}
    for n in GRID:
        if not H[n]:
            continue
        r = np.mean(H[n], axis=0) / np.maximum(th, 1e-9)
        print('%-5d ' % n + ' '.join('%6.2f' % x for x in r[:14]))
        for b in range(14):
            if r[b] >= 1.0 and b not in first:
                first[b] = n
    print('smallest N reaching parity per bin: ' + ' '.join('%d-%d:%s' % (BINS[b][0], BINS[b][1], first.get(b, '>%d' % NMAX)) for b in range(14)))
    rings = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 70)]
    sched = []
    for lo, hi in rings:
        bs = [b for b in range(14) if BINS[b][0] >= lo and BINS[b][1] <= hi]
        need = [first.get(b, NMAX + 1) for b in bs]
        sched.append((lo, max(need) if need else None))
    print('ring depth (max over its bins, %d+1 = not reached): %s' % (NMAX, sched))
