"""Per-5-m-bin parity ratios: nuScenes Singapore (n015) at N accumulated sweeps over KITTI (experiments_md T05 §5
item 42d.7, report 20261108_06).

Same pipeline as the parity-depth computation that gave Singapore -> KITTI N = 9 (experiments_md 20260922_05 §3,
`accumulation_matrix.py`, unchanged since b714c87): `domain_gap_analysis.build_platforms()` loaders on the TRAIN
infos, 360 deg (no camera-FOV crop), raw sensor frame (no SHIFT_COOR), nuScenes ego returns removed (1.5 m keyframe,
1.0 m sweeps, as the dataset class does), `mask_range` = POINT_CLOUD_RANGE [-75.2, -75.2, -2, 75.2, 75.2, 4],
`radial_hist` over the x-y range with 5 m bins (`linspace(0, 75, 16)`), averaged per frame. KITTI: `sample(40)`
frames. Singapore: `sample(anchors)` keyframes, each accumulated with its STORED sweeps (`Platform.accumulate`, i.e.
`MAX_SWEEPS: N`, N counting the anchor). `accumulation_matrix.py` cannot be imported (it runs at import), so its
two aggregation lines are repeated here on the same registry functions.

The depth rule reads the 5-70 m band (bins 1-13); this also prints 0-5 m. Only target POINT CLOUDS are read.
`--anchors 12` gives the 12-anchor sample of `accumulation_deep_nuscenes.py` / `gates/per_ring_parity_depth.py`
(for N <= 10 their keyframe chain is exactly the anchor plus its stored sweeps, and their anchors are the same
stride over the same info order), which is the T07 §5 "9 vs 10" conflict.

    python analysis/parity_ratio_bins.py --out <json> [--anchors 20] [--kitti-frames 40] [--depths 9,10]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from domain_gap_analysis import MAX_DIST, mask_range, radial_hist  # noqa: E402

EDGES = np.linspace(0, MAX_DIST, 16)  # accumulation_matrix.EDGES
N_BINS = 14                           # 0-70 m; the 70-75 m bin is outside the rule's band
BAND = slice(1, 14)                   # accumulation_matrix.BAND, 5-70 m
SINGAPORE = 'nuScenes n015 Singapore'


def mean_hist(clouds, edges=EDGES):
    """Mean per-frame radial histogram of range-masked clouds (each [M, >=3])."""
    return np.mean([radial_hist(mask_range(p), edges) for p in clouds], axis=0)


def ratio_rows(target_hist, source_hists, n_bins=N_BINS, edges=EDGES):
    """One row per bin [edges[b], edges[b+1]) for b < n_bins: target pts/frame, source pts/frame and ratio per depth."""
    rows = []
    for b in range(n_bins):
        row = {'lo': float(edges[b]), 'hi': float(edges[b + 1]), 'target': float(target_hist[b])}
        for n, h in sorted(source_hists.items()):
            row[f'src_{n}'] = float(h[b])
            row[f'ratio_{n}'] = float(h[b] / target_hist[b]) if target_hist[b] > 0 else float('nan')
        rows.append(row)
    return rows


def smallest_parity_depth(target_hist, source_hists, band=BAND):
    """accumulation_matrix.need_global: smallest N whose histogram meets the target in every bin of the band."""
    for n in sorted(source_hists):
        if np.all(source_hists[n][band] >= target_hist[band]):
            return n
    return None


def run(kitti_frames, anchors, depths):
    from domain_gap_analysis import build_platforms
    P = build_platforms()
    kitti, sg = P['KITTI'], P[SINGAPORE]
    k_infos, s_infos = kitti.sample(kitti_frames), sg.sample(anchors)
    th = mean_hist(kitti.frame(i).points for i in k_infos)
    sh = {n: mean_hist(sg.accumulate(i, n) for i in s_infos) for n in depths}
    return {
        'kitti_frames': len(k_infos), 'kitti_ids': [i['point_cloud']['lidar_idx'] for i in k_infos],
        'singapore_anchors': len(s_infos), 'singapore_tokens': [i['token'] for i in s_infos],
        'n_singapore_train_keyframes': len(sg), 'n_kitti_train_frames': len(kitti),
        'edges': EDGES.tolist(), 'kitti_hist': th.tolist(), 'singapore_hist': {n: h.tolist() for n, h in sh.items()},
        'bins_met_5_70': {n: int(np.sum(h[BAND] >= th[BAND])) for n, h in sh.items()},
        'smallest_parity_depth_5_70': smallest_parity_depth(th, sh),
        'rows': ratio_rows(th, sh),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--kitti-frames', type=int, default=40)
    ap.add_argument('--anchors', type=int, default=20)
    ap.add_argument('--depths', default='1,2,3,4,5,6,7,8,9,10')
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(',')]
    assert max(depths) <= 10, 'stored sweeps end at 10; deeper needs keyframe chaining (accumulation_deep_nuscenes)'
    res = run(args.kitti_frames, args.anchors, depths)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1)
    show = [n for n in (9, 10) if n in depths] or depths[-2:]
    print(f"KITTI train {res['kitti_frames']} frames; Singapore train {res['singapore_anchors']} anchors "
          f"(of {res['n_singapore_train_keyframes']} keyframes)")
    print('bin(m)   KITTI ' + ' '.join(f'  SG{n:<3d}  r{n:<3d}' for n in show))
    for r in res['rows']:
        print(f"{r['lo']:2.0f}-{r['hi']:2.0f}  {r['target']:7.0f} "
              + ' '.join(f"{r[f'src_{n}']:7.0f} {r[f'ratio_{n}']:5.2f}" for n in show))
    print('bins met (5-70 m, of 13):', res['bins_met_5_70'])
    print('smallest N at parity in every 5-70 m bin:', res['smallest_parity_depth_5_70'])


if __name__ == '__main__':
    main()
