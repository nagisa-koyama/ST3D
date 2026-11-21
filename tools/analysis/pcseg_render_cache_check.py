"""Check an S3b render cache through the TRAINING loader path (experiments_md episodes/20261116_01 §8.9; the bug-prevention
checklist's rule 8: run a swap key at its identity setting against the base over >= 40 frames, points and labels exact).

For every cached frame (or every --every-th), PandasetSeg with FACTOR_OPS {RENDER: {CACHE: <dir>}} is loaded beside the
base PandasetSeg (same device, SHIFT_COOR, class map), and each rendered point is compared with the base frame's point
it names (src_off, src_idx):
  - label and intensity: must be EQUAL (they follow the index);
  - coordinates: |rendered - source| per point. For the identity cache (gt_on_gtrays_k0, one frame onto its own rays)
    and for frame-f winners of gt_on_gtrays, a point is re-placed on its own ray at its own range, so it must come back
    within 1 mm, up to near-duplicate rays (the share above 1 mm must stay <= 1%);
  - identity cache: the rendered frame must keep >= 97% of the frame's points (near-duplicate returns merge).

    python3 analysis/pcseg_render_cache_check.py --cache /home/koyama/data/pcseg/render_cache/v1/gt_on_gtrays_k0
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from pcseg.pandaset_seg import PandasetSeg                                                        # noqa: E402

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cfgs/pcseg/spunet-pandaset64.yaml')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--every', type=int, default=1)
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.cache, 'manifest.json')))
    rows = json.load(open(os.path.join(args.cache, 'frames.json')))[::args.every]
    d = yaml.safe_load(open(CFG))['DATA']
    seqs = sorted({r['seq'] for r in rows})
    base_cfg = {'ROOT': d['ROOT'], 'LIDAR_DEVICE': man['source_device'], 'SHIFT_COOR': d['SHIFT_COOR'],
                'CLASS_MAP': d['CLASS_MAP'], 'GRID_SIZE': d['GRID_SIZE'], 'SEQUENCES': seqs}
    base = PandasetSeg(base_cfg, training=True)
    rend = PandasetSeg(dict(base_cfg, FACTOR_OPS={'RENDER': {'CACHE': args.cache}}), training=True)
    index = {fr: i for i, fr in enumerate(base.frames)}
    identity = man['kind'] == 'gt_on_gtrays' and man['K'] == 0
    exact_f = man['kind'] == 'gt_on_gtrays'
    out = {'cache': args.cache, 'kind': man['kind'], 'K': man['K'], 'frames': 0, 'points': 0, 'label_mismatch': 0,
           'intensity_mismatch': 0, 'coord_dev_pct_by_offset': {}, 'share_above_1mm_frame_f': None,
           'kept_share_min': None, 'failures': []}
    dev = {o: [] for o in (-1, 0, 1)}
    kept = []
    for r in rows:
        seq, f = r['seq'], r['frame']
        rec = rend.render.record(seq, f)
        coord, strength, segment, _ = rend.load(index[(seq, f)])
        if len(coord) != len(rec['src_idx']):
            out['failures'].append('%s/%02d: loader returns %d points, the cache %d' % (seq, f, len(coord),
                                                                                        len(rec['src_idx'])))
            continue
        for o in (-1, 0, 1):
            sel = rec['src_off'] == o
            if not sel.any():
                continue
            bc, bs, bl, _ = base.load(index[(seq, f + o)])
            i = rec['src_idx'][sel].astype(np.int64)
            out['label_mismatch'] += int((segment[sel] != bl[i]).sum())
            out['intensity_mismatch'] += int((strength[sel, 0] != bs[i, 0]).sum())
            if o == 0 or identity:
                dev[o].append(np.linalg.norm(coord[sel] - bc[i], axis=1))
            if identity:
                kept.append(len(np.unique(i)) / len(bc))
        out['frames'] += 1
        out['points'] += len(coord)
    for o, v in dev.items():
        if v:
            v = np.concatenate(v)
            out['coord_dev_pct_by_offset'][str(o)] = {
                'p50': float(np.median(v)), 'p99': float(np.percentile(v, 99)), 'p999': float(np.percentile(v, 99.9)),
                'max': float(v.max()), 'share_above_1mm': float((v > 1e-3).mean()), 'n': int(len(v))}
    if exact_f and '0' in out['coord_dev_pct_by_offset']:
        out['share_above_1mm_frame_f'] = out['coord_dev_pct_by_offset']['0']['share_above_1mm']
        if out['share_above_1mm_frame_f'] > 0.01:
            out['failures'].append('frame-f winners: %.4f above 1 mm (> 1%%)' % out['share_above_1mm_frame_f'])
    if identity:
        out['kept_share_min'] = float(min(kept))
        out['kept_share_median'] = float(np.median(kept))
        if out['kept_share_min'] < 0.97:
            out['failures'].append('identity keeps only %.4f of a frame (< 0.97)' % out['kept_share_min'])
    if out['label_mismatch'] or out['intensity_mismatch']:
        out['failures'].append('labels %d / intensities %d differ from the indexed source'
                               % (out['label_mismatch'], out['intensity_mismatch']))
    if out['frames'] < 40:
        out['failures'].append('only %d frames checked (< 40)' % out['frames'])
    path = os.path.join(args.cache, 'check.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(json.dumps(out, indent=1))
    if out['failures']:
        sys.exit(1)


if __name__ == '__main__':
    main()
