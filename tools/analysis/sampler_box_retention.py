"""Paired per-box point retention of a point sampler against its base config (same frames, same Car boxes).

Why: a sampler that de-clumps by emptying cars would pass a per-box ARRANGEMENT check at matched point counts
(box_pattern_source_vs_target.py bins by points in the box, so a thinned box simply moves to a lower bin). This reads
the same frames through the base config and each arm (augmentation off), checks the GT boxes are identical, and reports
per ring and per BASE point-count bin: boxes, median base points, median arm/base ratio, share of boxes keeping >= 10
points. ANALYSIS: reads source Car GT boxes.

    python analysis/sampler_box_retention.py <base cfg> <arm cfg> [<arm cfg> ...] <frames per source>
"""
import sys
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

PTS_BINS = [(10, 50), (50, 200), (200, 10**9)]
RINGS = [(0, 20), (20, 40), (40, 75)]
cfg_files, n = sys.argv[1:-1], int(sys.argv[-1])
logger = common_utils.create_logger()
cfgs = []
for f in cfg_files:
    c = EasyDict(); cfg_from_yaml_file(f, c); cfgs.append(c)


def car_counts(d):
    b = d['gt_boxes']; b = b[b[:, 7] == 1]
    if not len(b):
        return b, np.zeros(0, int)
    m = roiaware_pool3d_utils.points_in_boxes_cpu(d['points'][:, :3], b[:, :7])
    return b, (m > 0).sum(1)


rows, mismatched = [], 0
for key in cfgs[0].DATA_CONFIGS:
    dss = []
    for c in cfgs:
        ds, _, _ = build_dataloader(dataset_cfg=c.DATA_CONFIGS[key], class_names=c.CLASS_NAMES, batch_size=1, dist=False,
                                    workers=0, logger=logger, training=True, model_ontology=c.get('ONTOLOGY'))
        dss.append(ds)
    step = max(1, len(dss[0]) // n)
    ctx = [_augmentation_off(ds) for ds in dss]
    for cm in ctx:
        cm.__enter__()
    for i in range(0, len(dss[0]), step)[:n]:
        out = [car_counts(ds[i]) for ds in dss]
        b0 = out[0][0]
        if any(o[0].shape != b0.shape or not np.allclose(o[0], b0) for o in out[1:]):
            mismatched += 1
            continue
        r = np.hypot(b0[:, 0], b0[:, 1])
        rows.append(np.column_stack([r] + [o[1] for o in out]))
    for cm in ctx:
        cm.__exit__(None, None, None)
    del dss
rows = np.concatenate(rows)
print(f'\nframes skipped for mismatched boxes: {mismatched}; Car boxes: {len(rows)}')
print('base:', cfg_files[0])
for k, f in enumerate(cfg_files[1:], 1):
    print(f'\n## arm {k}: {f}')
    print('| ring | base points | boxes | median base pts | median arm/base | boxes keeping >= 10 pts |')
    print('|---|---|---|---|---|---|')
    for lo, hi in RINGS:
        for a, z in PTS_BINS:
            s = (rows[:, 0] >= lo) & (rows[:, 0] < hi) & (rows[:, 1] >= a) & (rows[:, 1] < z)
            if s.sum() < 5:
                continue
            ratio = rows[s, 1 + k] / rows[s, 1]
            print(f'| {lo}-{hi} | {a}-{z - 1 if z < 10**9 else ""} | {s.sum()} | {np.median(rows[s, 1]):.0f} | '
                  f'{np.median(ratio):.2f} | {(rows[s, 1 + k] >= 10).mean():.2f} |')
