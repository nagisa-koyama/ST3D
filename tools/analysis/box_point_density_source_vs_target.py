"""Points per Car box AS THE MODEL SEES THEM: the accumulated source (training mode) against the target (eval mode).

Why: the nuScenes -> Waymo residual is mostly missed SPARSE objects (experiments_md/20261007_03): 10-199-point
Waymo cars are 47% of the GT and the accumulated-source model does not propose them. This asks whether the
densified source still contains such cars at all. Counting is done on the points that reach the detector (after
the data processor, inside POINT_CLOUD_RANGE), so both sides are comparable; the source is read in training mode
(augmentation on, as trained), the target in eval mode. ANALYSIS: target labels are read to diagnose only.

    python analysis/box_point_density_source_vs_target.py <cfg> <n_frames_per_source> <n_target_frames>
"""
import sys
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

cfg_file, n_src, n_tgt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
RINGS = [0, 10, 20, 30, 40, 50, 75]
BINS = [0, 1, 10, 50, 200, 10**9]
logger = common_utils.create_logger()


def collect(ds, n, car_label=1):
    """Per Car box: (range, points inside, after the processor)."""
    out = []
    step = max(1, len(ds) // n)
    for i in range(0, len(ds), step)[:n]:
        d = ds[i]
        b = d['gt_boxes']
        if not len(b):
            continue
        b = b[b[:, 7] == car_label]
        if not len(b):
            continue
        c = roiaware_pool3d_utils.points_in_boxes_cpu(d['points'][:, :3], b[:, :7]).sum(1)
        out.append(np.stack([np.linalg.norm(b[:, :2], axis=1), c], 1))
    return np.concatenate(out)


def table(name, arr):
    print(f'\n{name}: {len(arr)} Car boxes')
    print('| ring | boxes | ' + ' | '.join(f'{BINS[i]}-{BINS[i+1]-1 if BINS[i+1] < 10**9 else ""}' for i in range(len(BINS) - 1)) + ' | median pts |')
    print('|---' * (len(BINS) + 2) + '|')
    for k in range(len(RINGS) - 1):
        s = (arr[:, 0] >= RINGS[k]) & (arr[:, 0] < RINGS[k + 1])
        if not s.any():
            continue
        c = arr[s, 1]
        shares = [((c >= BINS[i]) & (c < BINS[i + 1])).mean() for i in range(len(BINS) - 1)]
        print(f'| {RINGS[k]}-{RINGS[k+1]} | {s.sum()} | ' + ' | '.join(f'{v:.2f}' for v in shares) + f' | {np.median(c):.0f} |')
    c = arr[:, 1]
    print('| all | ' + f'{len(c)} | ' + ' | '.join(f'{((c >= BINS[i]) & (c < BINS[i+1])).mean():.2f}' for i in range(len(BINS) - 1)) + f' | {np.median(c):.0f} |')


for key, dc in cfg.DATA_CONFIGS.items():
    ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
    table(f'SOURCE {key} (training mode, MAX_SWEEPS {dc.get("MAX_SWEEPS")})', collect(ds, n_src))
    del ds

dst, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                             workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
table('TARGET Waymo val (eval mode)', collect(dst, n_tgt))
