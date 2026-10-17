"""Mean box size per model class over a nuScenes TRAINING split (SOURCE labels), for a point detector's box coder.

IA-SSD's box coder takes per-class `mean_size` and regresses residuals against it. Under the UDA-legality rule it must
come from the SOURCE's own train labels. The nuScenes-source IA-SSD configs quote [[4.63, 1.96, 1.74], [0.73, 0.67, 1.77],
[1.70, 0.61, 1.30]] (experiments_md 20261006_01 §3.6); the Car row matches 20260922_02's full-train-split mean. This tool
re-derives all three rows from the dataset's own train infos (experiments_md 20261017_08, item B2's mean_size check), the
nuScenes counterpart of `pandaset_source_mean_sizes.py`.

Boxes come from the dataset object's `infos` (train split, platform filter applied, no balanced resampling expected),
names mapped by the dataset -> model ontology as `prepare_data` maps them. Two counts are printed per class: every box,
and the boxes the loader keeps under the block's FILTER_MIN_POINTS_IN_GT (num_lidar_pts >= that value). No range or
point-budget filter. Read-only, CPU; it loads the info pickle, so run it as a Slurm CPU job.

    python analysis/nuscenes_source_mean_sizes.py <cfg> [DATA_CONFIG | DATA_CONFIGS key, default DATA_CONFIG]
"""
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import os  # noqa: E402
os.chdir(TOOLS)
import _init_path  # noqa: E402,F401
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

cfg = cfg_from_yaml_file(sys.argv[1], EasyDict())
key = sys.argv[2] if len(sys.argv) > 2 else 'DATA_CONFIG'
dcfg = cfg.DATA_CONFIG if key == 'DATA_CONFIG' else cfg.DATA_CONFIGS[key]
assert dcfg.DATASET == 'NuScenesDataset', dcfg.DATASET
assert not dcfg.get('BALANCED_RESAMPLING', False), 'balanced resampling duplicates infos: read the pickle instead'
assert not dcfg.get('CLASS_MAPPING', {}), 'CLASS_MAPPING set: this tool does not apply it'
ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                            logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
to_model = ds.map_ontology_dataset_to_model or {}
min_pts = dcfg.get('FILTER_MIN_POINTS_IN_GT', 0) or 0
sizes = {c: [] for c in cfg.CLASS_NAMES}
kept = {c: [] for c in cfg.CLASS_NAMES}
for info in ds.infos:
    names, boxes, npts = info['gt_names'], info['gt_boxes'], info['num_lidar_pts']
    for name, b, n in zip(names, boxes, npts):
        m = to_model.get(name, name)
        if m in sizes:
            sizes[m].append(b[3:6])
            if n >= min_pts:
                kept[m].append(b[3:6])
print(f'nuScenes {key} ({dcfg.get("VERSION")}, VEHICLE {dcfg.get("VEHICLE", "all")}, LOCATION {dcfg.get("LOCATION", "all")}), '
      f'{len(ds.infos)} TRAIN frames; ontology map {dict((k, v) for k, v in to_model.items() if v in sizes)}')
print(f'mean dx / dy / dz per model class (all boxes | boxes with num_lidar_pts >= {min_pts}, what the loader trains on):')
for c in cfg.CLASS_NAMES:
    out = []
    for v in (sizes[c], kept[c]):
        v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        ok = np.all(v > 0, axis=1) if len(v) else np.zeros(0, bool)
        mu = v[ok].mean(0) if ok.any() else np.full(3, np.nan)
        out.append(f'n={len(v):7d} (degenerate {int((~ok).sum())}) [{mu[0]:.2f}, {mu[1]:.2f}, {mu[2]:.2f}]')
    print(f'  {c:<11} ' + ' | '.join(out))
mean_size = cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG.get('mean_size') if cfg.get('MODEL') else None
print(f'config mean_size: {mean_size}')
