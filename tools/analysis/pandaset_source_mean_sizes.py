"""Mean box size per model class over a PandaSet TRAINING split (SOURCE labels), for a point detector's box coder.

IA-SSD's box coder takes per-class `mean_size` and regresses residuals against it. Under the UDA-legality rule it must
come from the SOURCE's own train labels (experiments_md 20261011_07); this is the PandaSet counterpart of the Waymo and
nuScenes means quoted in the IA-SSD configs. Boxes come from the dataset's own `_get_annotations` (the cuboids that
`__getitem__` would train on, `LIDAR_DEVICE` filter included), names mapped by the dataset -> model ontology, every box
counted (no range or point filter, as for the Waymo means). Sizes are rotation-invariant, so the world-frame cuboids
need no pose. CPU; a pass over the cuboid files, so run it as a Slurm CPU job.

    python analysis/pandaset_source_mean_sizes.py <cfg> [DATA_CONFIG key, default DATA_CONFIG]
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
ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                            logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
to_model = ds.map_ontology_dataset_to_model or {}
sizes = {c: [] for c in cfg.CLASS_NAMES}
n_frames = len(ds.pandaset_infos)
for k, info in enumerate(ds.pandaset_infos):
    boxes, labels, _ = ds._get_annotations(info, ds._get_pose(info))
    for b, name in zip(boxes, labels):
        m = to_model.get(name, name)
        if m in sizes:
            sizes[m].append(b[3:6])
    if k % 500 == 0:
        print(f'{k}/{n_frames} frames', flush=True)
print(f'PandaSet {key} (LIDAR_DEVICE {dcfg.get("LIDAR_DEVICE", 0)}), {n_frames} TRAIN frames; mean dx / dy / dz per class:')
for c, v in sizes.items():
    v = np.asarray(v, dtype=np.float64)
    if len(v):
        ok = np.all(v > 0, axis=1)
        mu = v[ok].mean(0)
        print(f'  {c:<11} n={len(v):6d} (degenerate {int((~ok).sum())}): [{mu[0]:.2f}, {mu[1]:.2f}, {mu[2]:.2f}]')
    else:
        print(f'  {c:<11} n=0')
