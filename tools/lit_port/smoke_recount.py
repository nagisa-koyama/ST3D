"""Per class: labelled boxes in the stored infos vs boxes left after the LIT_TRANSLATED recount (zero-point filter on the
translated cloud), over the frames of the smoke config. experiments_md 20261015_02 step 4.
    python analysis-style: python3 lit_port/smoke_recount.py <cfg> <n_frames>"""
import sys
sys.path.insert(0, '.')
import _init_path  # noqa
import numpy as np
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

cfg = EasyDict(); cfg_from_yaml_file(sys.argv[1], cfg)
ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                            logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
names = {'Vehicle': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist'}
before = dict.fromkeys(cfg.CLASS_NAMES, 0); after = dict.fromkeys(cfg.CLASS_NAMES, 0)
n = min(int(sys.argv[2]), len(ds))
for i in range(n):
    an = ds.infos[i]['annos']
    for nm in an['name']:
        if nm in names and names[nm] in before:
            before[names[nm]] += 1
    out = ds[i]
    for lab in out['gt_boxes'][:, 7]:
        after[cfg.CLASS_NAMES[int(abs(lab)) - 1]] += 1
print(f'{n} frames')
for c in cfg.CLASS_NAMES:
    print(f'  {c:11s} labelled {before[c]:5d}  kept after recount {after[c]:5d}  ({100 * after[c] / max(before[c], 1):.1f}%)')
