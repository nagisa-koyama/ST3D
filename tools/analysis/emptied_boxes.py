"""Share of TRAINING ground-truth boxes that hold >= 1 point BEFORE the data processor and 0 after it.

Why it exists (experiments_md/20261003_04 sections 16-17): `DatasetTemplate.prepare_data` applies the
zero-point GT filter (MIN_POINTS_OF_GT) BEFORE `data_processor.forward`, so any processor step that drops
points (sample_points_hist_based, sample_points_learned) leaves the boxes it empties in the labels, and the
detector is trained to fire on empty space. The first S3 learned sampler (L5, job 27261) emptied 80% of Car
training boxes this way. Run it on any new point-dropping row BEFORE launching it.

    python analysis/emptied_boxes.py <cfg> <DATA_CONFIGS key or DATA_CONFIG> <n_frames>
(PandaSet sources need the extra container bind --bind /home/koyama/code/ST3D:/root/ST3D.)
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

cfg_file, key, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
dcs = cfg.get('DATA_CONFIGS') or {'DATA_CONFIG': cfg.DATA_CONFIG}
ds, _, _ = build_dataloader(dataset_cfg=dcs[key], class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                            logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
proc = ds.data_processor
orig = proc.forward
captured = {}
def spy(data_dict):
    captured['before'] = data_dict['points'][:, :3].copy()
    out = orig(data_dict=data_dict)
    return out
proc.forward = spy
tot = {}; emp = {}; npts_b = []; npts_a = []
step = max(1, len(ds) // n)
for i in range(0, len(ds), step)[:n]:
    d = ds[i]
    b = d['gt_boxes']; b = b[b[:, 7] > 0] if len(b) else b
    if not len(b): continue
    pa = d['points'][:, :3] if 'points' in d else None
    if pa is None: continue
    cb = roiaware_pool3d_utils.points_in_boxes_cpu(captured['before'][:, :3], b[:, :7]).sum(1)
    ca = roiaware_pool3d_utils.points_in_boxes_cpu(pa[:, :3], b[:, :7]).sum(1)
    npts_b.append(len(captured['before'])); npts_a.append(len(pa))
    for c in np.unique(b[:, 7]).astype(int):
        m = (b[:, 7] == c) & (cb > 0)
        tot[c] = tot.get(c, 0) + m.sum(); emp[c] = emp.get(c, 0) + (m & (ca == 0)).sum()
print(f'{cfg_file} [{key}] frames {len(npts_b)}, points kept {np.sum(npts_a) / np.sum(npts_b):.3f}')
for c in sorted(tot):
    print(f'  {cfg.CLASS_NAMES[c - 1]:10s} boxes with points before: {tot[c]:6d}, emptied by the processor: {emp[c]:5d} ({emp[c] / max(tot[c], 1):.1%})')
