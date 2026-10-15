"""test.py with DALI's test-time scale s (T-RO2024-DALI tools/test.py: args.scale = 1.20), without touching test.py.

DALI multiplies the SHIFT_COOR-shifted point coordinates by s before voxelisation and divides the predicted box
centre and size (columns 0-5) by s before the KITTI prediction dicts are made (eval_mine_utils.py, kitti_dataset.py:381).
Here: get_lidar returns (p + shift) * s - shift, so the unchanged __getitem__ adds the shift back and gets (p + shift) * s;
generate_prediction_dicts divides pred_boxes[:, :6] by s first. s comes from the env var EVAL_POINT_SCALE.
ANALYSIS ONLY: s = 1.20 is a target box-size statistic.
Usage (from ST3D/tools):  EVAL_POINT_SCALE=1.2 python3 analysis/released_test_scaled.py <test.py arguments>
"""
import importlib.util
import os
import sys

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
SCALE = float(os.environ['EVAL_POINT_SCALE'])

spec = importlib.util.spec_from_file_location('st3d_test_entry', os.path.join(TOOLS, 'test.py'))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

import numpy as np
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

_get_lidar = KittiDataset.get_lidar
_gen = KittiDataset.generate_prediction_dicts


def get_lidar_scaled(self, idx):
    points = _get_lidar(self, idx)
    shift = np.array(self.dataset_cfg.get('SHIFT_COOR', [0, 0, 0]), dtype=np.float32)
    points[:, 0:3] = (points[:, 0:3] + shift) * SCALE - shift
    return points


def gen_scaled(self, batch_dict, pred_dicts, class_names, output_path=None):
    for p in pred_dicts:
        p['pred_boxes'][:, 0:6] /= SCALE
    return _gen(self, batch_dict, pred_dicts, class_names, output_path=output_path)


KittiDataset.get_lidar = get_lidar_scaled
KittiDataset.generate_prediction_dicts = gen_scaled
print('released_test_scaled: EVAL_POINT_SCALE =', SCALE, flush=True)
T.main()
