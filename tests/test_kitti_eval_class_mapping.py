"""
Regression test for pcdet/datasets/kitti/kitti_dataset.py::KittiDataset.evaluation()'s
CLASS_MAPPING remap.

Bug (found 2026-08-29 while confirming job results from the 2026-08-23 report series): ST3D's
`kitti_dataset.py` never ported UADA3D's `CLASS_MAPPING` remap in `evaluation()`. This is needed
whenever a model's `CLASS_NAMES` are in the SOURCE dataset's native casing (e.g.
`nuscenes2kitti_models/centerpoint-rospm-C.yaml` uses nuScenes-style lowercase names like 'car')
but evaluation happens against a real KITTI target dataset, whose `kitti_eval.get_official_eval_result()`
only recognizes capitalized native KITTI names ('Car', 'Pedestrian', 'Cyclist', ...). Without the
remap, evaluating with class_names=['car', 'pedestrian', 'bicycle'] crashes with
`KeyError: 'car'` in kitti_eval's `name_to_class` lookup (see job 22819).

Fix: ported UADA3D's CLASS_MAPPING remap block into KittiDataset.evaluation(), and added the
CLASS_MAPPING dict back into nuscenes2kitti_models/centerpoint-rospm-C.yaml's DATA_CONFIG_TAR
(dropped during the ST3D port).
"""
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.kitti.kitti_dataset import KittiDataset  # noqa: E402

# The real kitti_object_eval_python/eval.py transitively imports rotate_iou.py, which uses
# numba.cuda with eager compilation at import time - this crashes on any host/container without
# an actual GPU driver (e.g. this repo's CPU-only master node), even under `singularity exec --nv`.
# To keep this test CPU-only, we inject a fake module directly into sys.modules under the exact
# import path so KittiDataset.evaluation()'s local `from .kitti_object_eval_python import eval as
# kitti_eval` resolves to our stub instead of ever importing/executing the real module.
_FAKE_MODULE_PATH = 'pcdet.datasets.kitti.kitti_object_eval_python.eval'


def _make_bare_kitti_dataset(class_mapping):
    """Builds a KittiDataset instance without running __init__ (no real data on disk needed)."""
    dataset = object.__new__(KittiDataset)
    dataset.dataset_cfg = EasyDict({'CLASS_MAPPING': class_mapping})
    dataset.kitti_infos = [{'annos': {'name': np.array(['Car'])}}]
    dataset.draw_conf_calib_curve = False
    dataset.run_conf_calib = False
    return dataset


def _fake_kitti_eval_module(captured):
    fake_module = types.ModuleType(_FAKE_MODULE_PATH)

    def fake_get_official_eval_result(gt_annos, dt_annos, current_classes, **kwargs):
        captured['current_classes'] = current_classes
        captured['dt_names'] = [d['name'].tolist() for d in dt_annos]
        return 'result_str', {}

    fake_module.get_official_eval_result = fake_get_official_eval_result
    return fake_module


def test_class_mapping_remaps_predicted_and_gt_class_names_before_kitti_eval():
    class_mapping = {
        'car': 'Car', 'pedestrian': 'Pedestrian', 'bicycle': 'Cyclist',
        'Car': 'Car', 'Pedestrian': 'Pedestrian', 'Cyclist': 'Cyclist',
    }
    dataset = _make_bare_kitti_dataset(class_mapping)

    det_annos = [{'name': np.array(['car', 'bicycle'])}]
    class_names = ['car', 'pedestrian', 'bicycle']

    captured = {}
    with mock.patch.dict(sys.modules, {_FAKE_MODULE_PATH: _fake_kitti_eval_module(captured)}):
        dataset.evaluation(det_annos, class_names)

    # class_names passed to kitti_eval must be capitalized KITTI names, not the raw nuScenes-style ones.
    assert captured['current_classes'] == ['Car', 'Pedestrian', 'Cyclist']
    # Predicted box names must also be remapped before being handed to kitti_eval.
    assert captured['dt_names'] == [['Car', 'Cyclist']]


def test_no_class_mapping_is_a_no_op():
    dataset = _make_bare_kitti_dataset(class_mapping={})

    det_annos = [{'name': np.array(['Car'])}]
    class_names = ['Car', 'Pedestrian', 'Cyclist']

    captured = {}
    with mock.patch.dict(sys.modules, {_FAKE_MODULE_PATH: _fake_kitti_eval_module(captured)}):
        dataset.evaluation(det_annos, class_names)

    assert captured['current_classes'] == ['Car', 'Pedestrian', 'Cyclist']
