"""A DataProcessor that has ALREADY voxelised a frame must still pickle (DDP spawns its loader workers).

Job 26326 (2 GPUs, on-the-fly global calibration) died at `iter(train_loader)` with
`TypeError: cannot pickle 'spconv...Point2VoxelCPU' object`: the pre-training calibration ran frames
through the processor in the main process, creating the spconv generator that the original lazy
design meant to keep out of the pickled state. `DataProcessor.__getstate__` now drops it.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import _init_path  # noqa: F401,E402
from pcdet.datasets.processor.data_processor import DataProcessor  # noqa: E402

CFG = [EasyDict(NAME='transform_points_to_voxels', VOXEL_SIZE=[0.1, 0.1, 0.15],
                MAX_POINTS_PER_VOXEL=10, MAX_NUMBER_OF_VOXELS={'train': 150000, 'test': 150000})]
RANGE = np.array([-75.2, -75.2, -2, 75.2, 75.2, 4], dtype=np.float32)


def _frame():
    rng = np.random.default_rng(0)
    pts = rng.uniform([-70, -70, -1.5], [70, 70, 3.5], size=(5000, 3)).astype(np.float32)
    return {'points': pts, 'use_lead_xyz': True}


def test_processor_pickles_after_voxelising_once():
    dp = DataProcessor(CFG, RANGE, training=True, num_point_features=3)
    out = dp.forward(_frame())
    assert dp.voxel_generator is not None, 'precondition: the generator was created in this process'
    n_before = len(out['voxels'])

    clone = pickle.loads(pickle.dumps(dp))          # this is what spawn does
    assert clone.voxel_generator is None
    out2 = clone.forward(_frame())                  # re-created lazily, same parameters
    assert len(out2['voxels']) == n_before
    assert dp.voxel_generator is not None, 'pickling must not disturb the live object'
