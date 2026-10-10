"""ACCUMULATION_DEPTH_BY_RANGE: a per-range depth schedule for nuScenes sweep accumulation.

One MAX_SWEEPS over-accumulates the near field (3.4x the target at N = 15) and leaves 30-50 m short
(20260922_08; 20260929_06 §1 puts the remaining nuScenes -> KITTI gap there). The schedule keeps
sweep k only beyond the range whose depth asks for more than k frames. The anchor is always whole.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(ROOT))
import _init_path  # noqa: F401,E402
from pcdet.datasets.nuscenes.nuscenes_dataset import sweep_min_range, sweep_range_mask  # noqa: E402

SCHED = [[0, 5], [20, 10], [40, 15]]


def test_min_range_per_sweep_age():
    assert sweep_min_range(SCHED, 1) == 0.0          # every ring wants at least 2 frames
    assert sweep_min_range(SCHED, 4) == 0.0          # the 5th frame is still wanted everywhere
    assert sweep_min_range(SCHED, 5) == 20.0         # the 6th only from 20 m
    assert sweep_min_range(SCHED, 9) == 20.0
    assert sweep_min_range(SCHED, 10) == 40.0        # the 11th only from 40 m
    assert sweep_min_range(SCHED, 14) == 40.0
    assert sweep_min_range(SCHED, 15) is None        # a 16th sweep is never wanted


def test_mask_keeps_far_points_only_for_old_sweeps():
    pts = np.array([[5, 0, 0], [25, 0, 0], [45, 0, 0]], dtype=np.float32)
    assert sweep_range_mask(pts, 1, SCHED).tolist() == [True, True, True]
    assert sweep_range_mask(pts, 7, SCHED).tolist() == [False, True, True]
    assert sweep_range_mask(pts, 12, SCHED).tolist() == [False, False, True]
    assert sweep_range_mask(pts, 20, SCHED).tolist() == [False, False, False]


def test_schedule_must_be_non_decreasing():
    with pytest.raises(AssertionError):
        sweep_min_range([[0, 10], [20, 5]], 1)


def test_loader_applies_the_mask_after_compensation():
    src = (ROOT / 'pcdet/datasets/nuscenes/nuscenes_dataset.py').read_text(encoding='utf-8')
    body = src[src.index('def get_lidar_with_sweeps'):src.index('def __len__')]
    # j is the rank among the SELECTED sweeps (83a64fe, SWEEP_SELECTION); in the default consecutive mode it equals
    # the old stored-sweep index k, so the schedule still reads sweep age.
    assert body.index('compensate_sweep(') < body.index('sweep_range_mask(points_sweep, j + 1, schedule)')
    assert "get('ACCUMULATION_DEPTH_BY_RANGE', None)" in body
