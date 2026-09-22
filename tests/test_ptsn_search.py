"""Tests for the PTSN sweep's plumbing, driven by a stub model so no GPU is needed.

`mean_predicted_size` is the function that spends the search's GPU hour. Nothing about it is
subtle, which is exactly why it is worth testing: per-class thresholds, class selection, the
unscaling and the frame cap are all cheap to get wrong and expensive to discover in a job log -
this repo has already lost 16 GPU-hours to a plumbing fault that a seconds-long CPU check would
have caught (see 20260921_04, and the preflight tool that came out of it).

The stub returns a fixed box in the SCALED frame, which is what a detector whose size prior is
frozen at the source does. So these tests also pin the search end of the 1/s claim that
test_ptsn.py pins at the transform end.

See experiments_md/20260922_04 section 2.6.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parent.parent / 'tools'
sys.path.insert(0, str(TOOLS.parent))
sys.path.insert(0, str(TOOLS))

from analysis import ptsn_search  # noqa: E402

SOURCE_MEAN = np.array([4.60, 1.95, 1.72])


class StubModel:
    """A detector with a frozen size prior: the same box, in the scaled frame, at every scale."""

    def __init__(self, boxes_per_frame, scores, labels, frames=100):
        self.boxes_per_frame = boxes_per_frame
        self.scores = np.asarray(scores, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.frames = frames
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        preds = []
        for _ in range(batch['batch_size']):
            preds.append({
                'pred_boxes': torch.from_numpy(np.asarray(self.boxes_per_frame, dtype=np.float32)),
                'pred_scores': torch.from_numpy(self.scores),
                'pred_labels': torch.from_numpy(self.labels),
            })
        return preds, {}


def _loader(num_batches, batch_size):
    return [{'batch_size': batch_size} for _ in range(num_batches)]


def _box(dims, score_dummy=None):
    return np.concatenate([np.zeros(3), np.asarray(dims), np.zeros(1)])


def _noop(batch):
    """Stands in for load_data_to_gpu."""
    return batch


def _run(model, loader, scale, wanted, thresholds, frames):
    return ptsn_search.mean_predicted_size(
        model, loader, scale, np.asarray(wanted), np.asarray(thresholds), frames, to_device=_noop)


# --- the 1/s claim, seen from the search's end -------------------------------------------------

@pytest.mark.parametrize('scale', [0.8, 1.0, 1.25])
def test_reported_mean_size_is_source_mean_over_s(scale):
    model = StubModel([_box(SOURCE_MEAN)], scores=[0.9], labels=[1])
    mean, n, _ = _run(model, _loader(4, 2), scale, [1], [0.1, 0.1, 0.1], frames=8)
    assert n == 8
    assert np.allclose(mean, SOURCE_MEAN / scale)


# --- selection has to match what pseudo-labelling would keep -----------------------------------

def test_boxes_below_their_class_threshold_are_dropped():
    model = StubModel([_box([4.0, 2.0, 1.5]), _box([8.0, 2.0, 1.5])],
                      scores=[0.9, 0.05], labels=[1, 1])
    mean, n, _ = _run(model, _loader(1, 1), 1.0, [1], [0.1, 0.1, 0.1], frames=1)
    assert n == 1
    assert np.allclose(mean, [4.0, 2.0, 1.5])


def test_thresholds_are_applied_per_class():
    """A per-class threshold vector must be indexed by label, not applied as one scalar."""
    model = StubModel([_box([4.0, 2.0, 1.5]), _box([8.0, 2.0, 1.5])],
                      scores=[0.3, 0.3], labels=[1, 2])
    mean, n, _ = _run(model, _loader(1, 1), 1.0, [1, 2], [0.1, 0.5, 0.1], frames=1)
    assert n == 1                      # class 2 needs 0.5 and scored 0.3
    assert np.allclose(mean, [4.0, 2.0, 1.5])


def test_other_classes_are_excluded():
    model = StubModel([_box([4.0, 2.0, 1.5]), _box([0.8, 0.6, 1.7])],
                      scores=[0.9, 0.9], labels=[1, 2])
    mean, n, _ = _run(model, _loader(1, 1), 1.0, [1], [0.1, 0.1, 0.1], frames=1)
    assert n == 1
    assert np.allclose(mean, [4.0, 2.0, 1.5])


# --- the frame cap and the degenerate cases ----------------------------------------------------

def test_frame_cap_stops_the_sweep_early():
    model = StubModel([_box(SOURCE_MEAN)], scores=[0.9], labels=[1])
    _, _, seen = _run(model, _loader(50, 4), 1.0, [1], [0.1, 0.1, 0.1], frames=10)
    assert seen == 12                  # stops at the first batch that reaches the cap
    assert model.calls == 3


def test_no_surviving_box_reports_nan_and_zero():
    """This is the row select_scale must never pick; see test_ptsn.py."""
    model = StubModel([_box([4.0, 2.0, 1.5])], scores=[0.01], labels=[1])
    mean, n, seen = _run(model, _loader(2, 2), 1.0, [1], [0.1, 0.1, 0.1], frames=4)
    assert n == 0 and seen == 4
    assert not np.isfinite(mean).any()


def test_a_frame_with_no_predictions_is_skipped_not_counted_as_zero():
    model = StubModel([], scores=[], labels=[])
    mean, n, seen = _run(model, _loader(1, 1), 1.0, [1], [0.1, 0.1, 0.1], frames=1)
    assert n == 0 and seen == 1
    assert not np.isfinite(mean).any()


# --- class and threshold resolution ------------------------------------------------------------

def test_class_indices_are_one_based_and_case_insensitive():
    assert ptsn_search.class_indices(['Car', 'Pedestrian', 'Cyclist'], 'car') == [1]
    assert ptsn_search.class_indices(['Car', 'Pedestrian'], 'PEDEST') == [2]


def test_class_indices_match_head_prefixed_names():
    """Multi-head configs carry names like 'nuscenes:car'."""
    assert ptsn_search.class_indices(['nuscenes:car', 'nuscenes:pedestrian'], 'car') == [1]


def test_class_indices_returns_empty_when_nothing_matches():
    assert ptsn_search.class_indices(['Car'], 'truck') == []


def test_score_thresholds_prefer_the_self_train_vector():
    cfg = EasyDict({'SELF_TRAIN': {'SCORE_THRESH': [0.4, 0.3, 0.2]}})
    assert np.allclose(ptsn_search.score_thresholds(cfg, None, 3), [0.4, 0.3, 0.2])


def test_score_thresholds_override_wins():
    cfg = EasyDict({'SELF_TRAIN': {'SCORE_THRESH': [0.4, 0.3, 0.2]}})
    assert np.allclose(ptsn_search.score_thresholds(cfg, 0.05, 3), [0.05] * 3)


def test_score_thresholds_fall_back_when_the_config_has_none():
    assert np.allclose(ptsn_search.score_thresholds(EasyDict({}), None, 2), [0.1, 0.1])


def test_score_thresholds_broadcast_a_mismatched_vector():
    cfg = EasyDict({'SELF_TRAIN': {'SCORE_THRESH': [0.25]}})
    assert np.allclose(ptsn_search.score_thresholds(cfg, None, 3), [0.25] * 3)


# --- which dataset the sweep runs over ---------------------------------------------------------

def test_target_config_prefers_data_config_tar():
    cfg = EasyDict({'DATA_CONFIG': {'DATASET': 'src'}, 'DATA_CONFIG_TAR': {'DATASET': 'tgt'}})
    assert ptsn_search.target_config(cfg).DATASET == 'tgt'


def test_target_config_falls_back_to_data_config():
    cfg = EasyDict({'DATA_CONFIG': {'DATASET': 'src'}})
    assert ptsn_search.target_config(cfg).DATASET == 'src'


def test_source_configs_reads_the_single_source_shape():
    cfg = EasyDict({'DATA_CONFIG': {'DATASET': 'src'}})
    assert [c.DATASET for c in ptsn_search.source_configs(cfg)] == ['src']


def test_source_configs_reads_the_multi_source_shape():
    """The da-MIRU2025 family carries DATA_CONFIGS, not DATA_CONFIG.

    Reading only DATA_CONFIG raised AttributeError on exactly the configs a DALI row would be
    built from - caught by the --dry_run preflight, which is what it is for.
    """
    cfg = EasyDict({'DATA_CONFIGS': {'NUSCENES_CONFIG': {'DATASET': 'NuScenesDataset'}}})
    assert [c.DATASET for c in ptsn_search.source_configs(cfg)] == ['NuScenesDataset']


def test_source_configs_returns_every_source():
    cfg = EasyDict({'DATA_CONFIGS': {'A': {'DATASET': 'a'}, 'B': {'DATASET': 'b'}}})
    assert sorted(c.DATASET for c in ptsn_search.source_configs(cfg)) == ['a', 'b']


def test_source_configs_rejects_a_config_with_neither():
    with pytest.raises(AssertionError):
        ptsn_search.source_configs(EasyDict({'CLASS_NAMES': ['Car']}))


def test_target_config_falls_back_to_a_multi_source_config():
    cfg = EasyDict({'DATA_CONFIGS': {'A': {'DATASET': 'a'}}})
    assert ptsn_search.target_config(cfg).DATASET == 'a'
