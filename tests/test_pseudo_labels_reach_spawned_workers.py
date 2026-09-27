"""Pseudo-labels must reach a DataLoader worker that was SPAWNED, not forked.

`fill_pseudo_labels` used to read `self_training_utils.PSEUDO_LABELS`, a module global. Under
fork a worker inherits the main process's dict by copy-on-write, so that worked. Under DDP the
workers are spawned (`init_dist_pytorch` forces the start method), a spawned worker re-imports the
module, and its PSEUDO_LABELS is a fresh EMPTY dict - every lookup raised
`ValueError: Cannot find pseudo label for frame`, which is why no self-training row had ever run on
two GPUs. The dict now travels ON the dataset (`DatasetTemplate.set_pseudo_labels`), which is
pickled into each worker whatever the start method.

The spawn tests here use a real DataLoader with `multiprocessing_context='spawn'`, because the
whole point is what a fresh interpreter sees. Sibling of test_pseudo_label_worker_staleness.py
(the fork side of the same family).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
import _init_path  # noqa: F401,E402
from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.utils import self_training_utils as stu  # noqa: E402

CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']


def _labels():
    # two frames; frame 1 carries one positive Car and one IGNORED Car (negative class index)
    box = lambda cls, score: [1.0, 2.0, 0.5, 4.0, 1.8, 1.5, 0.3, float(cls), score]  # noqa: E731
    return {
        'f0': {'gt_boxes': np.array([box(1, 0.9)], dtype=np.float32)},
        'f1': {'gt_boxes': np.array([box(1, 0.9), box(-1, 0.15)], dtype=np.float32)},
    }


class Carrier(Dataset):
    """Only what fill_pseudo_labels reads off a DatasetTemplate: class_names and pseudo_labels.

    __getitem__ calls the REAL unbound method and reports back what the worker process saw:
    boxes filled, the per-class counters, and the size of the module global in that process.
    """

    def __init__(self, pseudo_labels):
        self.class_names = CLASS_NAMES
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return 2

    def __getitem__(self, index):
        d = {'frame_id': 'f%d' % index, 'num_points_in_gt': np.zeros(1)}
        DatasetTemplate.fill_pseudo_labels(self, d)
        return torch.tensor([d['gt_boxes'].shape[0], d['pos_ps_bbox'][0], d['ign_ps_bbox'][0],
                             len(stu.PSEUDO_LABELS)], dtype=torch.float32)


# ---------------------------------------------------------------- the lookup itself

def test_load_ps_label_reads_the_dict_it_is_given(monkeypatch):
    monkeypatch.setattr(stu, 'PSEUDO_LABELS', {})          # the spawned-worker situation
    out = stu.load_ps_label('f1', _labels())
    assert out.shape == (2, 9)


def test_load_ps_label_falls_back_to_the_module_global(monkeypatch):
    """The fork path (and the main process) never set the attribute; behaviour must be unchanged."""
    monkeypatch.setattr(stu, 'PSEUDO_LABELS', _labels())
    assert stu.load_ps_label('f0', None).shape == (1, 9)


def test_a_missing_frame_names_the_spawn_cause_only_on_the_global_path(monkeypatch):
    monkeypatch.setattr(stu, 'PSEUDO_LABELS', {})
    with pytest.raises(ValueError, match='SPAWNED worker'):
        stu.load_ps_label('nope', None)
    with pytest.raises(ValueError) as e:
        stu.load_ps_label('nope', _labels())
    assert 'SPAWNED' not in str(e.value), 'a carried dict that lacks the frame is a different bug'


def test_fill_pseudo_labels_uses_the_carried_dict_and_counts_ignored_boxes(monkeypatch):
    monkeypatch.setattr(stu, 'PSEUDO_LABELS', {})
    d = {'frame_id': 'f1', 'num_points_in_gt': np.zeros(1)}
    ds = Carrier(_labels())
    DatasetTemplate.fill_pseudo_labels(ds, d)
    assert d['gt_boxes'].shape == (2, 7)
    assert list(d['gt_names']) == ['Car', 'Car']
    assert d['pos_ps_bbox'].tolist() == [1, 0, 0] and d['ign_ps_bbox'].tolist() == [1, 0, 0]
    assert 'num_points_in_gt' not in d


def test_set_pseudo_labels_keeps_a_reference_not_a_copy():
    """gather_and_dump rewrites PSEUDO_LABELS in place; a copy would freeze at install time."""
    live = {}
    ds = Carrier(None)
    DatasetTemplate.set_pseudo_labels(ds, live)
    live.update(_labels())                                   # what a generation pass does
    assert ds.pseudo_labels is live and 'f1' in ds.pseudo_labels


# ---------------------------------------------------------------- what a spawned worker sees

def _spawn_loader(dataset):
    return DataLoader(dataset, batch_size=2, num_workers=1, shuffle=False,
                      multiprocessing_context='spawn')


def test_spawned_worker_reads_the_carried_labels_while_its_module_global_is_empty():
    batch = next(iter(_spawn_loader(Carrier(_labels()))))
    n_boxes, pos, ign, n_global = batch.T
    assert n_boxes.tolist() == [1, 2]
    assert pos.tolist() == [1, 1] and ign.tolist() == [0, 1]
    assert n_global.tolist() == [0, 0], (
        'the worker re-imported self_training_utils and holds an EMPTY global - the labels can '
        'only have come from the dataset attribute')


def test_spawned_worker_without_carried_labels_fails_the_old_way():
    """The pre-fix behaviour, kept as the negative control for the test above."""
    with pytest.raises(ValueError, match='Cannot find pseudo label'):
        next(iter(_spawn_loader(Carrier(None))))


# ---------------------------------------------------------------- the trainer installs it

def test_train_model_st_installs_the_live_dict_before_any_target_loader_is_iterated():
    src = (ROOT / 'tools/train_utils/train_st_utils.py').read_text(encoding='utf-8')
    src = src[src.index('def train_model_st('):]      # train_one_epoch_st above also iterates
    install = src.index('set_pseudo_labels(self_training_utils.PSEUDO_LABELS)')
    first_gen_loader = src.index('ps_gen_loader = build_inference_dataloader(')
    first_target_iter = src.index('dataloader_iter = iter(target_loader)')
    assert install < first_gen_loader < first_target_iter
