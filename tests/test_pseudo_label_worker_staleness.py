"""Pseudo-label updates must reach the training workers.

`save_pseudo_label_epoch` rewrites the module-level `PSEUDO_LABELS` dict in the main process
(`clear()` then `update()`), and `fill_pseudo_labels` reads that same global. DataLoader workers
forked at the first `iter()` hold a copy-on-write snapshot of it, so a later rewrite is invisible
to them: they would serve the epoch-0 labels for the whole run and `UPDATE_PSEUDO_LABEL_INTERVAL`
would be silently dead.

These tests exercise a real DataLoader with real worker processes rather than re-implementing the
semantics, because the whole point is what the fork does. Same family as
experiments_md/20260921_02_persistent_workers_stale_dataset_state.md.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets import restart_persistent_workers  # noqa: E402

# stands in for self_training_utils.PSEUDO_LABELS - a module global the workers snapshot on fork
LABELS = {'v': 0}


class ReadsTheGlobal(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return LABELS['v']


def _loader():
    return DataLoader(ReadsTheGlobal(), batch_size=4, num_workers=1, persistent_workers=True)


@pytest.mark.skipif(torch.multiprocessing.get_start_method() != 'fork',
                    reason='copy-on-write snapshot semantics assume fork')
def test_workers_do_not_see_a_later_global_rewrite():
    LABELS['v'] = 0
    loader = _loader()
    try:
        assert int(next(iter(loader))[0]) == 0        # forks here
        LABELS['v'] = 7                               # what save_pseudo_label_epoch does
        assert int(next(iter(loader))[0]) == 0, 'workers unexpectedly saw the rewrite'
    finally:
        restart_persistent_workers(loader)


@pytest.mark.skipif(torch.multiprocessing.get_start_method() != 'fork',
                    reason='copy-on-write snapshot semantics assume fork')
def test_restart_persistent_workers_makes_the_rewrite_visible():
    LABELS['v'] = 0
    loader = _loader()
    try:
        assert int(next(iter(loader))[0]) == 0
        LABELS['v'] = 7
        restart_persistent_workers(loader)            # what the fix adds after every update
        assert int(next(iter(loader))[0]) == 7
    finally:
        restart_persistent_workers(loader)
