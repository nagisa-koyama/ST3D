"""`get_no_evaluated_ckpt` must SORT by epoch, not INDEX by it.

Job 25817 (KITTI source-only, 152 epochs) trained cleanly for 21 h and then died in evaluation:

    ckpt_list_sorted[int(float(num_list[-1])) - 1] = cur_ckpt
    IndexError: list assignment index out of range

The list was sized by the NUMBER of checkpoint files and assigned at `epoch - 1`, on the comment's
stated assumption that epoch numbers are "1-index" and "incremental". That holds only while every
checkpoint is retained. `--max_ckpt_save_num` defaults to 100, so a 152-epoch run keeps epochs
53-152 and index 151 runs off a list of length 100.

It fires for any row where NUM_EPOCHS > max_ckpt_save_num. In the da-ieee-access family that is
KITTI (152) and both PandaSet rows (115), but not Lyft (30), nuScenes (20) or Waymo (7) - which is
why it stayed hidden until the first long row reached its evaluation.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))


def _load_get_no_evaluated_ckpt():
    """Import the function without importing tools/test.py as a whole (it pulls in CUDA)."""
    src = (Path(__file__).resolve().parent.parent / 'tools' / 'test.py').read_text()
    start = src.index('def get_no_evaluated_ckpt(')
    end = src.index('\ndef ', start + 1)
    ns = {'glob': __import__('glob'), 'os': __import__('os'), 're': re}
    exec(compile(src[start:end], 'test_py_fragment', 'exec'), ns)
    return ns['get_no_evaluated_ckpt']


class _Args:
    start_epoch = 0
    max_waiting_mins = 0


def _make(tmp_path, epochs, evaluated=()):
    ckpt_dir = tmp_path / 'ckpt'
    ckpt_dir.mkdir()
    for e in epochs:
        (ckpt_dir / f'checkpoint_epoch_{e}.pth').write_text('x')
    record = tmp_path / 'record.txt'
    record.write_text(''.join(f'{e}\n' for e in evaluated))
    return str(ckpt_dir), str(record)


def test_retained_window_shorter_than_the_epoch_number(tmp_path):
    """The exact shape of job 25817: 152 epochs, only the last 100 kept."""
    ckpt_dir, record = _make(tmp_path, range(53, 153))
    epoch_id, ckpt = _load_get_no_evaluated_ckpt()(ckpt_dir, record, _Args())
    assert epoch_id == '53', 'the earliest unevaluated retained checkpoint must come first'
    assert ckpt.endswith('checkpoint_epoch_53.pth')


def test_epochs_are_returned_in_numeric_not_lexicographic_order(tmp_path):
    """`checkpoint_epoch_9` must not sort after `checkpoint_epoch_100`."""
    ckpt_dir, record = _make(tmp_path, [9, 10, 100], evaluated=[9.0])
    epoch_id, _ = _load_get_no_evaluated_ckpt()(ckpt_dir, record, _Args())
    assert epoch_id == '10', f'expected epoch 10 next, got {epoch_id}'


def test_optimizer_checkpoints_are_ignored(tmp_path):
    """`*_optim.pth` files sit in the same directory and must not be parsed as epochs."""
    ckpt_dir, record = _make(tmp_path, [115])
    (Path(ckpt_dir) / 'checkpoint_epoch_115_optim.pth').write_text('x')
    epoch_id, ckpt = _load_get_no_evaluated_ckpt()(ckpt_dir, record, _Args())
    # Check the BASENAME: pytest names tmp_path after the test, so the directory itself contains
    # the substring "optim" and a naive `'optim' not in ckpt` fails on the path, not the file.
    assert epoch_id == '115'
    assert Path(ckpt).name == 'checkpoint_epoch_115.pth', f'picked {Path(ckpt).name}'


def test_pandaset_length_also_exceeds_the_default_retention(tmp_path):
    """115 > 100, so both PandaSet rows are in the failing regime, not just KITTI."""
    ckpt_dir, record = _make(tmp_path, range(16, 116), evaluated=[float(e) for e in range(16, 115)])
    epoch_id, _ = _load_get_no_evaluated_ckpt()(ckpt_dir, record, _Args())
    assert epoch_id == '115'
