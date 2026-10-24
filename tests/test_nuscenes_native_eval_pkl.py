"""tools/analysis/nuscenes_native_eval_pkl.py: the log-block parser that ties a stored result.pkl to the evaluation that
wrote it (through '\\r' progress bars, /storage paths, two scorings in one log), and the 0.02 validity rule.
CPU only, synthetic logs; no nuScenes table is read (experiments_md 20261024_04 item 2).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))

import nuscenes_native_eval_pkl as m  # noqa: E402


def block(epoch, bev07, d307, bev05, d305, result_dir, bar=True):
    progress = 'eval: 10%|#   | 1/10\reval: 100%|####| 10/10\r' if bar else ''
    return (f'[x] *************** EPOCH {epoch} of nuscenes EVALUATION start.*****************\n{progress}\n'
            'Car AP@0.70, 0.70, 0.70:\nbbox AP:90.0, 90.0, 90.0\nbev  AP:99.0000, 99.0000, 99.0000\n'
            '3d   AP:98.0000, 98.0000, 98.0000\naos  AP:50.0, 50.0, 50.0\n'
            f'Car AP_R40@0.70, 0.70, 0.70:\nbbox AP:89.0, 89.0, 89.0\nbev  AP:{bev07}, {bev07}, {bev07}\n'
            f'3d   AP:{d307}, {d307}, {d307}\naos  AP:50.0, 50.0, 50.0\n'
            'Car AP@0.70, 0.50, 0.50:\nbbox AP:90.0, 90.0, 90.0\nbev  AP:97.0, 97.0, 97.0\n'
            '3d   AP:96.0, 96.0, 96.0\naos  AP:50.0, 50.0, 50.0\n'
            f'Car AP_R40@0.70, 0.50, 0.50:\nbbox AP:89.0, 89.0, 89.0\nbev  AP:{bev05}, {bev05}, {bev05}\n'
            f'3d   AP:{d305}, {d305}, {d305}\naos  AP:50.0, 50.0, 50.0\n'
            f'[x] eval_utils.py 195  INFO]  Result is save to {result_dir}\n'
            f'[x] *************** EPOCH {epoch} of nuscenes EVALUATION end.*****************\n')


RUN = '/storage/wandb/run-1-abc/files/eval/eval_with_train/epoch_{}/val'
OUT = '/home/koyama/code/ST3D/output/x/eval/epoch_7/val/{}'


def test_in_run_log_picks_the_epoch_whose_directory_holds_the_pickle():
    text = 'header\n' + block(6, '23.4103', '11.2077', '32.8928', '28.2246', RUN.format(6)) \
        + block(7, '25.9213', '13.7448', '35.3877', '30.7285', RUN.format(7))
    pkl = '/home/koyama/data/wandb/run-1-abc/files/eval/eval_with_train/epoch_7/val/result.pkl'
    b = m.logged_block_for(text, pkl)
    assert (b['bev_07'], b['3d_07'], b['bev_05'], b['3d_05']) == (25.9213, 13.7448, 35.3877, 30.7285)
    assert b['result_dir'] == '/home/koyama/data/wandb/run-1-abc/files/eval/eval_with_train/epoch_7/val'
    lines = text.replace('\r', '\n').split('\n')     # line numbers are those of the converted text
    assert lines[b['bev_07_header_line'] - 1] == 'Car AP_R40@0.70, 0.70, 0.70:'
    assert lines[b['bev_05_header_line'] - 1] == 'Car AP_R40@0.70, 0.50, 0.50:'
    assert lines[b['bev_07_header_line'] + 1].startswith('bev  AP:25.9213')


def test_two_scorings_in_one_log_are_told_apart_by_directory():
    text = block(7, '22.5790', '11.3405', '31.3416', '28.4286', OUT.format('rescore2').replace('x', 'c1')) \
        + block(7, '23.3247', '12.3786', '31.5570', '28.7895', OUT.format('rescore2').replace('x', 'ctl'))
    b = m.logged_block_for(text, OUT.format('rescore2').replace('x', 'ctl') + '/result.pkl')
    assert (b['bev_07'], b['3d_05']) == (23.3247, 28.7895)


def test_zero_or_two_matching_blocks_raise():
    text = block(7, '1.0', '1.0', '1.0', '1.0', OUT.format('a'))
    with pytest.raises(ValueError, match='0 evaluation blocks'):
        m.logged_block_for(text, OUT.format('b') + '/result.pkl')
    with pytest.raises(ValueError, match='2 evaluation blocks'):
        m.logged_block_for(text + text, OUT.format('a') + '/result.pkl')


def test_a_block_without_the_r40_07_lines_raises():
    text = ('*** EPOCH 7 of nuscenes EVALUATION start.***\nPedestrian AP_R40@0.50, 0.50, 0.50:\n'
            f'Result is save to {OUT.format("a")}\n')
    with pytest.raises(ValueError, match='no Car AP_R40'):
        m.logged_block_for(text, OUT.format('a') + '/result.pkl')


def test_storage_paths_are_read_as_host_paths():
    assert m.host_path('/storage/wandb/r/') == '/home/koyama/data/wandb/r'
    assert m.host_path('/home/koyama/code/ST3D/output/a') == '/home/koyama/code/ST3D/output/a'


@pytest.mark.parametrize('rec, logged, ok', [(25.9413, 25.9213, True), (25.9013, 25.9213, True),
                                             (25.9414, 25.9213, False), (25.9012, 25.9213, False)])
def test_validity_tolerance_is_inclusive_at_0_02(rec, logged, ok):
    d, passed = m.validity(rec, logged, 0.02)
    assert passed is ok and d == pytest.approx(rec - logged)
