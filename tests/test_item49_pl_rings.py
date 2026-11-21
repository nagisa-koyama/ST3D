"""Item 49's components and reading rule (experiments_md 20261003_05 §5 item 49, design fixed 2026-11-21 21:10 JST). CPU only."""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS / 'analysis'))
import item49_pl_rings as m  # noqa: E402


def test_components_add_up_to_pl_minus_t():
    v = {('T', 'none'): [70.0, 80.0], ('T', 'src'): [69.0, 79.5], ('E', 'src'): [72.0, 83.0], ('E', 'none'): [72.5, 82.0],
         ('PL', 'none'): [74.0, 85.0]}
    c = m.components(v)
    assert c['O'] == [-1.0, -0.5] and c['G'] == [3.0, 3.5] and c['S'] == [0.5, -1.0] and c['L'] == [1.5, 3.0]
    for i in range(2):
        assert abs(c['O'][i] + c['G'][i] + c['S'][i] + c['L'][i] - c['PL-T'][i]) < 1e-9


def test_classify():
    assert m.classify([0.5, 0.0, 1.0, 4.0, 6.0]) == 'acts far'        # far 5.0, near 0.25
    assert m.classify([-1.0, -1.0, 0.0, 2.5, 2.0]) == 'acts far'       # negative near counts as 0
    assert m.classify([0.5, 0.0, 1.0, 1.0, 2.0]) == 'level or spread'  # far 1.5 < 2
    assert m.classify([3.0, 3.0, 0.0, 2.0, 2.0]) == 'level or spread'  # far 2.0 < 2 x near 3.0, near 3.0 < 2 x far
    assert m.classify([6.0, 4.0, 0.0, 1.0, 0.0]) == 'acts near'


def test_result_dirs_follow_item46_tags():
    assert str(m.result_dir('PL', 'src')).endswith('centerpoint-accum-legaldepth-st3d-nuscenes2kitti/i46_Ps2_src/eval/epoch_20/i46_Ps2_src')
    assert str(m.result_dir('T', 'none')).endswith('centerpoint-accum-legaldepth-nuscenes2kitti/i46_Ts2_none/eval/epoch_20/i46_Ts2_none')


def test_seed_667_rows_share_the_teacher_and_use_the_49b_tags():
    assert str(m.result_dir('T', 'none', 667)) == str(m.result_dir('T', 'none', 666))  # 27468 is the teacher of both seeds
    assert str(m.result_dir('E', 'src', 667)).endswith('/i46_E2s2_src/eval/epoch_20/i46_E2s2_src')
    assert str(m.result_dir('PL', 'none', 667)).endswith('/i46_P2s2_none/eval/epoch_20/i46_P2s2_none')
    assert m.JOB[667]['PL'] == 28283 and m.JOB[667]['T'] == 27468
