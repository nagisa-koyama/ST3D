"""Item 48's reading rule (experiments_md 20261003_05 §5 item 48, design fixed 2026-11-21 20:09 JST). CPU only, no files."""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS / 'analysis'))
import item48_rings as m  # noqa: E402


def seed(own, other):
    return m.per_seed(own, other)


def test_per_seed_is_relative_to_the_arm_on_its_own_lattice():
    s, r = seed([80, 80, 60, 40, 20], [70, 80, 30, 20, 10])
    assert s == [10, 0, 30, 20, 10]
    assert r == [10 / 80, 0.0, 0.5, 0.5, 0.5]


def test_far_needs_both_seeds_twice_the_near_share_and_2_ap():
    far = seed([90, 90, 70, 50, 30], [88, 88, 60, 35, 20])         # near r ~0.02, far r ~0.32, far s 12.5
    assert m.classify([far, far]) == 'residual far'
    small = seed([90, 90, 70, 10, 4], [90, 90, 70, 9, 3.5])          # far r high but mean far s 0.75 < 2
    assert m.classify([small, small]) == 'over the whole range'
    assert m.classify([far, seed([90] * 5, [80] * 5)]) == 'over the whole range'  # one seed uniform


def test_near_and_negative_shares():
    near = seed([90, 90, 70, 50, 30], [60, 60, 65, 49, 30])
    assert m.classify([near, near]) == 'residual near'
    neg_near = seed([90, 90, 70, 50, 30], [95, 95, 60, 35, 20])     # near negative counts as 0
    assert m.classify([neg_near, neg_near]) == 'residual far'


def test_missing_rows_are_not_read():
    far = seed([90, 90, 70, 50, 30], [88, 88, 60, 35, 20])
    assert m.classify([far, None]) == 'not read'
    assert m.classify([far]) == 'not read'


def test_unreproduced_rows_are_dropped():
    rows = [dict(group='fwd', stats='target', row='L', seed=666, reproduced=False, rings=[dict(bev=1.0)] * 5),
            dict(group='fwd', stats='target', row='P', seed=666, reproduced=True, rings=[dict(bev=1.0)] * 5)]
    assert m.ring_values(rows, 'fwd', 'target', 'L', 666) is None
    assert m.ring_values(rows, 'fwd', 'target', 'P', 666) == [1.0] * 5
