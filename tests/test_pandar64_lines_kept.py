"""tools/analysis/pandar64_lines_kept.py (experiments_md 20261003_05 §5 item 42d record 5): the line count, the frame
transform and the range cut it reports with, on synthetic input."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))
import pandar64_lines_kept as L  # noqa: E402


def test_count_lines_counts_distinct_channels_with_a_point():
    labels = np.array([0, 0, 5, 5, 5, 7, 63])
    assert L.count_lines(labels) == 4
    assert L.count_lines(labels, labels != 5) == 3
    assert L.count_lines(labels, np.zeros(len(labels), dtype=bool)) == 0


def test_count_lines_never_counts_unlabelled_points():
    # ring_pattern_mask drops label < 0; a -1 must not become a 65th "line" (bug_prevention_checklist rule 1)
    labels = np.array([-1, -1, 3, 3, 9])
    assert L.count_lines(labels) == 2
    assert L.count_lines(labels, np.array([True, True, False, False, False])) == 0


def test_to_normative_matches_the_loader_axis_swap_and_shift():
    ego = np.array([[1.0, 2.0, 3.0]])                      # x right, y forward, z up
    out = L.to_normative(ego, [0.0, 0.0, 0.30])
    np.testing.assert_allclose(out, [[2.0, -1.0, 3.30]])   # x forward, y left, + SHIFT_COOR


def test_in_range_cuts_z_as_the_voxeliser_and_x_y_as_the_processor():
    pcr = [-75.2, -75.2, -2, 75.2, 75.2, 4]
    p = np.array([[0, 0, 0], [0, 0, 4.0], [0, 0, -2.0], [75.3, 0, 0], [0, -75.2, 0], [0, 0, 3.99]])
    np.testing.assert_array_equal(L.in_range(p, pcr), [True, False, True, False, True, True])


def test_summary_quartiles():
    s = L.summary([1, 2, 3, 4, 5])
    assert (s['median'], s['q1'], s['q3'], s['iqr'], s['min'], s['max'], s['n']) == (3, 2, 4, 2, 1, 5, 5)
    assert s['mean'] == 3
