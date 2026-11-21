"""The PandaSet repeated-scan census (tools/analysis/pcseg_pandaset_duplicate_scans.py; experiments_md 20261116_01 §8.9)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'analysis'))

import pcseg_pandaset_duplicate_scans as census                                                   # noqa: E402


def test_pairs_of_compares_every_pair_not_only_neighbours():
    spans = {0: (10.0, 10.1), 1: (10.0, 10.1), 2: (10.1, 10.2), 5: (10.1, 10.2), 9: (20.0, 20.1)}
    assert census.pairs_of(spans) == [(0, 1), (2, 5)]


def test_same_scan_needs_both_ends_within_tolerance():
    assert census.same_scan((1.0, 1.1), (1.0005, 1.1005))
    assert not census.same_scan((1.0, 1.1), (1.0, 1.102))
    assert not census.same_scan((1.0, 1.1), (1.002, 1.1))


def test_three_copies_give_three_pairs():
    spans = {0: (5.0, 5.1), 1: (5.0, 5.1), 2: (5.0, 5.1)}
    assert census.pairs_of(spans) == [(0, 1), (0, 2), (1, 2)]
