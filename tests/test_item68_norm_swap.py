"""Item 68's rows and readings (tools/analysis/item68_norm_swap.py; experiments_md 20261003_05 §5 item 68, design fixed
2026-11-28 16:42 JST). Run from the repo root."""
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item68_norm_swap as m  # noqa: E402


def row(name, seed, bev):
    return dict(row=name, seed=seed, reproduced=True, logged_bev=bev, pooled={'bev': bev, '3d': bev / 2},
                rings=[{'bev': bev} for _ in range(5)])


def rows_with(ln, gn):
    out = []
    for name, vals in (('LN', ln), ('GN', gn), ('BN_none', m.REF['BN_none']), ('BN_ada', m.REF['BN_ada'])):
        out += [row(name, s, v) for s, v in zip(m.SEEDS, vals)]
    return out


SRC = [dict(row='LN_src', seed=s, logged_bev=44.9) for s in m.SEEDS] + \
      [dict(row='GN_src', seed=s, logged_bev=40.0) for s in m.SEEDS]


def test_reference_means_are_the_designs():
    for k in m.REF:
        assert sum(m.REF[k]) / 3 == pytest.approx(m.REF_MEAN[k], abs=0.005)


def test_words():
    assert m.word([45.0, 44.0, 43.0], 38.15, 'LN') == (pytest.approx(5.85), 'LN beats BN')
    assert m.word([45.0, 44.0, 37.0], 38.15, 'LN')[1] == 'no verdict'               # d 5.18 but one seed below
    assert m.word([39.0, 37.0, 40.0], 38.15, 'LN')[1] == 'level'
    assert m.word([30.0, 32.0, 33.0], 38.15, 'GN')[1] == 'GN is below BN'
    assert m.word([30.0, 32.0, 40.0], 38.15, 'GN')[1] == 'no verdict'               # d -6.48 but one seed above


def test_readings_and_source_flag():
    r = m.readings(rows_with((70.0, 71.0, 72.0), (40.0, 39.0, 38.0)), SRC)
    assert r['arms']['LN']['H_none'] == 'LN beats BN' and r['arms']['LN']['H_ada'] == 'LN beats BN'
    assert r['arms']['GN']['H_none'] == 'level' and r['arms']['GN']['H_ada'] == 'GN is below BN'
    assert not r['arms']['LN']['source_flag'] and r['arms']['GN']['source_flag']    # 40.0 < 44.94 - 3


def test_nothing_is_read_when_a_reference_or_a_reproduction_is_off():
    rows = rows_with((70.0, 71.0, 72.0), (40.0, 39.0, 38.0))
    rows[-1]['pooled']['bev'] = 67.0                                                  # BN_ada 668 off by 0.73
    r = m.readings(rows, SRC)
    assert r['references_off'] and r['arms']['LN']['H_none'] == 'not read'
    rows = rows_with((70.0, 71.0, 72.0), (40.0, 39.0, 38.0))
    rows[0]['reproduced'] = False
    assert m.readings(rows, SRC)['arms']['GN']['H_ada'] == 'not read'


def test_specs():
    k = m.kitti_specs()
    assert ('item68', 'GN', 667, 37419, 'i68_kitti_gn_s667') in k
    assert ('item68', 'BN_ada', 668, 32435, 'i41_adabn_1gpu_s668') in k and len(k) == 12
    assert ('item68', 'LN_src', 666, 37409, None) in m.source_specs() and len(m.source_specs()) == 9


@pytest.mark.skipif(not (TOOLS / 'logs' / 'error_32435_i41ev8.txt').exists(), reason='no item 41a logs')
def test_bn_rows_resolve_from_41a_logs():
    import item38_kitti_rings as m38
    got = {(r[1], r[2]): m38.resolve(*r) for r in m.kitti_specs() if r[1].startswith('BN')}
    for (name, s), r in got.items():
        assert r['status'] == 'ok' and r['epoch'] == 20, (name, s, r)
        assert r['logged_bev'] == pytest.approx(m.REF[name][m.SEEDS.index(s)], abs=0.006), (name, s, r['logged_bev'])
    src = {r[2]: m38.resolve(*r) for r in m.source_specs() if r[1] == 'BN_src'}
    assert [round(src[s]['logged_bev'], 2) for s in m.SEEDS] == [44.81, 44.84, 45.18]
