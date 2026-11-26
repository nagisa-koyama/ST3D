"""Item 56's readings and row table (experiments_md 20261003_05 §5 item 56, design fixed 2026-11-27 00:02 JST). CPU only, no
files."""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item56_halves as m  # noqa: E402

RINGS = m.RING_CELLS


def table(o_full, o_rows, o_az, o_sp, p_sp, few_rows=None, few_p=None, n=500):
    """v[(det, cloud, seed)] = {cell: bev}; same values at both seeds."""
    v = {}
    for s in (666, 667):
        for det, cloud, vals in (('O', 'full', o_full), ('O', 'rows', o_rows), ('O', 'az', o_az), ('O', 'sprime', o_sp),
                                 ('P', 'sprime', p_sp)):
            cells = dict(zip(RINGS, vals))
            if few_rows is not None:
                src = few_rows if (det, cloud) == ('O', 'rows') else few_p if (det, cloud) == ('P', 'sprime') else [50] * 5
                cells.update({'%s few' % r: x for r, x in zip(RINGS, src)})
            v[(det, cloud, s)] = cells
    nn = {c: n for c in RINGS}
    nn.update({'%s few' % r: n for r in RINGS})
    return v, nn


FULL, SP, P = [90, 84, 55, 22, 4], [87, 82, 54, 22, 3.5], [92, 93, 72, 45, 16]


def test_rows_restoring_the_far_loss_reads_azimuth():
    v, _ = table(FULL, [90, 90, 68, 40, 13], [70, 60, 40, 20, 3], SP, P)
    r, d = m.reading1(v)
    assert r == 'azimuth carries most of the far loss'
    assert d['rings_read'] == ['ring 30-40', 'ring 40-70.4']                  # the far rings with P - O >= 5 at both seeds
    assert abs(d['far'][666] - ((40 - 22) / 23 + (13 - 3.5) / 12.5) / 2) < 1e-9


def test_rows_not_restoring_reads_no_and_in_between_partly():
    v, _ = table(FULL, [88, 83, 55, 24, 4], [70, 60, 40, 20, 3], SP, P)
    assert m.reading1(v)[0] == 'azimuth does not carry it'
    v, _ = table(FULL, [88, 83, 60, 30, 8], [70, 60, 40, 20, 3], SP, P)
    assert m.reading1(v)[0] == 'partly'


def test_gap_under_5_ap_is_not_read():
    v, _ = table(FULL, [90, 90, 68, 40, 13], FULL, SP, [91, 93, 72, 45, 16])
    r, d = m.reading1(v)
    assert d['rho'][666][0] is None and d['rho'][666][3] is not None           # 0-10 m: P - O = 4 < 5


def test_reading2_uses_item_54_rule_on_rows_against_p():
    v, n = table(FULL, FULL, FULL, SP, P, few_rows=[0, 70, 66, 60, 18], few_p=[0, 80, 75, 65, 20])
    r, d = m.reading2(v, n)
    assert r == 'range-free'                                                   # q 0.875 / 0.88 / 0.92 / 0.90
    v, n = table(FULL, FULL, FULL, SP, P, few_rows=[0, 70, 50, 30, 5], few_p=[0, 80, 75, 65, 20])
    assert m.reading2(v, n)[0] == 'range also matters'


def test_reading3_which_axis():
    v, _ = table(FULL, [90, 84, 54, 20, 3], [80, 70, 40, 10, 1], SP, P)       # far costs: rows 1.5, az 7.5
    r, d = m.reading3(v)
    assert r == 'azimuth larger'
    assert d['far']['rows s666'] == 1.5 and d['far']['az s666'] == 7.5
    v, _ = table(FULL, [80, 70, 40, 10, 1], [90, 84, 54, 20, 3], SP, P)
    assert m.reading3(v)[0] == 'vertical larger'
    v, _ = table(FULL, [85, 80, 50, 18, 2], [85, 80, 50, 17, 2.5], SP, P)
    assert m.reading3(v)[0] == 'comparable'


def test_eval_rows_and_paths():
    rows = m.eval_rows()
    assert len(rows) == 8 and len({r[0] for r in rows}) == 8
    assert all(r[3] == 152 for r in rows) and {r[4] for r in rows} == {'rows', 'az'}
    p = m.result_pkl('O', 'rows', 666, 26843)
    assert p.parts[-6:] == ('centerpoint-item56-kitti-o-on-nuscenes-rows', 'i56K_rows_O_none_s666_26843', 'eval', 'epoch_152',
                            'i56K_rows_O_none_s666_26843', 'result.pkl')
