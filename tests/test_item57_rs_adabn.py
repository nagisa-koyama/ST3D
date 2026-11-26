"""Item 57's readings (experiments_md 20261003_05 §5 item 57, design fixed 2026-11-27 00:10 JST). CPU only, no files."""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item57_rs_adabn as m  # noqa: E402


def row(name, seed, rings, pooled=70.0, logged=None, ok=True):
    return dict(row=name, seed=seed, reproduced=ok, logged_bev=logged if logged is not None else pooled,
                pooled={'bev': pooled, '3d': 40.0}, rings=[{'bev': x} for x in rings])


L = [92.4, 95.8, 83.5, 65.7, 32.3]
P1 = [90.8, 91.7, 74.4, 55.5, 31.7]


def table(p5, p1=P1, rs1_pooled=69.4, ada=69.4):
    out = []
    for s in (666, 667):
        out += [row('L_ada', s, L), row('P_ada', s, P1, pooled=ada), row('P_RS1', s, p1, pooled=rs1_pooled), row('P_RS5', s, p5)]
    return out


def read(rows):
    seeds = [m.per_seed(rows, s) for s, _ in m.CKPTS]
    return m.reading1(seeds), m.reading2(seeds), seeds


def test_far_residual_removed():
    r1, r2, seeds = read(table([90.8, 91.7, 81.0, 62.0, 31.7]))         # s at 20-30 / 30-40: 9.1 / 10.2 -> 2.5 / 3.7
    assert r1 == 'RS-AdaBN removes most of the far residual' and r2 == 'near kept'
    assert abs(seeds[0]['delta'] - ((9.1 - 2.5) + (10.2 - 3.7)) / 2) < 1e-9


def test_not_removed_and_partly():
    assert read(table([90.8, 91.7, 74.9, 55.9, 31.7]))[0] == 'does not remove it'
    assert read(table([90.8, 91.7, 77.5, 58.5, 31.7]))[0] == 'partly'


def test_near_guard():
    assert read(table([86.0, 88.0, 81.0, 62.0, 31.7]))[1] == 'costs near'


def test_rs1_must_reproduce_adabn():
    r1, _, seeds = read(table([90.8, 91.7, 81.0, 62.0, 31.7], rs1_pooled=69.8, ada=69.4))
    assert r1 == 'not read' and seeds[0]['check'].endswith('not read')
    r1, _, _ = read(table([90.8, 91.7, 81.0, 62.0, 31.7], rs1_pooled=69.65, ada=69.4))
    assert r1 == 'RS-AdaBN removes most of the far residual'


def test_runs_and_paths():
    rs = m.runs()
    assert len(rs) == 4 and {r[2] for r in rs} == {'global', 'item57'}
    d = m.result_dir('item57', 666, 32283)
    assert d.parts[-4:] == ('i57_item57_s666_32283', 'eval', 'epoch_152', 'i57_item57_s666_32283')
    assert d.parts[-5] == 'centerpoint-factorswap-s2-p-pattern'
