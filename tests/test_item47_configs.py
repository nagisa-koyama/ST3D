"""Item 47 configs and row table (experiments_md 20261003_05 §5 item 47, design fixed 2026-11-21 19:30 JST). CPU only, no
dataset files.

Item 47 scores item 42a's checkpoints on S', the dense sensor's own clouds thinned to nuScenes' lattice by P's own training
cut, with stats `none` and `target`. `adabn_eval.py --stats target` re-estimates BatchNorm on the TRAIN split of the eval config's
DATA_CONFIG_TAR in the eval-mode view, so the design needs, per eval config:
  - DATA_CONFIG_TAR = the target oracle's block (as item 42a pinned it) plus exactly ONE lattice key, and nothing else that
    perturbs points or labels;
  - that key equal to P's training cut on that target (KITTI: P_K's TRAIN_FACTOR_SWAP.RING_THIN; Pandar64: P's RING_PATTERN
    with MODE pattern; Waymo: P's RING_PATTERN with APPLY_AT_EVAL);
  - every arm of a target reading the same block, and the eval config building the trained model.
B rows (nuScenes val): their training config, whose DATA_CONFIG_TAR is unperturbed nuScenes.
"""
import sys
from pathlib import Path

import pytest
import yaml
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'analysis'))
import _init_path  # noqa: E402,F401
import make_item42_configs as m42  # noqa: E402
import make_item47_configs as m47  # noqa: E402
from test_item42_configs import PERTURBING, ORACLE, plain, keys_anywhere  # noqa: E402

CUT_KEY = {'K': 'EVAL_FACTOR_SWAP', 'P': 'EVAL_RING_THIN', 'W': 'RING_PATTERN'}
A_ROWS = [r for r in m47.ROWS if r[0] in ('K', 'W', 'P')]
B_ROWS = [r for r in m47.ROWS if r[0] in ('BW', 'BP')]
_CACHE = {}


def resolved(path):
    from pcdet.config import cfg_from_yaml_file
    if path not in _CACHE:
        _CACHE[path] = cfg_from_yaml_file(path, EasyDict())
    return _CACHE[path]


def rid(r):
    return m47.tag(*r[:4], r[5])


def test_row_table_covers_the_design():
    for part in ('K', 'W', 'P'):
        for row in ('O', 'P', 'L', 'C1'):
            for stats in ('none', 'target'):
                seeds = sorted(r[2] for r in m47.ROWS if r[0] == part and r[1] == row and r[5] == stats)
                assert seeds == [666, 667], (part, row, stats, seeds)
    for part in ('BW', 'BP'):
        assert sorted({r[1] for r in m47.ROWS if r[0] == part}) == ['C1', 'L', 'P']
        assert len([r for r in m47.ROWS if r[0] == part]) == 12
    extra = [r for r in m47.ROWS if r[5] == 'none2']
    assert [r[:5] for r in extra] == [('W', 'P', 666, 27807, 7)]  # procedure check 4, the Waymo redraw
    assert m47.stats_mode('none2') == 'none' and m47.stats_mode('target') == 'target'
    tags = [rid(r) for r in m47.ROWS]
    assert len(set(tags)) == len(tags) == 73  # per-row output dirs (adabn_eval keys its output by config + epoch)
    assert m47.ROWS[0][:6] == ('K', 'P', 666, 32283, 152, 'none')  # the smoke: P_K on its own lattice


def test_checkpoints_are_item_42a_ones():
    m42rows = {(r[0], r[1], r[2]): (r[3], r[4], r[6]) for r in m42.ROWS}
    for r in m47.ROWS:
        part = r[0][-1] if r[0].startswith('B') else r[0]
        assert m42rows[(part, r[1], r[2])] == (r[3], r[4], r[7]), r


@pytest.mark.parametrize('twin', m47.written_twins(), ids=lambda t: '%s_%s' % t[:2])
def test_twins_are_generated_not_hand_edited(twin):
    part, row, base = twin
    assert (TOOLS / m47.twin_path(part, row)).read_text() == m47.render(part, row, base)
    assert yaml.safe_load(open(TOOLS / m47.twin_path(part, row)))['_BASE_CONFIG_'] == base


@pytest.mark.parametrize('twin', m47.written_twins(), ids=lambda t: '%s_%s' % t[:2])
def test_twin_resolves_to_its_item42a_eval_config_plus_the_cut(monkeypatch, twin):
    monkeypatch.chdir(TOOLS)
    part, row, base = twin
    got = plain(resolved(m47.twin_path(part, row)))
    want = plain(resolved(base))
    assert CUT_KEY[part] not in want['DATA_CONFIG_TAR']
    got['DATA_CONFIG_TAR'].pop(CUT_KEY[part])
    assert got == want


def test_cuts_are_p_s_own_training_cuts(monkeypatch):
    monkeypatch.chdir(TOOLS)
    p_k = yaml.safe_load(open(TOOLS / m47.P_K_TRAIN))['DATA_CONFIG']['TRAIN_FACTOR_SWAP']
    rp_p = yaml.safe_load(open(TOOLS / m47.P_PANDASET_TRAIN))['DATA_CONFIG']['RING_PATTERN']
    rp_w = yaml.safe_load(open(TOOLS / m47.P_WAYMO_TRAIN))['DATA_CONFIG']['RING_PATTERN']
    for r in A_ROWS:
        cut = plain(resolved(r[6]).DATA_CONFIG_TAR[CUT_KEY[r[0]]])
        if r[0] == 'K':
            assert cut == p_k, r
        elif r[0] == 'P':
            assert cut == dict(rp_p, MODE='pattern'), r
        else:
            assert cut == dict(rp_w, APPLY_AT_EVAL=True), r


@pytest.mark.parametrize('part', ['K', 'W', 'P'])
def test_every_arm_of_a_target_reads_one_block_the_oracle_s_plus_the_cut(monkeypatch, part):
    monkeypatch.chdir(TOOLS)
    blocks = {plain_json(plain(resolved(r[6]).DATA_CONFIG_TAR)) for r in A_ROWS if r[0] == part}
    assert len(blocks) == 1, part
    block = plain(resolved([r for r in A_ROWS if r[0] == part][0][6]).DATA_CONFIG_TAR)
    block.pop(CUT_KEY[part])
    assert block == plain(resolved(ORACLE[part]).DATA_CONFIG_TAR)
    assert not (keys_anywhere(block) & set(PERTURBING))
    assert block['DATA_SPLIT']['train'] == 'train'


def plain_json(d):
    import json
    return json.dumps(d, sort_keys=True, default=str)


@pytest.mark.parametrize('row', B_ROWS, ids=rid)
def test_b_rows_read_unperturbed_nuscenes(monkeypatch, row):
    monkeypatch.chdir(TOOLS)
    assert row[6] == row[7]  # scored with the config it was trained with, as its in-run evaluation
    block = plain(resolved(row[6]).DATA_CONFIG_TAR)
    assert block['DATASET'] == 'NuScenesDataset'
    assert not (keys_anywhere(block) & set(PERTURBING)), keys_anywhere(block) & set(PERTURBING)
    assert block['DATA_SPLIT']['train'] == 'train'


@pytest.mark.parametrize('row', [r for r in m47.ROWS if r[5] == 'none'], ids=rid)
def test_eval_config_builds_the_trained_model(monkeypatch, row):
    monkeypatch.chdir(TOOLS)
    ev, tr = resolved(row[6]), resolved(row[7])
    assert plain(ev.MODEL) == plain(tr.MODEL)
    assert list(ev.CLASS_NAMES) == list(tr.CLASS_NAMES)


def test_rule_a_rows_point_at_the_adabn_output_dirs():
    rows = [r for r in m47.ROWS if r[0] == 'P']
    paths = [str(m47.result_pkl(*r[:7])) for r in rows]
    assert len(set(paths)) == len(rows) == 16
    for r, p in zip(rows, paths):
        t = m47.tag(*r[:4], r[5])
        assert p.endswith('/%s/eval/epoch_115/%s/result.pkl' % (t, t)), p
