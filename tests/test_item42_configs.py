"""Item 42a / 42a' configs (experiments_md 20261003_05 §5 item 42, design fixed 2026-11-08 22:40 JST). CPU only, no dataset files.

42a: every row is rescored after full AdaBN with `analysis/adabn_eval.py --stats target`, which re-estimates the BatchNorm
statistics on the TRAIN split of the eval config's DATA_CONFIG_TAR (DATA_CONFIG if there is none) in the eval-mode view. The design
requires those statistics to come from the target's UNPERTURBED training clouds: no TRAIN_FACTOR_SWAP / RING_PATTERN / density
calibration / evaluation-time cut acting on that loader. Checked here per eval config:
  - the block adabn_eval reads exists as DATA_CONFIG_TAR and equals the target oracle's DATA_CONFIG_TAR exactly (so every row of a
    target reads the same frames the same way);
  - it carries no key that perturbs points or labels;
  - the eval config's MODEL and CLASS_NAMES equal those of the config the checkpoint was trained with.
42a': the L_K eval twin carries L_K's training RANGE_KEEP exactly and resolves to the oracle base plus that key only.
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

PERTURBING = ('TRAIN_FACTOR_SWAP', 'EVAL_FACTOR_SWAP', 'EVAL_RING_THIN', 'EVAL_TOP_THIN', 'RING_PATTERN', 'RING_FOV_CUT',
              'TRAIN_TOP_THIN', 'BEAM_DROP', 'BEAM_DISTILL', 'ROW_EXTRUSION', 'LIT_TRANSLATED', 'LIDAR_INDICES',
              'USE_PSEUDO_LABEL', 'KEEP_FRAMES', 'LEVEL_COOR', 'REMOVE_ORIGIN_GTS', 'HIST_DIST_REFERENCE_CFG')
ORACLE = {'K': m42.BASE, 'W': 'cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml',
          'P': 'cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml',
          'L': 'cfgs/da-ieee-access-analysis/factorswap-lyft40/centerpoint-factorswap-lyft40-oracle.yaml'}
ADABN_ROWS = [r for r in m42.ROWS if r[0] != 'A']
_CACHE = {}


def resolved(path):
    from pcdet.config import cfg_from_yaml_file
    if path not in _CACHE:
        _CACHE[path] = cfg_from_yaml_file(path, EasyDict())
    return _CACHE[path]


def plain(d):
    if isinstance(d, dict):
        return {k: plain(v) for k, v in d.items() if k != '_BASE_CONFIG_'}
    if isinstance(d, list):
        return [plain(v) for v in d]
    return d


def keys_anywhere(d):
    out = set()
    if isinstance(d, dict):
        for k, v in d.items():
            out.add(k)
            out |= keys_anywhere(v)
    elif isinstance(d, list):
        for v in d:
            out |= keys_anywhere(v)
    return out


def test_row_table_covers_the_design():
    # T05 item 42a: KITTI O, P_K, L_K, C1_K = R3, R1, R2; Waymo O2, P, L_W, C1; Pandar64 O_P, P, L, C1; Lyft-40 O, P, L, C1
    want = {'K': {'O', 'P', 'L', 'C1', 'R1', 'R2'}, 'W': {'O', 'P', 'L', 'C1'}, 'P': {'O', 'P', 'L', 'C1'},
            'L': {'O', 'P', 'L', 'C1'}, 'A': {'L'}}
    for part, rows in want.items():
        for row in rows:
            seeds = sorted(r[2] for r in m42.ROWS if r[0] == part and r[1] == row)
            assert seeds == [666, 667], (part, row, seeds)
    tags = [m42.tag(*r[:4]) for r in m42.ROWS]
    assert len(set(tags)) == len(tags)  # per-row output dirs (adabn_eval keys its output by config + epoch)
    assert m42.ROWS[0][:4] == ('K', 'O', 666, 26843)  # the smoke goes first


@pytest.mark.parametrize('row', ADABN_ROWS, ids=lambda r: m42.tag(*r[:4]))
def test_adabn_statistics_block_is_the_unperturbed_target_train_split(monkeypatch, row):
    monkeypatch.chdir(TOOLS)
    part, _, _, _, _, eval_cfg, _ = row
    cfg = resolved(eval_cfg)
    assert cfg.get('DATA_CONFIG_TAR', None), 'adabn_eval would read DATA_CONFIG (the training block)'
    block = plain(cfg.DATA_CONFIG_TAR)
    assert block == plain(resolved(ORACLE[part]).DATA_CONFIG_TAR)
    assert not (keys_anywhere(block) & set(PERTURBING)), keys_anywhere(block) & set(PERTURBING)
    assert 'train' in block['DATA_SPLIT'] and block['DATA_SPLIT']['train'] == 'train'
    assert not any(k.startswith('HIST') or 'DENSITY' in k or 'CALIB' in k for k in keys_anywhere(block))


@pytest.mark.parametrize('row', m42.ROWS, ids=lambda r: m42.tag(*r[:4]))
def test_eval_config_builds_the_trained_model(monkeypatch, row):
    monkeypatch.chdir(TOOLS)
    _, _, _, _, _, eval_cfg, train_cfg = row
    ev, tr = resolved(eval_cfg), resolved(train_cfg)
    assert plain(ev.MODEL) == plain(tr.MODEL)
    assert list(ev.CLASS_NAMES) == list(tr.CLASS_NAMES)


def test_42a_prime_twin_carries_l_k_training_cut_exactly():
    twin = yaml.safe_load(open(TOOLS / m42.TWIN_42AP))
    lk = yaml.safe_load(open(m42.L_K))
    assert twin['_BASE_CONFIG_'] == m42.BASE
    assert twin['DATA_CONFIG_TAR']['EVAL_FACTOR_SWAP'] == lk['DATA_CONFIG']['TRAIN_FACTOR_SWAP']
    assert (TOOLS / m42.TWIN_42AP).read_text() == m42.render()  # generated, not hand-edited


def test_42a_prime_twin_resolves_to_the_oracle_base_plus_the_key(monkeypatch):
    monkeypatch.chdir(TOOLS)
    base = plain(resolved(m42.BASE))
    twin = plain(resolved(m42.TWIN_42AP))
    assert 'EVAL_FACTOR_SWAP' not in base['DATA_CONFIG_TAR']
    twin['DATA_CONFIG_TAR'].pop('EVAL_FACTOR_SWAP')
    assert twin == base


def test_42a_prime_twin_is_l_k_minus_its_training_arm_plus_the_eval_arm(monkeypatch):
    monkeypatch.chdir(TOOLS)
    lk = plain(resolved(str(m42.L_K.relative_to(TOOLS))))
    twin = plain(resolved(m42.TWIN_42AP))
    arm = lk['DATA_CONFIG'].pop('TRAIN_FACTOR_SWAP')
    assert twin['DATA_CONFIG_TAR'].pop('EVAL_FACTOR_SWAP') == arm
    assert twin == lk


def test_rule_a_rows_point_at_the_adabn_output_dirs():
    import json
    rows = [r for r in m42.ROWS if r[0] == 'P']
    paths = [str(m42.result_pkl(*r[:6])) for r in rows]
    assert len(set(paths)) == len(rows) == 8
    for r, p in zip(rows, paths):
        t = m42.tag(*r[:4])
        assert p.endswith('/%s/eval/epoch_115/%s/result.pkl' % (t, t)), p
    assert m42.result_pkl(*m42.ROWS[-1][:6]).parts[-6:] == (
        m42.EXTRA_TAG_42AP, 'eval', 'epoch_152', 'val', m42.tag(*m42.ROWS[-1][:4]), 'result.pkl')
    json.dumps([[str(p), 0, False] for p in paths])
