"""Item-39 configs (experiments_md 20261108_03 §2 B2 / C): the control's eval-only twins carry exactly the training cut they
name and differ from the control's evaluation config only in DATA_CONFIG_TAR.EVAL_FACTOR_SWAP. CPU only, no dataset files."""
import sys
from pathlib import Path

import pytest
import yaml
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import _init_path  # noqa: E402,F401

I39 = TOOLS / 'cfgs/da-ieee-access-analysis/item39'
S2 = TOOLS / 'cfgs/da-ieee-access-analysis/factorswap-s2'
CONTROL = 'cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml'
TWINS = {
    'r3-density': ('centerpoint-factorswap-s2-r3-density.yaml', 'TRAIN'),
    'r3-halfway': ('centerpoint-factorswap-s2-eval-r3-halfway.yaml', 'EVAL'),
    'p-pattern': ('centerpoint-factorswap-s2-p-pattern.yaml', 'TRAIN'),
}


def without_base_keys(d):
    if isinstance(d, dict):
        return {k: without_base_keys(v) for k, v in d.items() if k != '_BASE_CONFIG_'}
    if isinstance(d, list):
        return [without_base_keys(v) for v in d]
    return d


@pytest.mark.parametrize('name', sorted(TWINS))
def test_twin_carries_the_named_cut_exactly(name):
    src, kind = TWINS[name]
    s = yaml.safe_load(open(S2 / src))
    want = s['DATA_CONFIG']['TRAIN_FACTOR_SWAP'] if kind == 'TRAIN' else s['DATA_CONFIG_TAR']['EVAL_FACTOR_SWAP']
    got = yaml.safe_load(open(I39 / ('centerpoint-control-eval-%s.yaml' % name)))
    assert got['_BASE_CONFIG_'] == CONTROL
    assert got['DATA_CONFIG_TAR']['EVAL_FACTOR_SWAP'] == want


@pytest.mark.parametrize('name', sorted(TWINS))
def test_twin_resolves_to_the_control_plus_the_key(monkeypatch, name):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(TOOLS)
    base = cfg_from_yaml_file(CONTROL, EasyDict())
    twin = cfg_from_yaml_file(str(I39 / ('centerpoint-control-eval-%s.yaml' % name)), EasyDict())
    assert 'EVAL_FACTOR_SWAP' in twin.DATA_CONFIG_TAR and 'EVAL_FACTOR_SWAP' not in base.DATA_CONFIG_TAR
    twin.DATA_CONFIG_TAR.pop('EVAL_FACTOR_SWAP')
    assert without_base_keys(twin) == without_base_keys(base)


ROW_27566 = TOOLS / 'cfgs/da-ieee-access/centerpoint-accum-legaldepth-st3d-nuscenes2kitti.yaml'
C_ROWS = {s: I39 / ('centerpoint-control-st3d-t%s-nuscenes2kitti.yaml' % s) for s in ('666', '667', '668')}


def below_base(path):
    lines = path.read_text().splitlines()
    i = [k for k, l in enumerate(lines) if l.startswith('_BASE_CONFIG_:')][0]
    return lines[i], lines[i + 1:]


@pytest.mark.parametrize('seed', sorted(C_ROWS))
def test_c_row_is_27566_verbatim_but_its_base_and_two_cut_lines(seed):
    base, mine = below_base(C_ROWS[seed])
    _, ref = below_base(ROW_27566)
    assert base == '_BASE_CONFIG_: ' + CONTROL
    assert len(mine) == len(ref)
    differ = [(a, b) for a, b in zip(ref, mine) if a != b]
    assert [a.split(':')[0].strip() for a, _ in differ] == ['SCORE_THRESH', 'NEG_THRESH']


@pytest.mark.parametrize('seed', sorted(C_ROWS))
def test_c_row_resolves_to_the_control_plus_self_training_with_valid_cuts(monkeypatch, seed):
    from pcdet.config import cfg_from_yaml_file
    monkeypatch.chdir(TOOLS)
    ctrl = without_base_keys(cfg_from_yaml_file(CONTROL, EasyDict()))
    row = without_base_keys(cfg_from_yaml_file(str(C_ROWS[seed]), EasyDict()))
    st, tar = row.pop('SELF_TRAIN'), row['DATA_CONFIG_TAR'].pop('USE_PSEUDO_LABEL')
    assert tar is True and row == ctrl
    teacher = dict(st['MODEL_TEACHER'])
    assert teacher.pop('CLASS_NAMES') == ['Car', 'Pedestrian', 'Cyclist'] and teacher.pop('ONTOLOGY') == 'kitti'
    assert teacher == ctrl['MODEL']
    score, neg = st['SCORE_THRESH'], st['NEG_THRESH']
    assert len(score) == len(neg) == 3
    assert all(isinstance(v, float) and 0 < v < 1 for v in score + neg)
    assert all(s >= n for s, n in zip(score, neg))
    assert st['FROZEN_TEACHER_SINGLE_PASS'] is True


def test_teacher_666_cuts_reproduce_28641():
    st = yaml.safe_load(open(C_ROWS['666']))['SELF_TRAIN']
    assert st['SCORE_THRESH'] == [0.589, 0.511, 0.482] and st['NEG_THRESH'] == [0.589, 0.511, 0.482]
