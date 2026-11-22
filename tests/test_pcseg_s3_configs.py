"""The S3 configs added for the approved plan (experiments_md episodes/20261116_01 §8.10) and the arm-smoke checker
(tools/analysis/pcseg_arm_smoke_check.py; bug_prevention_checklist rules 4 and 9).

- b0 / b1 / b1-reliance: the block reaches only its split, the config differs from the PandarGT oracle's only in that
  block, and the cache it names is the declared kind (source device, depth, both splits, all 4,800 frames).
- The six smoke configs: each equals its arm config except the smoke overrides of spunet-pandaset64-smoke.yaml, so the
  smoke runs the arm's block.
- The checker's comparisons flag a changed dump, a leak into the untouched split, a density arm that does not thin or
  grows a class, and a config with no block, both blocks or an unknown operator.
"""
import importlib.util
import json
import os
import sys

import pytest

from pcseg import pandaset_seg as ps_mod

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, 'tools')
CACHE = '/home/koyama/data/pcseg/render_cache/v1'
B_ARMS = {  # config: (block, cache kind)
    'spunet-p1-b0-identity': ('FACTOR_SWAP', 'gt_on_gtrays'),
    'spunet-p1-b1-lines': ('FACTOR_SWAP', 'gt_on_p64ring'),
    'spunet-p1-b1-lines-reliance': ('EVAL_PERTURB', 'gt_on_p64ring'),
}
SMOKES = ['spunet-p1-a1-density', 'spunet-p1-a2-intensity', 'spunet-p1-a3-cone', 'spunet-p1-a4-intensity',
          'spunet-p1-b0-identity', 'spunet-p1-b1-lines']
SMOKE_OPT = {'BATCH_SIZE': 4, 'NUM_EPOCHS': 2, 'SAVE_EVERY': 1, 'MAX_CKPT': 2, 'LOG_EVERY': 5}


def _load_cfg(name):
    sys.path.insert(0, TOOLS)
    import train_seg
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        return train_seg.load_cfg('cfgs/pcseg/%s.yaml' % name)
    finally:
        os.chdir(cwd)


class _Recorder:
    """As in test_pcseg_factor_ops.py: records the dataset config; 45 / 15 sequences pass EXPECT_SEQUENCES."""

    def __init__(self, cfg, training, logger=None):
        self.cfg, self.training = cfg, training
        self.sequences = list(cfg['SEQUENCES'])[:45 if training else 15]


def _routed(monkeypatch, cfg):
    sys.path.insert(0, TOOLS)
    import train_seg
    monkeypatch.setattr(ps_mod, 'PandasetSeg', _Recorder)
    sets = train_seg.build_datasets(cfg)
    return sets['train'].cfg, sets['val'].cfg


def _checker():
    spec = importlib.util.spec_from_file_location('pcseg_arm_smoke_check',
                                                  os.path.join(TOOLS, 'analysis', 'pcseg_arm_smoke_check.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip(cfg):
    for x in (cfg,):
        x.pop('TAG', None)
        x.pop('_BASE_CONFIG_', None)
    return cfg


# --- b0 / b1 / b1-reliance ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(B_ARMS))
def test_b_block_reaches_only_its_split(monkeypatch, name):
    block, kind = B_ARMS[name]
    cfg = _load_cfg(name)
    tr, va = _routed(monkeypatch, cfg)
    on, off = (tr, va) if block == 'FACTOR_SWAP' else (va, tr)
    assert 'FACTOR_OPS' not in off
    assert dict(on['FACTOR_OPS']) == {'RENDER': {'CACHE': os.path.join(CACHE, kind)}}


@pytest.mark.parametrize('name', sorted(B_ARMS))
def test_b_config_differs_from_the_pandargt_oracle_only_in_its_block(name):
    block, kind = B_ARMS[name]
    a, b = _strip(_load_cfg(name)), _strip(_load_cfg('spunet-pandasetgt'))
    assert a.DATA.LIDAR_DEVICE == 1
    other = 'EVAL_PERTURB' if block == 'FACTOR_SWAP' else 'FACTOR_SWAP'
    assert other not in a.DATA
    da, db = dict(a.DATA), dict(b.DATA)
    da.pop(block)
    assert da == db
    ra, rb = dict(a), dict(b)
    ra.pop('DATA'), rb.pop('DATA')
    assert ra == rb


@pytest.mark.parametrize('name', sorted(B_ARMS))
def test_b_cache_is_the_declared_kind(name):
    block, kind = B_ARMS[name]
    path = _load_cfg(name).DATA[block].RENDER.CACHE
    if not os.path.isfile(os.path.join(path, 'manifest.json')):
        pytest.skip('render cache absent')
    m = json.load(open(os.path.join(path, 'manifest.json')))
    assert m['kind'] == kind and int(m['source_device']) == 1 and int(m['depth']) == 3
    assert sorted(m['splits']) == ['train', 'val'] and m['n_frames'] == 4800 and m['frames'] is None
    from pcseg.render_cache import RenderCache
    RenderCache({'CACHE': path}, 1)                                    # loads; the device is checked


# --- smoke configs ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('arm', SMOKES)
def test_smoke_equals_its_arm_but_the_smoke_overrides(arm):
    s, a = _strip(_load_cfg(arm + '-smoke')), _strip(_load_cfg(arm))
    assert s.DATA.SPLITS == {'train': ['035', '041'], 'val': ['097']}
    assert s.DATA.EXPECT_SEQUENCES == {'train': 2, 'val': 1}
    for k, v in SMOKE_OPT.items():
        assert s.OPTIMIZATION[k] == v
    assert s.EVAL.EVERY == 1
    for blk in ('FACTOR_SWAP', 'EVAL_PERTURB'):
        assert s.DATA.get(blk, None) == a.DATA.get(blk, None)
    ds, da = dict(s.DATA), dict(a.DATA)
    for k in ('SPLITS', 'EXPECT_SEQUENCES'):
        ds.pop(k, None), da.pop(k, None)
    assert ds == da
    os_, oa = dict(s.OPTIMIZATION), dict(a.OPTIMIZATION)
    for k in SMOKE_OPT:
        os_.pop(k, None), oa.pop(k, None)
    assert os_ == oa
    es, ea = dict(s.EVAL), dict(a.EVAL)
    es.pop('EVERY', None), ea.pop('EVERY', None)
    assert es == ea
    assert s.MODEL == a.MODEL


# --- the checker --------------------------------------------------------------------------------------------------------

def _row(frame, points, cnt, inv=None):
    r = {'frame': frame, 'points': points, 'voxels': points, 'ignored_points': 0, 'ignored_voxels': 0,
         'point_counts': dict(cnt), 'voxel_counts': dict(cnt)}
    if inv is not None:
        r['inverse_agreement'] = inv
    return r


def test_dump_diff_finds_a_changed_count_a_leak_and_a_missing_split():
    ck = _checker()
    a = {'train': [_row('035/00', 10, {'road': 10})], 'val': [_row('097/00', 5, {'car': 5}, 1.0)]}
    assert ck.dump_diff(a, json.loads(json.dumps(a))) == []
    b = json.loads(json.dumps(a))
    b['val'][0]['point_counts']['car'] = 4
    assert ck.dump_diff(a, b) and 'val' in ck.dump_diff(a, b)[0]
    b = json.loads(json.dumps(a))
    b['val'][0]['inverse_agreement'] = 0.999
    assert ck.dump_diff(a, b)
    assert ck.dump_diff(a, {'train': a['train']})


def test_check_subset_flags_no_thinning_and_a_grown_class():
    ck = _checker()
    base = _row('035/00', 10, {'road': 6, 'car': 4})
    assert ck.check_subset(_row('035/00', 5, {'road': 3, 'car': 2}), base) == []
    assert ck.check_subset(_row('035/00', 10, {'road': 6, 'car': 4}), base)           # nothing thinned
    assert ck.check_subset(_row('035/00', 6, {'road': 1, 'car': 5}), base)            # car grew
    assert ck.check_subset(_row('035/00', 3, {'road': 1, 'bus': 2}), base)            # a class the base lacks


def test_check_equal_counts():
    ck = _checker()
    r = _row('035/00', 4, {'road': 3, 'car': 1})
    assert ck.check_equal_counts(r, 4, {'road': 3, 'car': 1}, 'x') == []
    assert ck.check_equal_counts(r, 5, {'road': 3, 'car': 1}, 'x')
    assert ck.check_equal_counts(r, 4, {'road': 4}, 'x')


def test_arm_block_and_strip_blocks():
    ck = _checker()
    cfg = _load_cfg('spunet-p1-a1-density')
    assert ck.arm_block(cfg) == ('train', ['DENSITY'])
    assert ck.arm_block(_load_cfg('spunet-p1-b1-lines-reliance')) == ('val', ['RENDER'])
    base = ck.strip_blocks(cfg)
    assert 'FACTOR_SWAP' not in base.DATA and 'FACTOR_SWAP' in cfg.DATA            # a copy, the input untouched
    with pytest.raises(ValueError):
        ck.arm_block(base)
    both = ck.strip_blocks(cfg)
    both.DATA.FACTOR_SWAP = {'CONE': {'FOV_DEG': 60.0}}
    both.DATA.EVAL_PERTURB = {'CONE': {'FOV_DEG': 60.0}}
    with pytest.raises(ValueError):
        ck.arm_block(both)
    odd = ck.strip_blocks(cfg)
    odd.DATA.FACTOR_SWAP = {'JITTER': {}}
    with pytest.raises(ValueError):
        ck.arm_block(odd)
