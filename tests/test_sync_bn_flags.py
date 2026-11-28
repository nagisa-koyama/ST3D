"""`--sync_bn` guard (pcdet/utils/norm_flags.py; user 2026-11-28: 2-GPU rows use SyncBatchNorm; experiments_md 20261003_14 §3).
train.py converts to SyncBatchNorm OR to DSNorm (if / elif), so a DSNorm row given --sync_bn would silently lose DSNorm; the
guard must name it, and RANGE_BN / NORM_SWAP. The consumers of BN statistics must still see converted layers. CPU only."""
import sys
from pathlib import Path

import torch
from easydict import EasyDict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'tools'))
from pcdet.utils.norm_flags import sync_bn_conflicts  # noqa: E402


def test_plain_rows_have_no_conflict():
    assert sync_bn_conflicts(EasyDict(MODEL=EasyDict(NAME='CenterPoint'))) == []
    assert sync_bn_conflicts(EasyDict(MODEL=EasyDict(), SELF_TRAIN=EasyDict(DSNORM=False, SCORE_THRESH=0.1))) == []


def test_dsnorm_range_bn_and_norm_swap_are_named():
    ds = sync_bn_conflicts(EasyDict(MODEL=EasyDict(), SELF_TRAIN=EasyDict(DSNORM=True)))
    assert len(ds) == 1 and 'DSNORM' in ds[0] and 'skipped' in ds[0]
    rb = sync_bn_conflicts(EasyDict(MODEL=EasyDict(RANGE_BN=EasyDict(KNOTS=[5.0]))))
    assert len(rb) == 1 and 'RANGE_BN' in rb[0]
    ns = sync_bn_conflicts(EasyDict(MODEL=EasyDict(NORM_SWAP=EasyDict(TYPE='layer'))))
    assert len(ns) == 1 and 'NORM_SWAP' in ns[0]


def test_train_py_refuses_before_converting():
    src = (REPO / 'tools/train.py').read_text()
    i, j = src.index('bad = sync_bn_conflicts(cfg)'), src.index('model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)')
    assert i < j and src.index("elif cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):") > j


def test_converted_layers_are_still_found_by_the_adabn_tools():
    """Teacher AdaBN (adabn_utils.reestimate_bn) runs after the conversion in train.py: it must still see every layer."""
    from pcdet.utils import adabn_utils
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Conv2d(4, 4, 1), torch.nn.BatchNorm2d(4))
    s = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
    assert sum(isinstance(x, torch.nn.SyncBatchNorm) for x in s.modules()) == 2
    assert len(adabn_utils.bn_layers(s)) == 2
    keys = set(s.state_dict())                                 # a SyncBN checkpoint loads into the plain-BN eval model
    assert keys == set(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Conv2d(4, 4, 1),
                                           torch.nn.BatchNorm2d(4)).state_dict())
