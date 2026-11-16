"""pcseg core (experiments_md episodes/20261116_01 §7): the IGNORE sentinel (-1) against every consumer, grid sampling's
index contract, collate offsets, and the SpUNet port against Pointcept's original source. On CUDA only: the port's
forward against Pointcept's, and evaluate() running the SpUNet (the smoke 33127 eval crash).

Sentinel coverage (bug_prevention_checklist rule 1): class map (raises on undeclared / negative / out-of-range raw ids,
maps declared-ignored ids to -1), ce_loss (ignored points add nothing; all-ignored is finite with zero gradient),
ConfusionMatrix (an ignored point never lands in any cell, in particular not in the last class, which is where a
negative index would wrap to).
"""
import os

import numpy as np
import pytest
import torch

from pcseg import IGNORE
from pcseg.class_map import (ClassMap, build_class_map, PANDASET_SCALR17_RAW_TO_NAME, PANDASET_SCALR17_NAMES,
                             PANDASET_3DLP8_RAW_TO_NAME)
from pcseg.grid_sample import grid_sample
from pcseg.losses_metrics import ce_loss, ConfusionMatrix
from pcseg.pandaset_seg import augment, collate, key_seed

POINTCEPT_SPUNET = '/home/koyama/data/released_repos/Pointcept/pointcept/models/sparse_unet/spconv_unet_v1m1_base.py'


# --- class map ----------------------------------------------------------------------------------------------------------

def test_class_map_maps_and_ignores():
    cm = ClassMap(['a', 'b'], {0: None, 1: 'a', 2: 'b', 3: 'a'})
    assert cm(np.array([1, 2, 3, 0])).tolist() == [0, 1, 0, IGNORE]


@pytest.mark.parametrize('bad', [[-1], [4], [99]])
def test_class_map_raises_outside_range(bad):
    cm = ClassMap(['a', 'b'], {0: None, 1: 'a', 2: 'b', 3: 'a'})
    with pytest.raises(ValueError):
        cm(np.array(bad))


def test_class_map_raises_on_undeclared_gap():
    cm = ClassMap(['a'], {1: 'a', 3: None})        # 0 and 2 never declared
    with pytest.raises(ValueError):
        cm(np.array([2]))
    with pytest.raises(ValueError):
        cm(np.array([0]))


def test_class_map_rejects_unused_model_class():
    with pytest.raises(AssertionError):
        ClassMap(['a', 'b'], {1: 'a'})


@pytest.mark.parametrize('name, table', [('pandaset_scalr17', PANDASET_SCALR17_RAW_TO_NAME),
                                         ('pandaset_3dlp8', PANDASET_3DLP8_RAW_TO_NAME)])
def test_pandaset_maps_cover_classes_json(name, table):
    cm = build_class_map(name)
    assert sorted(table) == list(range(0, 43))                        # classes.json 1..42, plus the stored 0
    out = cm(np.arange(0, 43))
    assert out[0] == IGNORE and out.min() == IGNORE and out.max() == cm.num_classes - 1
    with pytest.raises(ValueError):
        cm(np.array([43]))


# ScaLR's MAPPING_CLASS (valeoai/ScaLR datasets/pandaset_for_scalr.py at 53c474d), as published: raw id -> ScaLR class id,
# 0 = noise (ignore), train id = class id - 1. Transcribed independently of class_map.py's table.
SCALR_MAPPING_CLASS = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 5, 6: 16, 7: 1, 8: 6, 9: 6, 10: 6, 11: 7, 12: 17, 13: 10, 14: 12,
                       15: 12, 16: 12, 17: 15, 18: 11, 19: 15, 20: 15, 21: 15, 22: 0, 23: 13, 24: 0, 25: 0, 26: 14, 27: 15,
                       28: 15, 29: 15, 30: 4, 31: 4, 32: 0, 33: 0, 34: 9, 35: 3, 36: 2, 37: 9, 38: 2, 39: 3, 40: 0, 41: 8,
                       42: 8}
SCALR_CLASS_NAME = ['road', 'traffic sign', 'barrier', 'pedestrian', 'vegetation', 'road marking', 'sidewalk', 'manmade',
                    'traffic cone', 'car', 'motorcycle', 'truck', 'bus', 'bicycle', 'oth. vehicle', 'ground', 'driveway']
# 3DLabelProp's "pandaset" target_to_common (JulesSanchez/3DLabelProp mapping/sk.py at 202127f), as published.
TDLP_TARGET_TO_COMMON = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: 6, 6: 4, 7: 2, 8: 2, 9: 2, 10: 2, 11: 3, 12: 2, 13: 7, 14: 7,
                         15: 7, 16: 7, 17: 7, 18: 0, 19: 7, 20: 7, 21: 0, 22: 7, 23: 7, 24: 0, 25: 0, 26: 0, 27: -1, 28: -1,
                         29: -1, 30: 1, 31: 1, 32: -1, 33: -1, 34: 5, 35: 5, 36: 5, 37: 5, 38: 5, 39: 5, 40: 5, 41: 5, 42: 5}


def test_scalr17_map_equals_published():
    cm = build_class_map('pandaset_scalr17')
    assert [n.replace('_', ' ') for n in PANDASET_SCALR17_NAMES] == [
        n.replace('oth. vehicle', 'other vehicle') for n in SCALR_CLASS_NAME]
    raw = np.arange(0, 43)
    expect = np.array([SCALR_MAPPING_CLASS[r] - 1 for r in raw])
    assert np.array_equal(cm(raw), np.where(expect < 0, IGNORE, expect))


def test_3dlp8_map_equals_published():
    cm = build_class_map('pandaset_3dlp8')
    raw = np.arange(0, 43)
    assert np.array_equal(cm(raw), np.array([TDLP_TARGET_TO_COMMON[r] for r in raw]))


# --- grid sampling ------------------------------------------------------------------------------------------------------

def _cloud(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-3, 3, size=(n, 3))).astype(np.float32)


def test_grid_sample_index_contract():
    coord = _cloud()
    idx, grid, inv = grid_sample(coord, 0.5, np.random.default_rng(1))
    vox = np.floor(coord / 0.5).astype(np.int64)
    vox -= vox.min(0)
    assert len(np.unique(vox, axis=0)) == len(idx)                    # one representative per occupied voxel
    assert np.array_equal(grid, vox[idx])                            # grid_coord is the representative's voxel
    assert np.array_equal(vox[idx][inv], vox)                        # every point maps to its own voxel's row
    assert np.array_equal(inv[idx], np.arange(len(idx)))             # a representative maps to itself
    assert grid.min() == 0


def test_grid_sample_reproducible_by_rng():
    coord = _cloud()
    a = grid_sample(coord, 0.5, np.random.default_rng(key_seed('001/00')))
    b = grid_sample(coord, 0.5, np.random.default_rng(key_seed('001/00')))
    c = grid_sample(coord, 0.5, np.random.default_rng(key_seed('001/01')))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])


def test_labels_follow_the_sampled_index():
    """A label that is a function of the point's position stays that function after sampling."""
    coord = _cloud()
    seg = (coord[:, 0] > 0).astype(np.int64)
    idx, _, _ = grid_sample(coord, 0.5, np.random.default_rng(2))
    assert np.array_equal(seg[idx], (coord[idx, 0] > 0).astype(np.int64))


def test_augment_moves_but_never_reorders():
    coord = _cloud(200)
    aug = {'ROTATE_Z_PROB': 1.0, 'SCALE': [0.9, 1.1], 'FLIP_PROB': 0.5, 'JITTER': [0.005, 0.02]}
    out = augment(coord, np.random.default_rng(3), aug)
    assert out.shape == coord.shape
    # radius in the xy-plane is preserved by rotation and flip up to the scale and the clipped jitter
    r0, r1 = np.linalg.norm(coord[:, :2], axis=1), np.linalg.norm(out[:, :2], axis=1)
    s = np.median(r1 / np.maximum(r0, 1e-6))
    assert 0.9 <= s <= 1.1
    assert np.all(np.abs(r1 - s * r0) <= 0.02 * np.sqrt(2) + 1e-5)


# --- collate ------------------------------------------------------------------------------------------------------------

def test_collate_offsets_and_inverse():
    samples = []
    for k in range(3):
        coord = _cloud(300 + 100 * k, seed=k)
        idx, grid, inv = grid_sample(coord, 0.5, np.random.default_rng(k))
        seg = np.full(len(coord), k, dtype=np.int64)
        samples.append({'grid_coord': grid, 'feat': np.ones((len(idx), 4), np.float32), 'segment': seg[idx],
                        'frame_key': str(k), 'inverse': inv, 'origin_segment': seg, 'origin_coord': coord})
    b = collate(samples)
    assert b['offset'].tolist() == np.cumsum([len(s['segment']) for s in samples]).tolist()
    # mapping every original point through the shifted inverse lands on a voxel of its own frame
    assert torch.equal(b['segment'][b['inverse']], b['origin_segment'])


# --- loss and metric against the sentinel -------------------------------------------------------------------------------

def test_ce_loss_ignores_sentinel():
    torch.manual_seed(0)
    logits = torch.randn(6, 4, requires_grad=True)
    tgt = torch.tensor([0, 1, IGNORE, 3, IGNORE, 2])
    ref = torch.nn.functional.cross_entropy(logits[tgt >= 0], tgt[tgt >= 0])
    assert torch.allclose(ce_loss(logits, tgt), ref)


def test_ce_loss_all_ignored_is_finite_zero_gradient():
    logits = torch.randn(5, 4, requires_grad=True)
    loss = ce_loss(logits, torch.full((5,), IGNORE))
    assert torch.isfinite(loss) and float(loss) == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_confusion_matrix_never_counts_sentinel():
    cm = ConfusionMatrix(3)
    cm.update(pred=np.array([2, 2, 0, 1]), target=np.array([IGNORE, IGNORE, 0, 1]))
    assert cm.mat.sum() == 2 and cm.mat[2].sum() == 0 and cm.mat[:, 2].sum() == 0
    assert cm.mat[0, 0] == 1 and cm.mat[1, 1] == 1


def test_confusion_matrix_rejects_bad_ids():
    cm = ConfusionMatrix(3)
    with pytest.raises(ValueError):
        cm.update(pred=np.array([0]), target=np.array([3]))
    with pytest.raises(ValueError):
        cm.update(pred=np.array([-1]), target=np.array([0]))


def test_miou_classes_without_gt():
    cm = ConfusionMatrix(3)
    cm.update(pred=np.array([0, 0, 1, 2]), target=np.array([0, 0, 1, 1]))
    iou = cm.iou()
    assert iou[0] == 1.0 and iou[1] == 0.5 and iou[2] == 0.0
    assert cm.miou(require_gt=True) == pytest.approx(0.75)           # class 2 has no ground truth: left out
    assert cm.miou(require_gt=False) == pytest.approx(0.5)


# --- SpUNet port against Pointcept's source ---------------------------------------------------------------------------

def _load_pointcept_spunet():
    """Exec Pointcept's original file with its registry, timm and torch_geometric imports stubbed (none of them is used
    by the decoder path), and return its SpUNetBase class."""
    import types
    import sys
    src = open(POINTCEPT_SPUNET).read()
    stubs = {
        'torch_geometric': types.ModuleType('torch_geometric'), 'torch_geometric.utils': types.ModuleType('u'),
        'timm': types.ModuleType('timm'), 'timm.layers': types.ModuleType('l'),
        'pointcept': types.ModuleType('pointcept'), 'pointcept.models': types.ModuleType('m'),
        'pointcept.models.builder': types.ModuleType('b'), 'pointcept.models.utils': types.ModuleType('mu'),
    }
    stubs['torch_geometric.utils'].scatter = None
    stubs['timm.layers'].trunc_normal_ = torch.nn.init.trunc_normal_

    class _Reg:
        def register_module(self, *a, **k):
            return lambda c: c
    stubs['pointcept.models.builder'].MODELS = _Reg()
    from pcseg.spunet import offset2batch
    stubs['pointcept.models.utils'].offset2batch = offset2batch
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        ns = {'__name__': 'pointcept_spunet_original'}
        exec(compile(src, POINTCEPT_SPUNET, 'exec'), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return ns['SpUNetBase']


@pytest.mark.skipif(not os.path.exists(POINTCEPT_SPUNET), reason='Pointcept clone absent')
def test_spunet_port_matches_pointcept():
    from pcseg.spunet import SpUNet
    Orig = _load_pointcept_spunet()
    kw = dict(in_channels=4, num_classes=16, channels=(32, 64, 128, 256, 256, 128, 96, 96),
              layers=(2, 3, 4, 6, 2, 2, 2, 2))
    ours, orig = SpUNet(**kw), Orig(**kw)
    so, sr = ours.state_dict(), orig.state_dict()
    assert list(so) == list(sr)
    assert all(so[k].shape == sr[k].shape for k in so)
    n_params = sum(p.numel() for p in ours.parameters())
    assert n_params == sum(p.numel() for p in orig.parameters())
    ours.load_state_dict(sr)

    if not torch.cuda.is_available():
        pytest.skip('forward comparison needs CUDA (spconv); state-dict equality checked (%d parameters)' % n_params)
    dev = torch.device('cuda')
    ours, orig = ours.to(dev).eval(), orig.to(dev).eval()
    rng = np.random.default_rng(0)
    coords = []
    for _ in range(2):
        c = _cloud(4000, seed=int(rng.integers(1 << 30))) * 5
        idx, grid, _ = grid_sample(c, 0.05, rng)
        coords.append((grid, np.concatenate([c[idx], rng.random((len(idx), 1), dtype=np.float32)], 1)))
    batch = {'grid_coord': torch.as_tensor(np.concatenate([g for g, _ in coords]), dtype=torch.int32, device=dev),
             'feat': torch.as_tensor(np.concatenate([f for _, f in coords]), device=dev),
             'offset': torch.as_tensor(np.cumsum([len(g) for g, _ in coords]), device=dev)}
    with torch.no_grad():
        a, b = ours(batch), orig(batch)
    assert a.shape == (len(batch['feat']), 16)
    assert torch.allclose(a, b, atol=1e-5)


def _spunet_eval_case(dev, n_frames=3):
    """The P1 SpUNet (17 classes) and an eval loader of synthetic frames in PandasetSeg's eval-sample format."""
    from pcseg.spunet import SpUNet
    nc = 17
    torch.manual_seed(0)
    model = SpUNet(in_channels=4, num_classes=nc, channels=(32, 64, 128, 256, 256, 128, 96, 96),
                   layers=(2, 3, 4, 6, 2, 2, 2, 2)).to(dev)
    rng = np.random.default_rng(3)
    samples = []
    for i in range(n_frames):
        c = _cloud(20000, seed=10 + i) * 10
        idx, grid, inv = grid_sample(c, 0.05, rng)
        seg = rng.integers(0, nc, size=len(c))
        seg[rng.random(len(c)) < 0.05] = IGNORE
        feat = np.concatenate([c[idx], rng.random((len(idx), 1), dtype=np.float32)], 1)
        samples.append({'grid_coord': grid, 'feat': feat, 'segment': seg[idx], 'frame_key': 'g%d' % i,
                        'inverse': inv, 'origin_segment': seg, 'origin_coord': c})
    loader = torch.utils.data.DataLoader(samples, batch_size=2, collate_fn=collate, shuffle=False)
    return model, loader, ['c%d' % k for k in range(nc)], samples


def test_evaluate_runs_spunet_on_gpu():
    """Smoke 33127 trained an epoch, then crashed in its first validation pass: evaluate() ran the SpUNet under autocast,
    and spconv's eval path cannot (engine.evaluate docstring; probe 33180). evaluate() at its defaults must run the real
    SpUNet on CUDA and put every non-ignored point and voxel in the matrices exactly once."""
    from pcseg import engine
    if not torch.cuda.is_available():
        pytest.skip('needs CUDA (spconv)')
    dev = torch.device('cuda')
    model, loader, names, samples = _spunet_eval_case(dev)
    metrics, mats = engine.evaluate(model, loader, names, dev, cone=(60.0, 0.0))
    assert metrics['val/frames'] == len(samples)
    assert mats['point'].sum() == sum(int((s['origin_segment'] >= 0).sum()) for s in samples)
    assert mats['voxel'].sum() == sum(int((s['segment'] >= 0).sum()) for s in samples)
    assert 0 < mats['cone'].sum() < mats['point'].sum()


def test_gpu_eval_case_reproduces_the_autocast_crash():
    """Keeps test_evaluate_runs_spunet_on_gpu discriminating: the same case under autocast (what evaluate() did before
    the fix) must still crash. If a new spconv build makes this pass, the hazard is gone; say so in 20261116_01 before
    allowing autocast in evaluate()."""
    from pcseg import engine
    if not torch.cuda.is_available():
        pytest.skip('needs CUDA (spconv)')
    dev = torch.device('cuda')
    model, loader, names, _ = _spunet_eval_case(dev, n_frames=1)
    with pytest.raises(RuntimeError, match='suitable algorithm'):
        with torch.autocast('cuda'):
            engine.evaluate(model, loader, names, dev)
