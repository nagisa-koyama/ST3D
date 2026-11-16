"""pcseg/tta.py, Pointcept's test protocol (experiments_md episodes/20261116_01 §5, the G0 pass rule), on CPU with
stand-in models; the SpUNet case runs on CUDA only.

- grid_parts (GridSample 'test' mode): every part holds exactly one point of every voxel, the parts cover every point,
  there are max-count parts, and grid_coord is each point's own voxel.
- predict_frame: the index chain (0.025 subsample -> views -> parts -> inverse) gives every original point the prediction
  of its 0.025 representative; a flipped view negates x and y; a scaled view scales; ten views by default.
- evaluate_tta: equals a per-frame loop over predict_frame; ignored points never land in the matrix; shards sum to the
  whole.
- tools/test_seg.py merge: sums a complete shard set and refuses an incomplete one.
"""
import numpy as np
import pytest
import torch

from pcseg import IGNORE
from pcseg import tta
from pcseg.losses_metrics import ConfusionMatrix
from pcseg.pandaset_seg import key_seed

NC = 4


def _cloud(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-4, 4, size=(n, 3)).astype(np.float32)


def test_grid_parts_cover_every_point_once_per_voxel():
    c = _cloud(3000, 0)
    parts = tta.grid_parts(c, 0.5)
    vox = np.floor(c / 0.5).astype(np.int64)
    vox -= vox.min(0)
    _, vid, count = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    vid = vid.reshape(-1)
    assert len(parts) == count.max()
    covered = np.zeros(len(c), dtype=bool)
    for idx, g in parts:
        assert len(idx) == len(count)                                   # one point per occupied voxel
        assert sorted(vid[idx].tolist()) == list(range(len(count)))     # ... and every voxel once
        assert np.array_equal(g, vox[idx])                              # grid_coord = the point's own voxel
        covered[idx] = True
    assert covered.all()


class _StrengthModel(torch.nn.Module):
    """Class = floor(strength * NC); ignores the coordinates, so every view agrees."""

    def forward(self, b):
        k = torch.clamp((b['feat'][:, 3] * NC).long(), 0, NC - 1)
        return torch.nn.functional.one_hot(k, NC).float() * 10


class _SignXModel(torch.nn.Module):
    axis = 0

    def forward(self, b):
        x = b['feat'][:, self.axis]
        return torch.stack([(x > 0).float(), (x <= 0).float()], 1) * 10


class _SignYModel(_SignXModel):
    axis = 1


class _AbsXModel(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, b):
        x = b['feat'][:, 0].abs()
        return torch.stack([(x > self.t).float(), (x <= self.t).float()], 1) * 10


def test_predict_frame_index_chain():
    c = _cloud(4000, 1)
    s = np.random.default_rng(2).random((len(c), 1)).astype(np.float32)
    rng_seed = 7
    pred, n_fwd = tta.predict_frame(_StrengthModel(), c, s, NC, torch.device('cpu'), np.random.default_rng(rng_seed))
    idx_s, _, inv_s = tta.grid_sample(c, tta.FINE_GRID, np.random.default_rng(rng_seed))
    want = np.clip((s[idx_s][inv_s, 0] * NC).astype(np.int64), 0, NC - 1)
    assert np.array_equal(pred, want)
    assert n_fwd == sum(len(tta.grid_parts(tta.view(c[idx_s], sc, fl), 0.05)) for sc, fl in tta.TTA_VIEWS)
    assert len(tta.TTA_VIEWS) == 10 and sum(fl for _, fl in tta.TTA_VIEWS) == 5


def test_predict_frame_views_flip_and_scale():
    c = _cloud(2000, 3)
    s = np.zeros((len(c), 1), np.float32)
    seed = 5
    idx_s, _, inv_s = tta.grid_sample(c, tta.FINE_GRID, np.random.default_rng(seed))
    x = c[idx_s][inv_s, 0]
    p, _ = tta.predict_frame(_SignXModel(), c, s, 2, 'cpu', np.random.default_rng(seed), views=[(1.0, True)])
    assert np.array_equal(p, (x >= 0).astype(np.int64))                 # the model sees -x: class 0 where x < 0
    y = c[idx_s][inv_s, 1]
    p, _ = tta.predict_frame(_SignYModel(), c, s, 2, 'cpu', np.random.default_rng(seed), views=[(1.0, True)])
    assert np.array_equal(p, (y >= 0).astype(np.int64))                 # ... and -y (RandomFlip p=1 flips both)
    p, _ = tta.predict_frame(_SignXModel(), c, s, 2, 'cpu', np.random.default_rng(seed), views=[(1.0, False)])
    assert np.array_equal(p, (x <= 0).astype(np.int64))
    p, _ = tta.predict_frame(_AbsXModel(2.0), c, s, 2, 'cpu', np.random.default_rng(seed), views=[(1.1, False)])
    assert np.array_equal(p, (np.abs(x) * np.float32(1.1) <= 2.0).astype(np.int64))


class _Frames:
    def __init__(self, n=5):
        self.n = n

    def __len__(self):
        return self.n

    def load(self, i):
        c = _cloud(1500 + 200 * i, 10 + i)
        rng = np.random.default_rng(20 + i)
        s = rng.random((len(c), 1)).astype(np.float32)
        seg = np.clip((s[:, 0] * NC).astype(np.int64), 0, NC - 1)
        seg[rng.random(len(c)) < 0.1] = IGNORE
        flip = rng.random(len(c)) < 0.2                                  # some labels the strength model gets wrong
        seg[flip & (seg >= 0)] = (seg[flip & (seg >= 0)] + 1) % NC
        return c, s, seg, 'f%d' % i


def _brute(ds, model, views):
    cm = ConfusionMatrix(NC)
    for i in range(len(ds)):
        c, s, seg, key = ds.load(i)
        p, _ = tta.predict_frame(model, c, s, NC, 'cpu', np.random.default_rng(key_seed(key, 'tta')), views=views)
        cm.update(p, seg)
    return cm.mat


def test_evaluate_tta_matches_brute_force_and_shards():
    ds = _Frames()
    views = tta.TTA_VIEWS[:3]
    mats, frames, fwd = tta.evaluate_tta(_StrengthModel(), ds, ['c%d' % k for k in range(NC)], 'cpu', views=views,
                                         cone=(60.0, 0.0))
    assert frames == len(ds) and fwd > 0
    assert np.array_equal(mats['point'], _brute(ds, _StrengthModel(), views))
    n_valid = sum(int((ds.load(i)[2] >= 0).sum()) for i in range(len(ds)))
    assert mats['point'].sum() == n_valid
    assert 0 < mats['cone'].sum() < mats['point'].sum()
    total = sum(tta.evaluate_tta(_StrengthModel(), ds, ['c%d' % k for k in range(NC)], 'cpu', views=views,
                                 indices=range(r, len(ds), 2))[0]['point'] for r in range(2))
    assert np.array_equal(total, mats['point'])


def test_test_seg_merge_sums_complete_shard_sets(tmp_path):
    import json
    import os
    import sys
    tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
    sys.path.insert(0, tools)
    import test_seg
    names = ['c%d' % k for k in range(NC)]
    rng = np.random.default_rng(0)
    mats = [rng.integers(0, 50, size=(NC, NC)) for _ in range(3)]
    for r in range(3):
        s = {'ckpt': '/x/ck.pth', 'epoch': 49, 'shard': r, 'num_shards': 3, 'commit': 'abc',
             'tta': {'frames': 2, 'forwards': 7, 'matrices': {'point': mats[r].tolist()}}}
        json.dump(s, open(tmp_path / ('shard_%03d_of_%03d.json' % (r, 3)), 'w'))
    out = test_seg.merge(None, names, tmp_path)
    assert np.array_equal(np.asarray(out['tta']['matrices']['point']), sum(mats))
    assert out['tta']['frames'] == 6 and out['tta']['forwards'] == 21 and 'plain' not in out
    cm = ConfusionMatrix(NC)
    cm.mat = sum(mats)
    assert out['tta']['val_point/miou'] == pytest.approx(cm.miou())
    os.remove(tmp_path / 'shard_001_of_003.json')
    with pytest.raises(ValueError, match='incomplete'):
        test_seg.merge(None, names, tmp_path)


def test_evaluate_tta_runs_spunet_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip('needs CUDA (spconv)')
    from pcseg.spunet import SpUNet
    dev = torch.device('cuda')
    torch.manual_seed(0)
    model = SpUNet(in_channels=4, num_classes=NC, channels=(32, 64, 128, 256, 256, 128, 96, 96),
                   layers=(2, 3, 4, 6, 2, 2, 2, 2)).to(dev)
    ds = _Frames(n=2)
    mats, frames, fwd = tta.evaluate_tta(model, ds, ['c%d' % k for k in range(NC)], dev, views=tta.TTA_VIEWS[4:6])
    assert frames == 2 and fwd >= 4
    assert mats['point'].sum() == sum(int((ds.load(i)[2] >= 0).sum()) for i in range(2))
