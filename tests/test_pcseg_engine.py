"""pcseg S1 (experiments_md episodes/20261116_01 §8): the training / evaluation loop and the entry point's config
handling, on CPU with stand-in models.

- evaluate(): per-point predictions come from each point's OWN frame's voxels through the shifted inverse map, for a
  batch of several frames, against a per-frame brute force; IGNORE never lands in the matrix; the cone matrix uses
  ST3D's cone convention (DatasetTemplate.extract_fov_gt). It never enters autocast (smoke 33127: spconv's eval
  path crashes the SpUNet under it; the GPU check is test_pcseg_core.py::test_evaluate_runs_spunet_on_gpu).
- ShardSampler: every eval frame exactly once across ranks.
- DataLoader workers draw distinct, torch-seeded numpy streams (torch reseeds numpy per worker; pcseg relies on it).
- OneCycle: Pointcept's lr shape (start lr/10, peak at 4%, end lr/1000).
- target_dump: raises on a voxel label its points do not carry.
- checkpoint round trip; non-finite loss raises.
- configs: _BASE_CONFIG_ resolves, the splits equal the detection split, the semseg counts are as expected.
"""
import os
import sys

import numpy as np
import pytest
import torch

from pcseg import IGNORE
from pcseg import engine
from pcseg.grid_sample import grid_sample
from pcseg.losses_metrics import ConfusionMatrix
from pcseg.pandaset_seg import collate

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
NC = 5


class _EvalSet(torch.utils.data.Dataset):
    """Synthetic eval samples shaped like PandasetSeg's (frame-dependent sizes, some IGNORE points)."""

    def __init__(self, n=7, grid=0.4):
        self.n, self.grid = n, grid

    def __len__(self):
        return self.n

    def load(self, i):
        rng = np.random.default_rng(100 + i)
        coord = rng.uniform(-6, 6, size=(300 + 50 * i, 3)).astype(np.float32)
        seg = (np.floor(coord[:, 0] / 1.5).astype(np.int64) % NC)
        seg[rng.random(len(seg)) < 0.1] = IGNORE
        return coord, np.ones((len(coord), 1), np.float32), seg, 'f%d' % i

    def __getitem__(self, i):
        coord, strength, seg, key = self.load(i)
        idx, grid, inv = grid_sample(coord, self.grid, np.random.default_rng(i))
        return {'grid_coord': grid, 'feat': np.concatenate([coord[idx], strength[idx]], 1), 'segment': seg[idx],
                'frame_key': key, 'inverse': inv, 'origin_segment': seg, 'origin_coord': coord}


class _HashModel(torch.nn.Module):
    """Logits that are a fixed function of each voxel's features, so the same voxel gets the same prediction whether
    it is scored alone or inside a batch."""

    def forward(self, batch):
        f = batch['feat']
        z = torch.stack([torch.sin(f[:, 0] * (k + 1) + f[:, 1] * (k + 2)) for k in range(NC)], 1)
        return z


def _brute(ds, model, cone=None):
    mp, mv, mc = (ConfusionMatrix(NC) for _ in range(3))
    for i in range(len(ds)):
        s = ds[i]
        logits = model({'feat': torch.as_tensor(s['feat'])})
        pv = logits.argmax(1).numpy()
        pp = pv[s['inverse']]
        mv.update(pv, s['segment'])
        mp.update(pp, s['origin_segment'])
        if cone is not None:
            m = engine.cone_mask(s['origin_coord'][:, :2], *cone)
            mc.update(pp[m], s['origin_segment'][m])
    return mp.mat, mv.mat, mc.mat


def test_evaluate_matches_per_frame_brute_force():
    ds = _EvalSet()
    loader = torch.utils.data.DataLoader(ds, batch_size=3, collate_fn=collate, shuffle=False)
    names = ['c%d' % k for k in range(NC)]
    metrics, mats = engine.evaluate(_HashModel(), loader, names, torch.device('cpu'), cone=(60.0, 0.0))
    bp, bv, bc = _brute(ds, _HashModel(), cone=(60.0, 0.0))
    assert np.array_equal(mats['point'], bp)
    assert np.array_equal(mats['voxel'], bv)
    assert np.array_equal(mats['cone'], bc)
    n_valid = sum(int((ds.load(i)[2] >= 0).sum()) for i in range(len(ds)))
    assert mats['point'].sum() == n_valid                        # every non-ignored point once, no ignored point
    assert metrics['val/frames'] == len(ds)
    assert 0 < mats['cone'].sum() < mats['point'].sum()


def test_evaluate_shards_sum_to_whole():
    ds = _EvalSet()
    names = ['c%d' % k for k in range(NC)]
    total = None
    for r in range(3):
        loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate,
                                             sampler=engine.ShardSampler(len(ds), r, 3))
        _, mats = engine.evaluate(_HashModel(), loader, names, torch.device('cpu'))
        total = mats['point'] if total is None else total + mats['point']
    assert np.array_equal(total, _brute(ds, _HashModel())[0])


def test_evaluate_never_enters_autocast(monkeypatch):
    """Smoke 33127 died in its first validation pass: evaluate() ran the SpUNet under autocast, which spconv's eval
    path cannot do (engine.evaluate docstring). On CPU, autocast('cuda') is a disabled no-op, so record the requests
    instead: evaluate() must never ask for an enabled autocast, and the model must see grad mode off."""
    requests, grad_seen = [], []
    real = torch.autocast

    def recording_autocast(*a, **k):
        requests.append(k.get('enabled', True))
        return real(*a, **k)

    class _Probe(_HashModel):
        def forward(self, batch):
            grad_seen.append(torch.is_grad_enabled())
            return super().forward(batch)

    monkeypatch.setattr(torch, 'autocast', recording_autocast)
    loader = torch.utils.data.DataLoader(_EvalSet(n=3), batch_size=2, collate_fn=collate, shuffle=False)
    engine.evaluate(_Probe(), loader, ['c%d' % k for k in range(NC)], torch.device('cpu'))
    assert not any(requests), requests
    assert grad_seen and not any(grad_seen)


def test_shard_sampler_partitions():
    for n in (1, 7, 1200):
        for w in (1, 2, 3):
            seen = sorted(i for r in range(w) for i in engine.ShardSampler(n, r, w))
            assert seen == list(range(n))


def test_cone_mask_matches_st3d_box_convention():
    from pcdet.datasets.dataset import DatasetTemplate
    rng = np.random.default_rng(0)
    xy = rng.uniform(-50, 50, size=(5000, 2))
    boxes = np.concatenate([xy, np.zeros((len(xy), 5))], 1)
    for fov, heading in ((60.0, 0.0), (60.0, 90.0), (120.0, -45.0), (30.0, 180.0)):
        ref = DatasetTemplate.extract_fov_gt(boxes.copy(), fov, heading)
        assert np.array_equal(engine.cone_mask(xy, fov, heading), ref), (fov, heading)


class _RandSet(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return np.random.randint(0, 2 ** 31 - 1)


@pytest.mark.parametrize('ctx', ['fork', 'spawn'])
def test_dataloader_workers_draw_distinct_numpy_streams(ctx):
    """PandasetSeg draws each training sample's augmentation seed from numpy's GLOBAL RNG and relies on torch (>= 1.9)
    reseeding numpy in every DataLoader worker from base_seed + worker_id. Guards that assumption: forked workers
    would otherwise share one numpy state and repeat each other's augmentation."""
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(_RandSet(), batch_size=1, num_workers=2, multiprocessing_context=ctx)
    a = [int(x) for x in loader]
    assert len(set(a)) == len(a)
    torch.manual_seed(0)
    assert [int(x) for x in loader] == a                          # reproducible from the torch seed
    assert [int(x) for x in loader] != a                          # and redrawn every epoch


def test_onecycle_shape_matches_pointcept():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=0.002, weight_decay=0.005)
    sch = engine.build_scheduler(opt, {'LR': 0.002}, total_steps=1000)
    lrs = []
    for _ in range(1000):
        lrs.append(opt.param_groups[0]['lr'])
        opt.step()
        sch.step()
    assert lrs[0] == pytest.approx(0.0002)
    assert int(np.argmax(lrs)) == pytest.approx(40, abs=1)
    assert max(lrs) == pytest.approx(0.002)
    assert lrs[-1] == pytest.approx(0.002 / 10 / 100, rel=0.05)


class _BadDumpSet(_EvalSet):
    """Points carry classes 0-2 only; one voxel claims class NC - 1, as an index bug moving labels would."""

    def load(self, i):
        coord, strength, seg, key = super().load(i)
        return coord, strength, np.where(seg >= 0, seg % 3, seg), key

    def __getitem__(self, i):
        s = super().__getitem__(i)
        s['segment'] = s['segment'].copy()
        s['segment'][0] = NC - 1
        return s


def test_target_dump_reports_and_raises():
    names = ['c%d' % k for k in range(NC)]
    dump = engine.target_dump({'val': _EvalSet()}, names, n_frames=3)
    rows = dump['val']
    assert len(rows) == 3 and all(0.8 < r['inverse_agreement'] <= 1.0 for r in rows)
    assert all(sum(r['voxel_counts'].values()) + r['ignored_voxels'] == r['voxels'] for r in rows)
    with pytest.raises(ValueError, match='not among point labels'):
        engine.target_dump({'val': _BadDumpSet(n=2)}, names, n_frames=2)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=0.002)
    sch = engine.build_scheduler(opt, {'LR': 0.002}, total_steps=100)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    for _ in range(7):
        opt.zero_grad()
        m(torch.randn(2, 4)).sum().backward()
        opt.step()
        sch.step()
    engine.save_checkpoint(tmp_path / 'checkpoint_epoch_3.pth', m, opt, sch, scaler, epoch=2)
    m2 = torch.nn.Linear(4, 3)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=0.002)
    sch2 = engine.build_scheduler(opt2, {'LR': 0.002}, total_steps=100)
    ep = engine.load_checkpoint(tmp_path / 'checkpoint_epoch_3.pth', m2, opt2, sch2, scaler)
    assert ep == 2
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    assert opt2.param_groups[0]['lr'] == opt.param_groups[0]['lr'] and sch2.last_epoch == sch.last_epoch
    for k in range(1, 6):
        engine.save_checkpoint(tmp_path / ('checkpoint_epoch_%d.pth' % (k + 3)), m, opt, sch, scaler, epoch=k)
    engine.prune_checkpoints(tmp_path, keep=2)
    assert sorted(p.name for p in tmp_path.glob('*.pth')) == ['checkpoint_epoch_7.pth', 'checkpoint_epoch_8.pth']


class _TrainSet(_EvalSet):
    def __getitem__(self, i):
        s = super().__getitem__(i)
        return {k: s[k] for k in ('grid_coord', 'feat', 'segment', 'frame_key')}


class _LinearModel(torch.nn.Module):
    def __init__(self, nan=False):
        super().__init__()
        self.lin = torch.nn.Linear(4, NC)
        self.nan = nan

    def forward(self, batch):
        out = self.lin(batch['feat'])
        return out * float('nan') if self.nan else out


@pytest.mark.parametrize('nan', [False, True])
def test_train_one_epoch_cpu(nan):
    torch.manual_seed(0)
    model = _LinearModel(nan)
    loader = torch.utils.data.DataLoader(_TrainSet(n=6), batch_size=2, collate_fn=collate, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    sch = engine.build_scheduler(opt, {'LR': 0.01}, total_steps=len(loader))
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    if nan:
        with pytest.raises(FloatingPointError):
            engine.train_one_epoch(model, loader, opt, sch, scaler, 0, torch.device('cpu'), amp=False, log_every=0)
        return
    st = engine.train_one_epoch(model, loader, opt, sch, scaler, 0, torch.device('cpu'), amp=False, log_every=0)
    assert st['iters'] == 3 and st['frames'] == 6 and np.isfinite(st['loss'])


# --- configs ----------------------------------------------------------------------------------------------------------

def _load(name):
    sys.path.insert(0, TOOLS)
    import train_seg
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        return train_seg.load_cfg('cfgs/pcseg/%s.yaml' % name)
    finally:
        os.chdir(cwd)


def test_pcseg_configs_resolve_and_use_the_detection_split():
    import yaml
    det = yaml.safe_load(open(os.path.join(TOOLS, 'cfgs/dataset_configs/pandaset_dataset.yaml')))['SEQUENCES']
    base = _load('spunet-pandaset64')
    assert list(base.DATA.SPLITS.train) == det['train'] and list(base.DATA.SPLITS.val) == det['val']
    smoke = _load('spunet-pandaset64-smoke')
    assert smoke.MODEL == base.MODEL and smoke.DATA.AUG == base.DATA.AUG and smoke.EVAL.CONE == base.EVAL.CONE
    assert smoke.OPTIMIZATION.LR == base.OPTIMIZATION.LR and smoke.OPTIMIZATION.NUM_EPOCHS == 2
    assert set(smoke.DATA.SPLITS.train) <= set(det['train']) and set(smoke.DATA.SPLITS.val) <= set(det['val'])


@pytest.mark.skipif(not os.path.isdir('/home/koyama/data/pandaset/dataset'), reason='PandaSet absent')
def test_expected_semseg_counts_hold():
    from pcseg.pandaset_seg import has_semseg
    for name in ('spunet-pandaset64', 'spunet-pandaset64-smoke'):
        c = _load(name)
        for split, n in c.DATA.EXPECT_SEQUENCES.items():
            assert sum(has_semseg(c.DATA.ROOT, s) for s in c.DATA.SPLITS[split]) == n, (name, split)
