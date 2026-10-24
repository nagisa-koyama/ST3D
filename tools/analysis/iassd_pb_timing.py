"""Rule PB cost fallback, timing smoke (experiments_md 20261024_05; T05 §5 "Rule PB, FIXED 15:53 JST", item 5): the
training-step time of a PB budget against its 16,384 parent's, on the SAME frames, the SAME GPU and in ONE process.

Rule PB item 5: a 1-GPU timing smoke (50 iterations) per new budget; a block whose iteration is > 3x its 16,384 parent's
gets the authors' voxel step before the budget. The two arms of a block differ in their processor chain FROM the budget
step on (the budget value; for the N oracle also the recount), so the frames are captured once, at the input of the
budget step, and each arm runs its own tail of the chain on copies of them. What is timed is the training step (forward,
backward, gradient clip, optimiser step, CUDA-synchronised), cycling over the captured batches. The loader's work before
the budget step is identical for both arms by construction and is not timed here (the parents' logs give it).

Two sub-commands, so the GPU allocation only does GPU work:

    # CPU: capture N frames entering the budget step of one block (training block: training mode, augmentation on, the
    # density calibration installed as train.py does with --calib_frames frames; eval block: evaluation mode, val split)
    python analysis/iassd_pb_timing.py capture --cfg_file <PB cfg> --block <key> --frames 12 --out <pkl>
    # GPU: time each arm on the captured frames (arm = a config whose block supplies the tail from its budget step on)
    python analysis/iassd_pb_timing.py time --capture <pkl> --model_cfg <PB cfg> --arm <cfg>[:<block>] ... \
        --batch_size 6 --warmup 10 --iters 50 [--eval]

`time` prints one RESULT line per arm: seconds per iteration (median and mean over the timed iterations), peak GPU
memory, the loss (training) and the points per frame the arm feeds the network; the ratio of each arm to the first.
"""
import argparse
import copy
import os
import pickle
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET_STEPS = ('sample_points', 'sample_points_by_voxels')


def _setup():
    os.chdir(TOOLS)
    sys.path.insert(0, TOOLS)
    import _init_path  # noqa: F401


def _block_cfg(cfg, block):
    if block == 'DATA_CONFIG_TAR':
        return cfg.DATA_CONFIG_TAR
    if block == 'DATA_CONFIG':
        return cfg.DATA_CONFIG
    return cfg.DATA_CONFIGS[block]


def budget_index(names_and_np, mode):
    """Index of the first fixed-count budget step in a resolved DATA_PROCESSOR list."""
    for i, p in enumerate(names_and_np):
        if p['NAME'] in BUDGET_STEPS and p['NUM_POINTS'][mode] != -1:
            return i
    raise RuntimeError('no fixed-count budget step')


def capture(a):
    _setup()
    import logging
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from analysis.iassd_pb_budget_census import build_block, budget_step_index, strided_indices
    logger = logging.getLogger('pb_timing'); logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.WARNING)
    cfg = cfg_from_yaml_file(a.cfg_file, EasyDict())
    np.random.seed(a.seed)
    split = 'val' if a.block == 'DATA_CONFIG_TAR' else 'train'
    ds, blk, how = build_block(cfg, a.block, split, logger, a.calib_frames)
    q = ds.data_processor.data_processor_queue
    at = budget_step_index(q, ds.data_processor.mode)
    rec = []

    def grab(data_dict=None, _f=q[at]):
        rec.append(copy.deepcopy(data_dict))
        return _f(data_dict=data_dict)
    q[at] = grab
    idx = strided_indices(len(ds), max(a.frames * 50, 1))[::50][:a.frames]
    for i in idx:
        ds[i]
    meta = dict(cfg_file=a.cfg_file, block=a.block, split=split, built=how, frames=idx, mode=ds.data_processor.mode,
                training=ds.training, num_point_features=ds.point_feature_encoder.num_point_features,
                grid_size=ds.grid_size, voxel_size=ds.voxel_size, point_cloud_range=ds.point_cloud_range,
                class_names=list(cfg.CLASS_NAMES), n_in=[len(r['points']) for r in rec])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'wb') as f:
        pickle.dump(dict(meta=meta, dicts=rec), f)
    print('captured %d frames of %s [%s, %s] -> %s; points entering the budget step %s' % (
        len(rec), os.path.basename(a.cfg_file), a.block, split, a.out, meta['n_in']))


def _tail(cfg_file, block, meta):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.processor.data_processor import DataProcessor
    cfg = cfg_from_yaml_file(cfg_file, EasyDict())
    blk = _block_cfg(cfg, block)
    procs = blk.DATA_PROCESSOR
    at = budget_index(procs, meta['mode'])
    proc = DataProcessor(procs, point_cloud_range=np.asarray(blk.POINT_CLOUD_RANGE, dtype=np.float32),
                         training=meta['training'], num_point_features=meta['num_point_features'])
    budget = procs[at]['NUM_POINTS'][meta['mode']]
    return proc.data_processor_queue[at:], [p['NAME'] for p in procs[at:]], budget, cfg


def time_arms(a):
    _setup()
    import types
    import torch
    if not getattr(a, 'dry', False):
        assert torch.cuda.is_available(), 'time needs a GPU (use --dry on CPU)'
    from easydict import EasyDict
    from torch.nn.utils import clip_grad_norm_
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.dataset import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu, model_fn_decorator
    from train_utils.optimization import build_optimizer
    with open(a.capture, 'rb') as f:
        cap = pickle.load(f)
    meta, dicts = cap['meta'], cap['dicts']
    mcfg = cfg_from_yaml_file(a.model_cfg, EasyDict())
    stub = types.SimpleNamespace(point_feature_encoder=types.SimpleNamespace(num_point_features=meta['num_point_features']),
                                 grid_size=meta['grid_size'], voxel_size=meta['voxel_size'],
                                 point_cloud_range=meta['point_cloud_range'], class_names=meta['class_names'],
                                 depth_downsample_factor=None)
    print('=== %s [%s, %s mode], %d captured frames, GPU %s, batch %d, warmup %d + %d timed %s ===' % (
        os.path.basename(meta['cfg_file']), meta['block'], meta['mode'], len(dicts), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
        a.batch_size, a.warmup, a.iters, 'inference' if a.eval else 'training steps'))
    results = []
    if a.dry:   # CPU check of the data path only: each arm's tail and the collate, no model
        for arm in a.arm:
            cfg_file, _, block = arm.partition(':')
            tail, names, budget, _ = _tail(cfg_file, block or meta['block'], meta)
            d = copy.deepcopy(dicts[0])
            for f in tail:
                d = f(data_dict=d)
            d.pop('num_points_in_gt', None)
            d2 = copy.deepcopy(dicts[-1])
            for f in tail:
                d2 = f(data_dict=d2)
            d2.pop('num_points_in_gt', None)
            batch = DatasetTemplate.collate_batch([d, d2])   # two DIFFERENT frames, so ragged keys show here
            for k, v in batch.items():
                assert not (isinstance(v, np.ndarray) and v.dtype == object and k not in ('frame_id', 'gt_names', 'metadata', 'calib')), k
            print('DRY arm=%s budget=%s tail=%s points out %d, distinct %d, batch keys %s' % (
                os.path.basename(cfg_file), budget, '>'.join(names), len(d['points']),
                len(np.unique(d['points'][:, :3], axis=0)), sorted(batch.keys())))
        return
    for arm in a.arm:
        cfg_file, _, block = arm.partition(':')
        block = block or meta['block']
        tail, names, budget, acfg = _tail(cfg_file, block, meta)
        for k in ('MODEL',):
            assert acfg[k] == mcfg[k], 'arm %s has a different %s from --model_cfg' % (cfg_file, k)
        torch.manual_seed(0)
        np.random.seed(0)
        model = build_network(model_cfg=mcfg.MODEL, num_class=len(mcfg.CLASS_NAMES), dataset=stub).cuda()
        optimizer = None if a.eval else build_optimizer(model, mcfg.OPTIMIZATION)
        model_func = model_fn_decorator()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        nb = max(1, len(dicts) // a.batch_size)
        times, losses, pts = [], [], []
        t_arm = None
        for step in range(a.warmup + a.iters):
            if step == a.warmup:
                t_arm = time.perf_counter()
            elif t_arm is not None and a.max_seconds and time.perf_counter() - t_arm > a.max_seconds:
                print('  arm %s: stopped after %d timed steps (> %.0f s); the per-step time stands on those' % (
                    os.path.basename(cfg_file), len(times), a.max_seconds))
                break
            b = step % nb
            samples = []
            for d in dicts[b * a.batch_size:(b + 1) * a.batch_size]:
                d = copy.deepcopy(d)
                for f in tail:
                    d = f(data_dict=d)
                d.pop('num_points_in_gt', None)   # what WaymoDataset.__getitem__ drops after prepare_data
                samples.append(d)
            batch = DatasetTemplate.collate_batch(samples)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if a.eval:
                model.eval()
                with torch.no_grad():
                    load_data_to_gpu(batch)
                    model(batch)
            else:
                model.train()
                loss = model_func(model, batch)[0]
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), mcfg.OPTIMIZATION.GRAD_NORM_CLIP)
                optimizer.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if step >= a.warmup:
                times.append(dt)
                pts.append(len(batch['points']) / len(samples))
                if not a.eval:
                    losses.append(float(loss.detach()))
        t = np.array(times)
        res = dict(arm=os.path.basename(cfg_file), block=block, tail=names, budget=budget, n=len(t), s_med=float(np.median(t)),
                   s_mean=float(t.mean()), peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                   loss=float(np.mean(losses)) if losses else float('nan'), finite=bool(np.isfinite(losses).all()) if
                   losses else True, pts=float(np.mean(pts)))
        results.append(res)
        del model, optimizer
        torch.cuda.empty_cache()
    base = results[0]
    for r in results:
        print('RESULT %s arm=%s block=%s budget=%s tail=%s s_per_it_median=%.4f s_per_it_mean=%.4f ratio_median=%.2f '
              'ratio_mean=%.2f timed_steps=%d peak_gb=%.2f loss=%.3f finite=%s pts_per_frame=%.0f' % (
                  'eval' if a.eval else 'train', r['arm'], r['block'], r['budget'], '>'.join(r['tail']), r['s_med'],
                  r['s_mean'], r['s_med'] / base['s_med'], r['s_mean'] / base['s_mean'], r['n'], r['peak_gb'], r['loss'],
                  r['finite'], r['pts']))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--cfg_file', required=True)
    c.add_argument('--block', required=True)
    c.add_argument('--frames', type=int, default=12)
    c.add_argument('--calib_frames', type=int, default=None)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--out', required=True)
    t = sub.add_parser('time')
    t.add_argument('--capture', required=True)
    t.add_argument('--model_cfg', required=True)
    t.add_argument('--arm', action='append', required=True, help='<cfg>[:<block>]; the FIRST arm is the reference')
    t.add_argument('--batch_size', type=int, default=6)
    t.add_argument('--warmup', type=int, default=10)
    t.add_argument('--iters', type=int, default=50)
    t.add_argument('--eval', action='store_true')
    t.add_argument('--max_seconds', type=float, default=0,
                   help='stop an arm\'s timed steps after this many seconds (0 = always --iters); reported per arm')
    t.add_argument('--dry', action='store_true', help='CPU: run each arm\'s tail and the collate only, no model')
    a = ap.parse_args()
    capture(a) if a.cmd == 'capture' else time_arms(a)


if __name__ == '__main__':
    main()
