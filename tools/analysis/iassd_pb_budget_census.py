"""Rule PB census (experiments_md 20261024_05; T05 §5, "Rule PB, FIXED 15:53 JST"): the in-range point count that
ENTERS a fixed-budget IA-SSD config's budget step, per data block, and the budget the rule derives from it. CPU only.

Rule PB, item 2: each data block's NUM_POINTS = the smallest multiple of 4,096 >= the 99th percentile of the block's
in-range point count entering the budget step (after every processor before it, the row's operators and augmentation
included), on 1,000 strided frames of the block's TRAIN split; the eval block takes the target TRAIN split's value; val
never sets a value. This tool measures exactly that and nothing is tuned:

- A TRAINING block (`DATA_CONFIG` or one `DATA_CONFIGS` key) is built in training mode, augmentation on, as train.py
  builds it; a `HIST_DIST_ON_THE_FLY` block first gets its density calibration installed as train.py does
  (`calibration_target_config` of the target, `HIST_DIST_FRAMES`, `HIST_DIST_BEFORE_POINT_BUDGET` honoured).
- The EVAL block (`DATA_CONFIG_TAR`) is built in evaluation mode, as test.py builds it, on the target's TRAIN split
  (`calibration_target_config(..., 'train')`: DATA_SPLIT / INFO_PATH 'train' copied into 'test'). `--split val`
  describes the evaluated split instead; such a run is labelled DESCRIPTIVE and its budget field is not a PB value.
- Frames: `range(0, n, max(1, n // frames))[:frames]`, the stride `compute_range_histogram` and `emptied_boxes.py` use.
- The budget step is the first `sample_points` / `sample_points_by_voxels` whose NUM_POINTS for the mode is not -1. The
  hook records the cloud it receives (every processor before it has run), then runs the step unchanged. A budget step
  not preceded by `mask_points_and_boxes_outside_range` is refused: the count would not be an in-range count.

Recorded per frame: points entering the step (`n_in`), distinct xyz positions among them and their multiplicity
histogram, and the occupied voxels at the authors' voxel step (0.1 x 0.1 x 0.15 m over POINT_CLOUD_RANGE; rule PB item 5,
the cost fallback, applies P99 to that step's output). Derived (exact expectations, no second sampling pass):
- the real-point fraction a budget B keeps = E[distinct positions out] / n_in, where a frame above B is sampled WITHOUT
  replacement (a position held k times is missed with probability prod_{i<k} (n-B-i)/(n-i)) and a frame at or below B
  keeps every row and is padded with copies (`DataProcessor.sample_points`);
- copies per distinct point = B / E[distinct positions out];
- the share of frames above B (subsampled).

    python analysis/iassd_pb_budget_census.py --cfg_file <cfg> --block <DATA_CONFIG | DATA_CONFIGS key | DATA_CONFIG_TAR>
        [--split train|val] [--frames 1000] [--workers 8] [--out <json>]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET_STEPS = ('sample_points', 'sample_points_by_voxels')
PB_STEP = 4096                  # rule PB item 2: a multiple of 4,096
OLD_BUDGET = 16384              # the 16,384-budget protocol
VOXEL = (0.1, 0.1, 0.15)        # the authors' voxel step (iassd-sourceonly-waymo2waymo-vox65k.yaml)


# ------------------------------------------------------------------------------------------------ pure functions
def pb_budget(p99, step=PB_STEP):
    """The smallest multiple of `step` >= p99 (rule PB item 2)."""
    if not np.isfinite(p99) or p99 < 0:
        raise ValueError('p99 must be a finite non-negative count, got %r' % (p99,))
    return int(max(1, math.ceil(float(p99) / step)) * step)


def strided_indices(n, frames):
    """`range(0, n, max(1, n // frames))[:frames]`: the stride of compute_range_histogram / emptied_boxes.py."""
    return list(range(0, n, max(1, n // frames)))[:frames]


def multiplicity_histogram(xyz):
    """{k: number of distinct positions held by exactly k rows} for an (N, 3) array."""
    if len(xyz) == 0:
        return {}
    _, counts = np.unique(np.ascontiguousarray(xyz, dtype=np.float32), axis=0, return_counts=True)
    ks, hs = np.unique(counts, return_counts=True)
    return {int(k): int(h) for k, h in zip(ks, hs)}


def expected_distinct_out(n_in, mult_hist, budget):
    """E[distinct positions in the output of sample_points(budget)] for a cloud of n_in rows with `mult_hist`."""
    u_in = sum(mult_hist.values())
    if n_in <= budget:          # every row kept, the rest are copies of them
        return float(u_in)
    total = 0.0
    for k, h in mult_hist.items():
        miss = 1.0
        for i in range(int(k)):
            miss *= (n_in - budget - i) / (n_in - i)
        total += h * (1.0 - max(miss, 0.0))
    return total


def occupied_voxels(xyz, pc_range, voxel=VOXEL):
    """Occupied voxels of the authors' voxel step (cap not applied) for points inside pc_range."""
    if len(xyz) == 0:
        return 0
    lo = np.asarray(pc_range[:3], dtype=np.float64)
    hi = np.asarray(pc_range[3:6], dtype=np.float64)
    vs = np.asarray(voxel, dtype=np.float64)
    dims = np.round((hi - lo) / vs).astype(np.int64)
    c = np.floor((np.asarray(xyz[:, :3], dtype=np.float64) - lo) / vs).astype(np.int64)
    ok = ((c >= 0) & (c < dims)).all(axis=1)
    c = c[ok]
    key = (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
    return int(len(np.unique(key)))


def summarize(rows, budget=None, old_budget=OLD_BUDGET):
    """Distribution of n_in, the PB budget, and what a budget keeps. `budget` defaults to the PB value of these rows."""
    n = np.array([r['n_in'] for r in rows], dtype=np.float64)
    vox = np.array([r['vox'] for r in rows], dtype=np.float64)
    p99 = float(np.percentile(n, 99))
    b = pb_budget(p99) if budget is None else int(budget)

    def kept(B):
        d = np.array([expected_distinct_out(r['n_in'], r['mult'], B) for r in rows])
        frac = d / np.maximum(n, 1)
        return d, frac

    d_old, f_old = kept(old_budget)
    d_pb, f_pb = kept(b)
    copies = b / np.maximum(d_pb, 1e-9)
    u = np.array([sum(r['mult'].values()) for r in rows], dtype=np.float64)
    return dict(
        frames=len(rows),
        n_in=dict(median=float(np.median(n)), p90=float(np.percentile(n, 90)), p99=p99, max=float(n.max()),
                  min=float(n.min()),
                  p99_lower=float(np.percentile(n, 99, method='lower')),
                  p99_higher=float(np.percentile(n, 99, method='higher'))),
        input_duplicate_share=float(1.0 - u.sum() / max(n.sum(), 1)),
        pb_budget=b,
        pb_budget_if_p99_lower=pb_budget(float(np.percentile(n, 99, method='lower'))),
        pb_budget_if_p99_higher=pb_budget(float(np.percentile(n, 99, method='higher'))),
        old=dict(budget=old_budget, real_point_fraction_median=float(np.median(f_old)),
                 real_point_fraction_mean=float(np.mean(f_old)), subsampled_share=float(np.mean(n > old_budget)),
                 copies_per_distinct_median=float(np.median(old_budget / np.maximum(d_old, 1e-9)))),
        pb=dict(budget=b, real_point_fraction_median=float(np.median(f_pb)), real_point_fraction_min=float(f_pb.min()),
                subsampled_share=float(np.mean(n > b)),
                copies_per_distinct_median=float(np.median(copies)),
                copies_per_distinct_p10=float(np.percentile(copies, 10)),
                copies_per_distinct_p90=float(np.percentile(copies, 90)),
                frames_below_half_budget=int((n < b / 2).sum())),
        voxel_step=dict(voxel=list(VOXEL), occupied_median=float(np.median(vox)), occupied_p90=float(np.percentile(vox, 90)),
                        occupied_p99=float(np.percentile(vox, 99)), occupied_max=float(vox.max()),
                        pb_budget_on_voxel_output=pb_budget(float(np.percentile(vox, 99))),
                        kept_by_voxel_step_median=float(np.median(vox / np.maximum(n, 1)))),
    )


def budget_step_index(queue, mode):
    """Index of the first fixed-count budget step in a DataProcessor queue (functools.partial entries)."""
    for i, f in enumerate(queue):
        func = getattr(f, 'func', None)
        if func is not None and func.__name__ in BUDGET_STEPS and f.keywords['config'].NUM_POINTS[mode] != -1:
            return i
    raise RuntimeError('no fixed-count budget step in the processor queue')


def check_range_mask_first(queue, at):
    names = [getattr(getattr(f, 'func', None), '__name__', '?') for f in queue[:at]]
    if 'mask_points_and_boxes_outside_range' not in names:
        raise RuntimeError('budget step at %d is not preceded by mask_points_and_boxes_outside_range (%s): its input '
                           'is not an in-range count' % (at, names))
    return names


# ------------------------------------------------------------------------------------------------ dataset plumbing
_STATE = {}


def _hook(ds):
    q = ds.data_processor.data_processor_queue
    at = budget_step_index(q, ds.data_processor.mode)
    before = check_range_mask_first(q, at)
    f = q[at]
    rec = {}

    def wrapped(data_dict=None, _f=f):
        rec['xyz'] = np.asarray(data_dict['points'][:, :3], dtype=np.float32).copy()
        return _f(data_dict=data_dict)

    q[at] = wrapped
    return rec, f.func.__name__, dict(f.keywords['config'].NUM_POINTS), before


def _work(args):
    chunk, seed = args
    ds, rec, pcr = _STATE['ds'], _STATE['rec'], _STATE['pcr']
    np.random.seed(seed)
    out = []
    for i in chunk:
        t0 = time.time()
        rec.clear()
        ds[i]
        xyz = rec['xyz']
        out.append(dict(idx=int(i), n_in=int(len(xyz)), mult=multiplicity_histogram(xyz),
                        vox=occupied_voxels(xyz, pcr), sec=round(time.time() - t0, 3)))
    return out


def build_block(cfg, block, split, logger, calib_frames=None):
    from pcdet.datasets import build_dataloader
    from pcdet.datasets.point_calibration import calibration_target_config, link_point_calibration
    if block == 'DATA_CONFIG_TAR':
        blk = cfg.DATA_CONFIG_TAR
        if split == 'train':
            blk, _ = calibration_target_config(blk, 'train')
        ds, _, _ = build_dataloader(dataset_cfg=blk, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                    logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY', None))
        return ds, blk, 'evaluation mode (test.py), %s split' % split
    if split != 'train':
        raise ValueError('a training block is measured on its TRAIN split only')
    blk = cfg.DATA_CONFIG if block == 'DATA_CONFIG' else cfg.DATA_CONFIGS[block]
    ds, _, _ = build_dataloader(dataset_cfg=blk, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY', None))
    how = 'training mode, augmentation on (train.py), train split'
    if blk.get('HIST_DIST_ON_THE_FLY', False):
        tcfg, tsplit = calibration_target_config(cfg.DATA_CONFIG_TAR)
        tcal, _, _ = build_dataloader(dataset_cfg=tcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                      workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY', None))
        nf = calib_frames or blk.get('HIST_DIST_FRAMES', 1000)
        link_point_calibration(ds, tcal, num_frames=nf, num_bins=blk.get('HIST_DIST_BINS', 50),
                               max_dist=blk.get('HIST_DIST_MAX_DIST', 75.0), logger=logger,
                               fov_degree=blk.get('HIST_DIST_FOV_DEGREE', None),
                               fov_heading=blk.get('HIST_DIST_FOV_HEADING', 0.0),
                               skip_point_budget=blk.get('HIST_DIST_BEFORE_POINT_BUDGET', False))
        rate = ds.data_processor.per_bin_sample_rate()
        how += '; density calibration installed as train.py (%d frames, target %s split, rate %.3f..%.3f)' % (
            nf, tsplit, float(rate.min()), float(rate.max()))
        del tcal
    return ds, blk, how


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--block', required=True)
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--frames', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--calib_frames', type=int, default=None, help='default: the block HIST_DIST_FRAMES (train.py)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    os.chdir(TOOLS)
    sys.path.insert(0, TOOLS)
    import _init_path  # noqa: F401
    import logging
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    logger = logging.getLogger('pb_census'); logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.WARNING)
    cfg = cfg_from_yaml_file(a.cfg_file, EasyDict())
    t0 = time.time()
    np.random.seed(a.seed)
    ds, blk, how = build_block(cfg, a.block, a.split, logger, a.calib_frames)
    rec, step_name, old_np, before = _hook(ds)
    idx = strided_indices(len(ds), a.frames)
    _STATE.update(ds=ds, rec=rec, pcr=np.asarray(blk.POINT_CLOUD_RANGE, dtype=np.float64))
    chunks = [idx[k::a.workers] for k in range(a.workers)]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.workers) as pool:
            parts = pool.map(_work, [(c, a.seed * 1000 + k) for k, c in enumerate(chunks)])
    else:
        parts = [_work((chunks[0], a.seed * 1000))]
    rows = sorted([r for p in parts for r in p], key=lambda r: r['idx'])
    summ = summarize(rows)
    res = dict(cfg_file=a.cfg_file, block=a.block, split=a.split, built=how, dataset=type(ds).__name__,
               dataset_len=len(ds), budget_step=step_name, steps_before_budget=before, parent_num_points=old_np,
               pb_value=(a.split == 'train'), seconds=round(time.time() - t0, 1), summary=summ,
               frames=[dict(idx=r['idx'], n_in=r['n_in'], u_in=sum(r['mult'].values()), vox=r['vox'], sec=r['sec'])
                       for r in rows])
    s = summ
    print('%s [%s, %s] %s' % (os.path.basename(a.cfg_file), a.block, a.split, how))
    print('  %d of %d frames; budget step %s (parent NUM_POINTS %s) after %s' % (
        s['frames'], len(ds), step_name, old_np, before))
    print('  points entering the budget step: median %.0f, p90 %.0f, p99 %.0f, max %.0f (min %.0f)' % (
        s['n_in']['median'], s['n_in']['p90'], s['n_in']['p99'], s['n_in']['max'], s['n_in']['min']))
    print('  PB budget %d%s (p99 lower/higher -> %d / %d); old %d keeps real-point fraction median %.3f' % (
        s['pb_budget'], '' if a.split == 'train' else ' [DESCRIPTIVE: val never sets a value]',
        s['pb_budget_if_p99_lower'], s['pb_budget_if_p99_higher'], OLD_BUDGET, s['old']['real_point_fraction_median']))
    print('  under %d: subsampled %.1f%% of frames, copies per distinct point median %.2f (p10 %.2f, p90 %.2f), '
          'frames below half the budget %d' % (s['pb']['budget'], 100 * s['pb']['subsampled_share'],
                                                s['pb']['copies_per_distinct_median'], s['pb']['copies_per_distinct_p10'],
                                                s['pb']['copies_per_distinct_p90'], s['pb']['frames_below_half_budget']))
    print('  voxel step (0.1 x 0.1 x 0.15 m, no cap): occupied median %.0f, p99 %.0f, max %.0f -> %d' % (
        s['voxel_step']['occupied_median'], s['voxel_step']['occupied_p99'], s['voxel_step']['occupied_max'],
        s['voxel_step']['pb_budget_on_voxel_output']))
    print('  input duplicate share %.4f; %.1f s' % (s['input_duplicate_share'], res['seconds']))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)
        print('  wrote', a.out)


if __name__ == '__main__':
    main()
