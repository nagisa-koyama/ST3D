"""Measure training throughput, the GPU-vs-dataloader bound, and peak memory for ONE
(batch_size, workers) combination of a config.

Answers three questions that decide how to make the da-ieee-access baselines faster:

  1. Is a run GPU-bound or DATALOADER-bound? Reported as the fraction of wall time the main
     process sits blocked in next(dataloader_iter). With num_workers > 0 the loader prefetches
     during compute, so a healthy GPU-bound run shows a data-wait near zero. A large wait means
     more --workers will help and a bigger batch will not.
  2. What batch size actually fits in the GPU? Reported as peak torch allocated/reserved bytes.
  3. What does each combination cost in samples/s, which is the only rate that matters when the
     budget is fixed in sample presentations.

Runs a real training step - the repo's own model_fn_decorator(), optimizer and grad clipping -
so the numbers include the dense CenterHead target generation, not just a forward pass.

ONE COMBO PER PROCESS, DELIBERATELY. An earlier version of this script looped over a whole
(batch_size x workers) grid in a single process, tearing down and rebuilding a persistent-worker
DataLoader between combos. That reproduced the exact /dev/shm cgroup segfault this repo's own
--mem 32G->64G->96G history exists to avoid (see singularity_usage_and_tips.md Gotcha #2) - not
because any one combo used too much memory, but because torn-down workers' shm pages do not
reliably free before the next combo's workers fork, so pressure accumulates across the sweep. A
fresh process per combo, via a bash driver (sweep_throughput.sh) calling `singularity exec`
once per combo, tears down the CUDA context and every worker completely and sidesteps this
class of failure entirely. Do not restore the multi-combo loop.

Prints exactly one machine-readable RESULT line on success, so the bash driver can grep it out
even though everything else on stdout is normal training-style logging. Prints nothing of the
sort on OOM or any other failure - the driver treats a missing RESULT line as a failed combo.

Usage (inside the container, from ST3D/tools):

    python analysis/profile_throughput.py --cfg_file cfgs/da-ieee-access/<config>.yaml \
        --batch_size 12 --workers 8 --warmup 15 --measure 60
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

# Equivalent to `import _init_path`, done inline because this script lives in analysis/, not
# tools/ itself - `import _init_path` only resolves when the INVOKED script's own directory is
# tools/ (Python puts that on sys.path[0], not cwd). Same two paths _init_path.py itself adds:
# tools/ (for train_utils, etc.) and the repo root (for the live pcdet, not the stale editable
# install baked into the container image - see singularity gotcha #1).
_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TOOLS_DIR)
sys.path.insert(0, os.path.join(_TOOLS_DIR, '..'))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--batch_size', type=int, required=True)
    ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--warmup', type=int, default=15, help='iterations discarded before timing')
    ap.add_argument('--measure', type=int, default=60, help='iterations actually timed')
    ap.add_argument('--budget', type=int, default=562600,
                    help='sample presentations, for the projected-hours column')
    return ap.parse_args()


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    logger.info('=== profiling %s  bs=%d workers=%d ===', args.cfg_file, args.batch_size, args.workers)
    logger.info('GPU: %s', torch.cuda.get_device_name(0))

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
        dist=False, workers=0, logger=logger, training=True,
        model_ontology=cfg.get('ONTOLOGY', None))
    logger.info('source: %s, %d train frames', cfg.DATA_CONFIG.DATASET, len(dataset))

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.cuda()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    model_func = model_fn_decorator()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers,
        shuffle=True, collate_fn=dataset.collate_batch, drop_last=False,
        sampler=None, timeout=0, persistent_workers=(args.workers > 0),
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    it = iter(loader)
    data_s, compute_s, n = 0.0, 0.0, 0
    losses = []

    try:
        for step in range(args.warmup + args.measure):
            timing = step >= args.warmup
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            t1 = time.perf_counter()

            model.train()
            loss = model_func(model, batch)[0]
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
            optimizer.step()
            # Without this the async launch queue makes compute look free and data look slow.
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            if timing:
                data_s += t1 - t0
                compute_s += t2 - t1
                n += 1
                losses.append(float(loss.detach()))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
            raise
        logger.info('OOM at bs=%d workers=%d', args.batch_size, args.workers)
        return  # no RESULT line - the driver reads this as a failed combo

    total_s = data_s + compute_s
    it_s = n / total_s
    samples_s = n * args.batch_size / total_s
    data_frac = data_s / total_s
    peak_alloc_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 2 ** 30
    hours = args.budget / samples_s / 3600.0
    iters = args.budget // args.batch_size

    logger.info(
        'bs=%-3d workers=%-2d  %6.2f it/s  %6.1f samples/s  data-wait %5.1f%%  '
        'peak %5.1f/%5.1f GB  -> %5.1f h, %7d iters  (loss %.2f)',
        args.batch_size, args.workers, it_s, samples_s, 100 * data_frac,
        peak_alloc_gb, peak_reserved_gb, hours, iters, float(np.mean(losses)))

    # Single machine-readable line, grepped by sweep_throughput.sh. Keep field order/names in
    # sync with that script's `awk`/`cut` parsing if this changes.
    print('RESULT cfg=%s bs=%d workers=%d it_s=%.4f samples_s=%.4f data_frac=%.4f '
          'peak_alloc_gb=%.2f peak_reserved_gb=%.2f hours=%.3f iters=%d loss=%.3f' % (
              os.path.basename(args.cfg_file), args.batch_size, args.workers, it_s, samples_s,
              data_frac, peak_alloc_gb, peak_reserved_gb, hours, iters, float(np.mean(losses))))


if __name__ == '__main__':
    main()
