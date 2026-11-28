"""Item 72's checks before any GPU row (experiments_md 20261003_05 §5 item 72, design fixed 2026-11-29 00:15 JST;
bug_prevention_checklist rules 4 and 8). For one target, the swap row's config against its x, y, z, intensity oracle,
through the REAL loaders. Any failed check raises (exit 1).

  1. IDENTITY (rule 8): the arm with an identity table (to_q = from_q = the real table's from_q, from_step 0), over
     --frames strided TRAIN frames WITH augmentation, each frame seeded alike: points and labelled boxes equal the
     oracle's exactly in x, y, z and boxes, the intensity column to 1e-6 (the float32 round trip of the map).
  2. RAW (real table, before prepare_data): every column but the intensity equal to the oracle's, the boxes too; the
     intensity changed.
  3. DIRECTION (real table, augmentation off, --frames TRAIN frames): per horizontal-range ring with >= 2000 points, the
     mapped intensity's median lies within 0.02 of the table's nuScenes median (to_q). Beside, not a gate: the oracle's
     median against the table's target median (from_q), i.e. whether the table describes this oracle's clouds.
  4. EVALUATION PATH: DATA_CONFIG_TAR equal to the oracle's; --frames strided val frames from it equal the oracle's
     (points and boxes); the swap's TRAINING block built in evaluation mode equals the oracle's on the same frames (the
     key is training-gated).
ANALYSIS (target TRAIN labels are read); CPU; a Slurm CPU job from ST3D/tools:

    python analysis/item72_intensity_checks.py --target kitti|waymo|pandaspin [--frames 40]
    python analysis/item72_intensity_checks.py --same_nuscenes_side A.npz B.npz ...    # to_q equal exactly
"""
import argparse
import contextlib
import copy
import io
import os
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
os.chdir(TOOLS)
import _init_path  # noqa: E402,F401
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

D = 'cfgs/da-ieee-access-analysis/item72/'
TARGETS = {   # target: (swap row, oracle)
    'kitti': (D + 'centerpoint-item72-kitti-imap.yaml', D + 'centerpoint-item72-kitti-oracle-xyzi.yaml'),
    'waymo': (D + 'centerpoint-item72-waymo-imap.yaml', 'cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo-xyzi.yaml'),
    'pandaspin': (D + 'centerpoint-item72-pandaspin-imap.yaml', 'cfgs/da-ieee-access/centerpoint-pandaset-spin2spin-xyzi.yaml'),
}
OUT = Path('/home/koyama/data/item72')
logger = common_utils.create_logger()
FAILS = []


def check(ok, msg):
    print('    [%s] %s' % ('ok' if ok else 'FAIL', msg), flush=True)
    if not ok:
        FAILS.append(msg)


def load(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return cfg_from_yaml_file(path, EasyDict())


def build(cfg, dcfg, training=True, aug_off=False):
    with contextlib.redirect_stdout(io.StringIO()):
        ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                    logger=logger, training=training, model_ontology=cfg.get('ONTOLOGY'))
    if training and aug_off:
        ds.data_augmentor.data_augmentor_queue = []
    return ds


def item(ds, i, seed):
    np.random.seed(seed)
    return ds[i]


def raw_input(ds, i):
    keep = ds.prepare_data
    ds.prepare_data = lambda data_dict: data_dict
    try:
        return ds[i]
    finally:
        ds.prepare_data = keep


def host(path):
    return str(path).replace('/storage/', '/home/koyama/data/', 1) if not os.path.exists(path) else str(path)


def strided(n, k):
    return np.unique(np.linspace(0, n - 1, k).astype(int))


def identity(swap_cfg, oracle_cfg, table, n, tag):
    t = np.load(host(table))
    p = OUT / ('identity_%s.npz' % tag)
    np.savez(p, edges=t['edges'], from_q=t['from_q'], to_q=t['from_q'], from_step=0.0)
    icfg = copy.deepcopy(swap_cfg)
    icfg.DATA_CONFIG.TRAIN_FACTOR_SWAP.INTENSITY_MAP.TABLES = str(p)
    s_ds, o_ds = build(icfg, icfg.DATA_CONFIG), build(oracle_cfg, oracle_cfg.DATA_CONFIG)
    same = worst = 0
    idx = strided(len(o_ds), n)
    for i in idx:
        o, s = item(o_ds, i, int(i)), item(s_ds, i, int(i))
        ok = (o['points'].shape == s['points'].shape and np.array_equal(o['points'][:, :3], s['points'][:, :3])
              and np.array_equal(o['gt_boxes'], s['gt_boxes']))
        if ok:
            d = float(np.abs(o['points'][:, 3:] - s['points'][:, 3:]).max()) if len(o['points']) else 0.0
            worst = max(worst, d)
            ok = d <= 1e-6
        same += ok
    print('  IDENTITY (identity table, augmentation ON, frames seeded alike): %d of %d TRAIN frames equal; largest '
          'intensity difference %.2e' % (same, len(idx), worst))
    check(same == len(idx), 'identity setting reproduces the oracle (x, y, z and boxes exact, intensity <= 1e-6)')


def raw_and_direction(swap_cfg, oracle_cfg, table, n):
    t = np.load(host(table))
    edges, from_q, to_q = t['edges'], t['from_q'], t['to_q']
    mid = from_q.shape[1] // 2
    s_ds, o_ds = build(swap_cfg, swap_cfg.DATA_CONFIG, aug_off=True), build(oracle_cfg, oracle_cfg.DATA_CONFIG, aug_off=True)
    idx = strided(len(o_ds), n)
    raw_ok, changed = 0, 0
    for i in idx[:min(10, len(idx))]:
        o, s = raw_input(o_ds, i), raw_input(s_ds, i)
        cols = [c for c in range(o['points'].shape[1]) if c != 3]
        raw_ok += (np.array_equal(o['points'][:, cols], s['points'][:, cols])
                   and np.array_equal(o['gt_boxes'], s['gt_boxes']) and np.array_equal(o['gt_names'], s['gt_names']))
        changed += not np.array_equal(o['points'][:, 3], s['points'][:, 3])
    k = min(10, len(idx))
    print('  RAW (real table, before prepare_data, %d frames, %d columns): other columns and boxes equal in %d, '
          'intensity changed in %d' % (k, o['points'].shape[1], raw_ok, changed))
    check(raw_ok == k and changed == k, 'the arm changes the intensity column only')
    per = {'swap': [[] for _ in range(len(edges) - 1)], 'oracle': [[] for _ in range(len(edges) - 1)]}
    for i in idx:
        for name, ds in (('swap', s_ds), ('oracle', o_ds)):
            p = item(ds, i, int(i))['points']
            r = np.clip(np.searchsorted(edges, np.hypot(p[:, 0], p[:, 1]), side='right') - 1, 0, len(edges) - 2)
            for b in range(len(edges) - 1):
                per[name][b].append(p[r == b, 3])
    print('  DIRECTION (real table, augmentation off, %d TRAIN frames): per-ring medians' % len(idx))
    print('    ring       n   swap  nuS(to_q) | oracle target(from_q)')
    for b in range(len(edges) - 1):
        sv, ov = np.concatenate(per['swap'][b]), np.concatenate(per['oracle'][b])
        if len(sv) < 2000:
            print('    %4.0f-%-4.0f %7d  (too few points, not gated)' % (edges[b], edges[b + 1], len(sv)))
            continue
        ms, mo = float(np.median(sv)), float(np.median(ov))
        print('    %4.0f-%-4.0f %7d  %.3f  %.3f     | %.3f  %.3f' % (edges[b], edges[b + 1], len(sv), ms, to_q[b, mid], mo,
                                                                  from_q[b, mid]))
        check(abs(ms - to_q[b, mid]) <= 0.02, 'ring %.0f-%.0f m: mapped median %.3f within 0.02 of nuScenes %.3f' % (
            edges[b], edges[b + 1], ms, to_q[b, mid]))


def eval_path(swap_cfg, oracle_cfg, n):
    check(swap_cfg.DATA_CONFIG_TAR == oracle_cfg.DATA_CONFIG_TAR, 'DATA_CONFIG_TAR equal to the oracle\'s')
    s_ds = build(swap_cfg, swap_cfg.DATA_CONFIG_TAR, training=False)
    o_ds = build(oracle_cfg, oracle_cfg.DATA_CONFIG_TAR, training=False)
    idx = strided(len(o_ds), n)
    same = 0
    for i in idx:
        o, s = item(o_ds, i, 0), item(s_ds, i, 0)
        same += np.array_equal(o['points'], s['points']) and np.array_equal(o['gt_boxes'], s['gt_boxes'])
    check(same == len(idx), 'evaluation loader: %d of %d val frames equal to the oracle\'s (points, boxes)' % (same, len(idx)))
    s_tr = build(swap_cfg, swap_cfg.DATA_CONFIG, training=False)
    o_tr = build(oracle_cfg, oracle_cfg.DATA_CONFIG, training=False)
    idx = strided(len(o_tr), n)
    same = sum(np.array_equal(item(o_tr, i, 0)['points'], item(s_tr, i, 0)['points']) for i in idx)
    check(same == len(idx), 'the swap\'s training block in evaluation mode: %d of %d frames equal (training-gated)' % (
        same, len(idx)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=sorted(TARGETS))
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--same_nuscenes_side', nargs='+', default=None)
    a = ap.parse_args()
    if a.same_nuscenes_side:
        ts = [np.load(p) for p in a.same_nuscenes_side]
        for p, t in zip(a.same_nuscenes_side[1:], ts[1:]):
            check(np.array_equal(t['to_q'], ts[0]['to_q']) and np.array_equal(t['edges'], ts[0]['edges']),
                  '%s: nuScenes side (to_q, edges) equal to %s\'s' % (os.path.basename(p),
                                                                   os.path.basename(a.same_nuscenes_side[0])))
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        swap_path, oracle_path = TARGETS[a.target]
        swap_cfg, oracle_cfg = load(swap_path), load(oracle_path)
        table = swap_cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP.INTENSITY_MAP.TABLES
        print('== item 72 checks, %s: %s against %s; table %s' % (a.target, swap_path, oracle_path, table), flush=True)
        identity(swap_cfg, oracle_cfg, table, a.frames, a.target)
        raw_and_direction(swap_cfg, oracle_cfg, table, a.frames)
        eval_path(swap_cfg, oracle_cfg, a.frames)
    print('== %s' % ('ALL PASSED' if not FAILS else 'FAILED: %d (%s)' % (len(FAILS), '; '.join(FAILS))), flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
