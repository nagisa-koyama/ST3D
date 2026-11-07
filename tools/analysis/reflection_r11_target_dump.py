"""Checks of the reflection rows R11 (threshold) / R11r (random twin) on one target before any GPU row (experiments_md
20261003_05 §5 item 35g; bug_prevention_checklist rules 4, 8, 9). The pattern of factorswap_pl_target_dump.py, whose
helpers it reuses.

For each of the target's two rows:
  - CONFIG: the row resolves to its oracle's file plus ONLY <DATA_CONFIG>.TRAIN_FACTOR_SWAP.*, and against the config the
    oracle JOB logged (26843 S2, 31624 N->W O2, 26389 Pandar64) it differs in those paths plus exactly what the oracle's
    own file has drifted since that job (listed);
  - TARGET DUMP on strided TRAIN frames, augmentation off, beside the oracle's loader on the same frames:
      the row's cloud before prepare_data = the oracle's minus exactly the arm's dropped returns (same order);
      unclassified returns (cell -1) never dropped; threshold: every dropped return has intensity <= T of its cell and
      every return below T in an applied cell is dropped; realised dropped share per (bin, class) against E (pooled over
      the dump frames; random: within 4 binomial SE);
      boxes: the raw boxes identical; after prepare_data the row's boxes = the oracle's minus exactly the boxes the arm
      emptied (>= 1 point before, 0 after), labels unchanged;
      targets of one collated batch (positives, heatmap peaks, median points per positive box), beside the oracle's;
  - IDENTITY (rule 8): the row with E = 0 in every cell against the oracle over IDENTITY_FRAMES strided TRAIN frames WITH
    augmentation, each frame seeded alike: points and labelled boxes exact;
  - the key inert at evaluation (DATA_CONFIG_TAR identical, its frame 0 identical, the row's training block built in eval
    mode identical);
  - loader time per frame.
Any failed check raises. ANALYSIS (target TRAIN labels are read); CPU; run as a Slurm CPU job from ST3D/tools:

    python analysis/reflection_r11_target_dump.py --target s2|n2w|pandaset
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import factorswap_pl_target_dump as P  # noqa: E402  (chdir to tools, _init_path, helpers)
import factorswap_pl_config_check as C  # noqa: E402
from pcdet.datasets import reflection_drop as R  # noqa: E402
from reflection_r11_dump_utils import gone_mask  # noqa: E402

A = 'cfgs/da-ieee-access-analysis/'
D = 'cfgs/da-ieee-access/'
TARGETS = {   # tag: (oracle config, oracle job whose logged config is the parent, directory)
    's2': (D + 'centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml', 26843, 'factorswap-s2'),
    'n2w': (D + 'centerpoint-sourceonly-waymo2waymo.yaml', 31624, 'factorswap-n2w'),
    'pandaset': (D + 'centerpoint-pandaset-spin2spin.yaml', 26389, 'factorswap-pandaset'),
}
ROWS = ('r11-reflection', 'r11r-reflection-random')
check = P.check

ARM_CALLS = []
FULL_KEEP = []
_ORIG_KEEP = R.keep_mask
_ORIG_WAYMO_KEEP = P.wfs.reflection_keep


def _spy(xyz_sensor, inten, cfg, frame_key):
    keep, cell = _ORIG_KEEP(xyz_sensor, inten, cfg, frame_key)
    ARM_CALLS.append(dict(keep=keep, cell=cell, inten=np.asarray(inten, dtype=np.float64), key=frame_key))
    return keep, cell


def _spy_waymo(*a, **k):
    keep = _ORIG_WAYMO_KEEP(*a, **k)
    FULL_KEEP.append(keep)
    return keep


def config_check(tag, row_path):
    oracle_path, job, _ = TARGETS[tag]
    row, ora = C.flat(C.load(row_path)), C.flat(C.load(oracle_path))
    swap = sorted(p for p in row if p.startswith('DATA_CONFIG.TRAIN_FACTOR_SWAP.'))
    check(len(swap) > 0, '%s: the key is present (%d paths)' % (Path(row_path).name, len(swap)))
    got = C.diff(row, ora)
    check(got == swap, '%s vs %s today: differs only in TRAIN_FACTOR_SWAP%s' % (
        Path(row_path).name, Path(oracle_path).name, '' if got == swap else ' (got %s)' % got))
    keys = ('DATA_CONFIG', 'DATA_CONFIG_TAR', 'MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'ONTOLOGY')
    if tag == 'pandaset':       # 26389 trained spin2flash (eval on flash): the training recipe only
        keys = ('DATA_CONFIG', 'MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'ONTOLOGY')
    lg = C.logged(job)
    drift = C.diff(ora, lg, keys)
    got = C.diff(row, lg, keys)
    check(sorted(set(got) - set(drift)) == swap and set(drift) <= set(got),
          '%s vs job %d logged: TRAIN_FACTOR_SWAP plus the oracle file\'s own drift since that job (%d paths: %s)'
          % (Path(row_path).name, job, len(drift), drift if len(drift) < 10 else drift[:10] + ['...']))


def boxes_emptied(base_raw, row_raw, waymo=False):
    """Raw boxes (7 values) holding >= 1 point of the oracle's raw cloud and none of the row's."""
    b = np.asarray(base_raw['gt_boxes'])[:, :7].astype(np.float32)
    if len(b) == 0:
        return b
    # the counting the zero-point filter does: prepare_data's points_in_boxes_cpu (KITTI, PandaSet); Waymo's
    # differential recount (waymo_factor_swap.points_per_box, degenerate boxes count 0)
    count = P.wfs.points_per_box if waymo else P.lfs.oracle_counts
    nb, nr = count(base_raw['points'], b), count(row_raw['points'], b)
    return b[(nb > 0) & (nr == 0)]


def dump_row(tag, row_path, base, args):
    base_cfg, path, base_ds, frames, b_items, base_t, base_tar0, base_eval0 = base
    cfg = P.load(row_path)
    swap = cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP
    sub = swap.REFLECTION_DROP
    E = R.check_cfg(sub)
    T = np.nan_to_num(R._table(sub, 'T'), nan=-np.inf) if sub.MODE == 'threshold' else None
    print('=' * 118)
    print(row_path, '\n  MODE %s; E ground %s; E above %s' % (sub.MODE, list(sub.E.ground), list(sub.E.above)))
    config_check(tag, row_path)
    ds = P.build(cfg, cfg.DATA_CONFIG)
    # --- raw clouds (before prepare_data) and the arm's own mask
    n_cell = np.zeros(R.N_BINS * 2); n_drop = np.zeros(R.N_BINS * 2)
    n_bad_thr = n_missed = n_untouched_drop = 0
    n_emptied = 0
    n_fg = np.zeros(2); n_bg = np.zeros(2)      # (oracle, row) points inside / outside any raw box
    R.keep_mask = _spy
    P.wfs.reflection_keep = _spy_waymo
    try:
        for k, i in enumerate(dump_idx(base_ds, args.share_frames)):
            ARM_CALLS.clear(); FULL_KEEP.clear()
            rr = P.raw_input(ds, i)
            assert len(ARM_CALLS) == 1, 'the arm ran %d times on one frame' % len(ARM_CALLS)
            call = ARM_CALLS[0]
            br = P.raw_input(base_ds, i)
            # KITTI / PandaSet: the arm's mask covers the whole cloud; Waymo: TOP returns only, the full mask from
            # waymo_factor_swap.reflection_keep
            keep_full = FULL_KEEP[0] if tag == 'n2w' else call['keep']
            same = keep_full is not None and len(keep_full) == len(br['points']) and \
                np.array_equal(rr['points'], br['points'][keep_full])
            if k < P.N_FRAMES or not same:
                print('  frame %s: points %d -> %d (dropped %.4f); boxes %d (raw)' % (
                    call['key'], len(br['points']), len(rr['points']), 1 - len(rr['points']) / max(len(br['points']), 1),
                    len(br['gt_boxes'])))
            check(same, 'frame %s: the row\'s raw cloud = the oracle\'s minus exactly the arm\'s dropped returns'
                  % call['key'])
            check(np.array_equal(np.asarray(rr['gt_boxes']), np.asarray(br['gt_boxes'])), 'raw boxes identical')
            cell, keep, inten = call['cell'], call['keep'], call['inten']
            v = cell >= 0
            n_cell += np.bincount(cell[v], minlength=R.N_BINS * 2)
            n_drop += np.bincount(cell[v & ~keep], minlength=R.N_BINS * 2)
            n_untouched_drop += int((~v & ~keep).sum())
            if T is not None:
                tt = np.where(v, T[np.where(v, cell, 0)], -np.inf)
                n_bad_thr += int(((~keep) & (inten > tt)).sum())
                n_missed += int((keep & v & (E[np.where(v, cell, 0)] > 0) & (inten < tt)).sum())
            n_emptied += len(boxes_emptied(br, rr, tag == 'n2w'))
            if len(br['gt_boxes']):
                for j, cloud in enumerate((br['points'], rr['points'])):
                    inside = P.in_any_box(cloud, np.asarray(br['gt_boxes'])[:, :7])
                    n_fg[j] += inside.sum(); n_bg[j] += (~inside).sum()
    finally:
        R.keep_mask = _ORIG_KEEP
        P.wfs.reflection_keep = _ORIG_WAYMO_KEEP
    check(n_untouched_drop == 0, 'unclassified returns (ego, beyond 75 m, outside both height classes) never dropped')
    if T is not None:
        check(n_bad_thr == 0, 'threshold: no dropped return brighter than T of its cell')
        check(n_missed == 0, 'threshold: every return below T in an applied cell dropped')
    real = n_drop / np.maximum(n_cell, 1)
    lines = []
    for kc in range(R.N_BINS * 2):
        if n_cell[kc] == 0:
            continue
        se = np.sqrt(max(E[kc] * (1 - E[kc]), 1e-12) / n_cell[kc])
        lines.append('%s %s: E %.3f realised %.4f (n %d)' % (P_BINS[kc // 2], R.CLASSES[kc % 2], E[kc], real[kc],
                                                             n_cell[kc]))
        if sub.MODE == 'random' and n_cell[kc] >= 1000:
            check(abs(real[kc] - E[kc]) <= 4 * se + 1e-9, 'random: realised share in %s %s within 4 SE of E'
                  % (P_BINS[kc // 2], R.CLASSES[kc % 2]))
        if E[kc] == 0:
            check(n_drop[kc] == 0, 'no drop where E = 0 (%s %s)' % (P_BINS[kc // 2], R.CLASSES[kc % 2]))
    print('  realised dropped share per cell over %d frames:\n    %s' % (args.share_frames, '\n    '.join(lines)))
    print('  overall dropped share of classified returns %.4f (expected %.4f); boxes the arm emptied %d (%.2f per frame); '
          'dropped share of the points inside annotated boxes (any class) %.4f, outside them %.4f'
          % (n_drop.sum() / max(n_cell.sum(), 1), (E * n_cell).sum() / max(n_cell.sum(), 1), n_emptied,
             n_emptied / args.share_frames, 1 - n_fg[1] / max(n_fg[0], 1), 1 - n_bg[1] / max(n_bg[0], 1)))
    # --- after prepare_data (augmentation off): boxes = the oracle's minus the emptied ones
    s_items = [P.item(ds, i) for i in frames]
    for i, (b, _), (s, _) in zip(frames, b_items, s_items):
        emptied = boxes_emptied(P.raw_input(base_ds, i), P.raw_input(ds, i), tag == 'n2w')
        bb = b['gt_boxes']
        gone = gone_mask(bb, emptied)
        expect = bb[~gone]
        check(len(expect) == len(s['gt_boxes']) and np.array_equal(expect, s['gt_boxes']),
              'frame %d: boxes after prepare_data = the oracle\'s minus the %d the arm emptied (labels unchanged)'
              % (i, int(gone.sum())))
    tb = np.median([P.item(base_ds, i)[1] for _ in range(2) for i in frames])
    ts = np.median([P.item(ds, i)[1] for _ in range(2) for i in frames])
    print('  loader time per frame (warm, augmentation off): oracle %.3f s, row %.3f s (x%.2f)' % (tb, ts, ts / tb))
    swap_t = P.targets(cfg, ds, [s for s, _ in s_items])
    P.report_targets('base', base_t)
    P.report_targets('row', swap_t)
    P.eval_inert(cfg, ['DATA_CONFIG'], base_cfg, path, base_tar0, base_eval0)
    identity(cfg, base_cfg, args.identity_frames)


def identity(cfg, base_cfg, n_frames):
    icfg = copy.deepcopy(cfg)
    sub = icfg.DATA_CONFIG.TRAIN_FACTOR_SWAP.REFLECTION_DROP
    sub.E = {'ground': [0.0] * R.N_BINS, 'above': [0.0] * R.N_BINS}
    ids = P.build(icfg, icfg.DATA_CONFIG, aug_off=False)
    base = P.build(base_cfg, base_cfg.DATA_CONFIG, aug_off=False)
    idx = np.linspace(0, len(base) - 1, n_frames).astype(int)
    n_same = 0
    for i in idx:
        b, _ = P.item(base, i, seed=int(i))
        s, _ = P.item(ids, i, seed=int(i))
        n_same += bool(np.array_equal(b['points'], s['points']) and np.array_equal(b['gt_boxes'], s['gt_boxes']))
    print('  IDENTITY (REFLECTION_DROP %s with E = 0 everywhere, augmentation ON, each frame seeded alike): %d of %d '
          'strided TRAIN frames identical to the oracle in points AND labelled boxes' % (sub.MODE, n_same, len(idx)))
    check(n_same == len(idx), 'identity setting reproduces the oracle exactly (%s)' % sub.MODE)


def dump_idx(ds, n):
    return list(np.linspace(0, len(ds) - 1, n).astype(int))


P_BINS = ['%g-%g' % (R.RANGE_EDGES[i], R.RANGE_EDGES[i + 1]) for i in range(R.N_BINS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=sorted(TARGETS))
    ap.add_argument('--identity-frames', type=int, default=48)
    ap.add_argument('--share-frames', type=int, default=40)
    args = ap.parse_args()
    oracle_path, _, d = TARGETS[args.target]
    base_cfg = P.load(oracle_path)
    base_ds = P.build(base_cfg, base_cfg.DATA_CONFIG)
    frames = list(np.linspace(0, len(base_ds) - 1, P.N_FRAMES).astype(int))
    b_items = [P.item(base_ds, i) for i in frames]
    base_t = P.targets(base_cfg, base_ds, [b for b, _ in b_items])
    base_tar0, _ = P.item(P.build(base_cfg, base_cfg.DATA_CONFIG_TAR, training=False), 0)
    base_eval0, _ = P.item(P.build(base_cfg, base_cfg.DATA_CONFIG, training=False), 0)
    print('oracle %s: %d TRAIN frames; dump frames %s' % (oracle_path, len(base_ds), frames))
    base = (base_cfg, ['DATA_CONFIG'], base_ds, frames, b_items, base_t, base_tar0, base_eval0)
    for r in ROWS:
        dump_row(args.target, A + '%s/centerpoint-%s-%s.yaml' % (d, d, r), base, args)
    print('=' * 118)
    if P.FAILS:
        print('FAILED CHECKS (%d):' % len(P.FAILS))
        for f in P.FAILS:
            print('  ' + f)
        raise SystemExit(1)
    print('ALL R11 CHECKS PASSED (%s: %d rows)' % (args.target, len(ROWS)))


if __name__ == '__main__':
    main()
