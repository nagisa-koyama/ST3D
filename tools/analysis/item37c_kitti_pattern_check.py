"""Checks before any GPU row of experiments_md 20261003_05 §5 item 37c (KITTI pattern row P_K and its count-matched control
L_K; bug_prevention_checklist rules 1, 4, 8). ANALYSIS, CPU; real KITTI TRAIN frames through the TRAINING loader with
augmentation switched off, beside the 26843 oracle config on the same frames.
  identity : the L_K arm (RANGE_KEEP) at RATE 1 everywhere gives exactly the oracle's points and labels on every frame;
  P_K      : every kept point is a point of the oracle's frame; boxes are the oracle's minus those the cut leaves with
             fewer than MIN_POINTS_OF_GT points (prepare_data's recount); rings kept per frame and the spacing of their
             elevations; at most one point per kept ring per 0.332 deg bin on the raw cut; points per 5-m bin in the
             voxel grid against the oracle's and nuScenes TRAIN's (item 28's measurement);
  L_K      : (if its config exists) the same box rule; kept share per frame against the rates' expected share;
  eval     : both keys are inert when the config is built for evaluation.
Prints one line per check and "ALL ITEM-37C CHECKS PASSED" or exits 1.
    python analysis/item37c_kitti_pattern_check.py [n_frames]       # from ST3D/tools, inside the container
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402

BASE = 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml'
P_K = 'cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml'
L_K = 'cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-l-latcount.yaml'
NUS_TARGET = '/home/koyama/data/arm_point_counts/p2n-lattice.json'   # its 'target' side = nuScenes TRAIN, in the grid
FAILS = []


def check(ok, msg):
    print('[%s] %s' % ('ok' if ok else 'FAIL', msg))
    if not ok:
        FAILS.append(msg)


def load(path):
    from pcdet.config import cfg_from_yaml_file
    return cfg_from_yaml_file(path, EasyDict())


def build(cfg, training):
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=common_utils.create_logger(), training=training,
                                model_ontology=cfg.get('ONTOLOGY'))
    if training:
        ds.data_augmentor.data_augmentor_queue = []
    return ds


def item(ds, i):
    np.random.seed(0)
    return ds[i]


def rows(a):
    return {tuple(np.round(r, 4)) for r in np.asarray(a, dtype=np.float64)}


def in_box(points, boxes):
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    import torch
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    m = roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(points[:, :3]).float(),
                                                   torch.from_numpy(boxes[:, :7]).float()).numpy()
    return m.sum(1)


def bins5(points, pcr):
    p = points[:, :3]
    g = np.all((p >= pcr[:3]) & (p < pcr[3:6]), axis=1)
    r = np.hypot(p[g, 0], p[g, 1])
    return np.histogram(r[r < 75.0], bins=np.arange(0.0, 80.0, 5.0))[0]


def box_rule(name, base_d, row_d, min_pts):
    """Row boxes = base boxes minus exactly those left with < min_pts row points."""
    bb, rb = np.asarray(base_d['gt_boxes']), np.asarray(row_d['gt_boxes'])
    n_in = in_box(row_d['points'], bb)
    want = rows(bb[n_in >= min_pts])
    return rows(rb) == want, len(bb), len(rb)


def main(n):
    from pcdet.datasets.kitti import kitti_rings
    base_cfg = load(BASE)
    base = build(base_cfg, True)
    idx = np.linspace(0, len(base) - 1, n).astype(int)
    pcr = np.asarray(base_cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float64)
    min_pts = int(base_cfg.DATA_CONFIG.get('MIN_POINTS_OF_GT', 1))
    base_items = [item(base, i) for i in idx]

    # ---- identity: the L_K arm at RATE 1
    ident = copy.deepcopy(base_cfg)
    ident.DATA_CONFIG.TRAIN_FACTOR_SWAP = EasyDict({'RANGE_KEEP': {'EDGES': [0, 75.0], 'RATE': [1.0]}})
    ids = build(ident, True)
    bad = 0
    for i, b in zip(idx, base_items):
        d = item(ids, i)
        same = np.array_equal(d['points'], b['points']) and np.array_equal(d['gt_boxes'], b['gt_boxes'])
        bad += not same
    check(bad == 0, 'identity: RANGE_KEEP at RATE 1 = the oracle (points and labels exact) on %d of %d frames'
          % (len(idx) - bad, len(idx)))

    # ---- P_K
    pk = build(load(P_K), True)
    sub_bad, box_bad, n_box_b, n_box_r = 0, 0, 0, 0
    b_cnt, p_cnt = np.zeros(15), np.zeros(15)
    for i, b in zip(idx, base_items):
        d = item(pk, i)
        sub_bad += not rows(d['points']) <= rows(b['points'])
        ok, nb, nr = box_rule('P_K', b, d, min_pts)
        box_bad += not ok
        n_box_b += nb
        n_box_r += nr
        b_cnt += bins5(b['points'], pcr)
        p_cnt += bins5(d['points'], pcr)
    check(sub_bad == 0, 'P_K: every kept point is a point of the oracle frame (%d of %d frames)' % (len(idx) - sub_bad, len(idx)))
    check(box_bad == 0, 'P_K: boxes = oracle boxes minus those left with < %d points (%d of %d frames; %d -> %d boxes)'
          % (min_pts, len(idx) - box_bad, len(idx), n_box_b, n_box_r))
    # rings and azimuth bins on the raw cut (the loader's own call)
    from pcdet.datasets.kitti import kitti_factor_swap
    swap = load(P_K).DATA_CONFIG.TRAIN_FACTOR_SWAP
    n_kept, spacings, per_bin_max = [], [], 0
    for i in idx[:20]:
        sid = base.kitti_infos[i]['point_cloud']['lidar_idx']
        raw = base.get_lidar(sid)
        ring, lev = kitti_rings.rings_from_order(raw[:, :3])
        kept_mask = kitti_rings.eval_ring_thin_mask(raw, swap.RING_THIN, kitti_factor_swap.frame_rng(sid, 'RING_THIN'))
        if not np.array_equal(raw[kept_mask], kitti_factor_swap.apply_points(raw, swap, sid)):
            FAILS.append('P_K: apply_points differs from the frame-seeded pattern mask on frame %s' % sid)
        kr = np.unique(ring[kept_mask])
        n_kept.append(len(kr))
        el = np.sort(lev[kr])
        spacings += np.diff(el).tolist()
        az = np.degrees(np.arctan2(raw[kept_mask, 1], raw[kept_mask, 0]))
        b = np.floor((az + 180.0) / 0.332).astype(np.int64)
        _, c = np.unique(ring[kept_mask] * 2000 + b, return_counts=True)
        per_bin_max = max(per_bin_max, int(c.max()) if len(c) else 0)
    print('     P_K rings kept per frame: min %d median %d max %d of 64; elevation spacing of kept rings median %.2f deg '
          '(IQR %.2f-%.2f)' % (min(n_kept), int(np.median(n_kept)), max(n_kept), np.median(spacings),
                               np.percentile(spacings, 25), np.percentile(spacings, 75)))
    check(18 <= np.median(n_kept) <= 22, 'P_K: about 20 rings kept (KITTI FOV ~26.8 deg / 1.33 deg)')
    check(abs(np.median(spacings) - 1.33) < 0.25, 'P_K: kept rings ~1.33 deg apart')
    check(per_bin_max <= 2, 'P_K: at most one point per kept ring per 0.332 deg bin (max %d; 2 = a bin edge shared '
          'by the binning here and the operator\'s phase)' % per_bin_max)
    nus = np.asarray(json.load(open(NUS_TARGET))['target']['points']['mean'])
    print('     in-grid points per frame, 5-m bins 0-75 m:')
    print('       oracle  ', ' '.join('%6.0f' % v for v in b_cnt / len(idx)), '| total %.0f' % (b_cnt.sum() / len(idx)))
    print('       P_K     ', ' '.join('%6.0f' % v for v in p_cnt / len(idx)), '| total %.0f' % (p_cnt.sum() / len(idx)))
    print('       nuScenes', ' '.join('%6.0f' % v for v in nus), '| total %.0f' % nus.sum())

    # ---- L_K (if built)
    if os.path.exists(L_K):
        lcfg = load(L_K)
        lk = build(lcfg, True)
        box_bad, share = 0, []
        for i, b in zip(idx, base_items):
            d = item(lk, i)
            ok, _, _ = box_rule('L_K', b, d, min_pts)
            box_bad += not ok
            share.append(len(d['points']) / max(len(b['points']), 1))
        check(box_bad == 0, 'L_K: boxes = oracle boxes minus those left with < %d points (%d of %d frames)'
              % (min_pts, len(idx) - box_bad, len(idx)))
        print('     L_K kept share of the loader points per frame: median %.3f (P_K %.3f in the grid)'
              % (np.median(share), p_cnt.sum() / max(b_cnt.sum(), 1)))

    # ---- eval inertness
    for path in (P_K, L_K):
        if not os.path.exists(path):
            continue
        ev, evb = build(load(path), False), build(base_cfg, False)
        same = all(np.array_equal(item(ev, i)['points'], item(evb, i)['points']) for i in idx[:8])
        check(same, '%s: inert in evaluation (8 val frames identical to the oracle\'s)' % os.path.basename(path))

    if FAILS:
        print('FAILED ITEM-37C CHECKS (%d):' % len(FAILS))
        for f in FAILS:
            print('  ' + f)
        raise SystemExit(1)
    print('ALL ITEM-37C CHECKS PASSED')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 48)
