"""Target dump for the FACTOR-SWAP N->W rows (experiments_md 20261003_05 §5 item 24, Waymo port; bug_prevention_checklist
rule 4). The pattern of factorswap_s2_target_dump.py, on Waymo.

For each config, real Waymo TRAIN frames go through the TRAINING loader with augmentation switched off, beside the 26398
oracle config's loader on the same frames (np.random.seed(0) before each item, so the point shuffle matches when the
counts do), and the script checks what the swap changed against what it should change:
  RS (LIDAR_INDICES [0] + drop_empty_gt_boxes): the cloud is exactly the TOP block (NLZ-filtered, in range) of the raw
     file, points by lidar; positive boxes = the oracle's minus those with no TOP point, split into "empty in the
     five-lidar cloud too" (the C1-convention confound) and "seen only by the side lidars";
  PITCH_DEG: the implied pitch from points (matched by their untouched y) and from box centres (matched by size / yaw),
     both about the pivot, equal to the frame's drawn angle; boxes the pitch would empty;
  GT_DIM_SCALE: per-class dims ratio, bottom shift 0, centres x / y and yaw unchanged, points identical;
  GT_IGNORE_BY_RANGE: only the chosen class turns into -99, every other label identical; the same config with KEEP all
     1.0 reproduces the oracle's labels exactly (frames with non-class 'Sign' boxes before class boxes counted); -99
     boxes are no heatmap peak and lie in the ignored footprint; the other classes' heatmaps are identical;
  REGION_THIN: in-box / out-of-box points (every box the loader keeps, enlarged by BOX_MARGIN), boxes the arm emptied;
and for every row CenterHead's assigned targets on a collated batch (peaks never exceed positive-label boxes), the key
inert at evaluation (DATA_CONFIG_TAR identical to the oracle's and its first frame identical; the row's DATA_CONFIG in
eval mode identical for TRAIN_FACTOR_SWAP rows), and the loader time per frame. Any failed check raises. ANALYSIS, CPU;
run as a Slurm CPU job:

    python analysis/factorswap_n2w_target_dump.py <cfg.yaml> [<cfg.yaml> ...]
"""
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
os.chdir(TOOLS)
import _init_path  # noqa: E402,F401
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.datasets.osens2_utils import enlarge_boxes, in_any_box  # noqa: E402
from pcdet.datasets.waymo import waymo_factor_swap as wfs  # noqa: E402
from pcdet.models.dense_heads.center_head import CenterHead  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

BASE = 'cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml'
FRAMES = [0, 15000, 30000, 45000, 60000, 75000]
LABEL_FRAMES = 200          # strided frames for the GT_IGNORE_BY_RANGE label checks
IGN = common_utils.IGNORE_CLASS_LABEL
logger = common_utils.create_logger()
FAILS = []


def check(ok, msg):
    print('    [%s] %s' % ('ok' if ok else 'FAIL', msg))
    if not ok:
        FAILS.append(msg)


def build(cfg, training, block='DATA_CONFIG', aug_off=True):
    ds, _, _ = build_dataloader(dataset_cfg=cfg[block], class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=logger, training=training, model_ontology=cfg.get('ONTOLOGY'))
    if training and aug_off:
        ds.data_augmentor.data_augmentor_queue = []      # deterministic comparison: no augmentation
    return ds


def item(ds, i):
    np.random.seed(0)
    t = time.time()
    d = ds[i]
    return d, time.time() - t


def raw_names(ds, i):
    a = ds.infos[i]['annos']
    return [n for n in a['name'] if n != 'unknown']


def sign_first(names, class_names_raw=('Vehicle', 'Pedestrian', 'Cyclist')):
    """A non-class box precedes a class box (the order that shifted gt_classes before ST3D 12b5147)."""
    seen_other = False
    for n in names:
        if n not in class_names_raw:
            seen_other = True
        elif seen_other:
            return True
    return False


def loader_boxes(ds, i):
    """Every box the loader keeps for frame i ('unknown' dropped), in the loader frame (SHIFT_COOR applied)."""
    a = ds.infos[i]['annos']
    keep = a['name'] != 'unknown'
    b = np.array(a['gt_boxes_lidar'][keep][:, :7], dtype=np.float32, copy=True)
    b[:, :3] += np.asarray(ds.dataset_cfg.get('SHIFT_COOR', None) or [0, 0, 0], np.float32)
    return b


def head_for(cfg, ds):
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.model_cfg = cfg.MODEL.DENSE_HEAD
    head.class_names = list(cfg.CLASS_NAMES)
    head.class_names_each_head = [[c for c in names if c in cfg.CLASS_NAMES]
                                  for names in cfg.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD]
    head.point_cloud_range = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
    head.voxel_size = ds.voxel_size
    head.separate_head_cfg = cfg.MODEL.DENSE_HEAD.SEPARATE_HEAD_CFG
    head.build_losses()
    stride = cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
    fmap = (ds.grid_size[:2] // stride)[::-1]
    return head, fmap, stride


def targets(cfg, ds, dicts):
    head, fmap, stride = head_for(cfg, ds)
    batch = ds.collate_batch(dicts)
    gt = batch['gt_boxes']
    t = head.assign_targets(torch.from_numpy(gt).float(), feature_map_size=list(fmap))
    assert len(head.class_names_each_head) == 1, 'one head expected (the oracle recipe)'
    hm = t['heatmaps'][0]
    pos = {c: int((gt[:, :, -1] == k + 1).sum()) for k, c in enumerate(cfg.CLASS_NAMES)}
    peaks = {c: int(hm[:, j].eq(1).sum()) for j, c in enumerate(head.class_names_each_head[0])}
    mask = t['heatmap_masks'][0]
    return {'gt': gt, 'hm': hm, 'mask': mask, 'pos': pos, 'peaks': peaks, 'ign': int((gt[:, :, -1] < 0).sum()),
            'ign_px': 0 if mask is None else int((mask == 0).sum()), 'stride': stride, 'ds': ds}


def centre_px(tg, box):
    pcr = tg['ds'].point_cloud_range
    vs = tg['ds'].voxel_size
    cx = int((box[0] - pcr[0]) / vs[0] / tg['stride'])
    cy = int((box[1] - pcr[1]) / vs[1] / tg['stride'])
    return cy, cx


def report_targets(cfg, base_t, swap_t):
    print('  targets, base : positive-label boxes %s, peaks %s, ignore-label boxes %d, ignored px %d'
          % (base_t['pos'], base_t['peaks'], base_t['ign'], base_t['ign_px']))
    print('  targets, swap : positive-label boxes %s, peaks %s, ignore-label boxes %d, ignored px %d'
          % (swap_t['pos'], swap_t['peaks'], swap_t['ign'], swap_t['ign_px']))
    check(all(swap_t['peaks'][c] <= swap_t['pos'][c] for c in swap_t['pos']),
          'no heatmap peak without a positive-label box (an ignore box never becomes a positive)')
    # every -99 box: no peak at its centre pixel (unless a positive box shares the pixel) and inside the ignored footprint
    gt, hm = swap_t['gt'], swap_t['hm']
    for b in range(gt.shape[0]):
        positives = {centre_px(swap_t, g) for g in gt[b] if g[-1] > 0}
        for g in gt[b]:
            if g[-1] >= 0:
                continue
            cy, cx = centre_px(swap_t, g)
            if (cy, cx) in positives:
                continue
            check(bool((hm[b, :, cy, cx] < 1).all()), 'ignore box at (%.1f, %.1f): no peak at its centre' % (g[0], g[1]))
            check(swap_t['mask'] is not None and bool((swap_t['mask'][b][..., cy, cx] == 0).all()),
                  'ignore box at (%.1f, %.1f): centre pixel in the ignored footprint (all classes)' % (g[0], g[1]))


def match_rows(a, b, cols):
    """index into b for each row of a whose `cols` match a row of b exactly (-1 when none)."""
    key = {tuple(r): k for k, r in enumerate(np.asarray(b)[:, cols].tolist())}
    return np.array([key.get(tuple(r), -1) for r in np.asarray(a)[:, cols].tolist()])


def angle_about(xyz, pivot):
    return np.degrees(np.arctan2(xyz[:, 2] - pivot[2], xyz[:, 0] - pivot[0]))


def per_box_counts(points, boxes):
    return wfs.points_per_box(points, boxes) if len(boxes) else np.zeros(0, np.int64)


# ------------------------------------------------------------------------------------------------- arms
def dump_rs(cfg, ds, base_ds, b_items, s_items):
    pcr = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, np.float32)
    for i, (b, _), (s, _) in zip(FRAMES, b_items, s_items):
        info = ds.infos[i]
        raw = ds._raw_frame(info)
        counts = np.asarray(info['num_points_of_each_lidar'])
        ends = np.cumsum(counts)
        per_lidar = []
        for k in range(len(counts)):
            blk = raw[ends[k] - counts[k]:ends[k]]
            blk = blk[blk[:, 5] == -1]
            per_lidar.append(int(common_utils.mask_points_by_range(blk[:, :3], pcr).sum()))
        top = raw[:counts[0]]
        top = top[top[:, 5] == -1][:, :3]
        top = top[common_utils.mask_points_by_range(top, pcr)]
        same = len(top) == len(s['points']) and np.array_equal(
            top[np.lexsort(top.T[::-1])], s['points'][np.lexsort(s['points'][:, :3].T[::-1]), :3])
        print('  frame %s: points %d -> %d; in range by lidar (TOP, FRONT, SIDE_L, SIDE_R, REAR) %s'
              % (b['frame_id'], len(b['points']), len(s['points']), per_lidar))
        check(same, 'swap cloud == the TOP block of the raw file (NLZ-filtered, in range)')
        check(len(b['points']) == sum(per_lidar), 'oracle cloud == all five lidars')
        bb = b['gt_boxes']
        n_all = per_box_counts(b['points'], bb)
        n_top = per_box_counts(s['points'], bb)
        positive = bb[:, -1] > 0
        expect = bb[~positive | (n_top >= 1)]
        sb = s['gt_boxes']
        same_boxes = len(expect) == len(sb) and np.array_equal(expect[np.lexsort(expect.T[::-1])],
                                                               sb[np.lexsort(sb.T[::-1])])
        dropped = positive & (n_top == 0)
        print('    positive boxes %d -> %d: dropped %d (no point in the five-lidar cloud either: %d [confound]; '
              'side lidars only: %d)' % (positive.sum(), (sb[:, -1] > 0).sum(), dropped.sum(),
                                         (dropped & (n_all == 0)).sum(), (dropped & (n_all > 0)).sum()))
        check(same_boxes, 'swap boxes == oracle boxes minus the positives with no TOP point')
    # what the recount drops, over strided frames: boxes empty in the five-lidar cloud too (the C1 convention, also a
    # change against the oracle) against boxes only the side lidars see (the factor)
    idx = np.linspace(0, len(ds) - 1, 300).astype(int)
    car = list(cfg.CLASS_NAMES).index('Car') + 1
    tot, conf, side, tot_c, conf_c, side_c = 0, 0, 0, 0, 0, 0
    for i in idx:
        b, _ = item(base_ds, i)
        s, _ = item(ds, i)
        bb = b['gt_boxes']
        pos = bb[:, -1] > 0
        n_all, n_top = per_box_counts(b['points'], bb), per_box_counts(s['points'], bb)
        d = pos & (n_top == 0)
        c = bb[:, -1] == car
        tot += pos.sum(); conf += (d & (n_all == 0)).sum(); side += (d & (n_all > 0)).sum()
        tot_c += c.sum(); conf_c += (d & c & (n_all == 0)).sum(); side_c += (d & c & (n_all > 0)).sum()
    print('  %d strided frames: the recount drops %d of %d positives (%.1f%%): %d empty in the five-lidar cloud too '
          '(%.1f%%, confound), %d seen only by the side lidars (%.1f%%); Car: %d / %d of %d (%.1f%% / %.1f%%)'
          % (len(idx), conf + side, tot, 100.0 * (conf + side) / max(tot, 1), conf, 100.0 * conf / max(tot, 1), side,
             100.0 * side / max(tot, 1), conf_c, side_c, tot_c, 100.0 * conf_c / max(tot_c, 1),
             100.0 * side_c / max(tot_c, 1)))


def trained_class_boxes(ds, i, swap=None):
    """The oracle's positive boxes of frame i before augmentation (class names, stored count >= MIN_POINTS_OF_GT, box
    corners in POINT_CLOUD_RANGE), optionally pitched first: what the loader should hand the head."""
    from pcdet.utils import box_utils
    a = ds.infos[i]['annos']
    names = wfs.model_names(a['name'][a['name'] != 'unknown'], ds.map_ontology_dataset_to_model)
    keep = a['name'] != 'unknown'
    b = np.asarray(a['gt_boxes_lidar'][keep][:, :7], np.float32)
    ok = np.isin(names, ds.class_names) & (a['num_points_in_gt'][keep] >= ds.dataset_cfg.get('MIN_POINTS_OF_GT', 1))
    if swap is not None:
        b = wfs.apply_boxes(b, names, swap, ds.infos[i]['frame_id'])
    b = b + np.r_[np.asarray(ds.dataset_cfg.get('SHIFT_COOR', None) or [0, 0, 0], np.float32), np.zeros(4, np.float32)]
    return ok & box_utils.mask_boxes_outside_range_numpy(b, ds.point_cloud_range)


def dump_pitch(cfg, swap, ds, b_items, s_items):
    pivot = np.asarray(swap.PIVOT, np.float64) + np.asarray(cfg.DATA_CONFIG.get('SHIFT_COOR', None) or [0, 0, 0])
    for i, (b, _), (s, _) in zip(FRAMES, b_items, s_items):
        drawn = wfs.pitch_of(swap, b['frame_id'])
        bp, sp = b['points'][:, :3].astype(np.float64), s['points'][:, :3].astype(np.float64)
        yb, cb = np.unique(bp[:, 1], return_counts=True)
        ys, cs = np.unique(sp[:, 1], return_counts=True)
        uniq = np.intersect1d(yb[cb == 1], ys[cs == 1])
        ib = {y: k for k, y in enumerate(bp[:, 1])}
        is_ = {y: k for k, y in enumerate(sp[:, 1])}
        pb = bp[[ib[y] for y in uniq]]
        ps = sp[[is_[y] for y in uniq]]
        far = np.hypot(pb[:, 0] - pivot[0], pb[:, 2] - pivot[2]) > 2.0
        d_pts = (angle_about(ps[far], pivot) - angle_about(pb[far], pivot) + 180) % 360 - 180
        bb, sb = b['gt_boxes'], s['gt_boxes']
        m = match_rows(bb, sb, [3, 4, 5, 6, 7])
        ok = m >= 0
        cb_, cs_ = bb[ok, :3].astype(np.float64), sb[m[ok], :3].astype(np.float64)
        farb = np.hypot(cb_[:, 0] - pivot[0], cb_[:, 2] - pivot[2]) > 2.0
        d_box = (angle_about(cs_[farb], pivot) - angle_about(cb_[farb], pivot) + 180) % 360 - 180
        n_before = per_box_counts(b['points'], bb[ok])
        n_after = per_box_counts(s['points'], sb[m[ok]])
        emptied = int(((n_before > 0) & (n_after == 0) & (bb[ok, -1] > 0)).sum())
        print('  frame %s: drawn %.2f deg; implied from %d points: median %.4f (max |dev| %.4f); from %d boxes: '
              'median %.4f (max |dev| %.4f); boxes %d -> %d (matched %d); positives the pitch would empty: %d'
              % (b['frame_id'], drawn, far.sum(), np.median(d_pts), np.abs(d_pts - drawn).max(), farb.sum(),
                 np.median(d_box) if len(d_box) else np.nan, np.abs(d_box - drawn).max() if len(d_box) else np.nan,
                 len(bb), len(sb), ok.sum(), emptied))
        check(far.sum() > 1000 and abs(np.median(d_pts) - drawn) < 1e-3 and np.percentile(np.abs(d_pts - drawn), 99) < 0.01,
              'points rotated by the drawn angle about the pivot')
        check(farb.sum() > 0 and np.abs(d_box - drawn).max() < 1e-3, 'box centres rotated by the same angle')
        check(np.array_equal(bb[ok][:, 3:], sb[m[ok]][:, 3:]), 'boxes upright: size, yaw and label unchanged')
        n_base, n_swap = trained_class_boxes(ds, i).sum(), trained_class_boxes(ds, i, swap).sum()
        check(len(bb) == n_base and len(sb) == n_swap and ok.sum() == len(sb),
              'box set = the oracle\'s, pitched, through the 3D range mask (%d -> %d; boxes leaving z [-2, 4]: %d)'
              % (n_base, n_swap, n_base - ok.sum()))
    # how many trained boxes the pitch pushes out of the z band, over strided frames (infos only)
    idx = np.linspace(0, len(ds) - 1, 2000).astype(int)
    nb = ns = 0
    for i in idx:
        nb += int(trained_class_boxes(ds, i).sum()); ns += int(trained_class_boxes(ds, i, swap).sum())
    print('  %d strided frames: trained class boxes %d -> %d (%.2f%% leave POINT_CLOUD_RANGE: pitched out of z [-2, 4])'
          % (len(idx), nb, ns, 100.0 * (nb - ns) / max(nb, 1)))


def dump_dims(cfg, swap, b_items, s_items):
    scale = {c: float(f) for c, f in swap.GT_DIM_SCALE.items()}
    for i, (b, _), (s, _) in zip(FRAMES, b_items, s_items):
        bb, sb = b['gt_boxes'], s['gt_boxes']
        check(np.array_equal(b['points'], s['points']), 'frame %s: points identical' % b['frame_id'])
        if len(bb) != len(sb):
            check(False, 'frame %s: box count %d -> %d' % (b['frame_id'], len(bb), len(sb)))
            continue
        lab = bb[:, -1].astype(int)
        ratio = sb[:, 3:6] / bb[:, 3:6]
        bottom = (sb[:, 2] - sb[:, 5] / 2) - (bb[:, 2] - bb[:, 5] / 2)
        n_before = per_box_counts(b['points'], bb)
        n_after = per_box_counts(s['points'], sb)
        msg = []
        for k, c in enumerate(cfg.CLASS_NAMES):
            m = lab == k + 1
            if m.any():
                f = scale.get(c, 1.0)
                msg.append('%s x%.4f (n %d)' % (c, float(np.median(ratio[m])), m.sum()))
                check(np.allclose(ratio[m], f, atol=1e-4), 'frame %s: %s dims x %.4f' % (b['frame_id'], c, f))
        print('  frame %s: dims ratio %s; bottom shift max |%.2e| m; positives emptied %d, gained a point %d'
              % (b['frame_id'], ', '.join(msg), np.abs(bottom).max() if len(bottom) else 0.0,
                 ((n_before > 0) & (n_after == 0)).sum(), ((n_before == 0) & (n_after > 0)).sum()))
        check(np.abs(bottom).max() < 1e-4 if len(bottom) else True, 'bottom kept')
        check(np.array_equal(bb[:, [0, 1, 6, 7]], sb[:, [0, 1, 6, 7]]), 'centre x / y, yaw and labels unchanged')


def dump_ignore(cfg, swap, ds, base_ds, b_items, s_items, base_t, swap_t):
    cls = swap.GT_IGNORE_BY_RANGE.CLASS
    k_cls = list(cfg.CLASS_NAMES).index(cls) + 1
    for i, (b, _), (s, _) in zip(FRAMES, b_items, s_items):
        bl, sl = b['gt_boxes'][:, -1].astype(int), s['gt_boxes'][:, -1].astype(int)
        r = np.hypot(b['gt_boxes'][:, 0], b['gt_boxes'][:, 1])
        changed = bl != sl
        print('  frame %s (Sign before a class box: %s): %d boxes, %s -> -99: %d (ranges %s)'
              % (b['frame_id'], sign_first(raw_names(ds, i)), len(bl), cls, changed.sum(),
                 np.round(r[changed], 1).tolist()))
        check(np.array_equal(b['gt_boxes'][:, :-1], s['gt_boxes'][:, :-1]) and np.array_equal(b['points'], s['points']),
              'frame %s: boxes and points identical' % b['frame_id'])
        check(bool(np.all((bl[changed] == k_cls) & (sl[changed] == IGN))), 'only %s labels changed, to -99' % cls)
    # targets: the other classes' heatmaps identical; the class loses one peak per new -99 box
    n_new = sum(int(((b['gt_boxes'][:, -1] == k_cls) & (s['gt_boxes'][:, -1] == IGN)).sum())
                for (b, _), (s, _) in zip(b_items, s_items))
    for j, c in enumerate(cfg.CLASS_NAMES):
        if c != cls:
            check(torch.equal(base_t['hm'][:, j], swap_t['hm'][:, j]), '%s heatmap identical' % c)
    diff = base_t['peaks'][cls] - swap_t['peaks'][cls]
    slack = base_t['pos'][cls] - base_t['peaks'][cls]           # coincident centre pixels in the base
    check(abs(diff - n_new) <= slack, '%s peaks %d -> %d for %d new ignore boxes (slack %d)'
          % (cls, base_t['peaks'][cls], swap_t['peaks'][cls], n_new, slack))
    # labels over many frames: real KEEP changes only the class to -99; KEEP all 1.0 reproduces the oracle exactly
    keep1 = copy.deepcopy(cfg)
    keep1.DATA_CONFIG.TRAIN_FACTOR_SWAP.GT_IGNORE_BY_RANGE.KEEP = [1.0] * len(swap.GT_IGNORE_BY_RANGE.KEEP)
    k1 = build(keep1, training=True)
    idx = np.linspace(0, len(ds) - 1, LABEL_FRAMES).astype(int)
    n_sign_first = n_equal = n_only_cls = n_changed = n_total = 0
    for i in idx:
        b, _ = item(base_ds, i)
        s, _ = item(ds, i)
        e, _ = item(k1, i)
        sf = sign_first(raw_names(ds, i))
        n_sign_first += sf
        n_equal += bool(np.array_equal(b['gt_boxes'], e['gt_boxes']))
        bl, sl = b['gt_boxes'][:, -1].astype(int), s['gt_boxes'][:, -1].astype(int)
        ch = bl != sl
        n_only_cls += bool(np.array_equal(b['gt_boxes'][:, :-1], s['gt_boxes'][:, :-1]) and
                           np.all((bl[ch] == k_cls) & (sl[ch] == IGN)))
        n_changed += int(ch.sum()); n_total += int((bl == k_cls).sum())
    print('  %d strided frames (%d with a non-class box before a class box): KEEP=1.0 == oracle labels on %d; '
          'real KEEP changes only %s -> -99 on %d (%d of %d %s boxes, kept share %.3f)'
          % (len(idx), n_sign_first, n_equal, cls, n_only_cls, n_changed, n_total, cls,
             1 - n_changed / max(n_total, 1)))
    check(n_equal == len(idx), 'KEEP=1.0 reproduces the oracle labels on every frame')
    check(n_only_cls == len(idx), 'real KEEP changes only %s labels' % cls)
    check(n_sign_first > 0, 'the label frames include non-class boxes before class boxes')
    return k1


def dump_region(cfg, swap, ds, b_items, s_items):
    side, keep = swap.REGION_THIN.SIDE, float(swap.REGION_THIN.KEEP)
    margin = float(swap.BOX_MARGIN)
    for i, (b, _), (s, _) in zip(FRAMES, b_items, s_items):
        boxes = enlarge_boxes(loader_boxes(ds, i), margin)
        ib, is_ = in_any_box(b['points'], boxes), in_any_box(s['points'], boxes)
        rin = is_.sum() / max(ib.sum(), 1)
        rout = (~is_).sum() / max((~ib).sum(), 1)
        bb, sb = b['gt_boxes'], s['gt_boxes']
        emptied = (per_box_counts(b['points'], bb) > 0) & (per_box_counts(s['points'], bb) == 0) & (bb[:, -1] > 0)
        expect = bb[~emptied]
        same_boxes = len(expect) == len(sb) and np.array_equal(expect, sb)
        print('  frame %s: points %d -> %d; in-box %d -> %d (x%.3f), out-of-box %d -> %d (x%.3f); boxes %d -> %d '
              '(emptied by the arm %d)' % (b['frame_id'], len(b['points']), len(s['points']), ib.sum(), is_.sum(), rin,
                                           (~ib).sum(), (~is_).sum(), rout, len(bb), len(sb), emptied.sum()))
        thinned, kept = (rin, rout) if side == 'foreground' else (rout, rin)
        check(abs(thinned - keep) < 0.03 and kept == 1.0, '%s points x%.2f, the other side untouched' % (side, keep))
        check(same_boxes, 'boxes == the oracle\'s minus the positives the arm emptied')
        if side == 'background':
            check(emptied.sum() == 0, 'background thinning empties no box')


# ------------------------------------------------------------------------------------------------- driver
def eval_inert(cfg, base_cfg, base_eval, base_tar_eval):
    check(cfg.DATA_CONFIG_TAR == base_cfg.DATA_CONFIG_TAR, 'DATA_CONFIG_TAR identical to the oracle\'s')
    tar = build(cfg, training=False, block='DATA_CONFIG_TAR')
    e, _ = item(tar, 0)
    be, _ = base_tar_eval
    check(np.array_equal(e['points'], be['points']) and np.array_equal(e['gt_boxes'], be['gt_boxes']),
          'evaluation (DATA_CONFIG_TAR) frame 0 identical')
    del tar
    ev = build(cfg, training=False)
    e, _ = item(ev, 0)
    be, _ = base_eval
    same = np.array_equal(e['points'], be['points'])
    if cfg.DATA_CONFIG.get('TRAIN_FACTOR_SWAP', None) is not None:
        check(same, 'the row\'s DATA_CONFIG built for evaluation: frame 0 identical (key training-gated)')
    else:
        print('    [info] the row\'s DATA_CONFIG built for evaluation differs: %s (LIDAR_INDICES is not training-gated; '
              'evaluation reads DATA_CONFIG_TAR)' % (not same))


def dump(cfg_path, base_cfg, base_ds, b_items, base_t, base_eval, base_tar_eval):
    cfg = cfg_from_yaml_file(cfg_path, EasyDict())
    print('=' * 110)
    swap = cfg.DATA_CONFIG.get('TRAIN_FACTOR_SWAP', None)
    print(cfg_path, '\n  TRAIN_FACTOR_SWAP:', None if swap is None else dict(swap),
          '| LIDAR_INDICES:', cfg.DATA_CONFIG.get('LIDAR_INDICES', None))
    ds = build(cfg, training=True)
    s_items = [item(ds, i) for i in FRAMES]
    # both timed warm (the files were read once already), 3 passes each, augmentation off
    tb = np.median([item(base_ds, i)[1] for _ in range(3) for i in FRAMES])
    ts = np.median([item(ds, i)[1] for _ in range(3) for i in FRAMES])
    print('  loader time per frame (warm, augmentation off, median of %d): oracle %.3f s, swap %.3f s (x%.2f)'
          % (3 * len(FRAMES), tb, ts, ts / tb))
    swap_t = targets(cfg, ds, [s for s, _ in s_items])
    report_targets(cfg, base_t, swap_t)
    if swap is None:
        dump_rs(cfg, ds, base_ds, b_items, s_items)
    else:
        arm = wfs.arm_of(swap)
        if arm == 'PITCH_DEG':
            dump_pitch(cfg, swap, ds, b_items, s_items)
        elif arm == 'GT_DIM_SCALE':
            dump_dims(cfg, swap, b_items, s_items)
        elif arm == 'GT_IGNORE_BY_RANGE':
            dump_ignore(cfg, swap, ds, base_ds, b_items, s_items, base_t, swap_t)
        elif arm == 'REGION_THIN':
            dump_region(cfg, swap, ds, b_items, s_items)
    eval_inert(cfg, base_cfg, base_eval, base_tar_eval)
    del ds


if __name__ == '__main__':
    base_cfg = cfg_from_yaml_file(BASE, EasyDict())
    base_ds = build(base_cfg, training=True)
    b_items = [item(base_ds, i) for i in FRAMES]
    base_t = targets(base_cfg, base_ds, [b for b, _ in b_items])
    base_eval = item(build(base_cfg, training=False), 0)
    base_tar_eval = item(build(base_cfg, training=False, block='DATA_CONFIG_TAR'), 0)
    print('oracle frames %s; non-class box before a class box: %s'
          % ([b['frame_id'] for b, _ in b_items], [sign_first(raw_names(base_ds, i)) for i in FRAMES]))
    for p in sys.argv[1:]:
        dump(p, base_cfg, base_ds, b_items, base_t, base_eval, base_tar_eval)
    print('=' * 110)
    if FAILS:
        print('FAILED CHECKS (%d):' % len(FAILS))
        for f in FAILS:
            print('  ' + f)
        raise SystemExit(1)
    print('ALL DUMPS PASSED')
