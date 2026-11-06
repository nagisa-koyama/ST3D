"""Target dump and identity check for the FACTOR-SWAP rows on PandaSet Pandar64, Lyft-64 and Lyft-40 (experiments_md
20261003_05 §5 item 34; bug_prevention_checklist rules 4 and 8). The pattern of factorswap_n2w_target_dump.py.

For one target, every new training row of its directory (the r* rows, Lyft-64's d-density, and the new oracles O_L64 /
O_L40 against their parents) goes through the TRAINING loader on strided TRAIN frames, beside the target oracle's
loader on the same frames, and the script checks what the row changed against what it should change:
  - targets (one collated batch, augmentation off): per-class positive-label boxes, CenterHead heatmap peaks (never more
    than positives), ignore-label boxes, ignored footprint pixels, median points per positive box;
  - PITCH_DEG: implied pitch from points (matched by their unchanged y) and from box centres, equal to the frame's drawn
    angle about the pivot; boxes upright; the box set = the oracle's; the fitted forward ground slope changes by the
    drawn angle with the named sign (the ground rises ahead);
  - GT_DIM_SCALE: per-class dims ratio, bottom kept, centre x / y, yaw and labels unchanged, points identical, box set =
    the oracle's (positives the scale leaves empty are reported, kept by design);
  - GT_IGNORE_BY_RANGE: only Car turns into -99; boxes and points identical; the other classes' heatmaps identical; over
    LABEL_FRAMES strided frames, the real KEEP changes only Car labels (frames with a non-class box before a class box
    counted: the order that misaligned gt_classes before ST3D 12b5147);
  - REGION_THIN background: points inside the enlarged boxes identical (same order), outside x KEEP, boxes identical;
  - D (Lyft-64 density): points x the expected kept share, boxes = the oracle's minus the positives the thinning emptied;
  - the IDENTITY check (rule 8): the row's arm at its identity setting (scale 1, KEEP 1, keep 1, angle 0; D: source
    histogram as target) against the oracle over IDENTITY_FRAMES strided TRAIN frames WITH augmentation, each frame
    seeded alike: points and labelled boxes compared exactly;
  - the key inert at evaluation: DATA_CONFIG_TAR identical to the oracle's, its first frame identical, and the row's
    training block built in eval mode identical (TRAIN_FACTOR_SWAP is training-gated);
  - loader time per frame (warm).
Any failed check raises. ANALYSIS (target TRAIN labels are read); CPU; run as a Slurm CPU job from ST3D/tools:

    python analysis/factorswap_pl_target_dump.py --target pandaset|lyft64|lyft40
"""
import argparse
import contextlib
import copy
import glob
import io
import json
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
from pcdet.datasets import loader_factor_swap as lfs  # noqa: E402
from pcdet.datasets.osens2_utils import enlarge_boxes, fit_plane, in_any_box, lowest_per_cell  # noqa: E402
from pcdet.datasets.waymo import waymo_factor_swap as wfs  # noqa: E402
from pcdet.models.dense_heads.center_head import CenterHead  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

A = 'cfgs/da-ieee-access-analysis/'
TARGETS = {   # target: (oracle, training block path, parent of a NEW oracle or None, frame-key function)
    'pandaset': ('cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml', ['DATA_CONFIG'], None),
    'lyft64': (A + 'factorswap-lyft64/centerpoint-factorswap-lyft64-oracle.yaml', ['DATA_CONFIG'],
               'cfgs/da-ieee-access/centerpoint-sourceonly-lyft2lyft.yaml'),
    'lyft40': (A + 'factorswap-lyft40/centerpoint-factorswap-lyft40-oracle.yaml', ['DATA_CONFIGS', 'LYFT_40BEAM'],
               'cfgs/da-ieee-access/centerpoint-global-recount-lyft40-lyft2nuscenes.yaml'),
}
N_FRAMES = 6
IGN = common_utils.IGNORE_CLASS_LABEL
logger = common_utils.create_logger()
FAILS = []


def check(ok, msg):
    print('    [%s] %s' % ('ok' if ok else 'FAIL', msg), flush=True)
    if not ok:
        FAILS.append(msg)


def load(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return cfg_from_yaml_file(path, EasyDict())


def block(cfg, path):
    b = cfg
    for k in path:
        b = b[k]
    return b


def train_block_of(cfg):
    return cfg.DATA_CONFIGS[list(cfg.DATA_CONFIGS)[0]] if cfg.get('DATA_CONFIG', None) is None else cfg.DATA_CONFIG


def build(cfg, dcfg, training=True, aug_off=True):
    with contextlib.redirect_stdout(io.StringIO()):
        ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                    logger=logger, training=training, model_ontology=cfg.get('ONTOLOGY'))
    if training and aug_off:
        ds.data_augmentor.data_augmentor_queue = []
    return ds


def item(ds, i, seed=0):
    np.random.seed(seed)
    t = time.time()
    d = ds[i]
    return d, time.time() - t


def raw_input(ds, i):
    """The loader's input_dict for frame i, before prepare_data (SHIFT_COOR applied, the swap of ds applied)."""
    keep = ds.prepare_data
    ds.prepare_data = lambda data_dict: data_dict
    try:
        d = ds[i]
    finally:
        ds.prepare_data = keep
    return d


def frame_key(ds, i):
    if hasattr(ds, 'pandaset_infos'):
        info = ds.pandaset_infos[i]
        return '%s/%02d' % (info['sequence'], int(info['frame_idx']))
    return Path(ds.infos[i]['lidar_path']).stem


def nonclass_first(names_model, class_names):
    seen_other = False
    for n in names_model:
        if n not in class_names:
            seen_other = True
        elif seen_other:
            return True
    return False


def head_for(cfg, ds):
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.model_cfg = cfg.MODEL.DENSE_HEAD
    head.class_names = list(cfg.CLASS_NAMES)
    head.class_names_each_head = [[c for c in names if c in cfg.CLASS_NAMES]
                                  for names in cfg.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD]
    head.point_cloud_range = np.asarray(ds.point_cloud_range, dtype=np.float32)
    head.voxel_size = ds.voxel_size
    head.separate_head_cfg = cfg.MODEL.DENSE_HEAD.SEPARATE_HEAD_CFG
    head.build_losses()
    stride = cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
    return head, list((ds.grid_size[:2] // stride)[::-1]), stride


def targets(cfg, ds, dicts):
    head, fmap, stride = head_for(cfg, ds)
    batch = ds.collate_batch(dicts)
    gt = batch['gt_boxes']
    t = head.assign_targets(torch.from_numpy(gt).float(), feature_map_size=fmap)
    hm, mask = t['heatmaps'][0], t['heatmap_masks'][0]
    pos = {c: int((gt[:, :, -1] == k + 1).sum()) for k, c in enumerate(cfg.CLASS_NAMES)}
    peaks = {c: int(hm[:, j].eq(1).sum()) for j, c in enumerate(head.class_names_each_head[0])}
    ppb = {}
    for k, c in enumerate(cfg.CLASS_NAMES):
        n = [lfs.oracle_counts(d['points'], d['gt_boxes'][d['gt_boxes'][:, -1] == k + 1]) for d in dicts]
        n = np.concatenate(n) if n else np.zeros(0)
        ppb[c] = float(np.median(n)) if len(n) else float('nan')
    return {'gt': gt, 'hm': hm, 'mask': mask, 'pos': pos, 'peaks': peaks, 'ign': int((gt[:, :, -1] < 0).sum()),
            'ign_px': 0 if mask is None else int((mask == 0).sum()), 'stride': stride, 'ds': ds, 'ppb': ppb}


def report_targets(name, t):
    print('  targets %-6s: positive-label boxes %s, peaks %s, ignore-label boxes %d, ignored px %d, median points per '
          'positive box %s' % (name, t['pos'], t['peaks'], t['ign'], t['ign_px'],
                               {c: round(v, 1) for c, v in t['ppb'].items()}))
    check(all(t['peaks'][c] <= t['pos'][c] for c in t['pos']), 'no heatmap peak without a positive-label box')


def centre_px(t, box):
    pcr, vs = t['ds'].point_cloud_range, t['ds'].voxel_size
    return int((box[1] - pcr[1]) / vs[1] / t['stride']), int((box[0] - pcr[0]) / vs[0] / t['stride'])


def check_ignore_targets(t):
    """Every -99 box whose centre lies on the heatmap: no peak there, the pixel ignored in all classes. A box kept by the
    3D range mask can have its centre beyond the grid (a corner inside): counted, not indexed (a negative index wraps)."""
    gt, hm = t['gt'], t['hm']
    H, W = hm.shape[-2:]
    off = 0
    for b in range(gt.shape[0]):
        positives = {centre_px(t, g) for g in gt[b] if g[-1] > 0}
        for g in gt[b]:
            if g[-1] >= 0:
                continue
            cy, cx = centre_px(t, g)
            if not (0 <= cy < H and 0 <= cx < W):
                off += 1
                continue
            if (cy, cx) in positives:
                continue
            check(bool((hm[b, :, cy, cx] < 1).all()) and t['mask'] is not None
                  and bool((t['mask'][b][..., cy, cx] == 0).all()),
                  'ignore box at (%.1f, %.1f): no peak at its centre, centre pixel ignored in all classes' % (g[0], g[1]))
    print('    ignore boxes with the centre beyond the heatmap (a corner inside POINT_CLOUD_RANGE): %d' % off)


def forward_slope(points):
    fit = fit_plane(lowest_per_cell(np.asarray(points[:, :3], dtype=np.float64)))
    return None if fit is None else float(np.degrees(np.arctan(fit[0][0])))


# ------------------------------------------------------------------------------------------------- arms
def dump_pitch(swap, ds, frames, b_items, s_items):
    pivot = np.asarray(swap.PIVOT, np.float64)
    for i, (b, _), (s, _) in zip(frames, b_items, s_items):
        drawn = wfs.pitch_of(swap, frame_key(ds, i))
        bp, sp = b['points'][:, :3].astype(np.float64), s['points'][:, :3].astype(np.float64)
        yb, cb = np.unique(bp[:, 1], return_counts=True)
        ys, cs = np.unique(sp[:, 1], return_counts=True)
        uniq = np.intersect1d(yb[cb == 1], ys[cs == 1])
        ib = {y: k for k, y in enumerate(bp[:, 1])}
        is_ = {y: k for k, y in enumerate(sp[:, 1])}
        pb, ps = bp[[ib[y] for y in uniq]], sp[[is_[y] for y in uniq]]
        far = np.hypot(pb[:, 0] - pivot[0], pb[:, 2] - pivot[2]) > 2.0
        ang = lambda p: np.degrees(np.arctan2(p[:, 2] - pivot[2], p[:, 0] - pivot[0]))   # noqa: E731
        d_pts = (ang(ps[far]) - ang(pb[far]) + 180) % 360 - 180
        bb, sb = b['gt_boxes'], s['gt_boxes']
        # order-preserving match (the loader never reorders boxes; several boxes can share dims / yaw / label)
        m, j = -np.ones(len(bb), dtype=int), 0
        for k in range(len(bb)):
            if j < len(sb) and np.array_equal(bb[k, 3:], sb[j, 3:]):
                m[k], j = j, j + 1
        ok = m >= 0
        # a base box without a partner must be one the pitch lifted or lowered out of POINT_CLOUD_RANGE (3D corner test)
        from pcdet.utils import box_utils
        lost = bb[~ok]
        lp = np.array(lost, copy=True)
        if len(lp):
            lp[:, :3] = wfs.pitch_about(lp[:, :3], drawn, pivot)
        left_range = bool(len(lp) == 0 or not box_utils.mask_boxes_outside_range_numpy(lp[:, :7],
                                                                                        ds.point_cloud_range).any())
        d_box = (ang(sb[m[ok], :3].astype(np.float64)) - ang(bb[ok, :3].astype(np.float64)) + 180) % 360 - 180
        farb = np.hypot(bb[ok, 0] - pivot[0], bb[ok, 2] - pivot[2]) > 2.0
        sl_b, sl_s = forward_slope(b['points']), forward_slope(s['points'])
        emptied = int(((lfs.oracle_counts(b['points'], bb[ok]) > 0) &
                       (lfs.oracle_counts(s['points'], sb[m[ok]]) == 0)).sum())
        print('  frame %s: drawn %.2f deg; from %d points median %.4f (max |dev| %.4f); boxes %d -> %d (matched %d, '
              'pitched out of range %d); forward ground slope %s -> %s; positives left empty by the pitch (kept) %d'
              % (frame_key(ds, i), drawn, far.sum(), np.median(d_pts), np.abs(d_pts - drawn).max(), len(bb), len(sb),
                 ok.sum(), (~ok).sum(), None if sl_b is None else round(sl_b, 2),
                 None if sl_s is None else round(sl_s, 2), emptied))
        check(far.sum() > 1000 and abs(np.median(d_pts) - drawn) < 1e-3 and np.percentile(np.abs(d_pts - drawn), 99) < 0.01,
              'points rotated by the drawn angle about the pivot')
        check(ok.sum() == len(sb) and left_range,
              'box set = the oracle\'s (sizes / yaw / labels unchanged), minus boxes pitched out of POINT_CLOUD_RANGE')
        if farb.any():
            check(np.abs(d_box[farb] - drawn).max() < 1e-3, 'box centres rotated by the same angle (max |dev| %.2e)'
                  % np.abs(d_box[farb] - drawn).max())
        check(sl_b is not None and sl_s is not None and abs((sl_s - sl_b) - drawn) < 0.3,
              'the forward ground slope changes by +%.2f (the ground rises ahead)' % drawn)


def dump_dims(cfg, swap, ds, frames, b_items, s_items):
    scale = {c: float(f) for c, f in swap.GT_DIM_SCALE.items()}
    for i, (b, _), (s, _) in zip(frames, b_items, s_items):
        bb, sb = b['gt_boxes'], s['gt_boxes']
        check(np.array_equal(b['points'], s['points']), 'frame %s: points identical' % frame_key(ds, i))
        if len(bb) != len(sb):
            check(False, 'frame %s: box count %d -> %d (the oracle\'s set must be kept)' % (frame_key(ds, i), len(bb),
                                                                                           len(sb)))
            continue
        lab = bb[:, -1].astype(int)
        ratio = sb[:, 3:6] / bb[:, 3:6]
        bottom = (sb[:, 2] - sb[:, 5] / 2) - (bb[:, 2] - bb[:, 5] / 2)
        n_b, n_s = lfs.oracle_counts(b['points'], bb), lfs.oracle_counts(s['points'], sb)
        msg = []
        for k, c in enumerate(cfg.CLASS_NAMES):
            m = lab == k + 1
            if m.any():
                f = scale.get(c, 1.0)
                msg.append('%s x%.4f (n %d)' % (c, float(np.median(ratio[m])), m.sum()))
                check(np.allclose(ratio[m], f, atol=1e-4), 'frame %s: %s dims x %.4f' % (frame_key(ds, i), c, f))
        print('  frame %s: dims ratio %s; bottom shift max |%.2e| m; positives emptied (kept) %d, gained a point %d'
              % (frame_key(ds, i), ', '.join(msg), np.abs(bottom).max() if len(bottom) else 0.0,
                 ((n_b > 0) & (n_s == 0)).sum(), ((n_b == 0) & (n_s > 0)).sum()))
        check(np.abs(bottom).max() < 1e-4 if len(bottom) else True, 'bottom kept')
        check(np.array_equal(bb[:, [0, 1, 6, 7]], sb[:, [0, 1, 6, 7]]), 'centre x / y, yaw and labels unchanged')


def dump_ignore(cfg, swap, ds, base_ds, frames, b_items, s_items, base_t, swap_t, label_frames):
    k_car = list(cfg.CLASS_NAMES).index('Car') + 1
    recount = any(p.NAME == 'drop_empty_gt_boxes' for p in ds.dataset_cfg.DATA_PROCESSOR)
    for i, (b, _), (s, _) in zip(frames, b_items, s_items):
        sb, extra = split_kept_empty_ignore(s, recount)
        bl, sl = b['gt_boxes'][:, -1].astype(int), sb[:, -1].astype(int)
        if len(bl) != len(sl):
            check(False, 'frame %s: %d oracle boxes against %d row boxes (after the empty ignore boxes)'
                  % (frame_key(ds, i), len(bl), len(sl)))
            continue
        r = np.hypot(b['gt_boxes'][:, 0], b['gt_boxes'][:, 1])
        ch = bl != sl
        print('  frame %s: %d boxes, Car -> -99: %d (ranges %s); empty -99 boxes the oracle\'s recount drops as '
              'positives, kept as ignore regions: %d' % (frame_key(ds, i), len(bl), ch.sum(), np.round(r[ch], 1).tolist(),
                                                        extra))
        check(np.array_equal(b['gt_boxes'][:, :-1], sb[:, :-1]) and np.array_equal(b['points'], s['points']),
              'frame %s: boxes and points identical' % frame_key(ds, i))
        check(bool(np.all((bl[ch] == k_car) & (sl[ch] == IGN))), 'only Car labels changed, to -99')
    for j, c in enumerate(cfg.CLASS_NAMES):
        if c != 'Car':
            check(torch.equal(base_t['hm'][:, j], swap_t['hm'][:, j]), '%s heatmap identical' % c)
    check_ignore_targets(swap_t)
    idx = np.linspace(0, len(ds) - 1, label_frames).astype(int)
    n_nf = n_only = n_ch = n_tot = n_extra = n_px = 0
    for i in idx:
        raw = raw_input(base_ds, i)
        n_nf += nonclass_first(wfs.model_names(raw['gt_names'], base_ds.map_ontology_dataset_to_model), cfg.CLASS_NAMES)
        b, _ = item(base_ds, i)
        s, _ = item(ds, i)
        sb, extra = split_kept_empty_ignore(s, recount)
        n_extra += extra
        bl, sl = b['gt_boxes'][:, -1].astype(int), sb[:, -1].astype(int)
        if len(bl) != len(sl):
            continue
        ch = bl != sl
        n_only += bool(np.array_equal(b['gt_boxes'][:, :-1], sb[:, :-1]) and np.all((bl[ch] == k_car) & (sl[ch] == IGN)))
        n_ch += int(ch.sum()); n_tot += int((bl == k_car).sum())
    print('  %d strided frames (%d with a non-class box before a class box): the real KEEP changes only Car -> -99 on %d; '
          '%d of %d Car boxes ignored (kept share %.3f); empty -99 boxes kept as ignore regions where the oracle\'s '
          'processor recount drops the positive: %d (%.2f per frame)'
          % (len(idx), n_nf, n_only, n_ch, n_tot, 1 - n_ch / max(n_tot, 1), n_extra, n_extra / len(idx)))
    check(n_only == len(idx), 'the real KEEP changes only Car labels on every frame')
    check(n_nf > 0, 'the label frames include non-class boxes before class boxes')


def split_kept_empty_ignore(s, recount):
    """(the row's boxes without the -99 boxes holding no point, how many those are). drop_empty_gt_boxes (Lyft-40's
    processor recount) keeps a -99 box whatever it holds but drops an empty POSITIVE: a Car the range crop emptied is
    dropped by the oracle and, once ignored by R7, kept as a zero-weight region. Only the oracle's dropped boxes can be
    empty here (a box the oracle keeps holds >= 1 point, and the points are identical)."""
    sb = s['gt_boxes']
    if len(sb) == 0 or not recount:
        return sb, 0
    empty = (sb[:, -1] == IGN) & (lfs.oracle_counts(s['points'], sb) == 0)
    return sb[~empty], int(empty.sum())


def dump_region(swap, ds, frames, b_items, s_items):
    keep, margin = float(swap.REGION_THIN.KEEP), float(swap.BOX_MARGIN)
    assert swap.REGION_THIN.SIDE == 'background'
    for i, (b, _), (s, _) in zip(frames, b_items, s_items):
        raw = raw_input(ds, i)          # every box the loader keeps (all classes), the boxes the arm used
        boxes = enlarge_boxes(np.asarray(raw['gt_boxes'])[:, :7], margin)
        ib, is_ = in_any_box(b['points'], boxes), in_any_box(s['points'], boxes)
        rout = (~is_).sum() / max((~ib).sum(), 1)
        print('  frame %s: points %d -> %d; in-box %d -> %d, out-of-box %d -> %d (x%.3f); boxes %d -> %d'
              % (frame_key(ds, i), len(b['points']), len(s['points']), ib.sum(), is_.sum(), (~ib).sum(), (~is_).sum(),
                 rout, len(b['gt_boxes']), len(s['gt_boxes'])))
        check(abs(rout - keep) < 0.03, 'background points x%.2f' % keep)
        # the processor shuffles: compare the in-box points as a set
        pin_b, pin_s = b['points'][ib], s['points'][is_]
        check(len(pin_b) == len(pin_s) and np.array_equal(pin_b[np.lexsort(pin_b.T[::-1])],
                                                          pin_s[np.lexsort(pin_s.T[::-1])]),
              'points inside the enlarged boxes identical')
        check(np.array_equal(b['gt_boxes'], s['gt_boxes']), 'boxes and labels identical')


def dump_density(ds, frames, b_items, s_items, hist_json):
    h = json.load(open(hist_json))
    for i, (b, _), (s, _) in zip(frames, b_items, s_items):
        bb, sb = b['gt_boxes'], s['gt_boxes']
        n_after = lfs.oracle_counts(s['points'], bb)
        positive = bb[:, -1] > 0
        expect = bb[~positive | (n_after >= 1)]
        same = len(expect) == len(sb) and np.array_equal(expect[np.lexsort(expect.T[::-1])], sb[np.lexsort(sb.T[::-1])])
        print('  frame %s: points %d -> %d (x%.3f, expected share %.3f); boxes %d -> %d (positives emptied %d); median '
              'points per positive Car box %s -> %s'
              % (frame_key(ds, i), len(b['points']), len(s['points']), len(s['points']) / max(len(b['points']), 1),
                 h['expected_kept_share'], len(bb), len(sb), int((positive & (n_after == 0)).sum()),
                 np.median(lfs.oracle_counts(b['points'], bb[bb[:, -1] == 1])) if (bb[:, -1] == 1).any() else None,
                 np.median(n_after[bb[:, -1] == 1]) if (bb[:, -1] == 1).any() else None))
        check(abs(len(s['points']) / max(len(b['points']), 1) - h['expected_kept_share']) < 0.08,
              'points kept near the expected share')
        check(same, 'boxes = the oracle\'s minus the positives the thinning emptied')


# ------------------------------------------------------------------------------------------------- identity (rule 8)
def identity_cfg(cfg, path):
    """The row's arm at its identity setting (None when the row has no identity form)."""
    c = copy.deepcopy(cfg)
    dc = block(c, path)
    swap = dc.get('TRAIN_FACTOR_SWAP', None)
    if swap is not None:
        arm = lfs.wfs.arm_of(swap)
        if arm == 'PITCH_DEG':
            swap.PITCH_DEG = 0.0
            swap.pop('PITCH_PROB', None)
        elif arm == 'GT_DIM_SCALE':
            swap.GT_DIM_SCALE = {k: 1.0 for k in swap.GT_DIM_SCALE}
        elif arm == 'GT_IGNORE_BY_RANGE':
            swap.GT_IGNORE_BY_RANGE.KEEP = [1.0] * len(swap.GT_IGNORE_BY_RANGE.KEEP)
        elif arm == 'REGION_THIN':
            swap.REGION_THIN.KEEP = 1.0
        return c, 'TRAIN_FACTOR_SWAP %s at identity' % arm
    if dc.get('HIST_DIST_LYFT_PATH', None):
        dc.HIST_DIST_NUSCENES_PATH = dc.HIST_DIST_LYFT_PATH
        return c, 'D with the source histogram as target (rate 1 in every bin)'
    return None, None


def identity_check(cfg, path, base_cfg, base_path, n_frames):
    icfg, what = identity_cfg(cfg, path)
    if icfg is None:
        return
    ids = build(icfg, block(icfg, path), aug_off=False)
    base = build(base_cfg, block(base_cfg, base_path), aug_off=False)
    idx = np.linspace(0, len(base) - 1, n_frames).astype(int)
    n_same = n_extra = 0
    for i in idx:
        b, _ = item(base, i, seed=int(i))
        s, _ = item(ids, i, seed=int(i))
        same_pts = np.array_equal(b['points'], s['points'])
        same_box = np.array_equal(b['gt_boxes'], s['gt_boxes'])
        if not same_box and what.startswith('D') and same_pts:
            # the recount right after the sampler may drop a positive the range crop emptied; nothing else may differ
            bb = b['gt_boxes']
            keep = (bb[:, -1] <= 0) | (lfs.oracle_counts(b['points'], bb) >= 1)
            if np.array_equal(bb[keep], s['gt_boxes']):
                n_extra += int((~keep).sum())
                same_box = True
        n_same += same_pts and same_box
    print('  IDENTITY (%s, augmentation ON, each frame seeded alike): %d of %d strided TRAIN frames identical to the '
          'oracle in points AND labelled boxes%s' % (what, n_same, len(idx),
                                                    '' if not n_extra else ' (%d boxes emptied by the range crop dropped '
                                                                           'by the recount)' % n_extra))
    check(n_same == len(idx), 'identity setting reproduces the oracle exactly (%s)' % what)


# ------------------------------------------------------------------------------------------------- driver
def eval_inert(cfg, path, base_cfg, base_path, base_tar0, base_eval0):
    check(cfg.DATA_CONFIG_TAR == base_cfg.DATA_CONFIG_TAR, 'DATA_CONFIG_TAR identical to the oracle\'s')
    e, _ = item(build(cfg, cfg.DATA_CONFIG_TAR, training=False), 0)
    check(np.array_equal(e['points'], base_tar0['points']) and np.array_equal(e['gt_boxes'], base_tar0['gt_boxes']),
          'evaluation (DATA_CONFIG_TAR) frame 0 identical')
    if block(cfg, path).get('TRAIN_FACTOR_SWAP', None) is not None:
        e, _ = item(build(cfg, block(cfg, path), training=False), 0)
        check(np.array_equal(e['points'], base_eval0['points']),
              'the row\'s training block built for evaluation: frame 0 identical (key training-gated)')


def dump_row(row_path, cfg_path, path, base, args):
    base_cfg, base_path, base_ds, frames, b_items, base_t, base_tar0, base_eval0 = base
    cfg = load(row_path)
    print('=' * 118)
    swap = block(cfg, path).get('TRAIN_FACTOR_SWAP', None)
    print(row_path, '\n  TRAIN_FACTOR_SWAP:', None if swap is None else json.loads(json.dumps(swap)))
    ds = build(cfg, block(cfg, path))
    s_items = [item(ds, i) for i in frames]
    tb = np.median([item(base_ds, i)[1] for _ in range(2) for i in frames])
    ts = np.median([item(ds, i)[1] for _ in range(2) for i in frames])
    print('  loader time per frame (warm, augmentation off): oracle %.3f s, row %.3f s (x%.2f)' % (tb, ts, ts / tb))
    swap_t = targets(cfg, ds, [s for s, _ in s_items])
    report_targets('base', base_t)
    report_targets('row', swap_t)
    if swap is not None:
        arm = lfs.wfs.arm_of(swap)
        if arm == 'PITCH_DEG':
            dump_pitch(swap, ds, frames, b_items, s_items)
        elif arm == 'GT_DIM_SCALE':
            dump_dims(cfg, swap, ds, frames, b_items, s_items)
        elif arm == 'GT_IGNORE_BY_RANGE':
            dump_ignore(cfg, swap, ds, base_ds, frames, b_items, s_items, base_t, swap_t, args.label_frames)
        elif arm == 'REGION_THIN':
            dump_region(swap, ds, frames, b_items, s_items)
    else:
        dump_density(ds, frames, b_items, s_items, '/home/koyama/data/factorswap_lyft/hist_lyft64_to_nuscenes.json')
    eval_inert(cfg, path, base_cfg, base_path, base_tar0, base_eval0)
    identity_check(cfg, path, base_cfg, base_path, args.identity_frames)


def dump_new_oracle(oracle_path, path, parent_path, frames):
    """O_L64 / O_L40 against the recipe they come from: same frames, same boxes up to SHIFT_COOR (and, for O_L40, the
    points the parent's density sampler would have thinned). Reported, with the targets of one batch."""
    cfg, par = load(oracle_path), load(parent_path)
    print('=' * 118)
    print('NEW ORACLE %s against %s' % (oracle_path, parent_path))
    ds = build(cfg, block(cfg, path))
    pds = build(par, train_block_of(par))
    if parent_path.endswith('lyft2lyft.yaml'):
        stems = {Path(i['lidar_path']).stem: k for k, i in enumerate(pds.infos)}
        check(all(Path(i['lidar_path']).name.split('_')[0] in ('host-a101', 'host-a102') for i in ds.infos)
              and len(ds.infos) == 3402, 'O_L64 trains on the 3,402 64-beam TRAIN frames only')
        for i in frames:
            d, _ = item(ds, i)
            p, _ = item(pds, stems[Path(ds.infos[i]['lidar_path']).stem])
            dz = float(cfg.DATA_CONFIG.SHIFT_COOR[2] - par.DATA_CONFIG.SHIFT_COOR[2])
            same = len(d['gt_boxes']) == len(p['gt_boxes']) and np.allclose(
                d['gt_boxes'][:, :3] - p['gt_boxes'][:, :3], [0, 0, dz], atol=1e-4) and np.array_equal(
                d['gt_boxes'][:, 3:], p['gt_boxes'][:, 3:])
            print('  frame %s: boxes %d vs lyft2lyft %d; points %d vs %d' % (frame_key(ds, i), len(d['gt_boxes']),
                                                                         len(p['gt_boxes']), len(d['points']),
                                                                         len(p['points'])))
            check(same, 'same boxes and labels as the lyft2lyft loader, z shifted by %.2f' % dz)
    else:
        check(len(ds) == len(pds) == 13926, 'O_L40 trains on the 13,926 kept 40-beam frames, as 30313')
        for i in frames:
            d, _ = item(ds, i)
            p, _ = item(pds, i)       # the parent's sampler is not installed here (no histograms): it no-ops
            same = np.array_equal(d['points'], p['points']) and np.array_equal(d['gt_boxes'], p['gt_boxes'])
            print('  frame %s: points %d, boxes %d' % (frame_key(ds, i), len(d['points']), len(d['gt_boxes'])))
            check(same, 'identical to the C1-analog loader with its sampler inactive (the only training-side change)')
    t = targets(cfg, ds, [item(ds, i)[0] for i in frames])
    report_targets('oracle', t)
    tar = build(cfg, cfg.DATA_CONFIG_TAR, training=False)
    print('  evaluation set: %d frames (%s, LIDAR_CONFIG %s, SHIFT_COOR %s)' % (len(tar), cfg.DATA_CONFIG_TAR.DATASET,
                                                                             cfg.DATA_CONFIG_TAR.get('LIDAR_CONFIG'),
                                                                             cfg.DATA_CONFIG_TAR.SHIFT_COOR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=sorted(TARGETS))
    ap.add_argument('--identity-frames', type=int, default=48)
    ap.add_argument('--label-frames', type=int, default=200)
    args = ap.parse_args()
    oracle_path, path, parent = TARGETS[args.target]
    base_cfg = load(oracle_path)
    base_ds = build(base_cfg, block(base_cfg, path))
    frames = list(np.linspace(0, len(base_ds) - 1, N_FRAMES).astype(int))
    b_items = [item(base_ds, i) for i in frames]
    base_t = targets(base_cfg, base_ds, [b for b, _ in b_items])
    base_tar0, _ = item(build(base_cfg, base_cfg.DATA_CONFIG_TAR, training=False), 0)
    base_eval0, _ = item(build(base_cfg, block(base_cfg, path), training=False), 0)
    print('oracle %s: %d TRAIN frames; dump frames %s' % (oracle_path, len(base_ds),
                                                         [frame_key(base_ds, i) for i in frames]))
    if parent is not None:
        dump_new_oracle(oracle_path, path, parent, frames)
    base = (base_cfg, path, base_ds, frames, b_items, base_t, base_tar0, base_eval0)
    rows = sorted(glob.glob(A + 'factorswap-%s/centerpoint-factorswap-%s-r*.yaml' % (args.target, args.target)))
    if args.target == 'lyft64':
        rows.append(A + 'factorswap-lyft64/centerpoint-factorswap-lyft64-d-density.yaml')
    for r in rows:
        dump_row(r, oracle_path, path, base, args)
    print('=' * 118)
    if FAILS:
        print('FAILED CHECKS (%d):' % len(FAILS))
        for f in FAILS:
            print('  ' + f)
        raise SystemExit(1)
    print('ALL DUMPS PASSED (%s: %d rows)' % (args.target, len(rows)))


if __name__ == '__main__':
    main()
