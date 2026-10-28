"""PL-MATCH (ANALYSIS; experiments_md 20261028_05, design T05 §5 item 23 fixed 2026-10-28 20:01, experiments_md 2e78a9c):
build the frozen MB / MB-FP pseudo-label files from 27704's stored epoch-0 pseudo-labels and the Waymo TRAIN ground
truth, and run the pre-launch checks of memory/repo/bug_prevention_checklist.md (rules 1, 3, 4).

    python analysis/plmatch_labels.py build --ps <27704 ps_label_e0.pkl> --gt <PL-GT file> --out_dir <dir> [--force]
    python analysis/plmatch_labels.py check --ps <...> --gt <...> --out_dir <dir> --out_json <json>
        [--empty_frames 400] [--control_frames 150] [--equal_frames 150] [--dump_frames 60]
    python analysis/plmatch_labels.py sizes --out_json <json>

build: per frame and per class (labels 1..3 = Car, Pedestrian, Cyclist; the GT is PL-GT's file, i.e. Waymo Vehicle ->
  Car, Pedestrian, Cyclist through PL-GT's filters), positives of 27704's file are matched one-to-one to GT, greedy by
  pseudo-label score descending, BEV IoU >= 0.5 (self_training_utils.pl_match_greedy); the ignore band is never matched.
  Writes three wrapper files (self_training_utils.ANALYSIS_PL_MATCH_FORMAT):
    plmatch_mb_27704e0_waymo_train_si10.pkl      MB: matched positives take their GT box (class and score kept)
    plmatch_mbfp_27704e0_waymo_train_si10.pkl    MB-FP: MB minus every unmatched positive
    plmatch_control_27704e0_waymo_train_si10.pkl CONTROL: 27704's labels unchanged, every box recounted (path control)
  plus build_stats.json: counts, match rates per class and range bin, and the matched-pair geometry (beside item).
check (no GPU):
  1. sentinels + counts through the trainer's own loader; an INDEPENDENT recount (27704's file, the PL-GT file and an
     exact row comparison, not the builder's bookkeeping) in every frame; the greedy matching re-verified by a separate
     implementation; MB's matched boxes == their GT boxes; MB-FP has no unmatched positive; the ignore band unchanged.
  2. matched GT boxes with no point of the loaded first-return cloud (the boxes the stored-count rule keeps).
  3. PATH CONTROL: 27704's own labels served through the new path (CONTROL file, the PL-MATCH key) against 27704's
     original path (its config, its ps_label_e0.pkl installed as the trainer installs it), on the same frames, with
     augmentation off and with augmentation on at a fixed seed: every served array and every assigned target equal.
  4. loader equality, augmentation off, MB and MB-FP: served boxes == stored boxes kept by the mixed zero-point rule and
     the range filter.
  5. one-batch target dump per row (augmentation on, as trained): per-class positives, ignore boxes, ignored pixels,
     heatmap peaks == distinct positive pixels, regression targets == positives, points per box.
sizes (beside item): per-dimension GT Car size, nuScenes TRAIN (Boston n008 / Singapore n015) against Waymo TRAIN, from
  the info pkls the loaders use, Car as each loader maps it.
"""
import argparse
import copy
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS.parent, TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(TOOLS)   # _BASE_CONFIG_ paths and ../data are relative to tools/
import _init_path  # noqa: F401,E402  (the checkout's pcdet, not the image's copy)
import torch  # noqa: E402
from easydict import EasyDict  # noqa: E402

from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.utils import box_utils, common_utils, self_training_utils as stu  # noqa: E402

NAMES = ['Car', 'Pedestrian', 'Cyclist']
C = len(NAMES)
RANGE_BINS = [(0.0, 30.0), (30.0, 50.0), (50.0, float('inf'))]
CFG_BASE = 'cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2waymo.yaml'   # 27704
CFG_ROW = {'mb': 'cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-plmatch-mb-nuscenes2waymo.yaml',
           'mbfp': 'cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-plmatch-mbfp-nuscenes2waymo.yaml'}
FILE = {'mb': 'plmatch_mb_27704e0_waymo_train_si10.pkl', 'mbfp': 'plmatch_mbfp_27704e0_waymo_train_si10.pkl',
        'control': 'plmatch_control_27704e0_waymo_train_si10.pkl'}
ROW_NAME = {'mb': 'MB', 'mbfp': 'MB-FP', 'control': 'CONTROL'}


def load_cfg(path):
    cfg = EasyDict()
    cfg_from_yaml_file(path, cfg)
    return cfg


def target_dataset(cfg):
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=common_utils.create_logger(), training=True,
                                model_ontology=cfg.get('ONTOLOGY'))
    return ds


def head_sha():
    try:
        return subprocess.check_output(['git', '-C', str(TOOLS.parent), 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


def bev_iou(a, b):
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
    a = np.ascontiguousarray(np.asarray(a)[:, :7], dtype=np.float32)
    b = np.ascontiguousarray(np.asarray(b)[:, :7], dtype=np.float32)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    out = boxes_bev_iou_cpu(a, b)
    return np.asarray(out.numpy() if hasattr(out, 'numpy') else out, dtype=np.float64)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rbin(xy):
    r = np.hypot(xy[:, 0], xy[:, 1])
    return np.digitize(r, [b[1] for b in RANGE_BINS[:-1]])


def load_gt(path):
    blob = pickle.load(open(path, 'rb'))
    assert blob.get('format') == stu.ANALYSIS_TARGET_GT_FORMAT, path
    return blob['labels'], blob['meta']


def summary(v):
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return {'n': 0}
    p25, p50, p75 = np.percentile(v, [25, 50, 75])
    return {'n': int(len(v)), 'median': round(float(p50), 4), 'p25': round(float(p25), 4), 'p75': round(float(p75), 4),
            'iqr': round(float(p75 - p25), 4), 'mean': round(float(v.mean()), 4)}


# ------------------------------------------------------------------------------------------------------------- build
def build(args):
    ps = pickle.load(open(args.ps, 'rb'))
    gt, gt_meta = load_gt(args.gt)
    assert list(gt_meta['class_names']) == NAMES, gt_meta['class_names']
    assert set(ps) == set(gt), 'the 27704 file and the PL-GT file cover different frames'
    os.makedirs(args.out_dir, exist_ok=True)
    outs = {k: os.path.join(args.out_dir, v) for k, v in FILE.items()}
    for p in outs.values():
        if os.path.exists(p) and not args.force:
            raise SystemExit('%s exists; it is FROZEN once written (pass --force only before any run read it)' % p)
    files = {'mb': {}, 'mbfp': {}, 'control': {}}
    # match statistics: per class and range bin; geometry of the matched pairs (PL - GT)
    n_pos = np.zeros((C, len(RANGE_BINS)), np.int64)
    n_pos_m = np.zeros((C, len(RANGE_BINS)), np.int64)
    n_gt = np.zeros((C, len(RANGE_BINS)), np.int64)
    n_gt_m = np.zeros((C, len(RANGE_BINS)), np.int64)
    geo = {c: {k: [] for k in ('dl', 'dw', 'dh', 'along', 'across', 'dz', 'dheading', 'iou', 'rbin')} for c in range(C)}
    for fid in sorted(ps):
        pe, ge = ps[fid], gt[fid]
        mb, info = stu.pl_match_entry(pe, ge, C, bev_iou, 0.5, drop_unmatched_positives=False)
        mbfp, info2 = stu.pl_match_entry(pe, ge, C, bev_iou, 0.5, drop_unmatched_positives=True)
        assert np.array_equal(info['match'], info2['match'])
        b = np.asarray(pe['gt_boxes'], np.float64).reshape(-1, 9)
        ctrl = {k: (None if v is None else np.asarray(v).copy()) for k, v in pe.items()}
        ctrl['gt_boxes'] = b.copy()
        ctrl['num_points_in_gt'] = np.full(len(b), stu.PL_MATCH_RECOUNT, dtype=np.int64)
        ctrl['match_gt_index'] = np.full(len(b), -1, dtype=np.int64)
        ctrl['match_bev_iou'] = np.zeros(len(b), dtype=np.float64)
        files['mb'][fid], files['mbfp'][fid], files['control'][fid] = mb, mbfp, ctrl
        g = np.asarray(ge['gt_boxes'], np.float64).reshape(-1, 9)
        m = info['match']
        lab = b[:, 7].astype(int)
        rb_p, rb_g = rbin(b[:, :2]), rbin(g[:, :2])
        g_taken = np.zeros(len(g), bool)
        g_taken[m[m >= 0]] = True
        for c in range(C):
            for r in range(len(RANGE_BINS)):
                sp = (lab == c + 1) & (rb_p == r)
                n_pos[c, r] += sp.sum()
                n_pos_m[c, r] += (sp & (m >= 0)).sum()
                sg = (g[:, 7].astype(int) == c + 1) & (rb_g == r)
                n_gt[c, r] += sg.sum()
                n_gt_m[c, r] += (sg & g_taken).sum()
        for i in np.where(m >= 0)[0]:
            c = lab[i] - 1
            p, q = b[i], g[m[i]]
            d = p[:2] - q[:2]
            th = q[6]
            geo[c]['dl'].append(p[3] - q[3])
            geo[c]['dw'].append(p[4] - q[4])
            geo[c]['dh'].append(p[5] - q[5])
            geo[c]['along'].append(d[0] * np.cos(th) + d[1] * np.sin(th))
            geo[c]['across'].append(-d[0] * np.sin(th) + d[1] * np.cos(th))
            geo[c]['dz'].append(p[2] - q[2])
            geo[c]['dheading'].append(wrap(p[6] - q[6]))
            geo[c]['iou'].append(info['match_iou'][i])
            geo[c]['rbin'].append(rb_g[m[i]])
    counts = {}
    for k in files:
        counts[k] = {kk: v.tolist() for kk, v in stu.validate_pl_match_pseudo_labels(files[k], NAMES).items()}
    nf = len(ps)
    rb_names = ['%g-%s m' % (lo, '' if hi == float('inf') else '%g' % hi) for lo, hi in RANGE_BINS]
    rates = {}
    for c in range(C):
        rates[NAMES[c]] = {
            'positives': int(n_pos[c].sum()), 'positives_matched': int(n_pos_m[c].sum()),
            'positive_match_rate': round(float(n_pos_m[c].sum() / max(n_pos[c].sum(), 1)), 4),
            'gt': int(n_gt[c].sum()), 'gt_matched': int(n_gt_m[c].sum()),
            'gt_match_rate': round(float(n_gt_m[c].sum() / max(n_gt[c].sum(), 1)), 4),
            'per_range': {rb_names[r]: {
                'positives': int(n_pos[c, r]), 'positives_matched': int(n_pos_m[c, r]),
                'positive_match_rate': round(float(n_pos_m[c, r] / max(n_pos[c, r], 1)), 4),
                'gt': int(n_gt[c, r]), 'gt_matched': int(n_gt_m[c, r]),
                'gt_match_rate': round(float(n_gt_m[c, r] / max(n_gt[c, r], 1)), 4)} for r in range(len(RANGE_BINS))}}
    geometry = {}
    for c in range(C):
        gc = {k: np.asarray(v) for k, v in geo[c].items()}
        dh = gc['dheading']
        ent = {k: summary(gc[k]) for k in ('dl', 'dw', 'dh', 'along', 'across', 'dz', 'dheading', 'iou')}
        ent['dheading_mod_pi'] = summary(wrap(2 * dh) / 2) if len(dh) else {'n': 0}
        ent['share_heading_flipped_gt_pi_2'] = round(float(np.mean(np.abs(dh) > np.pi / 2)), 4) if len(dh) else None
        ent['per_range'] = {rb_names[r]: {k: summary(gc[k][gc['rbin'] == r]) for k in ('dl', 'dw', 'dh', 'along',
                                                                                    'across', 'dz')}
                            for r in range(len(RANGE_BINS))} if len(dh) else {}
        geometry[NAMES[c]] = ent
    sha = head_sha()
    stamp = time.strftime('%Y-%m-%d %H:%M %Z')
    common = {
        'what': 'ANALYSIS PL-MATCH (experiments_md 20261028_05; T05 §5 item 23 fixed 2026-10-28 20:01, experiments_md '
                '2e78a9c): 27704 epoch-0 pseudo-labels matched against Waymo TRAIN GT. Target labels shape training '
                'targets; never a method.',
        'source_pseudo_labels': args.ps, 'source_gt': args.gt, 'n_frames': nf, 'class_names': NAMES,
        'matching': 'per frame and class; positives only, greedy by score descending (stable), one-to-one, '
                    'rotated BEV IoU >= 0.5 (boxes_bev_iou_cpu); ignore band never matched; unmatched GT not added',
        'num_points_in_gt': 'stored GT count on GT-replaced boxes; %d (recount on the loaded cloud) elsewhere'
                            % stu.PL_MATCH_RECOUNT,
        'st3d_head': sha, 'written': stamp}
    for k in files:
        meta = dict(common, row=ROW_NAME[k], counts=counts[k])
        tmp = outs[k] + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({'format': stu.ANALYSIS_PL_MATCH_FORMAT, 'meta': meta, 'labels': files[k]}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, outs[k])
    stats = dict(common, counts=counts,
                 per_frame={k: {kk: [round(x / nf, 4) for x in v] for kk, v in counts[k].items()} for k in counts},
                 match_rates=rates, matched_pair_geometry_pl_minus_gt=geometry,
                 boxes_per_frame_max={k: int(max(len(e['gt_boxes']) for e in files[k].values())) for k in files})
    with open(os.path.join(args.out_dir, 'build_stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: stats[k] for k in ('counts', 'per_frame', 'match_rates', 'boxes_per_frame_max')}, indent=1))
    print(json.dumps(geometry['Car'], indent=1))


# ------------------------------------------------------------------------------------------------------------- check
def greedy_reference(pb, ps_score, gb, iou):
    """A second, separately written greedy matcher (lists, explicit availability) for the re-verification."""
    order = sorted(range(len(pb)), key=lambda i: (-float(ps_score[i]), i))
    free = list(range(len(gb)))
    out = {}
    for i in order:
        best, bj = -1.0, None
        for j in free:
            if iou[i, j] > best:
                best, bj = iou[i, j], j
        if bj is not None and best >= 0.5:
            out[i] = bj
            free.remove(bj)
    return out


def make_head(cfg):
    from pcdet.models.dense_heads.center_head import CenterHead
    tar = cfg.DATA_CONFIG_TAR
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.model_cfg = cfg.MODEL.DENSE_HEAD
    head.class_names = list(cfg.CLASS_NAMES)
    head.class_names_each_head = [list(x) for x in cfg.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD]
    assert head.class_names_each_head == [NAMES], head.class_names_each_head
    pcr = np.array(tar.POINT_CLOUD_RANGE, dtype=np.float32)
    head.point_cloud_range = pcr
    head.voxel_size = [p for p in tar.DATA_PROCESSOR if p.NAME == 'transform_points_to_voxels'][0].VOXEL_SIZE
    stride = cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
    side = int(round((pcr[3] - pcr[0]) / head.voxel_size[0] / stride))
    return head, side, head.voxel_size[0] * stride, pcr


def targets(head, side, gt):
    return head.assign_targets(torch.from_numpy(np.asarray(gt)[None]).float().clone(), feature_map_size=[side, side])


def same_dict(a, b):
    """Every key and every array equal (NaN-aware); returns the list of differing keys."""
    diff = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            diff.append(k)
            continue
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.asarray(x), np.asarray(y)
            if x.shape != y.shape or x.dtype != y.dtype:
                diff.append(k)
            elif x.dtype.kind in 'fc':
                if not np.array_equal(x, y, equal_nan=True):
                    diff.append(k)
            elif not np.array_equal(x, y):
                diff.append(k)
        elif isinstance(x, dict) and isinstance(y, dict):
            if same_dict(x, y):
                diff.append(k)
        elif x != y:
            diff.append(k)
    return diff


def same_targets(t1, t2):
    diff = []
    for k in ('heatmaps', 'target_boxes', 'inds', 'masks', 'heatmap_masks'):
        for a, b in zip(t1[k], t2[k]):
            if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                diff.append(k)
    return diff


def seeded(ds, i, seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return ds[i]


def check(args):
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    res = {'st3d_head': head_sha(), 'started': time.strftime('%Y-%m-%d %H:%M %Z')}
    ps = pickle.load(open(args.ps, 'rb'))
    gt, _ = load_gt(args.gt)
    paths = {k: os.path.join(args.out_dir, v) for k, v in FILE.items()}
    loaded = {}
    # ---- 1. sentinels + counts through the trainer's loader; independent recount; matching re-verified
    for k in ('mb', 'mbfp', 'control'):
        labels, meta, counts = stu.load_analysis_pl_match_pseudo_labels(paths[k], NAMES, 1)
        loaded[k] = labels
        allb = np.concatenate([e['gt_boxes'] for e in labels.values()])
        res['1_%s_loader' % k] = {
            'row': meta['row'], 'n_frames': len(labels), 'n_boxes': int(len(allb)),
            'labels_unique': np.unique(allb[:, 7]).tolist(), 'zero': int((allb[:, 7] == 0).sum()),
            'minus99': int((allb[:, 7] == common_utils.IGNORE_CLASS_LABEL).sum()),
            'abs_gt_3': int((np.abs(allb[:, 7]) > C).sum()),
            'score_min': float(allb[:, 8].min()), 'score_max': float(allb[:, 8].max()),
            'counts': {kk: v.tolist() for kk, v in counts.items()},
            'per_frame': {kk: (v / len(labels)).round(4).tolist() for kk, v in counts.items()},
            'boxes_per_frame_max': int(max(len(e['gt_boxes']) for e in labels.values()))}
    # independent recount from 27704's file and the PL-GT file, per frame
    ind = {k: {'positive': np.zeros(C, np.int64), 'ignore': np.zeros(C, np.int64), 'gt_replaced': np.zeros(C, np.int64)}
           for k in ('mb', 'mbfp', 'control')}
    frame_ok = {k: 0 for k in ind}
    bad = {k: [] for k in ind}
    v = {'mb_rows_equal_gt_row_exactly': 0, 'mb_matched_rows': 0, 'mb_matched_eq_gt': 0, 'one_to_one_violations': 0,
         'matched_iou_lt_0.5': 0, 'greedy_reference_disagreements': 0, 'mbfp_unmatched_positive': 0,
         'ignore_band_changed_frames': 0, 'unmatched_positive_changed': 0, 'control_differs_from_27704': 0,
         'class_changed': 0, 'score_changed': 0, 'stored_count_ne_gt_count': 0}
    for fid in ps:
        b = np.asarray(ps[fid]['gt_boxes'], np.float64).reshape(-1, 9)
        g = np.asarray(gt[fid]['gt_boxes'], np.float64).reshape(-1, 9)
        gnp = np.asarray(gt[fid]['num_points_in_gt'])
        lab = b[:, 7].astype(int)
        # the reference matching, written separately, on recomputed IoU
        ref = {}
        for c in range(1, C + 1):
            pi, gi = np.where(lab == c)[0], np.where(g[:, 7].astype(int) == c)[0]
            if len(pi) and len(gi):
                iou = bev_iou(b[pi], g[gi])
                for a, j in greedy_reference(b[pi], b[pi, 8], g[gi], iou).items():
                    ref[int(pi[a])] = int(gi[j])
        e_mb, e_fp, e_ct = loaded['mb'][fid], loaded['mbfp'][fid], loaded['control'][fid]
        mbb, fpb = e_mb['gt_boxes'], e_fp['gt_boxes']
        # rows of MB equal to a GT row of the same class, found by exact comparison (not the stored audit trail)
        eq = {}
        for i in range(len(mbb)):
            if mbb[i, 7] <= 0:
                continue
            hit = np.where(np.all(g[:, :7] == mbb[i, :7], axis=1) & (g[:, 7] == mbb[i, 7]))[0]
            if len(hit):
                eq[i] = int(hit[0])
        v['mb_rows_equal_gt_row_exactly'] += len(eq)
        if len(set(eq.values())) != len(eq):
            v['one_to_one_violations'] += 1
        mi = np.asarray(e_mb['match_gt_index'])
        v['mb_matched_rows'] += int((mi >= 0).sum())
        if set(eq) != set(np.where(mi >= 0)[0].tolist()) or any(eq[i] != mi[i] for i in eq):
            bad['mb'].append(fid)
        if {i: eq[i] for i in eq} != ref:
            v['greedy_reference_disagreements'] += 1
        for i, j in ref.items():
            if bev_iou(b[i:i + 1], g[j:j + 1])[0, 0] < 0.5:
                v['matched_iou_lt_0.5'] += 1
            v['mb_matched_eq_gt'] += int(np.array_equal(mbb[i, :7], g[j, :7]))
            v['stored_count_ne_gt_count'] += int(e_mb['num_points_in_gt'][i] != gnp[j])
        v['class_changed'] += int(not np.array_equal(mbb[:, 7], b[:, 7]))
        v['score_changed'] += int(not np.array_equal(mbb[:, 8], b[:, 8]))
        unm = np.array([lab[i] > 0 and i not in ref for i in range(len(b))], dtype=bool)
        neg = lab < 0
        v['unmatched_positive_changed'] += int(not np.array_equal(mbb[unm], b[unm]))
        if not (np.array_equal(mbb[neg], b[neg]) and np.array_equal(fpb[fpb[:, 7] < 0], b[neg])):
            v['ignore_band_changed_frames'] += 1
        keep_fp = ~unm
        if not np.array_equal(fpb, mbb[keep_fp]):
            bad['mbfp'].append(fid)
        v['mbfp_unmatched_positive'] += int(((fpb[:, 7] > 0) & (np.asarray(e_fp['match_gt_index']) < 0)).sum())
        v['control_differs_from_27704'] += int(not np.array_equal(e_ct['gt_boxes'], b))
        # independent per-class counts for this frame
        n_ref = np.bincount([lab[i] - 1 for i in ref], minlength=C)
        exp = {'mb': (np.bincount(lab[lab > 0] - 1, minlength=C), np.bincount(-lab[neg] - 1, minlength=C), n_ref),
               'mbfp': (n_ref, np.bincount(-lab[neg] - 1, minlength=C), n_ref),
               'control': (np.bincount(lab[lab > 0] - 1, minlength=C), np.bincount(-lab[neg] - 1, minlength=C),
                           np.zeros(C, np.int64))}
        for k in ind:
            e = loaded[k][fid]
            got = stu.validate_pl_match_pseudo_labels({fid: e}, NAMES)
            ok = all(np.array_equal(got[n], x) for n, x in zip(('positive', 'ignore', 'gt_replaced'), exp[k]))
            frame_ok[k] += int(ok)
            if not ok and len(bad[k]) < 5:
                bad[k].append(fid)
            for n, x in zip(('positive', 'ignore', 'gt_replaced'), exp[k]):
                ind[k][n] += x
    res['1_independent'] = {
        'frames': len(ps),
        'recount': {k: {n: x.tolist() for n, x in ind[k].items()} for k in ind},
        'loader_equals_recount': {k: all(np.array_equal(np.asarray(res['1_%s_loader' % k]['counts'][n]), ind[k][n])
                                         for n in ind[k]) for k in ind},
        'frames_equal': frame_ok, 'bad_frames_first': {k: x[:5] for k, x in bad.items()},
        'verification': v}
    print(json.dumps({k: res[k] for k in res if k.startswith('1_')}, indent=1), flush=True)

    # ---- 2. matched GT boxes with no point of the loaded first-return cloud
    cfg_mb = load_cfg(CFG_ROW['mb'])
    ds = target_dataset(cfg_mb)
    step = max(1, len(ds.infos) // args.empty_frames)
    emp_gt, emp_pl, tot = np.zeros(C), np.zeros(C), np.zeros(C)
    shift = np.array(cfg_mb.DATA_CONFIG_TAR.get('SHIFT_COOR', [0, 0, 0]), dtype=np.float32)
    for info in ds.infos[::step][:args.empty_frames]:
        fid = info['frame_id']
        e = loaded['mb'][fid]
        m = np.asarray(e['match_gt_index']) >= 0
        if not m.any():
            continue
        pts = ds.get_lidar_with_sweeps(info)
        pts[:, :3] += shift
        tp = torch.from_numpy(np.ascontiguousarray(pts[:, :3], dtype=np.float32))
        ing = roiaware_pool3d_utils.points_in_boxes_cpu(
            tp, torch.from_numpy(e['gt_boxes'][m, :7].astype(np.float32))).numpy().sum(1)
        orig = np.asarray(ps[fid]['gt_boxes'])[m]
        inp = roiaware_pool3d_utils.points_in_boxes_cpu(
            tp, torch.from_numpy(orig[:, :7].astype(np.float32))).numpy().sum(1)
        lab = e['gt_boxes'][m, 7].astype(int)
        for c in range(C):
            tot[c] += (lab == c + 1).sum()
            emp_gt[c] += ((lab == c + 1) & (ing == 0)).sum()
            emp_pl[c] += ((lab == c + 1) & (inp == 0)).sum()
    res['2_matched_empty_in_loaded_cloud'] = {
        'frames': int(min(args.empty_frames, len(ds.infos[::step]))),
        'matched_pairs': dict(zip(NAMES, tot.astype(int).tolist())),
        'gt_box_empty_share (kept by the stored count)': dict(zip(NAMES, (emp_gt / np.maximum(tot, 1)).round(4).tolist())),
        'pl_box_empty_share (27704 dropped these by its recount)': dict(zip(NAMES, (emp_pl / np.maximum(tot, 1)).round(4).tolist()))}
    print(json.dumps(res['2_matched_empty_in_loaded_cloud'], indent=1), flush=True)

    # ---- 3. PATH CONTROL: CONTROL through the new path vs 27704's own path
    cfg_base = load_cfg(CFG_BASE)
    cfg_ctl = load_cfg(CFG_ROW['mb'])
    cfg_ctl.DATA_CONFIG_TAR[common_utils.ANALYSIS_TARGET_PL_MATCH_KEY] = paths['control']
    ds_o = target_dataset(cfg_base)
    ds_n = target_dataset(cfg_ctl)
    assert not ds_o.dataset_cfg.get(common_utils.ANALYSIS_TARGET_PL_MATCH_KEY, None)
    assert ds_n.dataset_cfg.get(common_utils.ANALYSIS_TARGET_PL_MATCH_KEY, None) == paths['control']
    assert [i['frame_id'] for i in ds_o.infos] == [i['frame_id'] for i in ds_n.infos]
    ds_o.set_pseudo_labels(dict(ps))                     # 27704: its ps_label_e0.pkl, as check_already_exsit loads it
    ds_n.set_pseudo_labels(dict(loaded['control']))      # the new path: the trainer's validated loader output
    head, side, cell, pcr = make_head(cfg_base)
    queue = {id(d): list(d.data_augmentor.data_augmentor_queue) for d in (ds_o, ds_n)}
    pc = {}
    for mode in ('aug_off', 'aug_on_fixed_seed'):
        for d in (ds_o, ds_n):
            d.data_augmentor.data_augmentor_queue = [] if mode == 'aug_off' else list(queue[id(d)])
        step = max(1, len(ds_o) // args.control_frames)
        idx = list(range(0 if mode == 'aug_off' else step // 2, len(ds_o), step))[:args.control_frames]
        st = {'frames': 0, 'frames_equal': 0, 'targets_equal': 0, 'failures': [],
              'served_pos': np.zeros(C), 'served_ign': np.zeros(C), 'ignored_px': 0, 'peaks': np.zeros(C)}
        for i in idx:
            a = seeded(ds_o, i, 20261028 + i)
            b = seeded(ds_n, i, 20261028 + i)
            dk = same_dict(a, b)
            ta, tb = targets(head, side, a['gt_boxes']), targets(head, side, b['gt_boxes'])
            dt = same_targets(ta, tb)
            st['frames'] += 1
            st['frames_equal'] += int(not dk)
            st['targets_equal'] += int(not dt)
            if (dk or dt) and len(st['failures']) < 5:
                st['failures'].append({'frame': ds_o.infos[i]['frame_id'], 'keys': dk, 'targets': dt})
            lab = b['gt_boxes'][:, 7].astype(int)
            for c in range(C):
                st['served_pos'][c] += (lab == c + 1).sum()
                st['served_ign'][c] += (lab == -(c + 1)).sum()
                st['peaks'][c] += int(tb['heatmaps'][0][0][c].eq(1).sum())
            if tb['heatmap_masks'][0] is not None:
                st['ignored_px'] += int((tb['heatmap_masks'][0][0] == 0).sum())
        for k in ('served_pos', 'served_ign', 'peaks'):
            st[k] = dict(zip(NAMES, st[k].astype(int).tolist()))
        pc[mode] = st
    for d in (ds_o, ds_n):
        d.data_augmentor.data_augmentor_queue = list(queue[id(d)])
    res['3_path_control'] = pc
    print(json.dumps(pc, indent=1), flush=True)
    del ds_o, ds_n

    # ---- 4. loader equality (augmentation off) and 5. assigned-target dump (augmentation on), MB and MB-FP
    rng_cfg = [p for p in cfg_mb.DATA_CONFIG_TAR.DATA_PROCESSOR if p.NAME == 'mask_points_and_boxes_outside_range'][0]
    for k in ('mb', 'mbfp'):
        cfg = load_cfg(CFG_ROW[k])
        assert cfg.DATA_CONFIG_TAR[common_utils.ANALYSIS_TARGET_PL_MATCH_KEY] == paths[k]
        d = ds if k == 'mb' else target_dataset(cfg)
        d.set_pseudo_labels(dict(loaded[k]))
        full_queue = list(d.data_augmentor.data_augmentor_queue)
        d.data_augmentor.data_augmentor_queue = []
        step = max(1, len(d) // args.equal_frames)
        eqs = {'frames': 0, 'frames_equal': 0, 'stored': 0, 'zero_point_dropped_recount': 0,
               'zero_point_dropped_gt_replaced': 0, 'outside_range': 0, 'failures': []}
        for i in list(range(0, len(d), step))[:args.equal_frames]:
            fid = d.infos[i]['frame_id']
            e = loaded[k][fid]
            s = e['gt_boxes']
            info = d.infos[i]
            pts = d.get_lidar_with_sweeps(info)
            pts[:, :3] += shift
            cnt = np.asarray(e['num_points_in_gt']).copy()
            rc = cnt == stu.PL_MATCH_RECOUNT
            if rc.any():
                cnt[rc] = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(np.ascontiguousarray(pts[:, :3], dtype=np.float32)),
                    torch.from_numpy(s[rc, :7].astype(np.float32))).numpy().sum(1)
            keep_pts = cnt >= 1
            keep_rng = box_utils.mask_boxes_outside_range_numpy(s[:, :7], pcr, rng_cfg.get('min_num_corners', 1)) \
                if len(s) else np.zeros(0, bool)
            exp = s[keep_pts & keep_rng]
            served = d[i]
            got = served['gt_boxes']
            same = got.shape[0] == exp.shape[0] and np.allclose(got[:, :6], exp[:, :6], atol=1e-4) and \
                np.allclose(np.abs(wrap(got[:, 6] - exp[:, 6])), 0, atol=1e-4) and np.array_equal(got[:, 7], exp[:, 7])
            lab = s[:, 7].astype(int)
            pos_exp = np.bincount(lab[lab > 0] - 1, minlength=C).astype(np.float32)
            ign_exp = np.bincount(-lab[lab < 0] - 1, minlength=C).astype(np.float32)
            cnt_ok = np.array_equal(served['pos_ps_bbox'], pos_exp) and np.array_equal(served['ign_ps_bbox'], ign_exp)
            eqs['frames'] += 1
            eqs['frames_equal'] += int(same and cnt_ok)
            eqs['stored'] += len(s)
            eqs['zero_point_dropped_recount'] += int((rc & ~keep_pts).sum())
            eqs['zero_point_dropped_gt_replaced'] += int((~rc & ~keep_pts).sum())
            eqs['outside_range'] += int((~keep_rng).sum())
            if not (same and cnt_ok) and len(eqs['failures']) < 5:
                eqs['failures'].append({'frame': fid, 'got': int(got.shape[0]), 'expected': int(exp.shape[0]),
                                        'counts_ok': bool(cnt_ok)})
        res['4_%s_loader_equality_no_aug' % k] = eqs
        print(json.dumps({k: eqs}, indent=1), flush=True)

        d.data_augmentor.data_augmentor_queue = full_queue
        t5 = {'pos': np.zeros(C), 'stored_pos': np.zeros(C), 'ign': np.zeros(C), 'stored_ign': np.zeros(C),
              'peaks': np.zeros(C), 'pixels': np.zeros(C), 'zero_or_bad_labels': 0, 'reg': 0, 'ign_px': 0,
              'ign_px_on_peak': 0, 'frames': 0, 'frames_with_ignore': 0, 'frames_ignore_masked': 0,
              'pts': {c: [] for c in range(C)}, 'fails': []}
        step = max(1, len(d) // args.dump_frames)
        for i in list(range(step // 3, len(d), step))[:args.dump_frames]:
            served = seeded(d, i, 777 + i)
            gtb = served['gt_boxes']
            lab = gtb[:, 7].astype(int)
            s = loaded[k][d.infos[i]['frame_id']]['gt_boxes']
            sl = s[:, 7].astype(int)
            t5['zero_or_bad_labels'] += int(((lab == 0) | (np.abs(lab) > C) | (gtb[:, 7] != np.round(gtb[:, 7]))).sum())
            for c in range(C):
                t5['pos'][c] += (lab == c + 1).sum()
                t5['ign'][c] += (lab == -(c + 1)).sum()
                t5['stored_pos'][c] += (sl == c + 1).sum()
                t5['stored_ign'][c] += (sl == -(c + 1)).sum()
            if (lab > 0).any():
                inside = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(np.ascontiguousarray(served['points'][:, :3], dtype=np.float32)),
                    torch.from_numpy(gtb[lab > 0, :7].astype(np.float32))).numpy().sum(1)
                for c in range(C):
                    t5['pts'][c] += inside[lab[lab > 0] == c + 1].tolist()
            t = targets(head, side, gtb)
            hm = t['heatmaps'][0][0]
            mask = t['heatmap_masks'][0]
            t5['frames_with_ignore'] += int((lab < 0).any())
            if mask is not None:
                mk = mask[0]
                t5['frames_ignore_masked'] += 1
                t5['ign_px'] += int((mk == 0).sum())
                t5['ign_px_on_peak'] += int(((mk == 0) & (hm.eq(1).any(0))).sum())
            pos_pix = {}
            for bx, l in zip(gtb, lab):
                if l > 0 and bx[3] > 0 and bx[4] > 0:
                    x = min(max((bx[0] - pcr[0]) / cell, 0), side - 0.5)
                    y = min(max((bx[1] - pcr[1]) / cell, 0), side - 0.5)
                    pos_pix.setdefault(l - 1, set()).add((int(y), int(x)))
            for c in range(C):
                pk, ex = int(hm[c].eq(1).sum()), len(pos_pix.get(c, ()))
                t5['peaks'][c] += pk
                t5['pixels'][c] += ex
                if pk != ex and len(t5['fails']) < 10:
                    t5['fails'].append('frame %d %s: peaks %d != distinct positive pixels %d' % (i, NAMES[c], pk, ex))
            t5['reg'] += int(t['masks'][0].sum())
            t5['frames'] += 1
        out = {
            'frames': t5['frames'],
            'stored_positives': dict(zip(NAMES, t5['stored_pos'].astype(int).tolist())),
            'served_positives': dict(zip(NAMES, t5['pos'].astype(int).tolist())),
            'stored_ignore': dict(zip(NAMES, t5['stored_ign'].astype(int).tolist())),
            'served_ignore': dict(zip(NAMES, t5['ign'].astype(int).tolist())),
            'zero_minus99_or_out_of_range_labels_served': t5['zero_or_bad_labels'],
            'heatmap_peaks': dict(zip(NAMES, t5['peaks'].astype(int).tolist())),
            'distinct_positive_pixels': dict(zip(NAMES, t5['pixels'].astype(int).tolist())),
            'regression_targets': t5['reg'], 'served_positives_total': int(t5['pos'].sum()),
            'frames_with_ignore_boxes': t5['frames_with_ignore'], 'frames_with_ignore_mask': t5['frames_ignore_masked'],
            'ignored_heatmap_pixels': t5['ign_px'], 'ignored_pixels_on_a_positive_peak': t5['ign_px_on_peak'],
            'points_per_positive_box_median': {NAMES[c]: float(np.median(t5['pts'][c])) if t5['pts'][c] else None
                                               for c in range(C)},
            'points_per_positive_box_empty_share': {NAMES[c]: round(float(np.mean(np.array(t5['pts'][c]) == 0)), 4)
                                                    if t5['pts'][c] else None for c in range(C)},
            'peak_failures': t5['fails'], 'n_peak_failures': len(t5['fails'])}
        res['5_%s_assigned_targets' % k] = out
        print(json.dumps({k: out}, indent=1), flush=True)
    res['finished'] = time.strftime('%Y-%m-%d %H:%M %Z')
    with open(args.out_json, 'w') as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('wrote', args.out_json)


# ------------------------------------------------------------------------------------------------------------- sizes
def sizes(args):
    """Per-dimension GT Car size (length dx, width dy, height dz): nuScenes TRAIN by vehicle (n008 Boston, n015
    Singapore) against Waymo TRAIN, from the info pkls the loaders use, Car as each loader maps it."""
    res = {'st3d_head': head_sha(), 'note': 'GT box sizes only. A difference here cannot separate the labelling '
           'convention from the vehicle fleet (a world term).'}
    # nuScenes: the loader's info pkl; FILTER_MIN_POINTS_IN_GT 1 (num_lidar_pts >= 1); nuscenes -> kitti: car -> Car
    from pcdet.utils.ontology_mapping import get_ontology_mapping
    nmap = get_ontology_mapping('nuscenes', 'kitti')
    car_names = sorted(n for n, m in nmap.items() if m == 'Car')
    root = Path(args.nuscenes_root)
    infos = pickle.load(open(root / 'nuscenes_infos_10sweeps_train.pkl', 'rb'))
    meta = root / 'v1.0-trainval'
    logs = {l['token']: l for l in json.load(open(meta / 'log.json'))}
    scene_log = {s['token']: logs[s['log_token']] for s in json.load(open(meta / 'scene.json'))}
    sample_log = {s['token']: scene_log[s['scene_token']] for s in json.load(open(meta / 'sample.json'))}
    dims = {'n008': [], 'n015': []}
    loc = {'n008': set(), 'n015': set()}
    nfr = {'n008': 0, 'n015': 0}
    for info in infos:
        log = sample_log[info['token']]
        veh = log['vehicle']
        nfr[veh] += 1
        loc[veh].add(log['location'])
        m = np.isin(info['gt_names'], car_names) & (info['num_lidar_pts'] >= 1)
        dims[veh].append(info['gt_boxes'][m, 3:6])
    res['nuscenes'] = {'info': str(root / 'nuscenes_infos_10sweeps_train.pkl'), 'car_names': car_names,
                       'filter': 'num_lidar_pts >= 1 (FILTER_MIN_POINTS_IN_GT 1)'}
    for veh, name in (('n008', 'nuScenes Boston (n008)'), ('n015', 'nuScenes Singapore (n015)')):
        d = np.concatenate(dims[veh])
        res[name] = {'frames': nfr[veh], 'locations': sorted(loc[veh]), 'cars': int(len(d)),
                     'length': summary(d[:, 0]), 'width': summary(d[:, 1]), 'height': summary(d[:, 2])}
    # Waymo: the target loader's own infos (DATA_CONFIG_TAR of 27704), every train frame and the SAMPLED_INTERVAL 10 set
    cfg = load_cfg(CFG_BASE)
    tar = copy.deepcopy(cfg.DATA_CONFIG_TAR)
    for si, name in ((1, 'Waymo TRAIN (all 158,081 frames)'), (10, 'Waymo TRAIN (SAMPLED_INTERVAL 10, 27704 / PL-GT)')):
        t = copy.deepcopy(tar)
        t.SAMPLED_INTERVAL = {'train': si, 'test': tar.SAMPLED_INTERVAL['test']}
        c2 = copy.deepcopy(cfg)
        c2.DATA_CONFIG_TAR = t
        ds = target_dataset(c2)
        wm = get_ontology_mapping('waymo', 'kitti')
        wcar = sorted(n for n, m in wm.items() if m == 'Car')
        dd = []
        for info in ds.infos:
            a = info['annos']
            m = np.isin(a['name'], wcar) & (a['num_points_in_gt'] >= 1)
            dd.append(a['gt_boxes_lidar'][m, 3:6])
        d = np.concatenate(dd)
        res[name] = {'frames': len(ds.infos), 'car_names': wcar, 'filter': 'stored num_points_in_gt >= 1',
                     'cars': int(len(d)), 'length': summary(d[:, 0]), 'width': summary(d[:, 1]),
                     'height': summary(d[:, 2])}
    w = res['Waymo TRAIN (all 158,081 frames)']
    res['waymo_minus_nuscenes_median'] = {
        city: {dim: round(w[dim]['median'] - res[city][dim]['median'], 4) for dim in ('length', 'width', 'height')}
        for city in ('nuScenes Boston (n008)', 'nuScenes Singapore (n015)')}
    with open(args.out_json, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('build', 'check'):
        p = sub.add_parser(name)
        p.add_argument('--ps', required=True, help="27704's ps_label_e0.pkl")
        p.add_argument('--gt', required=True, help="PL-GT's frozen file (the GT through PL-GT's filters)")
        p.add_argument('--out_dir', required=True)
        if name == 'build':
            p.add_argument('--force', action='store_true')
        else:
            p.add_argument('--out_json', required=True)
            p.add_argument('--empty_frames', type=int, default=400)
            p.add_argument('--control_frames', type=int, default=150)
            p.add_argument('--equal_frames', type=int, default=150)
            p.add_argument('--dump_frames', type=int, default=60)
    p = sub.add_parser('sizes')
    p.add_argument('--out_json', required=True)
    p.add_argument('--nuscenes_root', default='/home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval')
    a = ap.parse_args()
    {'build': build, 'check': check, 'sizes': sizes}[a.cmd](a)


if __name__ == '__main__':
    main()
