"""PL-GT (ANALYSIS; experiments_md 20261027_05, design T05 §5 fixed 2026-10-27 23:05): write the Waymo TRAIN ground truth
of the target frames as FROZEN pseudo-labels, and run the pre-launch checks of memory/repo/bug_prevention_checklist.md.

    python analysis/plgt_target_gt.py build <plgt cfg> [--force]
    python analysis/plgt_target_gt.py check <plgt cfg> --teacher_ps <27704 ps_label_e0.pkl> --out_json <json>
        [--empty_frames 400] [--equal_frames 300] [--dump_frames 60]

build: the target dataset exactly as the row builds it (DATA_CONFIG_TAR, training mode: SAMPLED_INTERVAL train 10,
  15,809 frames). Per frame, self_training_utils.target_gt_to_pseudo_label_entry applies the filters the Waymo ORACLE's
  training labels pass (26398: drop 'unknown', Waymo -> KITTI ontology, stored num_points_in_gt >= MIN_POINTS_OF_GT,
  CLASS_NAMES only, SHIFT_COOR); the range filter stays online. Writes the wrapper file the config names
  (DATA_CONFIG_TAR.ANALYSIS_TARGET_GT_AS_PSEUDO_LABELS) and <file>.stats.json. Refuses to overwrite unless --force.

check (no GPU):
  1. sentinels + counts: the file through the trainer's own loader (load_analysis_target_gt_pseudo_labels: labels 1..C,
     no ignore band, no -99, scores 1.0); per-class counts recounted INDEPENDENTLY from the per-sequence info pkls
     (not through the dataset or the conversion function); frame set == the served frames == 27704's stored frames.
  2. frame: 27704's stored epoch-0 pseudo-labels (teacher 27471) against the GT on every frame. Confident teacher Cars
     (positive label 1, score >= 0.5) should find a GT Car at high BEV IoU with agreeing heading and ~0 centre offset.
  3. empty share: stored-count >= 1 boxes holding no point of the loaded (first-return) cloud, on --empty_frames
     frames: the boxes the oracle trains on that a recount would have dropped (the loader keeps the stored count).
  4. loader equality, augmentation off: the row's target dataset with the labels installed as the trainer installs
     them; served gt_boxes == stored boxes inside POINT_CLOUD_RANGE (nothing else dropped, frame unchanged);
     pos_ps_bbox == stored per-class counts, ign_ps_bbox == 0.
  5. assigned targets, as trained (augmentation on): CenterHead.assign_targets with the row's head geometry; per-class
     positives, heatmap peaks == distinct positive pixels, no ignored pixel, regression targets, points per box.
"""
import argparse
import glob
import json
import os
import pickle
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


def build(args):
    cfg = load_cfg(args.cfg)
    tar = cfg.DATA_CONFIG_TAR
    out = tar.get(common_utils.ANALYSIS_TARGET_GT_KEY, None)
    assert out, 'the config does not name a PL-GT file (DATA_CONFIG_TAR.%s)' % common_utils.ANALYSIS_TARGET_GT_KEY
    if os.path.exists(out) and not args.force:
        raise SystemExit('%s exists; it is FROZEN once written (pass --force only before any run read it)' % out)
    ds = target_dataset(cfg)
    names = list(cfg.CLASS_NAMES)
    min_points = tar.get('MIN_POINTS_OF_GT', 1)
    shift = tar.get('SHIFT_COOR', None)
    labels, tot, raw_names = {}, {}, {}
    for info in ds.infos:
        a = info['annos']
        for n in a['name'].tolist():
            raw_names[n] = raw_names.get(n, 0) + 1
        entry, st = stu.target_gt_to_pseudo_label_entry(a['name'], a['gt_boxes_lidar'], a['num_points_in_gt'],
                                                        ds.map_ontology_dataset_to_model, names,
                                                        min_points=min_points, shift_coor=shift)
        assert info['frame_id'] not in labels, info['frame_id']
        labels[info['frame_id']] = entry
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
    counts = stu.validate_target_gt_pseudo_labels(labels, names, min_points=min_points)
    per_frame = np.array([len(e['gt_boxes']) for e in labels.values()])
    meta = {
        'what': 'ANALYSIS PL-GT: Waymo TRAIN ground truth as frozen pseudo-labels (experiments_md 20261027_05). '
                'Target labels are training targets; never a method.',
        'config': args.cfg, 'dataset': tar.DATASET, 'split': tar.DATA_SPLIT['train'],
        'sampled_interval_train': int(tar.SAMPLED_INTERVAL['train']), 'n_frames': len(labels),
        'class_names': names, 'map_dataset_to_model': dict(ds.map_ontology_dataset_to_model),
        'min_points_of_gt': int(min_points), 'shift_coor': list(shift) if shift is not None else None,
        'filters': "drop 'unknown'; dataset->model ontology; stored num_points_in_gt >= MIN_POINTS_OF_GT; "
                   "CLASS_NAMES only; SHIFT_COOR added. Range filter online (after augmentation), as for the oracle.",
        'counts': dict(zip(names, counts.tolist())), 'filter_totals': tot, 'raw_names': raw_names,
        'boxes_per_frame_max': int(per_frame.max()), 'frames_over_500': int((per_frame > 500).sum()),
        'st3d_head': head_sha(), 'written': time.strftime('%Y-%m-%d %H:%M %Z'),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump({'format': stu.ANALYSIS_TARGET_GT_FORMAT, 'meta': meta, 'labels': labels}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)
    with open(out + '.stats.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def check(args):
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    from pcdet.models.dense_heads.center_head import CenterHead
    cfg = load_cfg(args.cfg)
    tar = cfg.DATA_CONFIG_TAR
    names = list(cfg.CLASS_NAMES)
    C = len(names)
    path = stu.analysis_target_gt_pseudo_label_path(cfg)
    res = {'config': args.cfg, 'file': path, 'st3d_head': head_sha()}

    # 1. the trainer's loader (validates sentinels), independent recount, frame sets
    labels, meta, counts = stu.load_analysis_target_gt_pseudo_labels(path, names, tar.get('MIN_POINTS_OF_GT', 1))
    allb = np.concatenate([e['gt_boxes'] for e in labels.values()])
    res['1_sentinels'] = {
        'n_frames': len(labels), 'n_boxes': int(len(allb)), 'labels_unique': np.unique(allb[:, 7]).tolist(),
        'negative_labels': int((allb[:, 7] < 0).sum()), 'minus99': int((allb[:, 7] == common_utils.IGNORE_CLASS_LABEL).sum()),
        'zero_labels': int((allb[:, 7] == 0).sum()), 'scores_unique': np.unique(allb[:, 8]).tolist(),
        'counts_loader': dict(zip(names, counts.tolist())),
        'per_frame_mean': dict(zip(names, (counts / len(labels)).round(3).tolist())),
        'boxes_per_frame_max': int(max(len(e['gt_boxes']) for e in labels.values())),
        'num_max_objs': int(cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.NUM_MAX_OBJS)}
    # independent: straight from the per-sequence info pkls, own name map, own filter
    root = (TOOLS / tar.DATA_PATH).resolve() / tar.PROCESSED_DATA_TAG
    seqs = sorted({fid.rsplit('_', 1)[0] for fid in labels})
    indep = {'Vehicle': 0, 'Pedestrian': 0, 'Cyclist': 0, 'Sign': 0, 'unknown': 0}
    indep_kept = {'Vehicle': 0, 'Pedestrian': 0, 'Cyclist': 0}
    indep_empty, seen = 0, 0
    per_frame_ok = 0
    for s in seqs:
        infos = pickle.load(open(root / s / (s + '.pkl'), 'rb'))
        for info in infos:
            fid = info['frame_id']
            if fid not in labels:
                continue
            seen += 1
            nm, npts = info['annos']['name'], info['annos']['num_points_in_gt']
            for k in indep:
                indep[k] += int((nm == k).sum())
            ok = npts >= 1
            indep_empty += int(((nm != 'unknown') & ~ok).sum())
            fk = {k: int(((nm == k) & ok).sum()) for k in indep_kept}
            for k in indep_kept:
                indep_kept[k] += fk[k]
            lab = labels[fid]['gt_boxes'][:, 7]
            per_frame_ok += int([fk['Vehicle'], fk['Pedestrian'], fk['Cyclist']] ==
                                [int((lab == 1).sum()), int((lab == 2).sum()), int((lab == 3).sum())])
    ds = target_dataset(cfg)
    served = [info['frame_id'] for info in ds.infos]
    teacher = pickle.load(open(args.teacher_ps, 'rb'))
    res['1_counts'] = {
        'independent_raw_names': indep, 'independent_empty_dropped': indep_empty,
        'independent_kept': indep_kept, 'frames_seen_in_info_pkls': seen,
        'frames_per_class_counts_equal': per_frame_ok,
        'stored_equals_independent': [indep_kept['Vehicle'], indep_kept['Pedestrian'], indep_kept['Cyclist']] == counts.tolist(),
        'served_frames': len(served), 'stored_frames_eq_served': set(served) == set(labels),
        'stored_frames_eq_27704': set(teacher) == set(labels)}
    print(json.dumps({k: res[k] for k in ('1_sentinels', '1_counts')}, indent=1), flush=True)

    # 2. frame and heading against 27704's stored teacher pseudo-labels (all frames)
    best, dh, off = [], [], []
    n_t = 0
    for fid, e in labels.items():
        g = e['gt_boxes'][e['gt_boxes'][:, 7] == 1]
        t = teacher[fid]['gt_boxes']
        t = t[(t[:, 7] == 1) & (t[:, 8] >= 0.5)]
        n_t += len(t)
        if len(t) == 0:
            continue
        if len(g) == 0:
            best += [0.0] * len(t)
            continue
        iou = boxes_bev_iou_cpu(t[:, :7].astype(np.float32), g[:, :7].astype(np.float32))
        iou = iou.numpy() if hasattr(iou, 'numpy') else np.asarray(iou)
        j = iou.argmax(1)
        b = iou[np.arange(len(t)), j]
        best += b.tolist()
        m = b >= 0.5
        dh += np.abs(wrap(t[m, 6] - g[j[m], 6])).tolist()
        off += (t[m, :3] - g[j[m], :3]).tolist()
    best, dh, off = np.array(best), np.array(dh), np.array(off).reshape(-1, 3)
    res['2_frame_vs_27704'] = {
        'confident_teacher_cars': int(n_t), 'best_bev_iou_median': float(np.median(best)),
        'share_iou_ge_0.5': float((best >= 0.5).mean()), 'share_iou_ge_0.7': float((best >= 0.7).mean()),
        'matched_iou_ge_0.5': int(len(dh)), 'heading_absdiff_median_rad': float(np.median(dh)),
        'share_heading_within_0.2rad': float((dh < 0.2).mean()), 'share_heading_flipped_gt_2.9rad': float((dh > 2.9).mean()),
        'centre_offset_median_xyz_m': np.median(off, 0).round(3).tolist()}
    print(json.dumps(res['2_frame_vs_27704'], indent=1), flush=True)
    del teacher

    # 3. stored-count >= 1 boxes with no point of the loaded first-return cloud (what a recount would drop)
    step = max(1, len(ds.infos) // args.empty_frames)
    emp, tot = np.zeros(C), np.zeros(C)
    shift = np.array(tar.get('SHIFT_COOR', [0, 0, 0]), dtype=np.float32)
    for info in ds.infos[::step][:args.empty_frames]:
        e = labels[info['frame_id']]
        if len(e['gt_boxes']) == 0:
            continue
        pts = ds.get_lidar_with_sweeps(info)
        pts[:, :3] += shift
        inside = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(np.ascontiguousarray(pts[:, :3], dtype=np.float32)),
            torch.from_numpy(e['gt_boxes'][:, :7].astype(np.float32))).numpy().sum(1)
        lab = e['gt_boxes'][:, 7].astype(int)
        for c in range(C):
            tot[c] += (lab == c + 1).sum()
            emp[c] += ((lab == c + 1) & (inside == 0)).sum()
    res['3_empty_in_loaded_cloud'] = {'frames': int(min(args.empty_frames, len(ds.infos[::step]))),
                                      'boxes': dict(zip(names, tot.astype(int).tolist())),
                                      'empty_share': dict(zip(names, (emp / np.maximum(tot, 1)).round(4).tolist()))}
    print(json.dumps(res['3_empty_in_loaded_cloud'], indent=1), flush=True)

    # 4. loader equality with augmentation off, labels installed as the trainer installs them
    stu.PSEUDO_LABELS.clear()
    stu.PSEUDO_LABELS.update(labels)
    ds.set_pseudo_labels(stu.PSEUDO_LABELS)
    ds.data_augmentor.data_augmentor_queue = []
    pcr = np.array(tar.POINT_CLOUD_RANGE, dtype=np.float32)
    rng_cfg = [p for p in tar.DATA_PROCESSOR if p.NAME == 'mask_points_and_boxes_outside_range'][0]
    step = max(1, len(ds) // args.equal_frames)
    n_eq = n_fr = n_box = n_out = 0
    bad = []
    for i in list(range(0, len(ds), step))[:args.equal_frames]:
        d = ds[i]
        fid = ds.infos[i]['frame_id']
        s = labels[fid]['gt_boxes']
        keep = box_utils.mask_boxes_outside_range_numpy(s[:, :7], pcr, rng_cfg.get('min_num_corners', 1)) \
            if len(s) else np.zeros(0, bool)
        exp = s[keep]
        got = d['gt_boxes']
        same = got.shape[0] == exp.shape[0] and np.allclose(got[:, :6], exp[:, :6], atol=1e-4) and \
            np.allclose(np.abs(wrap(got[:, 6] - exp[:, 6])), 0, atol=1e-4) and np.array_equal(got[:, 7], exp[:, 7])
        cnt_ok = np.array_equal(d['pos_ps_bbox'], np.bincount(s[:, 7].astype(int) - 1, minlength=C).astype(np.float32)) \
            and not d['ign_ps_bbox'].any()
        n_fr += 1
        n_eq += int(same and cnt_ok)
        n_box += len(s)
        n_out += int((~keep).sum())
        if not (same and cnt_ok) and len(bad) < 5:
            bad.append({'frame': fid, 'got': int(got.shape[0]), 'expected': int(exp.shape[0]), 'counts_ok': bool(cnt_ok)})
    res['4_loader_equality_no_aug'] = {'frames': n_fr, 'frames_equal': n_eq, 'stored_boxes': n_box,
                                       'outside_range': n_out, 'failures': bad}
    print(json.dumps(res['4_loader_equality_no_aug'], indent=1), flush=True)

    # 5. assigned targets as trained (fresh dataset: augmentation on, as the row's target loader)
    ds = target_dataset(cfg)
    ds.set_pseudo_labels(stu.PSEUDO_LABELS)
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.model_cfg = cfg.MODEL.DENSE_HEAD
    head.class_names = names
    head.class_names_each_head = [list(x) for x in cfg.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD]
    assert head.class_names_each_head == [names], head.class_names_each_head
    head.point_cloud_range = pcr
    head.voxel_size = [p for p in tar.DATA_PROCESSOR if p.NAME == 'transform_points_to_voxels'][0].VOXEL_SIZE
    tac = cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG
    stride = tac.FEATURE_MAP_STRIDE
    side = int(round((pcr[3] - pcr[0]) / head.voxel_size[0] / stride))
    cell = head.voxel_size[0] * stride
    t5 = {'pos': np.zeros(C), 'stored': np.zeros(C), 'peaks': np.zeros(C), 'pixels': np.zeros(C), 'neg': 0,
          'reg': 0, 'ign_px': 0, 'frames': 0, 'pts': {c: [] for c in range(C)}}
    fails = []
    step = max(1, len(ds) // args.dump_frames)
    for i in list(range(0, len(ds), step))[:args.dump_frames]:
        d = ds[i]
        gt = d['gt_boxes']
        lab = gt[:, 7].astype(int)
        s = labels[ds.infos[i]['frame_id']]['gt_boxes']
        t5['neg'] += int((lab <= 0).sum())
        for c in range(C):
            t5['pos'][c] += (lab == c + 1).sum()
            t5['stored'][c] += (s[:, 7] == c + 1).sum()
        if len(gt):
            inside = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(np.ascontiguousarray(d['points'][:, :3], dtype=np.float32)),
                torch.from_numpy(gt[:, :7].astype(np.float32))).numpy().sum(1)
            for c in range(C):
                t5['pts'][c] += inside[lab == c + 1].tolist()
        t = head.assign_targets(torch.from_numpy(gt[None]).float().clone(), feature_map_size=[side, side])
        hm = t['heatmaps'][0][0]
        if t['heatmap_masks'][0] is not None:
            t5['ign_px'] += int((t['heatmap_masks'][0][0] == 0).sum())
        pos_pix = {}
        for b, l in zip(gt, lab):
            if l > 0 and b[3] > 0 and b[4] > 0:
                x = min(max((b[0] - pcr[0]) / cell, 0), side - 0.5)
                y = min(max((b[1] - pcr[1]) / cell, 0), side - 0.5)
                pos_pix.setdefault(l - 1, set()).add((int(y), int(x)))
        for c in range(C):
            pk, ex = int(hm[c].eq(1).sum()), len(pos_pix.get(c, ()))
            t5['peaks'][c] += pk
            t5['pixels'][c] += ex
            if pk != ex:
                fails.append('frame %d %s: peaks %d != distinct positive pixels %d' % (i, names[c], pk, ex))
        t5['reg'] += int(t['masks'][0].sum())
        t5['frames'] += 1
    res['5_assigned_targets'] = {
        'frames': t5['frames'], 'stored_positives': dict(zip(names, t5['stored'].astype(int).tolist())),
        'served_positives': dict(zip(names, t5['pos'].astype(int).tolist())),
        'served_over_stored': dict(zip(names, (t5['pos'] / np.maximum(t5['stored'], 1)).round(4).tolist())),
        'non_positive_labels_served': t5['neg'],
        'heatmap_peaks': dict(zip(names, t5['peaks'].astype(int).tolist())),
        'distinct_positive_pixels': dict(zip(names, t5['pixels'].astype(int).tolist())),
        'regression_targets': t5['reg'], 'served_positives_total': int(t5['pos'].sum()),
        'ignored_heatmap_pixels': t5['ign_px'],
        'points_per_box_median': {names[c]: float(np.median(t5['pts'][c])) if t5['pts'][c] else None for c in range(C)},
        'points_per_box_empty_share': {names[c]: float(np.mean(np.array(t5['pts'][c]) == 0)) if t5['pts'][c] else None
                                       for c in range(C)},
        'peak_failures': fails[:10], 'n_peak_failures': len(fails)}
    print(json.dumps(res['5_assigned_targets'], indent=1), flush=True)
    with open(args.out_json, 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', args.out_json)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('cfg')
    b.add_argument('--force', action='store_true')
    c = sub.add_parser('check')
    c.add_argument('cfg')
    c.add_argument('--teacher_ps', required=True)
    c.add_argument('--out_json', required=True)
    c.add_argument('--empty_frames', type=int, default=400)
    c.add_argument('--equal_frames', type=int, default=300)
    c.add_argument('--dump_frames', type=int, default=60)
    a = ap.parse_args()
    build(a) if a.cmd == 'build' else check(a)


if __name__ == '__main__':
    main()
