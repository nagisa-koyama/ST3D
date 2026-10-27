from collections import Counter
import torch
import os
import glob
import tqdm
import wandb
import numpy as np
import torch.distributed as dist
import torchinfo
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils, memory_ensemble_utils, ptsn_utils
import pickle as pkl
import re
from pcdet.models.model_utils.dsnorm import set_ds_target
from ..utils.ontology_mapping import get_ontology_mapping

PSEUDO_LABELS = {}
NEW_PSEUDO_LABELS = {}


def check_already_exsit_pseudo_label(ps_label_dir, start_epoch):
    """
    if we continue training, use this to directly
    load pseudo labels from exsiting result pkl

    if exsit, load latest result pkl to PSEUDO LABEL
    otherwise, return false and

    Args:
        ps_label_dir: dir to save pseudo label results pkls.
        start_epoch: start epoc
    Returns:

    """
    # support init ps_label given by cfg
    if start_epoch == 0 and cfg.SELF_TRAIN.get('INIT_PS', None):
        if os.path.exists(cfg.SELF_TRAIN.INIT_PS):
            init_ps_label = pkl.load(open(cfg.SELF_TRAIN.INIT_PS, 'rb'))
            PSEUDO_LABELS.update(init_ps_label)

            if cfg.LOCAL_RANK == 0:
                ps_path = os.path.join(ps_label_dir, "ps_label_e0.pkl")
                with open(ps_path, 'wb') as f:
                    pkl.dump(PSEUDO_LABELS, f)

            return cfg.SELF_TRAIN.INIT_PS

    ps_label_list = glob.glob(os.path.join(ps_label_dir, 'ps_label_e*.pkl'))
    if len(ps_label_list) == 0:
        return

    ps_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in ps_label_list:
        num_epoch = re.findall('ps_label_e(.*).pkl', cur_pkl)
        assert len(num_epoch) == 1

        # load pseudo label and return
        if int(num_epoch[0]) <= start_epoch:
            latest_ps_label = pkl.load(open(cur_pkl, 'rb'))
            PSEUDO_LABELS.update(latest_ps_label)
            return cur_pkl

    return None


def save_pseudo_label_epoch(model, val_loader, rank, leave_pbar, ps_label_dir, cur_epoch):
    """
    Generate pseudo label with given model.

    Args:
        model: model to predict result for pseudo label
        val_loader: data_loader to predict pseudo label
        rank: process rank
        leave_pbar: tqdm bar controller
        ps_label_dir: dir to save pseudo label
        cur_epoch
    """
    # Override point feature encoder to use only xyz
    point_feature_encoder_orig = val_loader.dataset.dataset.point_feature_encoder
    val_loader.dataset.dataset.point_feature_encoder.point_encoding_config.used_feature_list = ['x', 'y', 'z']
    val_loader.dataset.dataset.point_feature_encoder.used_feature_list = ['x', 'y', 'z']

    # DALI PTSN: the generation loader's dataset scales its points in the worker, so the boxes
    # coming back are in scaled metres and have to be divided by the same factor before anything
    # stores them. Read it off that dataset rather than from cfg a second time - a forward
    # transform and an inverse read from two places can drift apart. See pcdet/utils/ptsn_utils.py.
    ptsn_scale = float(getattr(val_loader.dataset.dataset, 'ptsn_scale', 1.0))

    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='generate_ps_e%d' % cur_epoch, dynamic_ncols=True)
        if ptsn_scale != 1.0:
            print('self-training: PTSN active, generating pseudo labels at input scale %.4f '
                  '(predicted boxes divided by it before storage)' % ptsn_scale)

    pos_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    ign_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))

    if cfg.SELF_TRAIN.get('DSNORM', None):
        model.apply(set_ds_target)

    model.eval()

    for cur_it in range(total_it_each_epoch):
        try:
            target_batch = next(val_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(val_loader)
            target_batch = next(target_dataloader_iter)

        # generate gt_boxes for target_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(target_batch)
            pred_dicts, ret_dict = model(target_batch)

        student_ontology = cfg.get('ONTOLOGY', None)
        pos_ps_batch_nmeters, ign_ps_batch_nmeters = save_pseudo_label_batch(
            target_batch, pred_dicts=pred_dicts,
            need_update=(cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE', None) and
                         cfg.SELF_TRAIN.MEMORY_ENSEMBLE.ENABLED and
                         cur_epoch > 0),
            student_ontology=student_ontology,
            ptsn_scale=ptsn_scale
        )

        # log to console and tensorboard
        pos_ps_nmeter.update(pos_ps_batch_nmeters)
        ign_ps_nmeter.update(ign_ps_batch_nmeters)
        pos_ps_result = pos_ps_nmeter.aggregate_result()
        ign_ps_result = ign_ps_nmeter.aggregate_result()

        disp_dict = {'pos_ps_box': pos_ps_result, 'ign_ps_box': ign_ps_result}

        if rank == 0:
            pbar.update()
            pbar.set_postfix(disp_dict)
            pbar.refresh()

    if rank == 0:
        pbar.close()

    # Set original point feature encoder back
    val_loader.dataset.dataset.point_feature_encoder = point_feature_encoder_orig

    gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch)


def gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch):
    commu_utils.synchronize()

    if dist.is_initialized():
        part_pseudo_labels_list = commu_utils.all_gather(NEW_PSEUDO_LABELS)

        new_pseudo_label_dict = {}
        for pseudo_labels in part_pseudo_labels_list:
            new_pseudo_label_dict.update(pseudo_labels)

        NEW_PSEUDO_LABELS.update(new_pseudo_label_dict)

    # dump new pseudo label to given dir
    if rank == 0:
        ps_path = os.path.join(ps_label_dir, "ps_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(NEW_PSEUDO_LABELS, f)

    commu_utils.synchronize()
    PSEUDO_LABELS.clear()
    PSEUDO_LABELS.update(NEW_PSEUDO_LABELS)
    teacher_class_names_concat = []
    for frame_id, pseudo_label in PSEUDO_LABELS.items():
        if 'teacher_classes' in pseudo_label:
            teacher_class_names_concat.extend(pseudo_label['teacher_classes'])
    # print("len(PSEUDO_LABELS.keys()):", len(PSEUDO_LABELS.keys()))
    # print("len(teacher_class_names_concat):", len(teacher_class_names_concat))
    # print(Counter(teacher_class_names_concat))
    if rank == 0:
        wandb.log({'train/ps_labels_total': len(teacher_class_names_concat)})
        for key, value in Counter(teacher_class_names_concat).items():
            wandb.log({'train/ps_labels_{}'.format(key): value})

    NEW_PSEUDO_LABELS.clear()


def save_pseudo_label_batch(input_dict,
                            pred_dicts=None,
                            need_update=True,
                            student_ontology=None,
                            ptsn_scale=1.0):
    """
    Save pseudo label for give batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.

    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
        need_update: Bool.
            If set to true, use consistency matching to update pseudo label
        ptsn_scale: float. DALI PTSN input scale the generation loader applied to the points.
            Predicted boxes arrive in that scaled frame and are divided by it here, so everything
            downstream - thresholding, memory ensembling, the stored labels - is in target metres.
    """
    pos_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    ign_ps_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))

    # If teacher model is given, use ontology mapping to convert teacher class to student class.
    if cfg.SELF_TRAIN.get('MODEL_TEACHER', None):
        teacher_class_names = cfg.SELF_TRAIN.MODEL_TEACHER.get('CLASS_NAMES', None)
        teacher_ontology = cfg.SELF_TRAIN.MODEL_TEACHER.get('ONTOLOGY', None)
    else:
        teacher_class_names = cfg.get('CLASS_NAMES', None)
        teacher_ontology = cfg.get('ONTOLOGY', None)
    student_class_names = cfg.get('CLASS_NAMES', None)

    ontology_mapping_teacher_to_student = get_ontology_mapping(teacher_ontology, student_ontology)

    batch_size = len(pred_dicts)
    for b_idx in range(batch_size):
        pred_cls_scores = pred_iou_scores = None
        if 'pred_boxes' in pred_dicts[b_idx]:
            # Exist predicted boxes passing self-training score threshold
            pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
            # Undo the PTSN input scaling. No-op at 1.0, which is every non-PTSN run.
            pred_boxes = ptsn_utils.unscale_boxes(pred_boxes, ptsn_scale)
            pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()

            # print("pred_labels:", pred_labels)
            # print("type(pred_labels):", type(pred_labels))
            # print("type(pred_labels - 1):", type(pred_labels - 1))
            # print("teacher_class_names:", teacher_class_names)
            # print("type(teacher_class_names):", type(teacher_class_names))
            # print("pred_teacher_labels:", pred_labels)
            pred_teacher_classes = np.array(teacher_class_names)[pred_labels - 1]
            # print("pred_teacher_classes:", pred_teacher_classes)
            pred_student_classes = [ontology_mapping_teacher_to_student[teacher_class] for teacher_class in pred_teacher_classes]
            pred_student_labels = [student_class_names.index(student_class) + 1 for student_class in pred_student_classes]
            # print("pred_student_classes:", pred_student_classes)
            # print("pred_student_labels:", pred_student_labels)
            pred_labels = np.array(pred_student_labels)
            # print("pred_student_labels:", pred_labels)

            # print('pred_dicts[b_idx].keys():', pred_dicts[b_idx].keys())
            pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()
            if 'pred_cls_scores' in pred_dicts[b_idx]:
                pred_cls_scores = pred_dicts[b_idx]['pred_cls_scores'].detach().cpu().numpy()
            if 'pred_iou_scores' in pred_dicts[b_idx]:
                pred_iou_scores = pred_dicts[b_idx]['pred_iou_scores'].detach().cpu().numpy()

            # remove boxes under negative threshold
            if cfg.SELF_TRAIN.get('NEG_THRESH', None) and len(pred_labels) > 0:
                labels_remove_scores = np.array(cfg.SELF_TRAIN.NEG_THRESH)[pred_labels - 1]
                remain_mask = pred_scores >= labels_remove_scores
                # print("labels_remove_scores:", labels_remove_scores)
                # print("pred_scores:", pred_scores)
                # print("remain_mask:", remain_mask)
                pred_labels = pred_labels[remain_mask]
                pred_scores = pred_scores[remain_mask]
                pred_boxes = pred_boxes[remain_mask]
                if 'pred_cls_scores' in pred_dicts[b_idx]:
                    pred_cls_scores = pred_cls_scores[remain_mask]
                else:
                    pred_cls_scores = pred_scores
                if 'pred_iou_scores' in pred_dicts[b_idx]:
                    pred_iou_scores = pred_iou_scores[remain_mask]
                else:
                    pred_iou_scores = pred_scores
                pred_teacher_classes = pred_teacher_classes[remain_mask]


            if len(pred_labels) > 0:
                labels_ignore_scores = np.array(cfg.SELF_TRAIN.SCORE_THRESH)[pred_labels - 1]
                ignore_mask = pred_scores < labels_ignore_scores
                pred_labels[ignore_mask] = -pred_labels[ignore_mask]
                pred_teacher_classes[ignore_mask] = ["Ignored"] * ignore_mask.sum()

            gt_box = np.concatenate((pred_boxes,
                                     pred_labels.reshape(-1, 1),
                                     pred_scores.reshape(-1, 1)), axis=1)
            # print("pred_boxes:", pred_boxes)
            # print("pred_labels:", pred_labels)
            # print("pred_scores:", pred_scores)
            # print("pred_iou_scores:", pred_iou_scores)
            # print("pred_cls_scores:", pred_cls_scores)

        else:
            # no predicted boxes passes self-training score threshold
            gt_box = np.zeros((0, 9), dtype=np.float32)

        gt_infos = {
            'gt_boxes': gt_box,
            'cls_scores': pred_cls_scores,
            'iou_scores': pred_iou_scores,
            'memory_counter': np.zeros(gt_box.shape[0]),
            'teacher_classes': pred_teacher_classes
        }

        # record pseudo label to pseudo label dict
        if need_update:
            ensemble_func = getattr(memory_ensemble_utils, cfg.SELF_TRAIN.MEMORY_ENSEMBLE.NAME)
            gt_infos = memory_ensemble_utils.memory_ensemble(
                PSEUDO_LABELS[input_dict['frame_id'][b_idx]], gt_infos,
                cfg.SELF_TRAIN.MEMORY_ENSEMBLE, ensemble_func
            )
            # print("Count(teacher_classes):", Counter(gt_infos['teacher_classes']))

        # counter the number of ignore boxes for each class
        for i in range(ign_ps_nmeter.n):
            num_total_boxes = (np.abs(gt_infos['gt_boxes'][:, 7]) == (i+1)).sum()
            ign_ps_nmeter.update((gt_infos['gt_boxes'][:, 7] == -(i+1)).sum(), index=i)
            pos_ps_nmeter.update(num_total_boxes - ign_ps_nmeter.meters[i].val, index=i)

        NEW_PSEUDO_LABELS[input_dict['frame_id'][b_idx]] = gt_infos

    return pos_ps_nmeter, ign_ps_nmeter


def load_ps_label(frame_id, pseudo_labels=None):
    """
    :param frame_id: file name of pseudo label
    :param pseudo_labels: the dict to read from. The caller (DatasetTemplate.fill_pseudo_labels)
        passes the copy it CARRIES, which is what a DataLoader worker holds after being pickled
        into a fresh interpreter. Falls back to this module's global for callers that never set
        one - in the main process, or in a FORKED worker, the two are the same dict object.
    :return gt_box: loaded gt boxes (N, 9) [x, y, z, w, l, h, ry, label, scores]
    """
    # Why the parameter exists: under DDP the loader workers are SPAWNED (init_dist_pytorch forces
    # it), and a spawned worker re-imports this module, so its PSEUDO_LABELS is a fresh EMPTY dict
    # however full the main process's is. Reading the global there raised for every frame; the
    # dict has to travel with the dataset object instead. Under fork the global is inherited by
    # copy-on-write and the fallback keeps that path bit-identical.
    labels = pseudo_labels if pseudo_labels is not None else PSEUDO_LABELS
    if frame_id in labels:
        gt_box = labels[frame_id]['gt_boxes']
    else:
        print('pseudo label keys (%d): %s' % (len(labels), list(labels.keys())[:10]))
        raise ValueError(
            'Cannot find pseudo label for frame: %s (%d labels visible in this process%s)'
            % (frame_id, len(labels),
               '' if pseudo_labels is not None else
               '; reading the module global - in a SPAWNED worker that is always empty, so the '
               'dataset must carry the dict: DatasetTemplate.set_pseudo_labels'))

    return gt_box


# ---------------------------------------------------------------------------------------------------------------------
# ANALYSIS ONLY: the target's ground truth as FROZEN pseudo-labels ("PL-GT"; experiments_md 20261027_05, design T05 §5
# fixed 2026-10-27 23:05 after the user's approval). Reached only through DATA_CONFIG_TAR.ANALYSIS_TARGET_GT_AS_PSEUDO_
# LABELS (common_utils.ANALYSIS_TARGET_GT_KEY), absent from every method config. Target labels become training targets:
# such a row never selects or feeds a method value and is labelled analysis everywhere (memory: uda_legality_rule.md).
#
# The file is written once (tools/analysis/plgt_build_target_gt_pseudo_labels.py) as a WRAPPER
#   {'format': ANALYSIS_TARGET_GT_FORMAT, 'meta': {...}, 'labels': {frame_id: entry}}
# so that handing it to any other pseudo-label path (INIT_PS, a ps_label_e*.pkl) fails loudly on the first frame
# ("Cannot find pseudo label") instead of training on it. Each entry has the keys a teacher pass writes
# (save_pseudo_label_batch) plus the stored per-box point count the oracle's zero-point filter reads.
ANALYSIS_TARGET_GT_FORMAT = 'analysis_target_gt_as_pseudo_labels/v1'


def analysis_target_gt_pseudo_label_path(config):
    """The frozen target-GT pseudo-label file an ANALYSIS row names, or None (every other config)."""
    tar = config.get('DATA_CONFIG_TAR', None)
    if tar is None:
        return None
    return tar.get(common_utils.ANALYSIS_TARGET_GT_KEY, None) or None


def target_gt_to_pseudo_label_entry(names, boxes, num_points_in_gt, map_dataset_to_model, class_names,
                                    min_points=1, shift_coor=None):
    """One frame's TARGET ground truth -> one PSEUDO_LABELS entry, through the filters a target ORACLE's training labels
    pass in the loader (WaymoDataset.__getitem__ + DatasetTemplate.prepare_data, as job 26398 trained):
      1. 'unknown' dropped (common_utils.drop_info_with_name);
      2. dataset -> model ontology (Waymo: Vehicle -> Car, Pedestrian, Cyclist, Sign -> Misc); a name missing from the
         map raises, as the loader does;
      3. the zero-point filter on the STORED count: num_points_in_gt >= min_points (MIN_POINTS_OF_GT);
      4. names outside class_names dropped (Misc, i.e. Sign);
      5. SHIFT_COOR added to the centre, as __getitem__ adds it to every box.
    The RANGE filter is not applied here: the loader applies it online after augmentation
    (mask_points_and_boxes_outside_range), to these labels exactly as to the oracle's.

    Returns (entry, stats). entry['gt_boxes'] is (N, 9) float64 [x, y, z, dx, dy, dz, heading, label, score]: label
    in 1..C (MODEL vocabulary, never negative - no ignore band - and never IGNORE_CLASS_LABEL), score 1.0.
    """
    names = np.asarray(names)
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 7)
    npts = np.asarray(num_points_in_gt).reshape(-1)
    assert len(names) == len(boxes) == len(npts), (len(names), len(boxes), len(npts))
    assert boxes.shape[1] >= 7, boxes.shape
    stats = {'raw': len(names)}
    known = names != 'unknown'
    stats['unknown'] = int((~known).sum())
    unmapped = sorted(set(names[known].tolist()) - set(map_dataset_to_model))
    if unmapped:
        raise KeyError('target GT names %s are not in the dataset -> model ontology map %s'
                       % (unmapped, sorted(map_dataset_to_model)))
    model_names = np.array([map_dataset_to_model[n] if k else '' for n, k in zip(names.tolist(), known)],
                           dtype=object)
    has_points = npts >= min_points
    stats['empty'] = int((known & ~has_points).sum())
    in_classes = np.array([n in class_names for n in model_names.tolist()], dtype=bool)
    stats['other_class'] = int((known & has_points & ~in_classes).sum())
    keep = known & has_points & in_classes
    out = boxes[keep, :7].copy()
    if shift_coor is not None:
        out[:, 0:3] += np.asarray(shift_coor, dtype=np.float64)
    kept_names = model_names[keep].astype(str)
    labels = np.array([list(class_names).index(n) + 1 for n in kept_names.tolist()], dtype=np.float64)
    n = len(out)
    gt_box = np.concatenate([out, labels.reshape(-1, 1), np.ones((n, 1), dtype=np.float64)], axis=1)
    entry = {
        'gt_boxes': gt_box,
        'cls_scores': np.ones(n, dtype=np.float32),
        'iou_scores': np.ones(n, dtype=np.float32),
        'memory_counter': np.zeros(n, dtype=np.float64),
        'teacher_classes': np.array(kept_names.tolist(), dtype='<U10'),
        'num_points_in_gt': npts[keep].astype(np.int64),
    }
    for i, c in enumerate(class_names):
        stats['kept_' + c] = int((labels == i + 1).sum())
    stats['kept'] = n
    return entry, stats


def validate_target_gt_pseudo_labels(labels, class_names, min_points=1):
    """Refuse a target-GT pseudo-label dict that is not exactly what the PL-GT design asks for; return per-class counts.

    Every label is a positive class 1..C (no ignore band -1..-C, no IGNORE_CLASS_LABEL, no background 0), every score
    is 1.0, every box has positive extents and a finite pose, and every stored point count passed the zero-point
    filter. Sentinels are tested here, before training, because CenterHead reads any negative label as an ignore
    region (memory: bug_prevention_checklist.md, rule 1)."""
    c = len(class_names)
    counts = np.zeros(c, dtype=np.int64)
    for frame_id, e in labels.items():
        b = np.asarray(e['gt_boxes'])
        assert b.ndim == 2 and b.shape[1] == 9, (frame_id, b.shape)
        lab = b[:, 7]
        assert np.all(lab == np.round(lab)) and np.all((lab >= 1) & (lab <= c)), \
            ('%s: labels must be positive classes 1..%d (no ignore band, no %d, no 0), got %s'
             % (frame_id, c, common_utils.IGNORE_CLASS_LABEL, np.unique(lab).tolist()))
        assert np.all(b[:, 8] == 1.0), (frame_id, 'scores must be 1.0')
        assert np.all(np.isfinite(b[:, :7])) and np.all(b[:, 3:6] > 0), (frame_id, 'non-finite box or extent <= 0')
        npts = np.asarray(e['num_points_in_gt'])
        assert len(npts) == len(b) and np.all(npts >= min_points), (frame_id, 'point counts')
        for k in ('cls_scores', 'iou_scores', 'memory_counter', 'teacher_classes'):
            assert len(e[k]) == len(b), (frame_id, k)
        assert [class_names[int(l) - 1] for l in lab] == [str(t) for t in e['teacher_classes']], \
            (frame_id, 'teacher_classes disagree with the labels')
        counts += np.bincount(lab.astype(np.int64) - 1, minlength=c)
    return counts


def load_analysis_target_gt_pseudo_labels(path, class_names, min_points=1):
    """Read and validate the frozen PL-GT file; returns (labels, meta, per-class counts)."""
    with open(path, 'rb') as f:
        blob = pkl.load(f)
    assert isinstance(blob, dict) and blob.get('format') == ANALYSIS_TARGET_GT_FORMAT, \
        '%s is not a %s file' % (path, ANALYSIS_TARGET_GT_FORMAT)
    meta = blob['meta']
    assert list(meta['class_names']) == list(class_names), (meta['class_names'], list(class_names))
    labels = blob['labels']
    counts = validate_target_gt_pseudo_labels(labels, list(class_names), min_points=min_points)
    return labels, meta, counts
