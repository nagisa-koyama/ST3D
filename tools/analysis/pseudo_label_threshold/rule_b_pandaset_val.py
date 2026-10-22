"""Rule B (source-val precision cut) for a PandaSet-source RING_PATTERN teacher, on PandaSet VAL through the pattern.

Pre-declared in experiments_md/20261022_07 §3.1 before any number. Same rule and constants as source_precision_cut.py
(imported from it): per class, the teacher's predictions on its labelled SOURCE val, GT with >= 1 point and centre within
|x|, |y| <= 75.2 m, greedy 1.0 m BEV centre match in descending score, precision P(t) on the 0.01 grid, cut = the smallest t
with P >= 0.50 wherever >= 50 predictions are kept, else 1.01.

Why its own script and not generate_pseudo_labels.py + source_precision_cut.py:
  (i)   PandasetDataset applies RING_PATTERN in TRAINING mode only (no APPLY_AT_EVAL), and the generator puts the set in
        eval mode, so the teacher would label the raw Pandar64 cloud instead of its training input;
  (ii)  its `frame_id` is the in-sequence index 0..79, so pseudo-labels keyed by it overwrite each other across sequences;
  (iii) PandaSet infos carry no boxes: cuboids are stored in WORLD coordinates and go through the frame's pose and the
        axis swap, so GT must be read in the loader's frame, not from the infos.
So this script subclasses the dataset FOR THIS PASS ONLY (nothing in pcdet is edited): raw device-0 points in the
normative frame, the training-path ring mask (`_ring_keep_mask`, on the exact pre-swap ego points, np.random phases as in
training), SHIFT_COOR on points and boxes, GT from `_get_annotations(info, pose)` mapped to the model vocabulary with the
dataset's own ontology map, frames keyed by (sequence, frame_idx). GT point counts are taken on the RAW Pandar64 cloud
(the analogue of Waymo's stored num_points_in_gt, the disclosed choice of 20261017_06 §3). The teacher is
SELF_TRAIN.MODEL_TEACHER of --cfg_file with the checkpoint AS TRAINED, eval mode, its configured post-processing.

Pre-check (bug-prevention rule 4, a new path): on --check frames, with the same np.random state, the thinned cloud equals
the training-mode `_get_lidar_points` cloud point for point; a mismatch aborts.

    python analysis/pseudo_label_threshold/rule_b_pandaset_val.py \
        --cfg_file cfgs/da-ieee-access/centerpoint-ringpattern-st3d-pandaset2nuscenes.yaml \
        --teacher_ckpt /storage/wandb/run-20261010_193142-lf9c36yn/files/ckpt/checkpoint_epoch_115.pth \
        --out_dir /storage/pseudo_labels/lf9c36yn_ep115_pandaset_val_ringpattern
"""
import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _init_path  # noqa: F401,E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_precision_cut import GRID, MATCH_DIST, MIN_KEPT, P_STAR, RANGE, match  # noqa: E402

EXTRA_KEYS = ('gt_pack', 'frame_key', 'n_raw', 'n_thin')


class PandasetValThroughPattern(PandasetDataset):
    """Eval-mode PandaSet whose points go through the TRAINING ring mask; GT packed beside, not through prepare_data."""

    def raw_points(self, info, pose):
        """Device-0 cloud in the normative frame, no thinning (the parent's eval path), no SHIFT_COOR."""
        assert not self.training and self.eval_ring_thin_cfg is None
        return PandasetDataset._get_lidar_points(self, info, pose)

    def thinned_points(self, info, pose, raw):
        """The parent's training-mode mask on the exact pre-swap ego points (normative x = ego y, normative y = -ego x)."""
        ego = np.stack([-raw[:, 1], raw[:, 0], raw[:, 2]], axis=1)
        keep = self._ring_keep_mask(info, ego)
        return raw[keep]

    def __getitem__(self, index):
        info = self.pandaset_infos[index]
        pose = self._get_pose(info)
        raw = self.raw_points(info, pose)
        pts = self.thinned_points(info, pose, raw)
        boxes, labels, _ = self._get_annotations(info, pose)
        shift = self.dataset_cfg.get('SHIFT_COOR', None)
        if shift:
            raw = raw.copy(); raw[:, 0:3] += np.array(shift, dtype=np.float32)
            pts = pts.copy(); pts[:, 0:3] += np.array(shift, dtype=np.float32)
            boxes = boxes.copy(); boxes[:, 0:3] += np.array(shift, dtype=np.float32)
        names = [self.map_ontology_dataset_to_model[n] for n in labels]  # strict, as prepare_data
        cls = np.array([self.class_names.index(n) + 1 if n in self.class_names else 0 for n in names], dtype=np.float32)
        if len(boxes):
            inb = roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(raw[:, :3]).float(),
                                                           torch.from_numpy(boxes[:, :7]).float())
            npts = inb.sum(1).numpy().astype(np.float32)
        else:
            npts = np.zeros(0, np.float32)
        pack = np.concatenate([boxes[:, :7].astype(np.float32), cls[:, None], npts[:, None]], axis=1) \
            if len(boxes) else np.zeros((0, 9), np.float32)
        # gt_boxes / gt_names go through prepare_data as in the parent's __getitem__ (its ontology remap needs them);
        # the rule reads `gt_pack`, never the processed gt_boxes (processors may drop or reorder them).
        data_dict = self.prepare_data(data_dict={'points': pts, 'gt_boxes': boxes, 'gt_names': labels,
                                                 'frame_id': '%s/%02d' % (info['sequence'], int(info['frame_idx']))})
        data_dict['gt_pack'] = pack
        data_dict['frame_key'] = (info['sequence'], int(info['frame_idx']))
        data_dict['n_raw'] = len(raw)
        data_dict['n_thin'] = len(pts)
        return data_dict


def collate(batch):
    extras = {k: [b.pop(k) for b in batch] for k in EXTRA_KEYS}
    out = PandasetDataset.collate_batch(batch)
    out.update(extras)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg_file', required=True, help='the P-PL config: DATA_CONFIG = the teacher\'s thinned PandaSet source')
    ap.add_argument('--teacher_ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=6)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--check', type=int, default=20, help='frames for the training-path equality pre-check')
    ap.add_argument('--classes', nargs='+', default=['Car', 'Pedestrian', 'Cyclist'])
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(out / 'rule_b.log')
    cfg_from_yaml_file(args.cfg_file, cfg)
    src = copy.deepcopy(cfg.DATA_CONFIG)
    assert src.DATASET == 'PandasetDataset' and src.get('RING_PATTERN', None) is not None, 'not a PandaSet RING_PATTERN source'
    val_infos = list(src.INFO_PATH['test'])
    src.DATA_SPLIT = {'train': 'val', 'test': 'val'}
    src.INFO_PATH = {'train': val_infos, 'test': val_infos}
    ds = PandasetValThroughPattern(dataset_cfg=src, class_names=cfg.CLASS_NAMES, root_path=None, training=False,
                                   logger=logger, model_ontology=cfg.get('ONTOLOGY', None))
    logger.info('PandaSet val through RING_PATTERN %s: %d frames, SHIFT_COOR %s, infos %s'
                % (dict(src.RING_PATTERN), len(ds), src.get('SHIFT_COOR'), val_infos))

    # Pre-check: the script's thinning equals the training path point for point under the same np.random state.
    n_ok = 0
    ratios = []
    for i in range(0, len(ds), max(1, len(ds) // max(args.check, 1)))[:args.check]:
        info = ds.pandaset_infos[i]; pose = ds._get_pose(info)
        raw = ds.raw_points(info, pose)
        st = np.random.get_state()
        mine = ds.thinned_points(info, pose, raw)
        np.random.set_state(st)
        ds.training = True
        try:
            theirs = PandasetDataset._get_lidar_points(ds, info, pose)
        finally:
            ds.training = False
        same = mine.shape == theirs.shape and np.array_equal(mine, theirs)
        n_ok += int(same)
        ratios.append(len(mine) / max(len(raw), 1))
        if not same:
            logger.error('pre-check MISMATCH on %s/%02d: mine %s vs training path %s' % (info['sequence'], info['frame_idx'], mine.shape, theirs.shape))
    logger.info('pre-check: thinned cloud equals the training-mode cloud on %d / %d frames; keep ratio mean %.3f'
                % (n_ok, min(args.check, len(ds)), float(np.mean(ratios))))
    if n_ok != min(args.check, len(ds)):
        raise SystemExit('pre-check failed: the pass does not reproduce the training-mode thinning')

    teacher_cfg = cfg.SELF_TRAIN.MODEL_TEACHER
    teacher_classes = teacher_cfg.get('CLASS_NAMES', None) or cfg.CLASS_NAMES
    model = build_network(model_cfg=teacher_cfg, num_class=len(teacher_classes), dataset=ds)
    model.load_params_from_file(filename=args.teacher_ckpt, to_cpu=False, logger=logger)
    model.cuda().eval()
    logger.info('teacher post-processing: SCORE_THRESH %s, MAX_OBJ %s, NMS %s' % (
        teacher_cfg.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH, teacher_cfg.DENSE_HEAD.POST_PROCESSING.MAX_OBJ_PER_SAMPLE,
        dict(teacher_cfg.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG)))

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate,
                        pin_memory=True)
    frames = {}
    n_raw = n_thin = 0
    with torch.no_grad():
        for it, batch in enumerate(loader):
            extras = {k: batch.pop(k) for k in EXTRA_KEYS}
            load_data_to_gpu(batch)
            pred_dicts, _ = model(batch)
            for b, pd in enumerate(pred_dicts):
                boxes = pd['pred_boxes'].cpu().numpy(); labels = pd['pred_labels'].cpu().numpy(); scores = pd['pred_scores'].cpu().numpy()
                frames[extras['frame_key'][b]] = {'pred': np.concatenate([boxes, labels[:, None], scores[:, None]], axis=1).astype(np.float32),
                                                  'gt': extras['gt_pack'][b]}
                n_raw += extras['n_raw'][b]; n_thin += extras['n_thin'][b]
            if it % 50 == 0:
                logger.info('pass %d / %d batches' % (it + 1, len(loader)))
    logger.info('wrote %d frames; points per frame raw %.0f -> thinned %.0f (ratio %.3f); predictions per frame %.1f'
                % (len(frames), n_raw / len(frames), n_thin / len(frames), n_thin / max(n_raw, 1),
                   np.mean([len(f['pred']) for f in frames.values()])))
    at_cap = sum(len(f['pred']) >= teacher_cfg.DENSE_HEAD.POST_PROCESSING.MAX_OBJ_PER_SAMPLE for f in frames.values())
    logger.info('frames at the rank cap (%d boxes): %d' % (teacher_cfg.DENSE_HEAD.POST_PROCESSING.MAX_OBJ_PER_SAMPLE, at_cap))
    with open(out / 'rule_b_pass.pkl', 'wb') as f:
        pickle.dump(frames, f)

    # Rule B, the code of source_precision_cut.py with GT from the pass.
    classes = {cfg.CLASS_NAMES.index(n) + 1: n for n in args.classes}
    rows = {c: [] for c in classes}
    gt_total = {c: 0 for c in classes}
    for fk, fr in frames.items():
        gt, p_all = fr['gt'], fr['pred']
        for c in classes:
            g = gt[(gt[:, 7] == c) & (gt[:, 8] >= 1)]
            g = g[(np.abs(g[:, 0]) <= RANGE) & (np.abs(g[:, 1]) <= RANGE)] if len(g) else g
            gt_total[c] += len(g)
            p = p_all[np.abs(p_all[:, 7]) == c]
            p = p[(np.abs(p[:, 0]) <= RANGE) & (np.abs(p[:, 1]) <= RANGE)]
            hit = match(p[:, :2], p[:, 8], g[:, :2] if len(g) else np.zeros((0, 2)))
            rows[c] += list(zip(p[:, 8], hit))
    print(f'source val frames read: {len(frames)} of {len(ds)} (pandaset, through RING_PATTERN; match {MATCH_DIST} m, '
          f'range {RANGE} m, P* {P_STAR}, min kept {MIN_KEPT})')
    cuts = {}
    for c, name in classes.items():
        a = np.array(rows[c], dtype=float).reshape(-1, 2)
        prec, kept = [], []
        for t in GRID:
            k = a[a[:, 0] >= t]
            kept.append(len(k))
            prec.append(k[:, 1].mean() if len(k) else np.nan)
        prec, kept = np.array(prec), np.array(kept)
        valid = kept >= MIN_KEPT
        cut = 1.01
        for i, t in enumerate(GRID):
            later = valid[i:]
            if valid[i] and np.all(prec[i:][later] >= P_STAR):
                cut = float(t)
                break
        cuts[name] = cut
        show = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        curve = '  '.join(f'{t:.2f}:{prec[GRID == t][0]:.2f}({kept[GRID == t][0]})' for t in show)
        print(f'{name:10s} GT boxes (>= 1 raw point, in range): {gt_total[c]}')
        print(f'{name:10s} precision(kept) at t: {curve}')
        print(f'{name:10s} cut = {cut:.2f}' + ('  (gated off: precision never holds >= 0.50)' if cut > 1 else ''))
    print('cuts:', cuts)


if __name__ == '__main__':
    main()
