"""Per-box records for the SELECTION-BIAS sweep: at which score cut does points-per-kept-box equal
points-per-GT-box, per range ring, and does that cut transfer from the labelled source to the target?

Writes one pickle per (dataset, pseudo-label file) holding every stored teacher box (ring, class,
score, points inside, matched-to-GT, centre radius, platform) and every GT box (ring, class, points,
matched, radius, platform) over N strided frames, no augmentation. `selection_bias_report.py` does
the arithmetic. Points per box is box-attributed (each box's own count, binned by its centre), over
OCCUPIED boxes (>= 1 point), the definition 20260927_01 section 4 used.

GT is read on the SOURCE (Lyft val) for selection - that is legal - and on the TARGET (nuScenes)
for diagnosis only, never for selection.

CPU, run from tools/ in the container:
    python analysis/pseudo_label_threshold/selection_bias_sweep.py --cfg_file <cfg> \
        --ps_label <ps_label_e0.pkl> --frames 1000 --out <records.pkl>
"""
import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))   # tools/  (for _init_path)
sys.path.insert(0, str(HERE.parents[1]))   # tools/analysis (for the audit's helpers)
import _init_path  # noqa: F401
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.utils import common_utils
from pseudo_label_foreground_audit import match, disable_augmentation, strided, ring_of  # noqa: E402

LYFT_64 = {'host-a101', 'host-a102'}


def platform_of(frame_id):
    host = str(frame_id).split('_')[0]
    if host in LYFT_64:
        return 64
    if host.startswith('host-'):
        return 40
    return 0


def in_fov(ds, d, boxes):
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    centres = boxes[:, :3].copy()
    shift = ds.dataset_cfg.get('SHIFT_COOR', None)
    if shift:
        centres -= np.asarray(shift, dtype=np.float32)
    calib, shape = d.get('calib'), d.get('image_shape')
    if calib is not None and shape is not None:
        return ds.get_fov_flag(calib.lidar_to_rect(centres), shape, calib, margin=5)
    return np.abs(np.arctan2(centres[:, 1], centres[:, 0])) <= np.deg2rad(45.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--ps_label', required=True)
    ap.add_argument('--frames', type=int, default=1000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fov', action='store_true',
                    help='keep only boxes whose centre projects into the camera image (KITTI: its '
                         'labels cover the camera FOV only, so a box behind the vehicle has no GT to '
                         'match and is not a false positive). Uses the dataset\'s own get_fov_flag on '
                         'the un-shifted centre, 5 px margin, like the eval-time FOV_FILTER.')
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    dc = copy.deepcopy(cfg.DATA_CONFIG_TAR)
    dc.USE_PSEUDO_LABEL = False
    ds, _, _ = build_dataloader(dc, cfg.CLASS_NAMES, 1, False, workers=0, logger=logger,
                                training=True, model_ontology=cfg.get('ONTOLOGY', None))
    disable_augmentation(ds)
    ps = pickle.load(open(args.ps_label, 'rb'))
    logger.info('%d frames in dataset, %d pseudo-labelled frames' % (len(ds), len(ps)))

    ps_rec, gt_rec = [], []
    frames = missing = 0
    for k, idx in enumerate(strided(len(ds), args.frames)):
        d = ds[idx]
        fid = d['frame_id']          # prepare_data may resample an empty frame: use what came back
        if fid not in ps:
            missing += 1
            continue
        frames += 1
        plat = platform_of(fid)
        pts = np.asarray(d['points'])[:, :3]
        gb = np.asarray(d['gt_boxes'], dtype=np.float32).reshape(-1, 8)
        gb = gb[(gb[:, 3:6] > 1e-3).all(axis=1)]
        pb = np.asarray(ps[fid]['gt_boxes'], dtype=np.float32).reshape(-1, 9)
        pb = pb[(pb[:, 3:6] > 1e-3).all(axis=1)]
        if args.fov:
            gb, pb = gb[in_fov(ds, d, gb)], pb[in_fov(ds, d, pb)]
        pcls = np.abs(pb[:, 7]).astype(int)
        gcls = gb[:, 7].astype(int)
        n_gt = DataProcessor.box_occupancy(pts, gb[:, :7])[1] if len(gb) else np.zeros(0)
        n_ps = DataProcessor.box_occupancy(pts, pb[:, :7])[1] if len(pb) else np.zeros(0)
        tp, used = match(pb[:, :7], pcls, pb[:, 8], gb[:, :7], gcls) if len(pb) and len(gb) else (
            np.zeros(len(pb), bool), np.zeros(len(gb), bool))
        rp = np.linalg.norm(pb[:, :2], axis=1)
        rg = np.linalg.norm(gb[:, :2], axis=1)
        for i in range(len(pb)):
            ps_rec.append((ring_of(rp[i]), pcls[i], pb[i, 8], n_ps[i], tp[i], rp[i], plat, frames))
        for j in range(len(gb)):
            gt_rec.append((ring_of(rg[j]), gcls[j], n_gt[j], used[j], rg[j], plat, frames))
        if (k + 1) % 100 == 0:
            logger.info('%d frames done (%d ps boxes, %d gt boxes)' % (k + 1, len(ps_rec), len(gt_rec)))

    out = {'ps': np.array(ps_rec, dtype=np.float64).reshape(-1, 8),
           'gt': np.array(gt_rec, dtype=np.float64).reshape(-1, 7),
           'ps_cols': ['ring', 'cls', 'score', 'npts', 'tp', 'r', 'platform', 'frame'],
           'gt_cols': ['ring', 'cls', 'npts', 'matched', 'r', 'platform', 'frame'],
           'frames': frames, 'missing': missing, 'classes': list(cfg.CLASS_NAMES),
           'cfg': args.cfg_file, 'ps_label': args.ps_label, 'fov': bool(args.fov)}
    pickle.dump(out, open(args.out, 'wb'))
    logger.info('wrote %s: %d frames (%d had no pseudo-labels), %d ps boxes, %d gt boxes'
                % (args.out, frames, missing, len(ps_rec), len(gt_rec)))


if __name__ == '__main__':
    main()
