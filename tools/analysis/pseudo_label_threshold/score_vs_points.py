"""Per-box (class, score, points inside, centre range, TP?) for job 26220's pseudo-labels.
Reuses the audit's frame selection, no-aug target and matching. Run from ST3D/tools."""
import copy, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path('analysis').resolve()))
sys.path.insert(0, str(Path('.').resolve()))
import _init_path  # noqa
from pseudo_label_foreground_audit import strided, disable_augmentation, match
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.utils import common_utils, self_training_utils

cfg_file, ps_path, out = sys.argv[1:4]
cfg_from_yaml_file(cfg_file, cfg)
log = common_utils.create_logger()
C = cfg.CLASS_NAMES
tp_ds, _, _ = build_dataloader(cfg.DATA_CONFIG_TAR, C, 1, False, workers=0, logger=log, training=True,
                               model_ontology=cfg.get('ONTOLOGY'))
rc = copy.deepcopy(cfg.DATA_CONFIG_TAR); rc.USE_PSEUDO_LABEL = False
gt_ds, _, _ = build_dataloader(rc, C, 1, False, workers=0, logger=log, training=True,
                               model_ontology=cfg.get('ONTOLOGY'))
disable_augmentation(tp_ds); disable_augmentation(gt_ds)
self_training_utils.PSEUDO_LABELS.update(pickle.load(open(ps_path, 'rb')))
rows = []   # class, score, npts, range, tp, frame
for k, idx in enumerate(strided(len(tp_ds), 1000)):
    dp, dg = tp_ds[idx], gt_ds[idx]
    pb = np.asarray(dp['gt_boxes']).reshape(-1, 8); pb = pb[pb[:, 7] > 0]
    raw = self_training_utils.PSEUDO_LABELS[dp['frame_id']]['gt_boxes']
    sc = np.array([raw[np.argmin(np.abs(raw[:, :3] - b[:3]).sum(1)), 8] for b in pb])
    gb = np.asarray(dg['gt_boxes']).reshape(-1, 8)
    keep = (pb[:, 3:6] > 1e-3).all(1); pb, sc = pb[keep], sc[keep]
    if not len(pb): continue
    _, per_box, _ = DataProcessor.box_occupancy(dp['points'], pb[:, :7])
    tp, _ = match(pb, pb[:, 7], sc, gb, gb[:, 7])
    r = np.linalg.norm(pb[:, :2], axis=1)
    rows += list(zip(pb[:, 7], sc, per_box, r, tp, [k] * len(pb)))
    if (k + 1) % 200 == 0: log.info('%d/1000' % (k + 1))
pickle.dump(np.array(rows, dtype=float), open(out, 'wb'))
