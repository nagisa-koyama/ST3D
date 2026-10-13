"""Predictions of one checkpoint on chosen Waymo TRAIN frames, WITHOUT evaluation (experiments_md 20261010_04 §5.1,
G1' reference, option A, pre-declared 2026-10-14 00:10 JST).

The evaluator is never called, so no AP is computed on these frames. The kept infos' annotation arrays are emptied to
length 0 before the loader sees them, so no label reaches the data dict (the loader's code path is the evaluation's). The model and the dataset are built exactly as tools/test.py builds
them, with the config's target block (DATA_CONFIG_TAR); only the split ('train') and the frame list change.

    python analysis/predict_waymo_frames.py <cfg> <ckpt.pth> <out result.pkl> [frame=50] [stride=4] [batch=4]

Frames: `sample_idx == frame` of every `stride`-th sequence of ImageSets/train.txt. Output: the list of prediction dicts
that eval_one_epoch would pickle as result.pkl (frame_id, name, score, boxes_lidar, ...), plus <out>.meta.json.
"""
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
import _init_path  # noqa


def main():
    import numpy as np
    import torch
    from easydict import EasyDict
    from torch.utils.data import DataLoader
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    cfg_file, ckpt, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    frame = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    stride = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    batch = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    tar = cfg.DATA_CONFIG_TAR
    assert tar.DATASET == 'WaymoDataset', tar.DATASET
    tar.DATA_SPLIT['test'] = 'train'
    tar.SAMPLED_INTERVAL['test'] = 1
    logger = common_utils.create_logger()
    ontology = cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None)
    ds, _, _ = build_dataloader(dataset_cfg=tar, class_names=cfg.CLASS_NAMES, batch_size=batch, dist=False, workers=0,
                                logger=logger, training=False, model_ontology=ontology)
    root = Path(ds.root_path)
    seqs = [s.strip().replace('.tfrecord', '') for s in open(root / 'ImageSets' / 'train.txt') if s.strip()][::stride]
    want = set(seqs)
    keep = [i for i in ds.infos if i['point_cloud']['lidar_sequence'] in want and int(i['point_cloud']['sample_idx']) == frame]
    # Every annotation array is emptied to length 0 (dtype and trailing shape kept), so the loader takes exactly the
    # path a normal evaluation takes (prepare_data remaps gt_names; num_points_of_each_lidar is set in the annos
    # branch) while no label content remains.
    for i in keep:
        i['annos'] = {k: np.asarray(v)[:0] for k, v in i['annos'].items()}
    ds.infos = keep
    found = {i['point_cloud']['lidar_sequence'] for i in keep}
    logger.info(f'kept {len(keep)} frames (sample_idx {frame}) of {len(seqs)} requested train sequences; '
                f'{len(want - found)} sequences without that frame')
    loader = DataLoader(ds, batch_size=batch, pin_memory=True, num_workers=4, shuffle=False, collate_fn=ds.collate_batch,
                        drop_last=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
        raise SystemExit('DSNorm configs are not supported here')
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()
    det = []
    t0 = time.time()
    with torch.no_grad():
        for batch_dict in loader:
            assert batch_dict['gt_boxes'].shape[1] == 0, 'a label reached the batch'
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)
            det += ds.generate_prediction_dicts(batch_dict, pred_dicts, ds.class_names, output_path=None)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(det, f)
    sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    meta = dict(cfg=cfg_file, ckpt=ckpt, frame=frame, stride=stride, sequences=len(seqs), frames=len(det),
                missing_sequences=sorted(want - found), seconds=time.time() - t0, git=sha,
                names=sorted({str(n) for d in det for n in np.asarray(d['name'])}),
                created=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    logger.info(f'saved {len(det)} frames of predictions to {out}: {json.dumps(meta)}')


if __name__ == '__main__':
    main()
