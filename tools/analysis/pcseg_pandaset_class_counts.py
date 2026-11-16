"""PandaSet semseg point counts per RAW class id, per sensor (Pandar64 d=0, PandarGT d=1), per split, over the whole
frame and inside the forward 60-degree cone (experiments_md episodes/20261116_01 §6: which classes each sensor actually
observes, before the P1 class set for the shares is fixed). Raw ids, so any class map can be applied afterwards.

Split = the detection split (cfgs/dataset_configs/pandaset_dataset.yaml SEQUENCES) restricted to sequences with semseg.
Points use pcseg's reader transform (world -> normative ego frame, identical to ST3D's reader); the cone is pcseg's
cone_mask (ST3D's extract_fov_gt convention) about the ego origin.

    python3 analysis/pcseg_pandaset_class_counts.py --out /home/koyama/data/pcseg/pandaset_class_counts_raw.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from pcseg.engine import cone_mask                                  # noqa: E402
from pcseg.pandaset_seg import N_FRAMES, has_semseg, load_poses, _ego_from_world   # noqa: E402

ROOT = '/home/koyama/data/pandaset/dataset'
NRAW = 43


def count_sequence(seq):
    poses = load_poses(ROOT, seq)
    out = np.zeros((2, 2, NRAW), dtype=np.int64)      # [device, region (0 full, 1 cone), raw id]
    for f in range(N_FRAMES):
        lidar = pd.read_pickle(os.path.join(ROOT, seq, 'lidar', '%02d.pkl.gz' % f))
        sem = pd.read_pickle(os.path.join(ROOT, seq, 'annotations', 'semseg', '%02d.pkl.gz' % f))
        assert len(lidar) == len(sem), (seq, f)
        d = lidar['d'].to_numpy()
        raw = sem['class'].to_numpy().astype(np.int64)
        assert raw.min() >= 0 and raw.max() < NRAW, (seq, f, raw.min(), raw.max())
        ego = _ego_from_world(lidar[['x', 'y', 'z']].to_numpy(), poses[f])
        cone = cone_mask(ego[:, :2], 60.0, 0.0)
        for dev in (0, 1):
            m = d == dev
            out[dev, 0] += np.bincount(raw[m], minlength=NRAW)
            out[dev, 1] += np.bincount(raw[m & cone], minlength=NRAW)
    return seq, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    args = ap.parse_args()
    det = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           '../cfgs/dataset_configs/pandaset_dataset.yaml')))['SEQUENCES']
    splits = {k: [s for s in det[k] if has_semseg(ROOT, s)] for k in ('train', 'val', 'test')}
    seqs = [s for v in splits.values() for s in v]
    with Pool(args.procs) as pool:
        per_seq = dict(pool.map(count_sequence, seqs))
    res = {'root': ROOT, 'frames_per_sequence': N_FRAMES, 'axes': '[device 0 Pandar64 / 1 PandarGT][full, cone60]'
           '[raw id 0..42]', 'splits': splits, 'counts': {}}
    for k, v in splits.items():
        res['counts'][k] = sum(per_seq[s] for s in v).tolist()
    res['per_sequence'] = {s: c.tolist() for s, c in per_seq.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f)
    print('wrote', args.out, {k: len(v) for k, v in splits.items()})


if __name__ == '__main__':
    main()
