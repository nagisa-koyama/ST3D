"""Unique intensity values of Lyft TRAIN clouds per platform, read exactly as LyftDataset.get_lidar_with_sweeps reads a
keyframe (float32, 5 columns, column 3). experiments_md 20261003_05 §5 item 37e: a reflection row needs a non-constant
intensity channel; the dataset is recorded as constant 100 (lyft/nuscenes-devkit#3). CPU only.
    python analysis/lyft_intensity_unique.py --frames 200        # from ST3D/tools, inside the container
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

HOSTS_64 = ('host-a101', 'host-a102')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='../data/lyft')
    ap.add_argument('--frames', type=int, default=200)
    a = ap.parse_args()
    root = Path(a.root)
    infos = pickle.load(open(root / 'lyft_infos_train.pkl', 'rb'))
    for name, want64 in (('64-beam', True), ('40-beam', False)):
        sel = [i for i in infos if (Path(i['lidar_path']).name.split('_')[0] in HOSTS_64) == want64]
        idx = np.linspace(0, len(sel) - 1, min(a.frames, len(sel))).astype(int)
        values, n_pts, per_frame = set(), 0, []
        for k in idx:
            p = np.fromfile(str(root / sel[k]['lidar_path']), dtype=np.float32, count=-1)
            p = p[: p.shape[0] - (p.shape[0] % 5)].reshape(-1, 5)
            u = np.unique(p[:, 3])
            per_frame.append(len(u))
            values.update(u.tolist())
            n_pts += p.shape[0]
        vals = sorted(values)
        print('%s: %d of %d TRAIN frames, %d points; unique intensities overall %d (%s); per frame min %d max %d'
              % (name, len(idx), len(sel), n_pts, len(vals), vals[:10], min(per_frame), max(per_frame)))


if __name__ == '__main__':
    main()
