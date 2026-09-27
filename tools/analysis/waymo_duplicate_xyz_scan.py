"""Fraction of distinct xyz per lidar in the processed Waymo frames, over sampled sequences of both
splits, with KITTI and nuScenes controls. Found the corrupt TOP cloud - experiments_md/20260927_05."""
import pickle, numpy as np
from pathlib import Path
ROOT = Path('/home/koyama/code/ST3D/data/waymo')
def stats(p):
    q = np.round(p[:, :3], 4); u = len(np.unique(q, axis=0)); d = np.all(np.diff(q, axis=0) == 0, axis=1).mean()
    return u / len(p), d
for split in ['train', 'val']:
    infos = pickle.load(open(ROOT / ('waymo_infos_%s.pkl' % split), 'rb'))
    seqs = sorted({i['point_cloud']['lidar_sequence'] for i in infos}); pick = seqs[::max(1, len(seqs) // 6)][:6]
    print('== %s: %d infos, %d sequences' % (split, len(infos), len(seqs)))
    for s in pick:
        fr = [i for i in infos if i['point_cloud']['lidar_sequence'] == s][:2]
        for i in fr:
            pts = np.load(ROOT / 'waymo_processed_data' / s / ('%04d.npy' % i['point_cloud']['sample_idx']))
            n = i['num_points_of_each_lidar']; top = pts[:int(n[0])]; rest = pts[int(n[0]):]
            ut, dt = stats(top); ur, dr = stats(rest) if len(rest) else (np.nan, np.nan)
            print('  %s f%04d: TOP unique-xyz %.3f (consec dup %.3f) | other lidars unique %.3f (dup %.3f) | n %s' % (s[8:28], i['point_cloud']['sample_idx'], ut, dt, ur, dr, list(map(int, n))))
print('== controls')
k = np.fromfile('/home/koyama/code/ST3D/data/kitti/training/velodyne/000008.bin', dtype=np.float32).reshape(-1, 4); print('  KITTI 000008: unique-xyz %.3f' % stats(k)[0])
import glob
f = sorted(glob.glob('/home/koyama/code/ST3D/data/nuscenes/samples/LIDAR_TOP/*.bin'))[0]; a = np.fromfile(f, dtype=np.float32); a = a[:len(a) - len(a) % 5].reshape(-1, 5); print('  nuScenes %s: unique-xyz %.3f' % (Path(f).name[-24:], stats(a)[0]))
