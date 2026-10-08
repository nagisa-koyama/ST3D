"""Cache Waymo TOP calibration (beam inclinations + extrinsic) per segment, for RING_PATTERN / BEAM_DISTILL on Waymo
(pcdet/datasets/waymo/waymo_rings.py). SOURCE calibration, legal. Writes /home/koyama/data/waymo_top_calib/top_calib.pkl
(outside the repo: jobs run from a snapshot). ~20 s with 16 processes.

    python analysis/waymo_top_calib_cache.py
"""
import sys, time, pickle
from pathlib import Path
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '/home/koyama/code/ST3D/tools')
from waymo_calib import laser_calibrations
RAW = Path('/home/koyama/code/ST3D/data/waymo/raw_data')
def one(p):
    c = laser_calibrations(p)['TOP']
    return p.name.replace('.tfrecord', ''), np.asarray(c['beam_inclinations'], np.float64), np.asarray(c['extrinsic'], np.float64)
if __name__ == '__main__':
    files = sorted(RAW.glob('*.tfrecord'))
    t = time.time()
    with Pool(16) as pool:
        out = pool.map(one, files, chunksize=8)
    d = {s: {'inclinations': i, 'extrinsic': e} for s, i, e in out}
    assert all(len(v['inclinations']) == 64 for v in d.values()), 'a TOP calibration without 64 inclinations'
    pickle.dump(d, open('/home/koyama/data/waymo_top_calib/top_calib.pkl', 'wb'))
    inc = np.stack([v['inclinations'] for v in d.values()]); ext = np.stack([v['extrinsic'] for v in d.values()])
    print(f'{len(d)} segments in {time.time()-t:.0f} s; distinct inclination vectors {len(np.unique(np.round(inc, 8), axis=0))}; '
          f'extrinsic max deviation from the first {np.abs(ext - ext[0]).max():.2e}; yaw deg {np.degrees(np.arctan2(ext[0,1,0], ext[0,0,0])):.3f}')
