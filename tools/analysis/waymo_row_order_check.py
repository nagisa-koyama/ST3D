"""Does our processed Waymo cloud still carry the range image's ROW ORDER - i.e. the beam identity?

Ring recovery from absolute inclination fails on Waymo (35.3% of points within 0.05 deg of a
declared beam vs a 32.1% null; per-return motion compensation moves each point's apparent
inclination). But `waymo_utils.save_lidar_points` concatenates the points exactly as
`tf.where(range_image_mask)` yields them - ROW-MAJOR, lidars sorted by name (TOP first) - and the
infos store `num_points_of_each_lidar`. If that order survives, a row boundary is where the azimuth
sweep wraps, and each row's identity comes from ORDER, which motion compensation does not touch.
That is what the LiDAR Distillation reference uses (range-image rows, every other row = 32 beams),
so a positive result makes Waymo 64 -> 32 -> nuScenes runnable from the processed data.

Run from ST3D/tools inside the container:
    python analysis/waymo_row_order_check.py [--seq <segment name>] [--frames 3]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from waymo_calib import laser_calibrations  # noqa: E402

ROOT = Path('/home/koyama/code/ST3D/data/waymo')


def rows_from_order(az):
    """Split an azimuth sequence into monotone sweeps: a new row starts where the sweep wraps."""
    # Column 0 of the range image is not at the +-180 deg seam, so every row contains one azimuth
    # wrap of its own and the row boundary is a small jump back to the starting azimuth. Unwrap
    # first: within a row the unwrapped azimuth then advances monotonically by ~2*pi, a stretch of
    # invalid pixels is a larger step in the SAME direction, and the next row is a jump of ~-2*pi
    # AGAINST the sweep. Only the last is a boundary.
    u = np.unwrap(az)
    d = np.diff(u)
    direction = np.sign(np.median(d))
    breaks = np.where((np.sign(d) == -direction) & (np.abs(d) > np.pi))[0] + 1
    return np.split(np.arange(len(az)), breaks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='segment-1005081002024129653_5313_150_5333_150_with_camera_labels')
    ap.add_argument('--frames', type=int, default=3)
    a = ap.parse_args()

    cal = laser_calibrations(ROOT / 'raw_data' / (a.seq + '.tfrecord'))['TOP']
    E = cal['extrinsic']                                # lidar -> vehicle
    Einv = np.linalg.inv(E)
    declared = np.sort(np.degrees(cal['beam_inclinations']))[::-1]   # range image rows run top -> bottom
    print('declared TOP beams: %d, %.2f .. %.2f deg' % (len(declared), declared.min(), declared.max()))

    infos = []
    for split in ['train', 'val']:
        infos += [i for i in pickle.load(open(ROOT / ('waymo_infos_%s.pkl' % split), 'rb'))
                  if i['point_cloud']['lidar_sequence'] == a.seq]
    infos = sorted(infos, key=lambda i: i['point_cloud']['sample_idx'])[:a.frames]
    assert infos, 'sequence not in the infos'

    for info in infos:
        idx = info['point_cloud']['sample_idx']
        pts = np.load(ROOT / 'waymo_processed_data' / a.seq / ('%04d.npy' % idx))
        n_top = int(info['num_points_of_each_lidar'][0])
        top = pts[:n_top, :3]
        # into the TOP sensor frame
        ps = (Einv[:3, :3] @ top.T).T + Einv[:3, 3]
        inc = np.degrees(np.arctan2(ps[:, 2], np.hypot(ps[:, 0], ps[:, 1])))
        az = np.arctan2(ps[:, 1], ps[:, 0])
        rows = rows_from_order(az)
        rows = [r for r in rows if len(r) >= 20]
        med = np.array([np.median(inc[r]) for r in rows])
        spread = np.array([np.percentile(inc[r], 84) - np.percentile(inc[r], 16) for r in rows])
        mono = np.all(np.diff(med) < 0)
        # nearest declared beam for each recovered row, in order
        resid = np.array([np.min(np.abs(declared - m)) for m in med])
        # one-to-one: does the k-th row match the k-th declared beam when 64 rows are found?
        seq_resid = np.abs(med - declared[:len(med)]) if len(med) == len(declared) else None
        print('\nframe %04d: TOP points %d of %d total; rows found by azimuth wrap: %d (>=20 pts each)'
              % (idx, n_top, len(pts), len(rows)))
        print('  row medians monotone top->bottom: %s; per-row 16-84%% inclination spread: median %.3f deg'
              % (mono, np.median(spread)))
        print('  nearest-declared-beam residual of row medians: median %.3f, p90 %.3f, max %.3f deg; '
              'within 0.05: %.0f%%, within 0.1: %.0f%%'
              % (np.median(resid), np.percentile(resid, 90), resid.max(),
                 100 * (resid < 0.05).mean(), 100 * (resid < 0.1).mean()))
        if seq_resid is not None:
            print('  k-th row vs k-th declared beam: median %.3f, max %.3f deg' % (np.median(seq_resid), seq_resid.max()))
        # what a per-point absolute assignment would have given (the failed method) on the same data
        pp = np.min(np.abs(inc[:, None] - declared[None, :]), axis=1)
        print('  per-POINT absolute nearest-beam residual < 0.05 deg: %.1f%% (the method that failed)'
              % (100 * (pp < 0.05).mean()))
        if len(rows) == len(declared):
            keep = np.concatenate([r for k, r in enumerate(rows) if k % 2 == 0])
            print('  reference-style 64->32 (every other row): keeps %.3f of TOP points' % (len(keep) / n_top))
        print('  row sizes: min %d, median %d, max %d; sum %d' % (min(map(len, rows)), int(np.median(list(map(len, rows)))), max(map(len, rows)), sum(map(len, rows))))


if __name__ == '__main__':
    main()
