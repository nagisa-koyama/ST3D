"""Quick check (20261010_07): pitch / roll / yaw rates and vertical speed of the sweeps behind the start-scan frames,
beside each frame's ring purity. Reads infos only."""
import json, pickle, sys, numpy as np
ROOT = '/home/koyama/data/level5-3d-object-detection/trainval/'
rows = json.load(open(sys.argv[1]))
infos = {i['lidar_path'].split('/')[-1]: i for i in pickle.load(open(ROOT + 'lyft_infos_train.pkl', 'rb'))}
for r in sorted(rows, key=lambda r: r['platform_purity']):
    inf = infos[r['file']]
    sw = [s for s in inf['sweeps'] if s.get('transform_matrix') is not None and s.get('time_lag', 0) > 0][0]
    T, lag = np.asarray(sw['transform_matrix']), sw['time_lag']
    R = T[:3, :3]
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2])); pitch = np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1)))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    print('pur %.3f rings %3d | lag %.3f s | rates deg/s roll %6.2f pitch %6.2f yaw %6.2f | vz %5.2f m/s | v %5.1f' % (
        r['platform_purity'], r['platform_rings'], lag, roll / lag, pitch / lag, yaw / lag, -T[2, 3] / lag, r['speed']))
