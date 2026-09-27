"""Pure-numpy Waymo range-image reader (no tensorflow) + comparison of a static-extrinsic
reconstruction against the stored processed points. Found the corrupt TOP cloud - see
experiments_md/20260927_05. Run inside the container from anywhere; edit `seq` for another segment.
RangeImage fields (v1.4 proto): 2 = range_image_compressed, 3 = camera_projection_compressed,
4 = range_image_pose_compressed (zlib MatrixFloat: data=1 packed float32, shape=2 -> dims=1).
"""
"""Parse the TOP range image (return 1) straight out of a tfrecord frame - no tensorflow - and
compare a pure-numpy reconstruction with the stored processed points."""
import sys, zlib, struct, pickle, numpy as np
sys.path.insert(0, '/home/koyama/code/ST3D/tools')
from waymo_calib import _fields, first_frame, laser_calibrations
from pathlib import Path
ROOT = Path('/home/koyama/code/ST3D/data/waymo'); seq = 'segment-1005081002024129653_5313_150_5333_150_with_camera_labels'
frame = first_frame(ROOT / 'raw_data' / (seq + '.tfrecord'))
# Frame.lasers = 2 ; Laser.name = 1, ri_return1 = 2 ; RangeImage.range_image_compressed = 1 (zlib MatrixFloat)
# MatrixFloat.data = 1 (packed float), shape = 2 -> MatrixShape.dims = 1 (packed int)
def matrix_float(blob):
    """MatrixFloat: data = 1 (packed floats), shape = 2 -> MatrixShape.dims = 1 (packed or repeated varints)."""
    data, dims = None, []
    for fn, wt, v in _fields(blob):
        if fn == 1 and wt == 2:
            data = np.frombuffer(v, dtype='<f4')
        elif fn == 2 and wt == 2:
            for f2, w2, v2 in _fields(v):
                if f2 == 1 and w2 == 0:
                    dims.append(int(v2))
                elif f2 == 1 and w2 == 2:
                    i = 0
                    while i < len(v2):
                        x = 0; sh = 0
                        while True:
                            b = v2[i]; i += 1; x |= (b & 0x7f) << sh; sh += 7
                            if not b & 0x80: break
                        dims.append(x)
    return data.reshape(dims)
ri = None
for fn, wt, v in _fields(frame):
    if fn == 5 and wt == 2:  # Frame.lasers = 5
        name = None; r1 = None
        for f2, w2, v2 in _fields(v):
            if f2 == 1 and w2 == 0: name = v2
            elif f2 == 2 and w2 == 2: r1 = v2
        if name == 1:  # TOP
            for f3, w3, v3 in _fields(r1):
                if f3 == 2 and w3 == 2: ri = matrix_float(zlib.decompress(v3))   # RangeImage.range_image_compressed = 2 (v1.4 proto)
print('TOP return-1 range image shape', ri.shape)
rng = ri[..., 0]; valid = rng > 0; print('valid pixels %d' % valid.sum())
cal = laser_calibrations(ROOT / 'raw_data' / (seq + '.tfrecord'))['TOP']
inc = np.sort(cal['beam_inclinations'])[::-1]                     # rows top -> bottom
H, W = rng.shape; az = np.pi - (np.arange(W) + 0.5) * 2 * np.pi / W - np.arctan2(cal['extrinsic'][1, 0], cal['extrinsic'][0, 0])  # column azimuth MINUS the extrinsic yaw (range_image_utils.compute_range_image_polar); without it TOP is rotated - the 2026-09-27 report's 23 m xyz figure came from that omission
INC, AZ = np.meshgrid(inc, az, indexing='ij')
x = rng * np.cos(INC) * np.cos(AZ); y = rng * np.cos(INC) * np.sin(AZ); z = rng * np.sin(INC)
P = np.stack([x, y, z], -1)[valid]                                 # sensor frame, no per-pixel pose
E = cal['extrinsic']; Pv = (E[:3, :3] @ P.T).T + E[:3, 3]           # vehicle frame (static extrinsic only)
infos = pickle.load(open(ROOT / 'waymo_infos_train.pkl', 'rb')) + pickle.load(open(ROOT / 'waymo_infos_val.pkl', 'rb'))
info = next(i for i in infos if i['point_cloud']['lidar_sequence'] == seq and i['point_cloud']['sample_idx'] == 0)
S = np.load(ROOT / 'waymo_processed_data' / seq / '0000.npy')[:int(info['num_points_of_each_lidar'][0]), :3]
print('stored TOP points %d vs reconstructed valid pixels %d' % (len(S), len(Pv)))
print('reconstructed: distinct xyz %.3f' % (len(np.unique(np.round(Pv, 4), axis=0)) / len(Pv)))
# same ORDER? compare element-wise (row-major gather order) - static extrinsic only, so expect cm-level offsets from missing pixel pose
if len(S) == len(Pv):
    d = np.linalg.norm(S - Pv, axis=1); print('element-wise |stored - reconstructed|: p50 %.3f, p90 %.3f, p99 %.3f m' % tuple(np.percentile(d, [50, 90, 99])))
# nearest-neighbour agreement regardless of order
from scipy.spatial import cKDTree
t = cKDTree(Pv); dn, _ = t.query(S[::20]); print('stored -> nearest reconstructed: p50 %.3f, p90 %.3f m' % tuple(np.percentile(dn, [50, 90])))
t2 = cKDTree(S); dn2, _ = t2.query(Pv[::20]); print('reconstructed -> nearest stored: p50 %.3f, p90 %.3f m (large = geometry lost in storage)' % tuple(np.percentile(dn2, [50, 90])))

# ---- (a) range-only comparison for TOP: rotation-invariant, pose-insensitive to ~1 m
Es = (Einv := np.linalg.inv(E))
Ss = (Einv[:3, :3] @ S.T).T + Einv[:3, 3]
r_stored = np.linalg.norm(Ss, axis=1); r_pix = rng[valid]
dr = np.abs(r_stored - r_pix); print('TOP element-wise |range_stored - range_pixel|: p50 %.2f, p90 %.2f m; within 1 m: %.1f%%' % (np.percentile(dr, 50), np.percentile(dr, 90), 100 * (dr < 1).mean()))
# ---- (b) control: the same reconstruction on the FRONT lidar (no pixel-pose branch in the converter)
allcal = laser_calibrations(ROOT / 'raw_data' / (seq + '.tfrecord'))
for fn, wt, v in _fields(frame):
    if fn == 5 and wt == 2:
        name = None; r1 = None
        for f2, w2, v2 in _fields(v):
            if f2 == 1 and w2 == 0: name = v2
            elif f2 == 2 and w2 == 2: r1 = v2
        if name == 2:  # FRONT
            for f3, w3, v3 in _fields(r1):
                if f3 == 2 and w3 == 2: rif = matrix_float(zlib.decompress(v3))
c = allcal['FRONT']; rngf = rif[..., 0]; vf = rngf > 0; Hf, Wf = rngf.shape
incf = np.linspace(c['inc_max'], c['inc_min'], Hf) if len(c['beam_inclinations']) == 0 else np.sort(c['beam_inclinations'])[::-1]
azf = np.pi - (np.arange(Wf) + 0.5) * 2 * np.pi / Wf - np.arctan2(Ef[1, 0], Ef[0, 0]) if False else np.pi - (np.arange(Wf) + 0.5) * 2 * np.pi / Wf - np.arctan2(c['extrinsic'][1, 0], c['extrinsic'][0, 0])
INCf, AZf = np.meshgrid(incf, azf, indexing='ij')
Pf = np.stack([rngf * np.cos(INCf) * np.cos(AZf), rngf * np.cos(INCf) * np.sin(AZf), rngf * np.sin(INCf)], -1)[vf]
Ef = c['extrinsic']; Pfv = (Ef[:3, :3] @ Pf.T).T + Ef[:3, 3]
n_top = int(info['num_points_of_each_lidar'][0]); n_front = int(info['num_points_of_each_lidar'][1])
Sf = np.load(ROOT / 'waymo_processed_data' / seq / '0000.npy')[n_top:n_top + n_front, :3]
print('FRONT: stored %d vs valid pixels %d (range image %dx%d)' % (len(Sf), vf.sum(), Hf, Wf))
if len(Sf) == len(Pfv):
    d = np.linalg.norm(Sf - Pfv, axis=1); print('FRONT element-wise |stored - reconstructed|: p50 %.3f, p90 %.3f m  <- control' % tuple(np.percentile(d, [50, 90])))
