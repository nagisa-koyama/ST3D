"""Time a pure-numpy re-extraction of one Waymo segment: tfrecord read, per-frame proto parse, range
image + per-pixel pose decompression, reconstruction of all five lidars through the library's
transform chain (sensor -> extrinsic -> pixel pose -> inverse frame pose), and the .npy write.

    python waymo_reextract_bench.py <segment.tfrecord> <out dir> [--frames N]

Prints per-stage seconds per frame and the projected total for the dataset. Also validates the
side lidars against the stored processed points (should match to cm) and reports how the TOP
reconstruction compares with the stored TOP (should NOT match - that is the defect).
"""
import argparse, struct, sys, time, zlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from waymo_calib import _fields, _doubles  # noqa: E402

NAMES = {1: 'TOP', 2: 'FRONT', 3: 'SIDE_LEFT', 4: 'SIDE_RIGHT', 5: 'REAR'}


def frames(path):
    with open(path, 'rb') as f:
        while True:
            h = f.read(8)
            if len(h) < 8:
                return
            (ln,) = struct.unpack('<Q', h); f.read(4)
            yield f.read(ln); f.read(4)


def varints(b):
    out, i = [], 0
    while i < len(b):
        x = 0; s = 0
        while True:
            c = b[i]; i += 1; x |= (c & 0x7f) << s; s += 7
            if not c & 0x80: break
        out.append(x)
    return out


def matrix_float(blob):
    data, dims = None, []
    for fn, wt, v in _fields(blob):
        if fn == 1 and wt == 2: data = np.frombuffer(v, dtype='<f4')
        elif fn == 2 and wt == 2:
            for f2, w2, v2 in _fields(v):
                if f2 == 1 and w2 == 0: dims.append(int(v2))
                elif f2 == 1 and w2 == 2: dims += varints(v2)
    return data.reshape(dims)


def doubles(blob):
    return np.frombuffer(blob, dtype='<f8')


def parse_frame(buf):
    """-> frame_pose (4x4), calibrations {name: (inclinations, inc_min, inc_max, extrinsic)},
          lasers {name: (range_image [H,W,4], pose [H,W,6] or None)}"""
    pose = None; cal = {}; lasers = {}
    for fn, wt, v in _fields(buf):
        if fn == 3 and wt == 2:                                      # Frame.pose -> Transform.transform = 1
            pose = np.asarray(_doubles(1, v)).reshape(4, 4)
        elif fn == 1 and wt == 2:                                    # Frame.context
            for f2, w2, v2 in _fields(v):
                if f2 == 3 and w2 == 2:                              # laser_calibrations
                    name = None; lo = hi = None; ext = None
                    for f3, w3, v3 in _fields(v2):
                        if f3 == 1 and w3 == 0: name = v3
                        elif f3 == 3 and w3 == 1: lo = struct.unpack('<d', v3)[0]
                        elif f3 == 4 and w3 == 1: hi = struct.unpack('<d', v3)[0]
                        elif f3 == 5 and w3 == 2: ext = np.asarray(_doubles(1, v3)).reshape(4, 4)   # Transform.transform, packed or not
                    inc = np.asarray(_doubles(2, v2), dtype=np.float64)                          # beam_inclinations, packed or not
                    cal[name] = (inc, lo, hi, ext)
        elif fn == 5 and wt == 2:                                    # Frame.lasers
            name = None; ri = None; rpose = None
            for f2, w2, v2 in _fields(v):
                if f2 == 1 and w2 == 0: name = v2
                elif f2 == 2 and w2 == 2:                            # ri_return1
                    for f3, w3, v3 in _fields(v2):
                        if f3 == 2 and w3 == 2: ri = matrix_float(zlib.decompress(v3))
                        elif f3 == 4 and w3 == 2: rpose = matrix_float(zlib.decompress(v3))
            lasers[name] = (ri, rpose)
    return pose, cal, lasers


def rot_rpy(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    R = np.empty(roll.shape + (3, 3))
    R[..., 0, 0] = cy * cp;              R[..., 0, 1] = cy * sp * sr - sy * cr; R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp;              R[..., 1, 1] = sy * sp * sr + cy * cr; R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp;                  R[..., 2, 1] = cp * sr;                R[..., 2, 2] = cp * cr
    return R


def reconstruct(ri, cal, rpose, frame_pose):
    inc, lo, hi, E = cal
    rng = ri[..., 0]; H, W = rng.shape
    if len(inc) == 0:
        inc = (0.5 + np.arange(H)) / H * (hi - lo) + lo
    inc = inc[::-1]                                                   # row 0 = top beam (library reverses)
    az = ((np.arange(W, 0, -1) - 0.5) / W * 2 - 1) * np.pi - np.arctan2(E[1, 0], E[0, 0])
    valid = rng > 0
    INC, AZ = np.meshgrid(inc, az, indexing='ij')
    r = rng[valid]; i = INC[valid]; a = AZ[valid]
    P = np.stack([r * np.cos(i) * np.cos(a), r * np.cos(i) * np.sin(a), r * np.sin(i)], -1)
    P = P @ E[:3, :3].T + E[:3, 3]                                    # sensor -> vehicle (at pixel time)
    if rpose is not None:
        pp = rpose[valid]                                             # [N, 6]: roll, pitch, yaw, x, y, z
        R = rot_rpy(pp[:, 0], pp[:, 1], pp[:, 2])
        P = np.einsum('nij,nj->ni', R, P) + pp[:, 3:6]                # vehicle(pixel time) -> global
        Finv = np.linalg.inv(frame_pose)
        P = P @ Finv[:3, :3].T + Finv[:3, 3]                          # global -> vehicle (frame time)
    return P.astype(np.float32), ri[..., 1][valid], ri[..., 2][valid], ri[..., 3][valid], valid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('tfrecord'); ap.add_argument('out'); ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--stored', default=None, help='processed segment dir to validate against')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t = {'read': 0.0, 'parse': 0.0, 'recon': 0.0, 'write': 0.0}; n = 0; sz = Path(a.tfrecord).stat().st_size
    t0 = time.time()
    for k, buf in enumerate(frames(a.tfrecord)):
        t1 = time.time(); t['read'] += t1 - t0
        pose, cal, lasers = parse_frame(buf); t2 = time.time(); t['parse'] += t2 - t1
        parts = []; counts = []
        for name in sorted(lasers):
            ri, rpose = lasers[name]
            P, inten, elong, nlz, valid = reconstruct(ri, cal[name], rpose if name == 1 else None, pose)
            parts.append(np.column_stack([P, inten, elong, nlz]).astype(np.float32)); counts.append(len(P))
        pts = np.concatenate(parts); t3 = time.time(); t['recon'] += t3 - t2
        np.save(out / ('%04d.npy' % k), pts); t4 = time.time(); t['write'] += t4 - t3
        if a.stored and k < 3:
            S = np.load(Path(a.stored) / ('%04d.npy' % k))
            if len(S) == len(pts):
                b = np.cumsum([0] + counts)
                for li, name in enumerate(sorted(lasers)):
                    d = np.linalg.norm(S[b[li]:b[li + 1], :3] - pts[b[li]:b[li + 1], :3], axis=1)
                    print('  frame %d %-10s n=%6d  |stored - new| p50 %.3f p90 %.3f m' % (k, NAMES[name], counts[li], np.percentile(d, 50), np.percentile(d, 90)))
            else:
                print('  frame %d: count mismatch stored %d vs new %d' % (k, len(S), len(pts)))
        n += 1; t0 = time.time()
        if a.frames and n >= a.frames: break
    tot = sum(t.values())
    print('segment %.2f GB, %d frames: %.2f s/frame (read %.3f, parse %.3f, reconstruct %.3f, write %.3f)' % (sz / 1e9, n, tot / n, *[t[k] / n for k in ['read', 'parse', 'recon', 'write']]))
    print('projected for 198,068 frames on ONE core: %.1f h; on 8 cores: %.1f h; on 32: %.1f h' % (tot / n * 198068 / 3600, tot / n * 198068 / 3600 / 8, tot / n * 198068 / 3600 / 32))


if __name__ == '__main__':
    main()
