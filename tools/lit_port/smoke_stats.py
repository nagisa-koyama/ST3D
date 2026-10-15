"""Smoke comparison of LiT-translated Waymo frames with real nuScenes frames (experiments_md 20261015_02 §3 item 6).
Source-side and unlabelled-target statistics only; the real nuScenes numbers use TRAIN keyframes.

    python3 -I lit_port/smoke_stats.py <sim_nuscenes_dir> <waymo_processed_data> <top_calib.pkl> <nuscenes_root> \
        <nuscenes_infos_train.pkl> <scenes comma list> <out.json> [n_real=100]

Compared, on the HDL-32E lattice of LiT's table (32 rows x 1090 columns, 3-70 m, about the sensor origin):
  points per frame; filled lattice cells per frame; rows holding any point; per-row point-share profile;
  ground height below the sensor (median z of points at 8-20 m with elevation < -5 deg, relative to the sensor);
  points in a Car box at 20-40 m (median, share with < 10 points), translated vs real nuScenes (`num_lidar_pts`).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

TABLE = np.array([10.65520, 9.33780, 7.97498, 6.65758, 5.34018, 3.97735, 2.65996, 1.34256, -0.02027, -1.33767, -2.65507,
                  -4.01789, -5.33529, -6.65269, -8.01552, -9.33292, -10.65032, -12.01314, -13.33054, -14.64794, -16.01077,
                  -17.32817, -18.64557, -20.00839, -21.32579, -22.64319, -24.00602, -25.32341, -26.64081, -28.00364,
                  -29.32104, -30.68386])
NC = 1090


def lattice(pts, origin):
    d = pts[:, :3].astype(np.float64) - origin
    rho = np.hypot(d[:, 0], d[:, 1]); rng = np.hypot(rho, d[:, 2])
    ok = (rng > 3.0) & (rng < 70.0)
    d, rho, rng = d[ok], rho[ok], rng[ok]
    el = np.degrees(np.arctan2(d[:, 2], rho)); az = np.arctan2(d[:, 1], d[:, 0])
    row = np.abs(el[:, None] - TABLE[None]).argmin(1)
    inside = np.abs(el - TABLE[row]) < 0.67                      # half a row gap
    col = np.clip(np.floor((az + np.pi) / (2 * np.pi) * NC).astype(int), 0, NC - 1)
    cells = np.unique(row[inside] * NC + col[inside]).size
    return dict(n=int(ok.sum()), cells=int(cells), rows=np.bincount(row[inside], minlength=32), d=d, rho=rho, el=el)


def frame_stats(pts, origin):
    L = lattice(pts, origin)
    g = (L['rho'] > 8) & (L['rho'] < 20) & (L['el'] < -5)
    return L, float(np.median(L['d'][g, 2])) if g.sum() > 20 else float('nan')


def box_pts(pts, boxes, origin, lo=20, hi=40):
    """Points inside each box (x, y, z, dx, dy, dz, heading) whose centre lies lo-hi m from the origin."""
    out = []
    for b in boxes:
        r = np.hypot(b[0] - origin[0], b[1] - origin[1])
        if not lo <= r < hi:
            continue
        c, s = np.cos(-b[6]), np.sin(-b[6])
        q = pts[:, :3] - b[:3]
        x = q[:, 0] * c - q[:, 1] * s; y = q[:, 0] * s + q[:, 1] * c
        out.append(int(((np.abs(x) <= b[3] / 2) & (np.abs(y) <= b[4] / 2) & (np.abs(q[:, 2]) <= b[5] / 2)).sum()))
    return out


def summarize(rows, ground, boxcounts, n):
    rows = np.array(rows, float); share = rows.sum(0) / max(rows.sum(), 1)
    bc = np.array(boxcounts) if boxcounts else np.zeros(0)
    return dict(frames=n, rows_filled_any=int((rows.sum(0) > 0).sum()), row_share=np.round(share, 4).tolist(),
                ground_z_below_sensor=float(np.nanmedian(ground)),
                car_boxes_20_40=int(len(bc)), car_pts_median=float(np.median(bc)) if len(bc) else None,
                car_share_lt10=float((bc < 10).mean()) if len(bc) else None)


def main(sim, wdata, calib, nroot, ninfos, scenes, out, n_real=100):
    with open(calib, 'rb') as f:
        calib = pickle.load(f)
    T = {}
    cnt = dict(n=[], cells=[], rows=[]); ground, bc = [], []
    for name in scenes.split(','):
        E = np.asarray(calib[name]['extrinsic'], np.float64).reshape(4, 4)
        origin = E[:3, 3]
        with open(Path(wdata) / name / (name + '.pkl'), 'rb') as f:
            infos = pickle.load(f)
        files = sorted((Path(sim) / name).glob('*.npz'))
        for fp in files[::5]:
            z = np.load(fp)
            idx = int(z['frame_index'])
            pts = z['local_points']
            L, g = frame_stats(pts, origin)
            cnt['n'].append(L['n']); cnt['cells'].append(L['cells']); cnt['rows'].append(L['rows']); ground.append(g)
            an = infos[idx]['annos']
            boxes = [b for b, nm in zip(an['gt_boxes_lidar'], an['name']) if nm == 'Vehicle']
            bc += box_pts(pts, boxes, origin)
        T['frames_checked'] = len(cnt['n'])
    res = {'translated': summarize(cnt['rows'], ground, bc, len(cnt['n']))}
    res['translated'].update(points_per_frame=float(np.mean(cnt['n'])), cells_filled_per_frame=float(np.mean(cnt['cells'])))
    with open(ninfos, 'rb') as f:
        ni = pickle.load(f)
    sel = ni[::max(1, len(ni) // n_real)][:n_real]
    rn, rc, rr, rg, rb = [], [], [], [], []
    for inf in sel:
        pts = np.fromfile(Path(nroot) / inf['lidar_path'], np.float32).reshape(-1, 5)
        L, g = frame_stats(pts, np.zeros(3))
        rn.append(L['n']); rc.append(L['cells']); rr.append(L['rows']); rg.append(g)
        gb = inf['gt_boxes']; names = inf['gt_names']; npts = inf.get('num_lidar_pts', None)
        for b, nm, k in zip(gb, names, npts if npts is not None else [None] * len(gb)):
            if nm == 'car' and 20 <= np.hypot(b[0], b[1]) < 40 and k is not None:
                rb.append(int(k))
    res['real_nuscenes_train'] = summarize(rr, rg, rb, len(sel))
    res['real_nuscenes_train'].update(points_per_frame=float(np.mean(rn)), cells_filled_per_frame=float(np.mean(rc)))
    Path(out).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, {a: b for a, b in v.items() if a != 'row_share'})
    print('row_share translated', np.round(res['translated']['row_share'], 3).tolist())
    print('row_share real      ', np.round(res['real_nuscenes_train']['row_share'], 3).tolist())


if __name__ == '__main__':
    main(*sys.argv[1:8], n_real=int(sys.argv[8]) if len(sys.argv) > 8 else 100)
