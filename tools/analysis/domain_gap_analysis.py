"""Reproduce the cross-dataset / cross-platform domain-gap measurements.

Every number and figure in experiments_md/20260922_01_dataset_and_platform_domain_gap_analysis.md
comes from this script. Run it from ST3D/tools/ inside the container:

    singularity exec --bind /home/koyama/data/:/storage <image>.sif \
        python analysis/domain_gap_analysis.py <analysis> [--frames N] [--out DIR]

Analyses:
    range       points per 10 m radial bin, per platform, against KITTI
    boxes       GT box dimensions, points-per-box vs range, objects per frame
    intensity   intensity vs range (median + IQR), in each loader's native units
    beams       point elevation angle (beam structure); sensor-frame datasets only
    accumulate  minimum sweep count N for a source to reach KITTI's radial profile
    correction  sample_points_hist_based before/after, globally and on GT boxes
    platforms   the per-platform summary table (Lyft 40/64, nuScenes n008/n015)

Notes that matter for interpreting the output:
  * Every loader applies exactly what its dataset class applies - nuScenes gets
    remove_ego_points(1.5) (26% of its raw cloud is ego returns), PandaSet is divided by
    255 and axis-swapped, Waymo gets tanh() on intensity and the NLZ filter.
  * Waymo's processed points and its stored annotations are geometrically inconsistent in
    this checkout (median counted/annotated = 0.06), so Waymo is excluded from every
    box-level analysis. Point-level analyses are unaffected.
  * Points-per-box is counted here rather than read from each dataset's own
    `num_points_in_gt`, so that both sides of a comparison get identical range masking.
    Small `--frames` values leave near bins empty (the >=10-box guard prints nan).
  * PandaSet PandarGT scores ~0 points per box by construction: its cuboids are annotated
    over the full 360 deg while the sensor itself only sees a forward cone.
"""
import argparse
import collections
import glob
import json
import pickle
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent.parent / 'data'
POINT_CLOUD_RANGE = np.array([-75.2, -75.2, -2, 75.2, 75.2, 4])
MAX_DIST = 75.0
LYFT_64_HOSTS = ('host-a101', 'host-a102')


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def mask_range(points):
    inside = np.all((points[:, :3] >= POINT_CLOUD_RANGE[:3]) &
                    (points[:, :3] < POINT_CLOUD_RANGE[3:]), axis=1)
    return points[inside]


def remove_ego_points(points, center_radius):
    return points[~((np.abs(points[:, 0]) < center_radius) &
                    (np.abs(points[:, 1]) < center_radius))]


def radial_hist(points, edges):
    return np.histogram(np.linalg.norm(points[:, :2], axis=1), bins=edges)[0].astype(float)


def quat_to_rot(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def points_in_boxes(points, boxes):
    """Exact point-in-rotated-box count, prefiltered by a KD-tree ball."""
    from scipy.spatial import cKDTree
    if len(boxes) == 0 or len(points) == 0:
        return np.zeros(len(boxes), int)
    tree = cKDTree(points[:, :3])
    counts = np.zeros(len(boxes), int)
    for i, box in enumerate(boxes):
        idx = tree.query_ball_point(box[:3], np.linalg.norm(box[3:6]) / 2)
        if not idx:
            continue
        q = points[idx, :3] - box[:3]
        ca, sa = np.cos(-box[6]), np.sin(-box[6])
        x = q[:, 0] * ca - q[:, 1] * sa
        y = q[:, 0] * sa + q[:, 1] * ca
        counts[i] = int(((np.abs(x) <= box[3] / 2) & (np.abs(y) <= box[4] / 2) &
                         (np.abs(q[:, 2]) <= box[5] / 2)).sum())
    return counts


# --------------------------------------------------------------------------------------
# platform registry - each yields Frame(points[x,y,z,intensity], boxes[N,7], names[N])
# --------------------------------------------------------------------------------------
Frame = collections.namedtuple('Frame', 'points boxes names')


def _nuscenes_log_map():
    meta = DATA / 'nuscenes/v1.0-trainval/v1.0-trainval'
    with open(meta / 'log.json') as f:
        logs = {l['token']: l for l in json.load(f)}
    with open(meta / 'scene.json') as f:
        scene_log = {s['token']: logs[s['log_token']] for s in json.load(f)}
    with open(meta / 'sample.json') as f:
        return {s['token']: scene_log[s['scene_token']] for s in json.load(f)}


class Platform:
    """One capture platform: a frame list plus the loader its dataset class would use."""

    def __init__(self, name, car_class, infos, load, rate_hz=None, sweeps=None):
        self.name, self.car_class, self.infos = name, car_class, infos
        self._load, self.rate_hz, self._sweeps = load, rate_hz, sweeps

    def __len__(self):
        return len(self.infos)

    def sample(self, n):
        step = max(1, len(self.infos) // n)
        return self.infos[::step][:n]

    def frame(self, info):
        return self._load(info)

    def accumulate(self, info, n_sweeps):
        """Points from this frame plus the n_sweeps-1 preceding, in this frame's coords."""
        return self._sweeps(info, n_sweeps)


def build_platforms(which=None):
    P = {}

    # ---- KITTI (target) -------------------------------------------------------------
    kitti_infos = pickle.load(open(DATA / 'kitti/kitti_infos_train.pkl', 'rb'))

    def kitti_load(info):
        idx = info['point_cloud']['lidar_idx']
        pts = np.fromfile(str(DATA / f'kitti/training/velodyne/{idx}.bin'),
                          dtype=np.float32).reshape([-1, 4])
        a = info['annos']
        boxes = np.asarray(a['gt_boxes_lidar'])
        names = np.asarray(a['name'])[:len(boxes)]
        return Frame(pts, boxes[:, :7] if len(boxes) else np.zeros((0, 7)), names)

    P['KITTI'] = Platform('KITTI', 'Car', kitti_infos, kitti_load, rate_hz=10)

    # ---- nuScenes, split by capture vehicle -----------------------------------------
    ns_root = DATA / 'nuscenes/v1.0-trainval'
    ns_infos = pickle.load(open(ns_root / 'nuscenes_infos_10sweeps_train.pkl', 'rb'))
    log_of = _nuscenes_log_map()

    def ns_raw(path, radius):
        p = np.fromfile(str(ns_root / path), dtype=np.float32).reshape([-1, 5])[:, :4]
        return remove_ego_points(p, radius)

    def ns_load(info):
        g = np.asarray(info['gt_boxes'])
        return Frame(ns_raw(info['lidar_path'], 1.5),
                     g[:, :7] if len(g) else np.zeros((0, 7)), np.asarray(info['gt_names']))

    def ns_sweeps(info, n):
        out = [ns_raw(info['lidar_path'], 1.5)]
        for s in info['sweeps'][:n - 1]:
            p = ns_raw(s['lidar_path'], 1.0)
            tm = s['transform_matrix']
            if tm is not None:
                p = np.column_stack([(tm[:3, :3] @ p[:, :3].T).T + tm[:3, 3], p[:, 3]])
            out.append(p)
        return np.concatenate(out)

    for vehicle, label in [('n008', 'nuScenes n008 Boston'), ('n015', 'nuScenes n015 Singapore')]:
        sub = [i for i in ns_infos if log_of[i['token']]['vehicle'] == vehicle]
        P[label] = Platform(label, 'car', sub, ns_load, rate_hz=20, sweeps=ns_sweeps)

    # ---- Lyft, split by lidar configuration -----------------------------------------
    ly_root = DATA / 'lyft/trainval'
    ly_infos = pickle.load(open(ly_root / 'lyft_infos_train.pkl', 'rb'))

    def ly_raw(path):
        a = np.fromfile(str(ly_root / path), dtype=np.float32)
        a = a[:len(a) - (len(a) % 5)]
        return a.reshape([-1, 5])[:, :4]

    def ly_load(info):
        g = np.asarray(info['gt_boxes'])
        return Frame(ly_raw(info['lidar_path']),
                     g[:, :7] if len(g) else np.zeros((0, 7)), np.asarray(info['gt_names']))

    def ly_sweeps(info, n):
        out = [ly_raw(info['lidar_path'])]
        for s in info['sweeps'][:n - 1]:
            p = ly_raw(s['lidar_path'])
            tm = s['transform_matrix']
            if tm is not None:
                p = np.column_stack([(tm[:3, :3] @ p[:, :3].T).T + tm[:3, 3], p[:, 3]])
            out.append(p)
        return np.concatenate(out)

    for want64, label in [(False, 'Lyft 40-beam'), (True, 'Lyft 64-beam')]:
        sub = [i for i in ly_infos
               if (Path(i['lidar_path']).name.split('_')[0] in LYFT_64_HOSTS) == want64]
        P[label] = Platform(label, 'car', sub, ly_load, rate_hz=5, sweeps=ly_sweeps)

    # ---- PandaSet, split by lidar device ---------------------------------------------
    pd_infos = pickle.load(open(DATA / 'pandaset/pandaset_infos_train.pkl', 'rb'))
    fix = lambda p: str(p).replace('/root/ST3D/data/pandaset', str(DATA / 'pandaset'))
    pose_cache = {}

    def pd_load_factory(device):
        def load(info):
            import pandas as pd
            seq = info['sequence']
            if seq not in pose_cache:
                pose_cache[seq] = json.load(open(DATA / f'pandaset/dataset/{seq}/lidar/poses.json'))
            pose = pose_cache[seq][info['frame_idx']]
            R = quat_to_rot(*[pose['heading'][k] for k in 'wxyz'])
            t = np.array([pose['position'][k] for k in 'xyz'])
            df = pd.read_pickle(fix(info['lidar_path']))
            if device != -1:
                df = df[df.d == device]
            ego = (R.T @ (df[['x', 'y', 'z']].to_numpy() - t).T).T
            pts = np.column_stack([ego[:, 1], -ego[:, 0], ego[:, 2],
                                   df['i'].to_numpy() / 255.0])       # normative frame
            cub = pd.read_pickle(fix(info['cuboids_path']))
            if device != -1:
                cub = cub[cub['cuboids.sensor_id'] != 1 - device]
            c = (R.T @ (cub[['position.x', 'position.y', 'position.z']].to_numpy() - t).T).T
            boxes = np.stack([c[:, 1], -c[:, 0], c[:, 2],
                              cub['dimensions.y'].to_numpy(), cub['dimensions.x'].to_numpy(),
                              cub['dimensions.z'].to_numpy(), cub['yaw'].to_numpy()], 1)
            return Frame(pts, boxes, cub['label'].to_numpy())
        return load

    for device, label in [(0, 'PandaSet Pandar64'), (1, 'PandaSet PandarGT')]:
        P[label] = Platform(label, 'Car', [i for i in pd_infos if i['frame_idx'] % 4 == 0],
                            pd_load_factory(device), rate_hz=10)

    # ---- Waymo (point-level only; see the module docstring) ---------------------------
    wa_infos = pickle.load(open(DATA / 'waymo/waymo_infos_train.pkl', 'rb'))

    def wa_load(info):
        pc = info['point_cloud']
        f = np.load(DATA / 'waymo/waymo_processed_data' / pc['lidar_sequence'] /
                    ('%04d.npy' % pc['sample_idx']))
        f = f[f[:, 5] == -1]
        pts = np.column_stack([f[:, :3], np.tanh(f[:, 3])])
        a = info.get('annos', {})
        g = np.asarray(a.get('gt_boxes_lidar', np.zeros((0, 7))))
        return Frame(pts, g[:, :7] if len(g) else np.zeros((0, 7)),
                     np.asarray(a.get('name', [])))

    P['Waymo'] = Platform('Waymo', 'Vehicle', wa_infos[::200], wa_load, rate_hz=10)

    return {k: v for k, v in P.items() if which is None or k in which}


# --------------------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------------------
def analysis_range(P, frames, **_):
    edges = np.arange(0, 90, 10)
    ref = None
    print(f"{'platform':26s} " + ' '.join(f'{int(edges[i])}-{int(edges[i+1])}m'.rjust(9)
                                          for i in range(len(edges) - 1)) + f"{'total':>10s}")
    for name, plat in P.items():
        h = np.mean([radial_hist(mask_range(plat.frame(i).points), edges)
                     for i in plat.sample(frames)], axis=0)
        if ref is None:
            ref = h
        print(f'{name:26s} ' + ' '.join(f'{v:9.0f}' for v in h) + f' {h.sum():10.0f}')
        if name != 'KITTI':
            print(f"{'  ratio to KITTI':26s} " + ' '.join(f'{v:9.2f}' for v in h / ref))


def analysis_boxes(P, frames, **_):
    edges = np.arange(0, 85, 10)
    print(f"{'platform':26s} {'Car l x w x h':>22s} {'obj/frame':>10s} "
          f"{'pts/box 10-20':>14s} {'20-30':>7s} {'30-40':>7s}")
    for name, plat in P.items():
        if name == 'Waymo':
            continue
        # boxes live in the infos for every platform except PandaSet, whose cuboids are
        # per-frame files - so PandaSet is sampled rather than swept.
        dims, per_frame = [], []
        source = plat.sample(frames * 5) if plat.name.startswith('PandaSet') else plat.infos
        for info in source:
            fr = plat.frame(info) if plat.name.startswith('PandaSet') else None
            if fr is None:                                  # cheap path: boxes live in the infos
                if 'gt_boxes' in info:
                    g, nm = np.asarray(info['gt_boxes']), np.asarray(info['gt_names'])
                elif 'annos' in info:
                    g = np.asarray(info['annos']['gt_boxes_lidar'])
                    nm = np.asarray(info['annos']['name'])[:len(g)]
                else:
                    continue
            else:
                g, nm = fr.boxes, fr.names
            m = nm == plat.car_class
            per_frame.append(int(m.sum()))
            if m.sum():
                dims.append(np.asarray(g)[m][:, 3:6])
        counted = []
        for info in plat.sample(frames):
            fr = plat.frame(info)
            m = fr.names == plat.car_class
            if m.sum():
                pts = mask_range(fr.points)
                counted.append(np.stack([np.linalg.norm(fr.boxes[m][:, :2], axis=1),
                                         points_in_boxes(pts, fr.boxes[m])], 1))
        D = np.concatenate(dims).mean(0) if dims else np.full(3, np.nan)
        med = np.full(8, np.nan)
        if counted:
            a = np.concatenate(counted)
            idx = np.digitize(a[:, 0], edges) - 1
            med = np.array([np.median(a[idx == i, 1]) if (idx == i).sum() >= 10 else np.nan
                            for i in range(8)])
        print(f'{name:26s} {D[0]:6.2f} x{D[1]:6.2f} x{D[2]:6.2f} {np.mean(per_frame):10.2f} '
              f'{med[1]:14.0f} {med[2]:7.0f} {med[3]:7.0f}')


def analysis_intensity(P, frames, **_):
    edges = np.arange(0, 160, 10)
    for name, plat in P.items():
        vals = collections.defaultdict(list)
        for info in plat.sample(frames):
            p = plat.frame(info).points
            r = np.linalg.norm(p[:, :2], axis=1)
            idx = np.digitize(r, edges) - 1
            for k in range(len(edges) - 1):
                m = idx == k
                if m.sum():
                    vals[k].append(p[m, 3])
        med = [np.median(np.concatenate(vals[k])) if k in vals else np.nan
               for k in range(len(edges) - 1)]
        allv = np.concatenate([v for vv in vals.values() for v in vv])
        print(f'{name:26s} unique={len(np.unique(allv)):6d}  median by 10 m bin: '
              + ' '.join('  nan' if np.isnan(m) else f'{m:6.3f}' for m in med[:8]))


def analysis_beams(P, frames, **_):
    edges = np.arange(-40, 20.02, 0.1)
    centres = (edges[:-1] + edges[1:]) / 2
    print('Elevation is measured about the frame origin, so it is only meaningful for')
    print('datasets whose point frame IS the sensor frame (KITTI, nuScenes, Lyft).')
    for name, plat in P.items():
        pts = np.concatenate([plat.frame(i).points for i in plat.sample(frames)])
        r = np.linalg.norm(pts[:, :2], axis=1)
        m = (r > 3) & (r < 60)
        h = np.histogram(np.degrees(np.arctan2(pts[m, 2], r[m])), bins=edges)[0].astype(float)
        span = centres[h > h.max() * 0.005]
        print(f'{name:26s} elevation span {span.min():7.1f} .. {span.max():5.1f} deg  '
              f'({span.max() - span.min():.1f} deg total)')


def analysis_accumulate(P, frames, **_):
    edges = np.linspace(0, MAX_DIST, 16)
    kitti = P['KITTI']
    hk = np.mean([radial_hist(mask_range(kitti.frame(i).points), edges)
                  for i in kitti.sample(40)], axis=0)
    band = slice(1, 14)                                   # 5-70 m: 96% of KITTI Car+Ped labels
    print(f"{'platform':26s} {'N(all 15 bins)':>15s} {'N(5-70 m)':>11s} {'window':>9s}")
    for name, plat in P.items():
        if plat._sweeps is None:
            continue
        acc = {n: [] for n in range(1, 11)}
        for info in plat.sample(frames):
            for n in acc:
                acc[n].append(radial_hist(mask_range(plat.accumulate(info, n)), edges))
        hs = {n: np.mean(v, axis=0) for n, v in acc.items()}
        pick = lambda sl: next((n for n in sorted(hs) if np.all(hs[n][sl] >= hk[sl])), None)
        a, c = pick(slice(0, 15)), pick(band)
        w = (c - 1) / plat.rate_hz if c else float('nan')
        print(f'{name:26s} {str(a) if a else ">10":>15s} {str(c) if c else ">10":>11s} {w:8.2f}s')


def analysis_correction(P, frames, **_):
    """Apply sample_points_hist_based verbatim, with self-measured histograms."""
    edges50 = np.linspace(0, MAX_DIST, 51)
    edges10 = np.arange(0, 85, 10)
    rng = np.random.RandomState(0)
    kitti = P['KITTI']
    ksamp = [mask_range(kitti.frame(i).points) for i in kitti.sample(40)]
    hk50 = np.mean([radial_hist(p, edges50) for p in ksamp], axis=0)
    k10 = np.mean([radial_hist(p, edges10) for p in ksamp], axis=0)
    kb = []
    for info in kitti.infos:
        a = info['annos']
        g = np.asarray(a['gt_boxes_lidar'])
        nm = np.asarray(a['name'])[:len(g)]
        m = nm == 'Car'
        if m.sum():
            kb.append(np.stack([np.linalg.norm(g[m][:, :2], axis=1),
                                np.asarray(a['num_points_in_gt'])[:len(g)][m]], 1))
    kb = np.concatenate(kb)
    med = lambda a: np.array([np.median(a[np.digitize(a[:, 0], edges10) - 1 == i, 1])
                              if (np.digitize(a[:, 0], edges10) - 1 == i).sum() >= 10 else np.nan
                              for i in range(8)])
    kbm = med(kb)

    def apply_correction(points, hs, ht):
        d = np.linalg.norm(points[:, 0:2], axis=1)
        idx = np.floor(np.clip(d, 0, MAX_DIST - 1e-4) / MAX_DIST * len(hs)).astype(np.int32)
        return points[rng.rand(len(points)) < (ht[idx] / hs[idx])]

    for name, plat in P.items():
        if name in ('KITTI', 'Waymo'):
            continue
        fr = [plat.frame(i) for i in plat.sample(frames)]
        pts = [mask_range(f.points) for f in fr]
        hs = np.mean([radial_hist(p, edges50) for p in pts], axis=0)
        pre = np.mean([radial_hist(p, edges10) for p in pts], axis=0)
        post, bpre, bpost = [], [], []
        for p, f in zip(pts, fr):
            q = apply_correction(p, hs, hk50)
            post.append(radial_hist(q, edges10))
            m = f.names == plat.car_class
            if m.sum():
                r = np.linalg.norm(f.boxes[m][:, :2], axis=1)
                bpre.append(np.stack([r, points_in_boxes(p, f.boxes[m])], 1))
                bpost.append(np.stack([r, points_in_boxes(q, f.boxes[m])], 1))
        post = np.mean(post, axis=0)
        print(f'--- {name}')
        print('    ALL POINTS  ratio to KITTI  before: ' + ' '.join(f'{v:5.2f}' for v in pre / k10))
        print('                                after : ' + ' '.join(f'{v:5.2f}' for v in post / k10))
        if bpre:
            a, b = med(np.concatenate(bpre)), med(np.concatenate(bpost))
            print('    PER CAR BOX ratio to KITTI  before: '
                  + ' '.join('  nan' if np.isnan(v) else f'{v:5.2f}' for v in a / kbm))
            print('                                after : '
                  + ' '.join('  nan' if np.isnan(v) else f'{v:5.2f}' for v in b / kbm))


def analysis_shift_coor(P, frames, **_):
    """Estimate the z shift that puts each platform's ground plane at z = 0.

    `anchor_bottom_heights: [0]` places anchors with their base on z = 0, and anchors have to match
    GT boxes - so the AUTHORITATIVE target is the median base of the Car boxes (z_centre - h/2),
    not the road surface. The road is measured too, as an independent sanity check: the two should
    sit within a few centimetres of each other, since cars rest on the road.

    The road estimate uses a low percentile of z in a 3-15 m annulus rather than the modal z. The
    mode is unstable in dense urban scenes, where vehicles and structures can out-vote the road
    surface inside the annulus.
    """
    print(f"{'platform':26s} {'road z':>14s} {'Car box base z':>16s} {'SHIFT_COOR':>12s} "
          f"{'configured':>11s} {'delta':>7s}")
    for name, plat in P.items():
        road, bases = [], []
        for info in plat.sample(frames):
            fr = plat.frame(info)
            r = np.linalg.norm(fr.points[:, :2], axis=1)
            m = (r > 3) & (r < 15)
            if m.sum() > 500:
                road.append(np.percentile(fr.points[m, 2], 10))
            sel = fr.names == plat.car_class
            if sel.sum():
                bases.append(fr.boxes[sel][:, 2] - fr.boxes[sel][:, 5] / 2)
        road_z = float(np.median(road)) if road else float('nan')
        road_mad = float(np.median(np.abs(np.array(road) - road_z))) if road else float('nan')
        if bases:
            bb = np.concatenate(bases)
            base_z = float(np.median(bb))
            base_mad = float(np.median(np.abs(bb - base_z)))
        else:
            base_z = base_mad = float('nan')
        shift = -base_z if not np.isnan(base_z) else -road_z
        cfg = CONFIGURED_SHIFT.get(name.split()[0], 0.0)
        print(f'{name:26s} {road_z:9.2f} +-{road_mad:.2f} {base_z:11.2f} +-{base_mad:.2f} '
              f'{shift:12.2f} {cfg:11.2f} {shift - cfg:+7.2f}')
    print('\n  road z   = 10th percentile of z in a 3-15 m annulus, median over frames (+- MAD)')
    print('  base z   = median Car box base, z_centre - height/2, over every sampled box (+- MAD)')
    print('  SHIFT_COOR = -base z, i.e. what puts Car box bases on z = 0 to match')
    print('               anchor_bottom_heights: [0]')

CONFIGURED_SHIFT = {'KITTI': 1.7, 'nuScenes': 1.75, 'Lyft': 1.6, 'PandaSet': 0.3, 'Waymo': 0.0}


def analysis_platforms(P, frames, **_):
    print(f"{'platform':26s} {'frames':>8s} {'pts/frame':>11s}")
    for name, plat in P.items():
        h = np.mean([len(mask_range(plat.frame(i).points)) for i in plat.sample(frames)])
        print(f'{name:26s} {len(plat):8d} {h:11,.0f}')


ANALYSES = dict(shift_coor=analysis_shift_coor, range=analysis_range, boxes=analysis_boxes, intensity=analysis_intensity,
                beams=analysis_beams, accumulate=analysis_accumulate,
                correction=analysis_correction, platforms=analysis_platforms)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('analysis', choices=sorted(ANALYSES))
    ap.add_argument('--frames', type=int, default=20, help='frames sampled per platform')
    ap.add_argument('--platforms', nargs='*', default=None, help='subset of platform names')
    args = ap.parse_args()
    P = build_platforms(args.platforms)
    ANALYSES[args.analysis](P, args.frames)


if __name__ == '__main__':
    main()
