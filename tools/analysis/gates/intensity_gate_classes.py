"""Intensity-calibration gate, every class, three pairs (experiments_md 20261002_02 section 4).

Pairs: S2 nuScenes -> KITTI (KITTI restricted to its +-40 deg camera FOV, the only region it labels);
S3 PandaSet Pandar64 (spin) -> PandarGT (flash) and S4 the reverse, both restricted to the flash
lidar's +-30 deg forward cone. GT boxes on both sides, diagnosis only. Classes in KITTI vocabulary via
the repo's own ontology maps (bicycle -> Cyclist; motorcycles are Misc and excluded from background).

Per class and range ring: AUC of intensity (class points vs background points) in source and target,
and the held-out distance (W1, target units) from mapped source class points to target class points
under four maps: raw (unit rescale only), global (one map per ring), pooled-foreground (one map for all
three classes' points), class-specific. Floor = target half A vs half B. Halves split by sequence
(PandaSet) or by strided frame (nuScenes, KITTI).

    python analysis/gates/intensity_gate_classes.py          (needs --bind /home/koyama/code/ST3D:/root/ST3D)
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, '/home/koyama/code/ST3D/tools'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _init_path  # noqa: F401
from pcdet.utils.ontology_mapping import get_ontology_mapping

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
rng = np.random.RandomState(0)
BG_KEEP = 0.15

def box_index(pts, boxes):
    """Index of the first box containing each point, -1 if none."""
    idx = np.full(len(pts), -1)
    for k, b in enumerate(boxes):
        d = pts[:, :3] - b[:3]; c, s = np.cos(-b[6]), np.sin(-b[6])
        x = d[:, 0] * c - d[:, 1] * s; y = d[:, 0] * s + d[:, 1] * c
        m = (np.abs(x) <= b[3] / 2) & (np.abs(y) <= b[4] / 2) & (np.abs(d[:, 2]) <= b[5] / 2) & (idx < 0)
        idx[m] = k
    return idx

def rows(pts, boxes, names, mapping, half):
    """(range, intensity, label, half): label 0..2 class, 3 background; points in other boxes dropped."""
    cls = np.array([CLASSES.index(mapping.get(n, '')) if mapping.get(n, '') in CLASSES else -1 for n in names], dtype=int)
    bi = box_index(pts, boxes) if len(boxes) else np.full(len(pts), -1)
    lab = np.where(bi < 0, 3, cls[np.maximum(bi, 0)] if len(cls) else 3)
    keep = (lab >= 0) & ((lab < 3) | (rng.rand(len(pts)) < BG_KEEP))
    r = np.linalg.norm(pts[:, :2], axis=1)
    return np.stack([r, pts[:, 3], lab, np.full(len(pts), half)], 1)[keep & (r < 75)]

def cone(p, deg):
    return p[(p[:, 0] > 0) & (np.abs(np.degrees(np.arctan2(p[:, 1], p[:, 0]))) < deg)]

def nuscenes(n=1000):
    infos = pickle.load(open('/home/koyama/code/ST3D/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl', 'rb'))
    mp = get_ontology_mapping('nuscenes', 'kitti'); out = []
    for j, i in enumerate(infos[::len(infos) // n][:n]):
        p = np.fromfile('/home/koyama/code/ST3D/data/nuscenes/v1.0-trainval/' + i['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :4]
        p = p[np.linalg.norm(p[:, :2], axis=1) > 1.5]
        out.append(rows(p, i['gt_boxes'][:, :7], np.asarray(i['gt_names']), mp, j % 2))
    return np.concatenate(out)

def kitti(n=1000):
    infos = pickle.load(open('/home/koyama/code/ST3D/data/kitti/kitti_infos_train.pkl', 'rb'))
    mp = {c: c for c in CLASSES}; out = []
    for j, i in enumerate(infos[::max(1, len(infos) // n)][:n]):
        p = cone(np.fromfile('/home/koyama/code/ST3D/data/kitti/training/velodyne/%s.bin' % i['point_cloud']['lidar_idx'], dtype=np.float32).reshape(-1, 4), 40)
        b = i['annos']['gt_boxes_lidar']
        out.append(rows(p, b, i['annos']['name'][:len(b)], mp, j % 2))
    return np.concatenate(out)

def pandaset(device, n=600):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset
    os.chdir('/home/koyama/code/ST3D/tools')
    cfg = cfg_from_yaml_file('cfgs/da-ieee-access/da_pandaset_%s_dataset.yaml' % ('spin' if device == 0 else 'flash'), EasyDict())
    import logging
    ds = PandasetDataset(cfg, CLASSES, training=False, model_ontology='kitti', logger=logging.getLogger('gate'))
    ds.pandaset_infos = pickle.load(open('/home/koyama/code/ST3D/data/pandaset/pandaset_infos_train.pkl', 'rb'))
    mp = get_ontology_mapping('pandaset', 'kitti'); out = []
    infos = ds.pandaset_infos[::max(1, len(ds.pandaset_infos) // n)][:n]
    for i in infos:
        pose = ds._get_pose(i)
        p = cone(ds._get_lidar_points(i, pose), 30)
        boxes, labels, _ = ds._get_annotations(i, pose)
        out.append(rows(p, boxes[:, :7], labels, mp, int(i['sequence']) % 2))
    return np.concatenate(out)

def dequantize(x, step):
    return x + rng.uniform(-step / 2, step / 2, len(x)) if step else x

def qmap(fs, ft, step):
    s = np.sort(dequantize(fs, step)); q = np.linspace(0, 1, 201); tq = np.quantile(ft, q)
    return lambda x: np.interp(np.searchsorted(s, dequantize(x, step)) / len(s), q, tq)

def w1(a, b):
    q = np.linspace(0, 1, 201); return np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q)))

def auc(pos, neg):
    if len(pos) < 50 or len(neg) < 50: return np.nan
    neg = np.sort(rng.choice(neg, min(len(neg), 200000), replace=False))
    return np.mean((np.searchsorted(neg, pos, 'left') + np.searchsorted(neg, pos, 'right')) / 2.0) / len(neg)

def gate(name, S, T, scale, step, edges, min_pts=200):
    print('\n=== %s ===   (W1 in target units; "-" = fewer than %d class points in a half)' % (name, min_pts))
    print('class       ring  | pts s/t        | AUC s / t     | W1: raw    global  pooled-fg  class   floor')
    A = lambda x: x[x[:, 3] == 0, 1]; B = lambda x: x[x[:, 3] == 1, 1]
    for c, cname in enumerate(CLASSES):
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = S[(S[:, 0] >= lo) & (S[:, 0] < hi)]; t = T[(T[:, 0] >= lo) & (T[:, 0] < hi)]
            sc, tc = s[s[:, 2] == c], t[t[:, 2] == c]
            sf, tf = s[s[:, 2] < 3], t[t[:, 2] < 3]
            head = '%-10s %3d-%-3d | %6d/%6d | %.3f / %.3f' % (cname, lo, hi, len(sc), len(tc),
                                                               auc(sc[:, 1], s[s[:, 2] == 3, 1]), auc(tc[:, 1], t[t[:, 2] == 3, 1]))
            if min(len(A(sc)), len(B(sc)), len(A(tc)), len(B(tc))) < min_pts:
                print(head + ' |  -'); continue
            # the global map must weight classes as the cloud does: background was subsampled to BG_KEEP
            sw = np.concatenate([A(s[s[:, 2] < 3])] + [A(s[s[:, 2] == 3])] * int(round(1 / BG_KEEP)))
            tw = np.concatenate([A(t[t[:, 2] < 3])] + [A(t[t[:, 2] == 3])] * int(round(1 / BG_KEEP)))
            gm, fm, cm = qmap(sw, tw, step), qmap(A(sf), A(tf), step), qmap(A(sc), A(tc), step)
            print(head + ' |     %.3f  %.3f   %.3f      %.3f   %.3f' % (
                w1(dequantize(B(sc), step) * scale, B(tc)), w1(gm(B(sc)), B(tc)), w1(fm(B(sc)), B(tc)),
                w1(cm(B(sc)), B(tc)), w1(A(tc), B(tc))))

def cached(name, fn):
    """GATE_CACHE=<dir> keeps the per-dataset point tables between runs (they take minutes to read)."""
    d = os.environ.get('GATE_CACHE')
    if not d: return fn()
    f = os.path.join(d, 'gate_%s.npy' % name)
    if not os.path.exists(f): np.save(f, fn())
    return np.load(f)

if __name__ == '__main__':
    NU, KI = cached('nuscenes', nuscenes), cached('kitti', kitti)
    gate('S2 nuScenes -> KITTI', NU, KI, 1 / 255.0, 1.0, [0, 10, 20, 30, 40, 70])
    SPIN, FLASH = cached('pandaset_spin', lambda: pandaset(0)), cached('pandaset_flash', lambda: pandaset(1))
    gate('S3 PandaSet spin -> flash (+-30 deg cone)', SPIN, FLASH, 1.0, 1 / 255.0, [0, 10, 20, 30, 50, 75])
    gate('S4 PandaSet flash -> spin (+-30 deg cone)', FLASH, SPIN, 1.0, 1 / 255.0, [0, 10, 20, 30, 50, 75])
    for nm, X in (('nuScenes', NU), ('KITTI', KI), ('PandaSet spin', SPIN), ('PandaSet flash', FLASH)):
        print('%-15s class points: Car %d  Ped %d  Cyc %d' % (nm, (X[:, 2] == 0).sum(), (X[:, 2] == 1).sum(), (X[:, 2] == 2).sum()))
