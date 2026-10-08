"""Scan-line spacing vs voxel height: a label-free descriptor of the scan-line term (experiments_md 20261008_03).

Hypothesis: the first sparse convolution sums over a 3x3x3 neighbourhood of 0.1 x 0.1 x 0.15 m voxels, so whether
adjacent scan lines land in VERTICALLY adjacent voxels (line spacing vs the 0.15 m z-voxel) sets the neighbourhood
statistics the detector learns. Accumulation and random thinning change point counts, not line spacing, so a pair whose
spacing ratio is large keeps a residual that count-matching cannot close.

Measured per dataset on TRAIN clouds AS THE FAMILY LOADS THEM (its own source config in training mode with the
augmentor queue emptied: SHIFT_COOR, ego removal, x,y,z, the data processor's range mask; MAX_SWEEPS / SWEEP_SELECTION
of the config), per range ring of the voxel centre:
  - share of occupied voxels with >= 1 occupied vertical neighbour (same x, y column, z +- 1);
  - mean occupied neighbours: vertical (of 2), same-layer (of 8), whole 3x3x3 cube (of 26).
Scene-wide numbers are label-free. The same inside Car GT boxes uses SOURCE labels (legal for a source; for a dataset
read as a target it is analysis).

    python analysis/vertical_neighbour_descriptor.py <frames> <out.npz> [name ...]
    python analysis/vertical_neighbour_descriptor.py report <out.npz> [...]
"""
import copy
import sys
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

CFG = 'cfgs/da-ieee-access/'
# name: (config, DATA_CONFIG key path, overrides). Each is the dataset AS A TRAINING SOURCE of the family.
SPECS = {
    'kitti': (CFG + 'centerpoint-sourceonly-kitti.yaml', 'DATA_CONFIG', {}),
    'nuscenes_n008': (CFG + 'centerpoint-sourceonly-nuscenes.yaml', 'DATA_CONFIG', {'VEHICLE': 'n008'}),
    'nuscenes_n015': (CFG + 'centerpoint-sourceonly-nuscenes.yaml', 'DATA_CONFIG', {'VEHICLE': 'n015'}),
    'lyft40': (CFG + 'centerpoint-sourceonly-lyft.yaml', 'DATA_CONFIG', {'LIDAR_CONFIG': 40}),
    'lyft64': (CFG + 'centerpoint-sourceonly-lyft.yaml', 'DATA_CONFIG', {'LIDAR_CONFIG': 64}),
    'waymo': (CFG + 'centerpoint-sourceonly-waymo.yaml', 'DATA_CONFIG', {}),
    'pandaset_spin': (CFG + 'centerpoint-sourceonly-pandaset.yaml', 'DATA_CONFIG', {}),
    'pandaset_flash': (CFG + 'centerpoint-pandaset-flash2spin.yaml', 'DATA_CONFIG', {}),
    # accumulated sources as trained (27468: consecutive Boston 15 / Singapore 10; 27705: displacement spread)
    'nus_acc_n008': (CFG + 'centerpoint-accum-legaldepth-nuscenes2kitti.yaml', 'DATA_CONFIGS.NUSCENES_N008', {}),
    'nus_acc_n015': (CFG + 'centerpoint-accum-legaldepth-nuscenes2kitti.yaml', 'DATA_CONFIGS.NUSCENES_N015', {}),
    'nus_disp_n008': (CFG + 'centerpoint-accum-legaldepth-disp-nuscenes2kitti.yaml', 'DATA_CONFIGS.NUSCENES_N008', {}),
    'nus_disp_n015': (CFG + 'centerpoint-accum-legaldepth-disp-nuscenes2kitti.yaml', 'DATA_CONFIGS.NUSCENES_N015', {}),
    'pandaset_spin_acc4': (CFG + 'centerpoint-accum-pandaset-spin2flash.yaml', 'DATA_CONFIG', {}),
}
RINGS = [0, 10, 20, 30, 40, 50, 75]
PCR = np.array([-75.2, -75.2, -2.0, 75.2, 75.2, 4.0]); VOX = np.array([0.1, 0.1, 0.15])
NX, NZ = 2000, 100
OFFS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]


def key_of(i, j, k):
    return (i * NX + j) * NZ + k


def neighbours(x):
    """Per occupied voxel: ring, n vertical (0-2), n same-layer (0-8), n cube (0-26); plus the voxel keys."""
    ins = np.all((x >= PCR[:3]) & (x < PCR[3:]), 1)
    ijk = np.floor((x[ins] - PCR[:3]) / VOX).astype(np.int64)
    keys = np.unique(key_of(ijk[:, 0], ijk[:, 1], ijk[:, 2]))
    i, j, k = keys // (NX * NZ), (keys // NZ) % NX, keys % NZ
    cen = PCR[:2] + (np.stack([i, j], 1) + 0.5) * VOX[:2]
    ring = np.digitize(np.hypot(cen[:, 0], cen[:, 1]), RINGS) - 1
    ring = np.where((ring >= 0) & (ring < len(RINGS) - 1), ring, -1)
    nv = np.zeros(len(keys), np.int32); nl = np.zeros(len(keys), np.int32); nc = np.zeros(len(keys), np.int32)
    for dx, dy, dz in OFFS:
        q = key_of(i + dx, j + dy, k + dz)
        pos = np.searchsorted(keys, q); pos = np.clip(pos, 0, len(keys) - 1)
        hit = (keys[pos] == q) & (k + dz >= 0) & (k + dz < NZ)
        nc += hit
        if dx == 0 and dy == 0:
            nv += hit
        elif dz == 0:
            nl += hit
    return ring, nv, nl, nc, keys, i, j, k


def accumulate(acc, ring, nv, nl, nc):
    for g in range(6):
        s = ring == g
        if not s.any():
            continue
        a = acc[g]; a[0] += s.sum(); a[1] += (nv[s] > 0).sum(); a[2] += nv[s].sum(); a[3] += nl[s].sum(); a[4] += nc[s].sum()


def measure(name, n):
    cfg_file, path, over = SPECS[name]
    cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
    dc = cfg
    for p in path.split('.'):
        dc = dc[p]
    dc = copy.deepcopy(dc); dc.update(over)
    ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
    step = max(1, len(ds) // n); frames = list(range(0, len(ds), step))[:n]
    scene = np.zeros((6, 5)); car = np.zeros((6, 5)); pts = []; nbox = 0
    with _augmentation_off(ds):
        for fi, i in enumerate(frames):
            d = ds[i]; x = d['points'][:, :3].astype(np.float64); pts.append(len(x))
            ring, nv, nl, nc, keys, vi, vj, vk = neighbours(x)
            accumulate(scene, ring, nv, nl, nc)
            b = d.get('gt_boxes')
            if b is not None and len(b):
                b = b[b[:, 7] == 1]
                if len(b):
                    cen = PCR[:3] + (np.stack([vi, vj, vk], 1) + 0.5) * VOX
                    inb = roiaware_pool3d_utils.points_in_boxes_cpu(cen, b[:, :7]).max(0) > 0
                    accumulate(car, ring[inb], nv[inb], nl[inb], nc[inb]); nbox += len(b)
            if (fi + 1) % 25 == 0:
                print(f'  {name}: {fi + 1} / {len(frames)}', flush=True)
    return dict(scene=scene, car=car, n=len(frames), pts=np.median(pts), nbox=nbox,
                sweeps=dc.get('MAX_SWEEPS', 1), sel=str(dc.get('SWEEP_SELECTION', None)))


def report(paths):
    R = {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        for name in z['names']:
            R[str(name)] = z[str(name)].item()
    hdr = ' | '.join(f'{RINGS[g]}-{RINGS[g + 1]}' for g in range(6))
    metrics = [(1, 'share of occupied voxels with >= 1 occupied vertical neighbour (z +- 1)'),
               (2, 'mean occupied vertical neighbours (of 2)'), (3, 'mean occupied same-layer neighbours (of 8)'),
               (4, 'mean occupied 3x3x3 neighbours (of 26)')]
    for what in ('scene', 'car'):
        print(f'\n## {what} voxels ({"label-free" if what == "scene" else "inside Car GT boxes: source labels; analysis when the dataset is a target"})')
        for col, title in metrics:
            print(f'\n{title}, per ring of the voxel centre (m):\n')
            print(f'| dataset (sweeps) | frames | pts/frame (k) | {hdr} |')
            print('|---' * 9 + '|')
            for name, r in R.items():
                a = r[what]; n = np.maximum(a[:, 0], 1)
                sel = f', {r["sel"]}' if r['sel'] != 'None' else ''
                print(f'| {name} ({r["sweeps"]}{sel}) | {r["n"]} | {r["pts"] / 1e3:.0f} | ' +
                      ' | '.join(f'{a[g, col] / n[g]:.2f}' for g in range(6)) + ' |')


if __name__ == '__main__':
    if sys.argv[1] == 'report':
        report(sys.argv[2:])
    else:
        n, out = int(sys.argv[1]), sys.argv[2]
        names = sys.argv[3:] or list(SPECS)
        res = {}
        for name in names:
            print(f'== {name}', flush=True)
            res[name] = measure(name, n)
        np.savez(out, names=np.array(list(res)), **{k: np.array(v, dtype=object) for k, v in res.items()})
        report([out])
