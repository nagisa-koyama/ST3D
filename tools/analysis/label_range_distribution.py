"""Label range distribution of the source and target of one config: how far out each dataset labels its objects.

Question (experiments_md T05 §12, the nuScenes -> Waymo residual, user 2026-10-17): do the source and target label
objects out to different maximum distances? Reads the info files the config's dataset blocks point at, takes the BEV
centre range sqrt(x^2 + y^2) of every box in the sensor frame, and prints, per raw class name: count, quantiles, max,
and the share beyond fixed radii. It also prints each block's POINT_CLOUD_RANGE, so a label beyond the grid is visible.

Reads labels only to describe datasets (analysis); nothing here feeds a method setting.

    python analysis/label_range_distribution.py --cfg_file cfgs/... [--split train|val ...] [--out file.json]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402

import numpy as np  # noqa: E402

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402

RADII = (30, 40, 50, 54, 60, 65, 70, 75, 80)
OVERRIDE = {}  # (DATASET, split) -> info file names; labels are the same in the 10- and 200-sweep nuScenes infos


def vehicle_filter(block, infos):
    """VEHICLE (nuScenes n008 / n015) by the log name that prefixes every lidar file name; the loader's own filter
    goes through the log table, and the two agree because each log belongs to one vehicle."""
    v = block.get('VEHICLE', None)
    if not v:
        return infos
    v = [v] if isinstance(v, str) else list(v)
    return [i for i in infos if any(Path(i['lidar_path']).name.startswith(x + '-') for x in v)]
QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)


def blocks(c):
    out = {}
    if c.get('DATA_CONFIGS', None):
        for name, b in c.DATA_CONFIGS.items():
            out[f'source:{name}'] = b
    elif c.get('DATA_CONFIG', None):
        out['source'] = c.DATA_CONFIG
    if c.get('DATA_CONFIG_TAR', None):
        out['target'] = c.DATA_CONFIG_TAR
    return out


def info_files(block, split):
    """Info files of one split, and the frame stride to apply. nuScenes / Lyft infos live under DATA_PATH / VERSION;
    Waymo has one info file per sequence, listed in ImageSets/<DATA_SPLIT[split]>.txt (waymo_dataset.py)."""
    root = (TOOLS / block.DATA_PATH).resolve() if not str(block.DATA_PATH).startswith('/') else Path(block.DATA_PATH)
    if block.get('INFO_PATH', None) is not None:
        if split not in block.INFO_PATH:
            return root, [], 1
        base = root / block.VERSION if block.get('VERSION', None) else root
        rels = OVERRIDE.get((block.DATASET, split), block.INFO_PATH[split])
        return root, [base / rel for rel in rels], 1
    if block.get('PROCESSED_DATA_TAG', None) is not None and split in block.get('DATA_SPLIT', {}):
        seqs = [s.strip() for s in open(root / 'ImageSets' / f'{block.DATA_SPLIT[split]}.txt') if s.strip()]
        names = [Path(s).stem for s in seqs]
        files = [root / block.PROCESSED_DATA_TAG / n / f'{n}.pkl' for n in names]
        return root, files, int(block.get('SAMPLED_INTERVAL', {}).get(split, 1))
    return root, [], 1


def boxes_of(info):
    """(names, boxes[:, :7], points or None) for nuScenes / Lyft style or Waymo / KITTI style infos."""
    if 'gt_boxes' in info and 'gt_names' in info:
        pts = info.get('num_lidar_pts', None)
        return np.asarray(info['gt_names']), np.asarray(info['gt_boxes'])[:, :7], (None if pts is None else np.asarray(pts))
    a = info['annos']
    boxes = a.get('gt_boxes_lidar', None)
    if boxes is None:
        return np.asarray(a['name']), None, None
    pts = a.get('num_points_in_gt', None)
    names = np.asarray(a['name'])
    boxes = np.asarray(boxes)[:, :7]
    if len(names) != len(boxes):  # KITTI keeps DontCare in names only
        names = names[:len(boxes)]
        pts = None if pts is None else np.asarray(pts)[:len(boxes)]
    return names, boxes, (None if pts is None else np.asarray(pts))


def summarise(r):
    if len(r) == 0:
        return dict(n=0)
    s = dict(n=int(len(r)), max=float(r.max()))
    for q in QUANTILES:
        s[f'p{q * 100:g}'] = float(np.quantile(r, q))
    for R in RADII:
        s[f'share_gt_{R}m'] = float((r > R).mean())
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--split', nargs='*', default=['train', 'test'], help='INFO_PATH keys to read')
    ap.add_argument('--out', default=None)
    ap.add_argument('--override', nargs='*', default=[], metavar='DATASET:SPLIT=FILE',
                    help='read FILE instead of the configured INFO_PATH, e.g. NuScenesDataset:train=nuscenes_infos_10sweeps_train.pkl')
    args = ap.parse_args()
    for o in args.override:
        k, f = o.split('=', 1)
        d, s = k.split(':', 1)
        OVERRIDE[(d, s)] = [f]
    cfg_from_yaml_file(args.cfg_file, cfg)

    result = {}
    for bname, block in blocks(cfg).items():
        print(f'=== {bname}: {block.DATASET}, POINT_CLOUD_RANGE {list(block.POINT_CLOUD_RANGE)}, '
              f'SHIFT_COOR {list(block.get("SHIFT_COOR", [0, 0, 0]))}')
        for split in args.split:
            root, files, stride = info_files(block, split)
            if not files:
                continue
            infos, missing = [], 0
            for p in files:
                if not p.exists():
                    missing += 1
                    continue
                with open(p, 'rb') as f:
                    infos.extend(pickle.load(f))
            if missing:
                print(f'  [{split}] {missing} of {len(files)} info files missing')
            infos = vehicle_filter(block, infos)[::stride]
            per = {}
            for info in infos:
                names, boxes, pts = boxes_of(info)
                if boxes is None or len(boxes) == 0:
                    continue
                r = np.hypot(boxes[:, 0], boxes[:, 1])
                for n in np.unique(names):
                    m = names == n
                    per.setdefault(n, {'r': [], 'r_pts': []})
                    per[n]['r'].append(r[m])
                    if pts is not None:
                        per[n]['r_pts'].append(r[m & (pts > 0)])
            key = f'{bname}|{split}'
            result[key] = dict(frames=len(infos), stride=stride, n_files=len(files), classes={})
            print(f'  [{split}] {len(infos)} frames (stride {stride}) from {len(files)} info file(s), e.g. {files[0]}')
            for n in sorted(per, key=lambda k: -sum(len(x) for x in per[k]['r'])):
                r = np.concatenate(per[n]['r'])
                rp = np.concatenate(per[n]['r_pts']) if per[n]['r_pts'] else np.zeros(0)
                s, sp = summarise(r), summarise(rp)
                result[key]['classes'][str(n)] = dict(all=s, with_points=sp)
                if s['n'] < 50:
                    continue
                print(f'    {str(n):22s} n={s["n"]:8d} p50={s["p50"]:6.1f} p90={s["p90"]:6.1f} p99={s["p99"]:6.1f} '
                      f'p99.9={s["p99.9"]:6.1f} max={s["max"]:6.1f} | >50 {s["share_gt_50m"]:.3f} >54 {s["share_gt_54m"]:.3f} '
                      f'>60 {s["share_gt_60m"]:.3f} >70 {s["share_gt_70m"]:.3f} >75 {s["share_gt_75m"]:.3f}'
                      + (f' | with points: n={sp["n"]} max={sp.get("max", 0):.1f} >50 {sp.get("share_gt_50m", 0):.3f}'
                         if sp['n'] else ''))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, 'w'), indent=1)
        print(f'written {args.out}')


if __name__ == '__main__':
    main()
