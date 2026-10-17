"""A4 stage-1 pre-flight (experiments_md 20261017_09 §2.2): how many nuScenes TRAIN frames exceed the voxel cap at the row's depths,
measured through the row's OWN source loaders.

    python analysis/a4_voxel_cap_check.py <cfg> [--frames 100]

For each source platform block in DATA_CONFIGS (NUSCENES_N008, NUSCENES_N015): --frames frames on a uniform stride over that
platform's train infos, served by the training dataset (MAX_SWEEPS, 200-sweep infos, motion compensation, range mask, as the
row trains). Occupied voxels are counted from the served `points` (after the range mask, before the voxeliser's cap) on the
row's VOXEL_SIZE and POINT_CLOUD_RANGE, and compared with MAX_NUMBER_OF_VOXELS (train). Two views:
- `unaugmented`: the dataset in eval mode (no augmentation), the view of 20261017_10 §3.1;
- `as trained`: training mode, augmentation on (world flip / rotation / scaling, object scaling).
The voxeliser's own output count is checked against min(count, cap). Label-free (source boxes only move with augmentation).
Bar (I3 of 20261017_03): at most 10% of frames above the cap.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _init_path  # noqa: F401,E402  (must precede pcdet)
from easydict import EasyDict  # noqa: E402

from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


def occupied(points, pc_range, vs):
    lo, hi = np.array(pc_range[:3]), np.array(pc_range[3:])
    p = points[:, :3]
    m = np.all((p >= lo) & (p < hi), axis=1)
    idx = np.floor((p[m] - lo) / np.array(vs)).astype(np.int64)
    return len(np.unique(idx, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg')
    ap.add_argument('--frames', type=int, default=100)
    a = ap.parse_args()
    cfg = EasyDict(); cfg_from_yaml_file(a.cfg, cfg)
    logger = common_utils.create_logger()
    for key, dcfg in cfg.DATA_CONFIGS.items():
        vox = [p for p in dcfg.DATA_PROCESSOR if p.NAME == 'transform_points_to_voxels'][0]
        cap = vox.MAX_NUMBER_OF_VOXELS['train']
        ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                    logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
        n = len(ds)
        frames = list(range(0, n, max(1, n // a.frames)))[:a.frames]
        print(f'{key}: MAX_SWEEPS {dcfg.MAX_SWEEPS}, INFO_PATH {list(dcfg.INFO_PATH.train)}, '
              f'compensation {dcfg.get("GT_BOXES_MOTION_COMPENSATION", None)}, {n} train frames, {len(frames)} sampled, '
              f'cap {cap}, voxel {list(vox.VOXEL_SIZE)}', flush=True)
        for view in ('unaugmented', 'as trained'):
            if view == 'unaugmented':
                ds.eval()
            else:
                ds.train()
            counts, mismatch, diffs = [], 0, []
            for i in frames:
                d = ds[i]
                c = occupied(d['points'], dcfg.POINT_CLOUD_RANGE, vox.VOXEL_SIZE)
                counts.append(c)
                if len(d['voxel_coords']) != min(c, cap):
                    mismatch += 1
                if c <= cap:
                    diffs.append(c - len(d['voxel_coords']))
            x = np.array(counts)
            print(f'  {view:11s}: above cap {np.mean(x > cap):.3f} ({int((x > cap).sum())} of {len(x)}); voxels median '
                  f'{int(np.median(x))}, p90 {int(np.percentile(x, 90))}, max {int(x.max())}; voxeliser count != '
                  f'min(count, cap) on {mismatch} frames', flush=True)
            if diffs:
                dd = np.array(diffs)
                print(f'    below the cap: count - voxeliser count median {np.median(dd):.0f}, min {dd.min()}, max {dd.max()}, '
                      f'max relative {np.max(np.abs(dd) / np.maximum(1, np.array(counts)[np.array(counts) <= cap])):.5f}',
                      flush=True)
        del ds


if __name__ == '__main__':
    main()
