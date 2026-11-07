"""Keep rates for the KITTI count-matched control L_K of experiments_md 20261003_05 §5 item 37c. ANALYSIS, CPU.

L_K thins KITTI TRAIN clouds at random per 1.5-m radius bin (TRAIN_FACTOR_SWAP RANGE_KEEP, frame-seeded as R3) to the
point count per bin of the lattice arm P_K (TRAIN_FACTOR_SWAP RING_THIN MODE pattern: nuScenes' 1.33 deg lattice,
0.332 deg azimuth step). Per bin b: RATE[b] = sum over frames of P_K's points / sum of the full scan's points, both
counted inside the voxel grid (POINT_CLOUD_RANGE on x, y AND z, after SHIFT_COOR), as HIST_DIST_IN_GRID does for the
other targets' L rows. Both counts are taken on the raw scan exactly as the training loader cuts it (the same
kitti_factor_swap.apply_points call, the same frame id), before augmentation; radius does not change under rotation or
flip. Bins with no full-scan point get rate 1. Beyond the last edge the last rate applies (radius_bin).
    python analysis/kitti_pattern_count_rates.py <P_K cfg> <frames> <out.json>     # from ST3D/tools, in the container
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402

EDGES = [round(1.5 * i, 1) for i in range(51)]       # 0, 1.5, ..., 75 m: 50 bins, as R3 and HIST_DIST_BINS 50


def main(cfg_file, n_frames, out):
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.datasets.kitti import kitti_factor_swap
    from pcdet.utils import common_utils
    cfg = cfg_from_yaml_file(cfg_file, EasyDict())
    dc = cfg.DATA_CONFIG
    swap = dc.TRAIN_FACTOR_SWAP
    assert kitti_factor_swap.arm_of(swap) == 'RING_THIN' and swap.RING_THIN.MODE == 'pattern', 'expects the P_K config'
    ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
    pcr = np.asarray(dc.POINT_CLOUD_RANGE, dtype=np.float64)
    shift = np.asarray(dc.get('SHIFT_COOR', [0, 0, 0]), dtype=np.float64)
    edges = np.asarray(EDGES)
    full, kept = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    idx = np.linspace(0, len(ds.kitti_infos) - 1, min(n_frames, len(ds.kitti_infos))).astype(int)

    def counts(points):
        p = points[:, :3].astype(np.float64) + shift
        g = np.all((p >= pcr[:3]) & (p < pcr[3:6]), axis=1)
        b = kitti_factor_swap.radius_bin(points[g, :2], EDGES)
        return np.bincount(b, minlength=len(edges) - 1)
    for k in idx:
        sid = ds.kitti_infos[k]['point_cloud']['lidar_idx']
        raw = ds.get_lidar(sid)
        full += counts(raw)
        kept += counts(kitti_factor_swap.apply_points(raw, swap, sid))
    rate = np.where(full > 0, np.minimum(kept / np.maximum(full, 1), 1.0), 1.0)
    res = dict(cfg=cfg_file, frames=len(idx), EDGES=EDGES, RATE=[round(float(r), 4) for r in rate],
               full_per_frame=(full / len(idx)).round(2).tolist(), kept_per_frame=(kept / len(idx)).round(2).tolist(),
               kept_share=float(kept.sum() / full.sum()))
    json.dump(res, open(out, 'w'), indent=1)
    print('frames %d, in-grid points per frame full %.0f, lattice %.0f, share %.3f; wrote %s'
          % (len(idx), full.sum() / len(idx), kept.sum() / len(idx), res['kept_share'], out))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
