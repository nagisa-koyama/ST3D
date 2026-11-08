"""Pandar64 laser lines kept per sample by RING_PATTERN's lattice thinning toward the HDL-32E (experiments_md 20261003_05
§5 item 42d record 5; report 20261108_06). Source TRAIN frames, training mode, the P->N row's own config and code path.

    python analysis/pandar64_lines_kept.py <ring-pattern cfg> <n frames> <out.json>

A "line" is a Pandar64 channel (0 = top laser) as labelled by the row's own ring cache (RING_PATTERN.LABEL_CACHE,
pandaset_rings.py: firing order + firing time + azimuth column). Per frame (every k-th TRAIN info, k = len // n):

- `raw`: distinct channels with >= 1 point in the full device-0 cloud (before the operator);
- `op`: distinct channels with >= 1 point right after the operator (PandasetDataset._ring_keep_mask, the call
  _get_lidar_points makes; it acts in PandaSet ego axes, before the axis swap, SHIFT_COOR, augmentation and the range
  crop);
- `raw_in_range` / `op_in_range`: the same on the points inside POINT_CLOUD_RANGE after the axis swap and SHIFT_COOR
  (x, y as mask_points_and_boxes_outside_range; z as the voxeliser, lo <= p < hi on all three) - what the detector
  receives, augmentation aside (world flip / rotation / x0.95-1.05 scaling only moves the crop boundary);
- `lattice`: the channel count the lattice targets for the sample's drawn vertical phase (kept_channels), the upper
  bound on `op`;
- point counts behind each, for the kept fractions.

The operator's phases come from the global np.random; each frame is seeded (1000 + index), and the script checks that
_get_lidar_points under the same seed returns exactly the operator's kept points, so the count is of the loader's cloud.
No labels are read (only the source's channel cache and clouds).
"""
import json
import sys

import numpy as np


def count_lines(labels, mask=None):
    """Distinct channels (labels >= 0) holding >= 1 point; `mask` selects points first. Negative labels (unlabelled
    points, which ring_pattern_mask drops) never count as a line."""
    lab = np.asarray(labels)
    if mask is not None:
        lab = lab[np.asarray(mask, dtype=bool)]
    lab = lab[lab >= 0]
    return int(len(np.unique(lab)))


def to_normative(ego_xyz, shift):
    """PandaSet ego axes (x right, y forward, z up) -> the loader's normative frame (x forward, y left), + SHIFT_COOR;
    the transform PandasetDataset._get_lidar_points and __getitem__ apply."""
    p = np.asarray(ego_xyz, dtype=np.float64)[:, [1, 0, 2]].copy()
    p[:, 1] = -p[:, 1]
    return p + np.asarray(shift if shift is not None else [0.0, 0.0, 0.0], dtype=np.float64)


def in_range(points, pcr):
    """lo <= p < hi on x, y, z: mask_points_and_boxes_outside_range's x-y cut and the voxeliser's z cut."""
    lo, hi = np.asarray(pcr[:3], dtype=np.float64), np.asarray(pcr[3:], dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)[:, :3]
    return np.all((p >= lo) & (p < hi), axis=1)


def summary(values):
    v = np.asarray(values, dtype=np.float64)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(mean=float(v.mean()), median=float(med), q1=float(q1), q3=float(q3), iqr=float(q3 - q1),
                min=float(v.min()), max=float(v.max()), n=int(len(v)))


def main():
    cfg_file, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, '.')
    import _init_path  # noqa: F401  the live repo's pcdet, before any pcdet import
    import pandas as pd
    import pandaset as ps
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.datasets.pandaset import pandaset_rings as R
    from pcdet.utils import common_utils

    logger = common_utils.create_logger()
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
    rp = cfg.DATA_CONFIG.RING_PATTERN
    assert ds.ring_pattern_cfg is not None and ds.training, 'not a RING_PATTERN training dataset'
    pcr = list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    shift = cfg.DATA_CONFIG.get('SHIFT_COOR', None)
    step = max(1, len(ds) // n)
    idx = list(range(0, len(ds), step))[:n]
    logger.info('%d TRAIN infos, %d frames, every %d-th (indices %d..%d)', len(ds), len(idx), step, idx[0], idx[-1])

    rows = []
    for i in idx:
        info = ds.pandaset_infos[i]
        pose = ds._get_pose(info)
        raw = pd.read_pickle(info['lidar_path'])
        raw = raw[raw.d == 0]
        ego = ps.geometry.lidar_points_to_ego(raw[['x', 'y', 'z']].to_numpy(), pose)
        labels = R.load_cached_labels(rp.LABEL_CACHE, info['sequence'], info['frame_idx'], len(ego))
        np.random.seed(1000 + i)
        keep = ds._ring_keep_mask(info, ego)
        np.random.seed(1000 + i)
        phase_v = np.random.random_sample()
        lattice = R.kept_channels(rp.SPACING_DEG, phase_v, rp.get('TARGET_FOV_DEG', None))
        np.random.seed(1000 + i)
        loader = ds._get_lidar_points(info, pose)   # the loader's own cloud: operator + axis swap, no SHIFT_COOR yet
        mine = to_normative(ego[keep], None)
        assert len(loader) == len(mine) and np.allclose(loader[:, :3], mine, atol=1e-3), \
            'frame %d: _get_lidar_points does not return the operator mask reproduced here' % i
        rng_mask = in_range(to_normative(ego, shift), pcr)
        rows.append(dict(
            index=i, sequence=str(info['sequence']), frame_idx=int(info['frame_idx']),
            raw=count_lines(labels), op=count_lines(labels, keep),
            raw_in_range=count_lines(labels, rng_mask), op_in_range=count_lines(labels, keep & rng_mask),
            lattice=int(len(lattice)), lattice_in_op=int(np.isin(lattice, np.unique(labels[keep])).sum()),
            pts_raw=int(len(labels)), pts_op=int(keep.sum()),
            pts_raw_in_range=int(rng_mask.sum()), pts_op_in_range=int((keep & rng_mask).sum()),
            unlabelled=int((labels < 0).sum())))

    agg = {k: summary([r[k] for r in rows]) for k in ('raw', 'op', 'raw_in_range', 'op_in_range', 'lattice')}
    pts = {k: int(sum(r[k] for r in rows)) for k in ('pts_raw', 'pts_op', 'pts_raw_in_range', 'pts_op_in_range')}
    frac = dict(kept_all=pts['pts_op'] / pts['pts_raw'], kept_in_range=pts['pts_op_in_range'] / pts['pts_raw_in_range'],
                per_frame_kept_in_range=summary([r['pts_op_in_range'] / r['pts_raw_in_range'] for r in rows]))
    res = dict(config=cfg_file, label_cache=rp.LABEL_CACHE, spacing_deg=rp.SPACING_DEG, az_res_deg=rp.AZ_RES_DEG,
               target_fov_deg=list(rp.get('TARGET_FOV_DEG', [])), point_cloud_range=pcr, shift_coor=shift,
               n_infos=len(ds), step=step, frames=len(rows), lines=agg, points=pts, kept_fraction=frac,
               op_below_lattice_frames=int(sum(r['op'] < r['lattice'] for r in rows)), rows=rows)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    for k, s in agg.items():
        logger.info('%-13s mean %.2f median %.1f IQR %.1f-%.1f min %d max %d', k, s['mean'], s['median'], s['q1'],
                    s['q3'], s['min'], s['max'])
    logger.info('points kept: all %.4f, in range %.4f; frames with op < lattice: %d', frac['kept_all'],
                frac['kept_in_range'], res['op_below_lattice_frames'])
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
