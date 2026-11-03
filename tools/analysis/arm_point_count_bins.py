"""Points and occupied voxels per 5-m range bin, as the detector sees them, in a scan-pattern (lattice) arm, its
density-matched twin and the target (experiments_md 20261003_05 §5 item 28; paper review round 3, B-2). CPU, analysis.

Per config: the SOURCE train split in training mode, as trained (augmentation on; a config with HIST_DIST_ON_THE_FLY
gets its density calibration installed as train.py does, over HIST_DIST_FRAMES frames, the target measured on its
TRAIN split), and the TARGET train split in eval mode (calibration_target_config, the view the calibration measures).
Counted after the full __getitem__, so after the range cut, the thinning, the recount and the voxelisation: points and
occupied voxels per frame in 5-m bins of horizontal range from the (shifted) origin, 0-75 m. No label is read beyond
what the training loader itself reads.

    python analysis/arm_point_count_bins.py measure <cfg> <frames> <out.json>
    python analysis/arm_point_count_bins.py report <lattice.json> <density.json> [<label>]
"""
import json
import sys

import numpy as np

EDGES = np.arange(0.0, 80.0, 5.0)     # 0, 5, ..., 75 m: 15 bins
DECISION_MAX = 70.0                   # the pre-declared reading covers 0-70 m (14 bins)
FRAME_BAND = (0.9, 1.1)               # lattice / density points per frame, 0-70 m
BIN_BAND = (0.8, 1.25)                # per bin, for bins holding >= 1% of the target's 0-70 m points
MIN_BIN_SHARE = 0.01


def bin_counts(xy):
    """Count per 5-m bin of horizontal range; points at or beyond 75 m are not counted."""
    r = np.hypot(xy[:, 0], xy[:, 1])
    return np.histogram(r[r < EDGES[-1]], bins=EDGES)[0]


def voxel_xy(coords, voxel_size, pc_range):
    """Voxel centres (x, y) from the voxel generator's (z, y, x) indices of ONE frame (no batch column)."""
    return np.stack([(coords[:, 2] + 0.5) * voxel_size[0] + pc_range[0],
                     (coords[:, 1] + 0.5) * voxel_size[1] + pc_range[1]], 1)


def strided(n_total, n):
    return list(range(0, n_total, max(1, n_total // n)))[:n]


def summarise(per_frame):
    a = np.asarray(per_frame, dtype=np.float64)
    return dict(frames=len(a), mean=a.mean(0).tolist(), per_frame_total_0_70=a[:, :int(DECISION_MAX // 5)].sum(1).tolist())


def reading(lat, den, tgt):
    """The pre-declared reading: do the lattice and density arms feed the detector a matched count?"""
    k = int(DECISION_MAX // 5)
    lt, dt, tt = (np.asarray(x['mean'][:k]) for x in (lat, den, tgt))
    frame_ratio = lt.sum() / dt.sum()
    counted = tt / tt.sum() >= MIN_BIN_SHARE
    bin_ratio = np.where(dt > 0, lt / np.maximum(dt, 1e-9), np.inf)
    outside = [int(i) for i in np.flatnonzero(counted & ((bin_ratio < BIN_BAND[0]) | (bin_ratio > BIN_BAND[1])))]
    matched = FRAME_BAND[0] <= frame_ratio <= FRAME_BAND[1] and not outside
    return dict(frame_ratio=float(frame_ratio), bins_outside=outside, matched=bool(matched))


def measure(cfg_file, n, out):
    sys.path.insert(0, '.')
    import _init_path  # noqa: F401
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.datasets.point_calibration import calibration_target_config, link_point_calibration
    from pcdet.utils import common_utils

    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger()
    dc = cfg.DATA_CONFIG
    src, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                 logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
    tcfg, split = calibration_target_config(cfg.DATA_CONFIG_TAR)
    tgt, _, _ = build_dataloader(dataset_cfg=tcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                 logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    calibrated = bool(dc.get('HIST_DIST_ON_THE_FLY', False))
    if calibrated:
        link_point_calibration(src, tgt, num_frames=dc.get('HIST_DIST_FRAMES', 1000),
                               num_bins=dc.get('HIST_DIST_BINS', 50), max_dist=dc.get('HIST_DIST_MAX_DIST', 75.0),
                               logger=logger, fov_degree=dc.get('HIST_DIST_FOV_DEGREE', None),
                               fov_heading=dc.get('HIST_DIST_FOV_HEADING', 0.0),
                               skip_point_budget=dc.get('HIST_DIST_BEFORE_POINT_BUDGET', False))
    np.random.seed(0)
    res = dict(cfg=cfg_file, calibrated=calibrated, target_split=split, edges=EDGES.tolist())
    for side, ds in (('source', src), ('target', tgt)):
        pts, vox = [], []
        for j, i in enumerate(strided(len(ds), n)):
            d = ds[i]
            pts.append(bin_counts(d['points'][:, :2]))
            vox.append(bin_counts(voxel_xy(d['voxel_coords'], ds.data_processor.voxel_size, ds.point_cloud_range)))
            if j % 50 == 0:
                logger.info('%s %s frame %d / %d' % (cfg_file, side, j, n))
        res[side] = dict(points=summarise(pts), voxels=summarise(vox))
    with open(out, 'w') as f:
        json.dump(res, f)
    logger.info('wrote %s' % out)


def report(lat_path, den_path, label=''):
    lat, den = (json.load(open(p)) for p in (lat_path, den_path))
    print(f'\n## {label or lat_path}\nlattice {lat["cfg"]}\ndensity {den["cfg"]} (calibrated {den["calibrated"]})\n')
    for unit in ('points', 'voxels'):
        rows = [('target', den['target'][unit]), ('lattice', lat['source'][unit]), ('density', den['source'][unit])]
        print(f'{unit} per frame | ' + ' | '.join(f'{int(a)}-{int(a) + 5}' for a in EDGES[:-1]) + ' | 0-70 |')
        print('|---' * (len(EDGES) + 1) + '|')
        for name, s in rows:
            m = np.asarray(s['mean'])
            print(f'| {name} | ' + ' | '.join(f'{v:.0f}' for v in m) + f' | {m[:14].sum():.0f} |')
        t, l, d = (np.asarray(s['mean']) for _, s in rows)
        for name, a, b in (('lattice / target', l, t), ('density / target', d, t), ('lattice / density', l, d)):
            print(f'| {name} | ' + ' | '.join(f'{x / max(y, 1e-9):.2f}' for x, y in zip(a, b))
                  + f' | {a[:14].sum() / max(b[:14].sum(), 1e-9):.2f} |')
        r = reading(lat['source'][unit], den['source'][unit], den['target'][unit])
        print(f'\n{unit}: lattice / density per frame (0-70 m) {r["frame_ratio"]:.2f}, bins outside {BIN_BAND}: '
              f'{[f"{5 * i}-{5 * i + 5}" for i in r["bins_outside"]]} -> {"MATCHED" if r["matched"] else "DIFFER"}\n')


if __name__ == '__main__':
    if sys.argv[1] == 'measure':
        measure(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == 'report':
        report(*sys.argv[2:5])
    else:
        sys.exit(__doc__)
