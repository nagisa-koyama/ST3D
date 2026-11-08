"""Pitch crop fraction (experiments_md 20261108_06; T05 §5 item 42d record 6): the share of the points an oracle receives
that the detector's range crop removes once the evaluation cloud is pitched by OSENS-2's arm A (+-1, +-2 deg).

The perturbation is OSENS-2's own (experiments_md 20261025_03 §3): `EVAL_OSENS2: {PITCH_DEG, SENSOR_ORIGIN, FORWARD_AXIS}`,
applied by `DatasetTemplate.prepare_data` as its first step through `osens2_utils.apply_in_prepare_data`, i.e. on the
cloud in the loader's frame (SHIFT_COOR applied), before the data processor's range crop. This script reuses that path
unchanged: it builds the oracle's evaluation loader from its own config (training=False), takes the data_dict that
prepare_data receives for each frame, and calls `apply_in_prepare_data` with the EVAL_OSENS2 block read from the OSENS-2
twin config of each angle (`cfgs/da-ieee-access-analysis/osens2/<oracle>-osens2-pitch{m2,m1,p1,p2}.yaml`). Points are
moved one to one, so "leaves" is per point: inside the range before the pitch, outside after it.

The range crop is the data processor's, as evaluation runs it:
  - x / y: `mask_points_and_boxes_outside_range` = `common_utils.mask_points_by_range` (closed interval; it does NOT
    look at z), then the voxel grid (half-open);
  - z: only the voxel grid drops it (spconv's point-to-voxel step discards a point whose index
    floor((z - z_min) / v_z) is outside [0, grid_z)), so "in z" means z_min <= z < z_max.
Reported per oracle and angle, over the points inside the full range BEFORE the pitch (what the unperturbed detector
receives): the share that leaves by z (split below z_min / at or above z_max), by x / y, by either (union), the share of
outside points that enters, and the net change. Beside: the share of the loader's points already outside before any
pitch. Each as the mean of per-frame shares and pooled (sum over frames / sum).

Frames: --frames evenly spread val indices, round(linspace(0, N - 1, frames)), of each oracle's own evaluation loader
(Waymo: its SAMPLED_INTERVAL test 5, 7,998 frames). ANALYSIS: no labels are read (GT boxes pass through untouched).

    python analysis/pitch_crop_fraction.py --out /home/koyama/data/analysis_42d/pitch_crop_fraction.json \
        [--frames 300] [--oracles waymo,nuscenes,kitti] [--check-frames 2]

`--check-frames k` also builds each twin's own loader and checks, on the first k selected frames, that the twin's
processed points equal this script's pitched cloud after mask_points_by_range (same code path, end to end).
"""
import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
import _init_path  # noqa: F401,E402
from analysis.make_osens2_configs import ORACLES, twin_path  # noqa: E402
from pcdet.datasets import osens2_utils as O  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

PITCH_TAGS = (('pitchm2', -2.0), ('pitchm1', -1.0), ('pitchp1', 1.0), ('pitchp2', 2.0))
COUNT_KEYS = ('n', 'in0', 'out0_xy', 'out0_z', 'out0_any', 'in1', 'leave_z', 'leave_z_low', 'leave_z_high', 'leave_xy',
              'leave_any', 'enter')


def cfg_of(path):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    return cfg_from_yaml_file(str(path), EasyDict())


def dataset_of(cfg):
    from pcdet.datasets import build_dataloader
    log = logging.getLogger('pitch_crop_fraction'); log.setLevel(logging.WARNING)
    ds, _, _ = build_dataloader(dataset_cfg=copy.deepcopy(cfg.DATA_CONFIG_TAR), class_names=cfg.CLASS_NAMES, batch_size=1,
                                dist=False, workers=0, logger=log, training=False,
                                model_ontology=cfg.get('ONTOLOGY', None))
    return ds


def base_config_path(oracle):
    return Path('cfgs/da-ieee-access') / f'{ORACLES[oracle][0]}.yaml'


def twin_osens2_cfgs(oracle):
    """[(tag, deg, EVAL_OSENS2 block)] read from the OSENS-2 pitch twins; each must be a pitch arm of the stated angle."""
    out = []
    for tag, deg in PITCH_TAGS:
        block = cfg_of(twin_path(oracle, tag)).DATA_CONFIG_TAR.EVAL_OSENS2
        assert float(block['PITCH_DEG']) == deg, (oracle, tag, block)
        assert O._check(block) == 'PITCH_DEG', (oracle, tag, block)
        out.append((tag, deg, block))
    return out


def loader_input(ds, idx):
    """The data_dict DatasetTemplate.prepare_data receives for frame `idx` (where the EVAL_OSENS2 hook acts first)."""
    ds.prepare_data = lambda data_dict: data_dict   # instance attribute shadows the method for this one call
    try:
        return ds[idx]
    finally:
        del ds.prepare_data


def pitched(data_dict, osens2_cfg):
    """The hook itself, on a copy: exactly what prepare_data does first under the twin config."""
    d = dict(data_dict)
    d['points'] = np.array(data_dict['points'], copy=True)
    return O.apply_in_prepare_data(d, osens2_cfg)['points']


def in_range_masks(points, pcr, voxel_size, grid_size):
    """(in_xy, in_z, z_low, z_high) of the evaluation crop: x / y by mask_points_by_range and the voxel grid, z by the
    voxel grid only (module docstring)."""
    pcr = np.asarray(pcr, np.float64)
    idx = np.floor((np.asarray(points[:, :3], np.float64) - pcr[:3]) / np.asarray(voxel_size, np.float64))
    grid = np.asarray(grid_size, np.int64)
    in_grid = (idx >= 0) & (idx < grid)
    in_xy = common_utils.mask_points_by_range(points, pcr) & in_grid[:, 0] & in_grid[:, 1]
    return in_xy, in_grid[:, 2], idx[:, 2] < 0, idx[:, 2] >= grid[2]


def frame_counts(points, points_pitched, pcr, voxel_size, grid_size):
    """Per-point counts for one frame (COUNT_KEYS); `points_pitched[i]` is `points[i]` after the pitch."""
    assert len(points) == len(points_pitched), 'the pitch moves points one to one'
    xy0, z0, _, _ = in_range_masks(points, pcr, voxel_size, grid_size)
    xy1, z1, low1, high1 = in_range_masks(points_pitched, pcr, voxel_size, grid_size)
    in0, in1 = xy0 & z0, xy1 & z1
    return dict(n=len(points), in0=int(in0.sum()), out0_xy=int((~xy0).sum()), out0_z=int((~z0).sum()),
                out0_any=int((~in0).sum()), in1=int(in1.sum()), leave_z=int((in0 & ~z1).sum()),
                leave_z_low=int((in0 & low1).sum()), leave_z_high=int((in0 & high1).sum()),
                leave_xy=int((in0 & ~xy1).sum()), leave_any=int((in0 & ~in1).sum()), enter=int((~in0 & in1).sum()))


def summarise(rows):
    """Mean of per-frame shares and pooled shares. leave / enter / net over in0; out0_* over n."""
    tot = {k: int(sum(r[k] for r in rows)) for k in COUNT_KEYS}
    out = {'frames': len(rows), 'totals': tot}
    for k in ('leave_z', 'leave_z_low', 'leave_z_high', 'leave_xy', 'leave_any', 'enter'):
        per = np.array([r[k] / r['in0'] for r in rows if r['in0']])
        out[k] = {'mean': float(per.mean()), 'pooled': tot[k] / tot['in0'], 'max': float(per.max()),
                  'median': float(np.median(per))}
    per = np.array([(r['in1'] - r['in0']) / r['in0'] for r in rows if r['in0']])
    out['net'] = {'mean': float(per.mean()), 'pooled': (tot['in1'] - tot['in0']) / tot['in0']}
    for k in ('out0_xy', 'out0_z', 'out0_any'):
        per = np.array([r[k] / r['n'] for r in rows if r['n']])
        out[k] = {'mean': float(per.mean()), 'pooled': tot[k] / tot['n']}
    return out


def frame_indices(n_frames, k):
    return np.unique(np.round(np.linspace(0, n_frames - 1, k)).astype(np.int64))


def check_twin_end_to_end(oracle, twins, base_ds, idxs):
    """The twin's own loader (hook + data processor) gives this script's pitched cloud after mask_points_by_range."""
    res = []
    for tag, _, block in twins:
        tds = dataset_of(cfg_of(twin_path(oracle, tag)))
        for i in idxs:
            raw = loader_input(base_ds, int(i))
            mine = pitched(raw, block)
            mine = mine[common_utils.mask_points_by_range(mine, base_ds.point_cloud_range)][:, :3]
            theirs = tds[int(i)]['points'][:, :3]
            ok = theirs.shape == mine.shape and np.array_equal(theirs, mine)
            res.append({'tag': tag, 'idx': int(i), 'n_twin': int(len(theirs)), 'n_mine': int(len(mine)), 'equal': bool(ok)})
            assert ok, (oracle, tag, i, theirs.shape, mine.shape)
    return res


def run_oracle(oracle, n_frames, check_frames):
    t0 = time.time()
    base_cfg = cfg_of(base_config_path(oracle))
    tb = base_cfg.DATA_CONFIG_TAR
    assert tb.get('EVAL_OSENS2', None) is None, 'the base oracle config must be unperturbed'
    twins = twin_osens2_cfgs(oracle)
    strip = lambda block: {k: v for k, v in block.items() if k not in ('EVAL_OSENS2', '_BASE_CONFIG_')}  # noqa: E731
    for tag, _, _ in twins:   # a twin differs from its oracle in EVAL_OSENS2 only (_BASE_CONFIG_ is bookkeeping)
        tt = strip(cfg_of(twin_path(oracle, tag)).DATA_CONFIG_TAR)
        assert json.dumps(tt, sort_keys=True, default=str) == json.dumps(strip(tb), sort_keys=True, default=str), (oracle, tag)
    ds = dataset_of(base_cfg)
    pcr, vs, grid = ds.point_cloud_range, ds.voxel_size, ds.grid_size
    idxs = frame_indices(len(ds), n_frames)
    rows = {tag: [] for tag, _, _ in twins}
    frame_ids = []
    for i in idxs:
        raw = loader_input(ds, int(i))
        frame_ids.append(str(raw['frame_id']))
        for tag, _, block in twins:
            rows[tag].append(frame_counts(raw['points'], pitched(raw, block), pcr, vs, grid))
    checks = check_twin_end_to_end(oracle, twins, ds, idxs[:check_frames]) if check_frames else []
    return {
        'oracle': oracle, 'config': str(base_config_path(oracle)), 'dataset': tb.DATASET,
        'point_cloud_range': [float(v) for v in pcr], 'voxel_size': [float(v) for v in vs],
        'grid_size': [int(v) for v in grid], 'shift_coor': [float(v) for v in tb.get('SHIFT_COOR', [0, 0, 0])],
        'max_sweeps': tb.get('MAX_SWEEPS', None), 'fov_points_only': tb.get('FOV_POINTS_ONLY', None),
        'used_features': list(tb.POINT_FEATURE_ENCODING.used_feature_list),
        'n_val_frames': int(len(ds)), 'frame_indices': [int(v) for v in idxs], 'frame_ids': frame_ids,
        'arms': {tag: {'pitch_deg': deg, 'eval_osens2': {k: (list(v) if isinstance(v, (list, tuple)) else v)
                                                         for k, v in block.items()},
                       'summary': summarise(rows[tag]), 'per_frame': rows[tag]} for tag, deg, block in twins},
        'twin_end_to_end_checks': checks, 'seconds': round(time.time() - t0, 1),
    }


def table(results):
    lines = ['| oracle | pitch | leave z (low / high) | leave x-y | leave either | enter | net | already out (z / x-y / either) |',
             '|---|---:|---|---:|---:|---:|---:|---|']
    pct = lambda v: f'{100 * v:.2f}%'  # noqa: E731
    for r in results:
        for tag, arm in r['arms'].items():
            s = arm['summary']
            lines.append(
                f"| {r['oracle']} | {arm['pitch_deg']:+.0f} | {pct(s['leave_z']['mean'])} ({pct(s['leave_z_low']['mean'])} / "
                f"{pct(s['leave_z_high']['mean'])}); pooled {pct(s['leave_z']['pooled'])} | {pct(s['leave_xy']['mean'])} | "
                f"{pct(s['leave_any']['mean'])}; pooled {pct(s['leave_any']['pooled'])} | {pct(s['enter']['mean'])} | "
                f"{pct(s['net']['mean'])} | {pct(s['out0_z']['mean'])} / {pct(s['out0_xy']['mean'])} / "
                f"{pct(s['out0_any']['mean'])} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--oracles', default='waymo,nuscenes,kitti')
    parser.add_argument('--check-frames', type=int, default=2)
    args = parser.parse_args()
    results = []
    for oracle in args.oracles.split(','):
        r = run_oracle(oracle, args.frames, args.check_frames)
        print(f"{oracle}: {len(r['frame_indices'])} of {r['n_val_frames']} val frames, PCR {r['point_cloud_range']}, "
              f"grid {r['grid_size']}, end-to-end checks {sum(c['equal'] for c in r['twin_end_to_end_checks'])}/"
              f"{len(r['twin_end_to_end_checks'])}, {r['seconds']} s", flush=True)
        results.append(r)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    md = table(results)
    out.with_suffix('.md').write_text(md + '\n')
    print(md)
    print(f'written {out} and {out.with_suffix(".md")}')


if __name__ == '__main__':
    main()
