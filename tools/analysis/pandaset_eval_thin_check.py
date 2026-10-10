"""Pre-launch check of the PandaSet oracle-degradation twins (experiments_md 20261011_02): is each manipulated
evaluation cloud what the twin says? CPU, PandaSet VAL frames, evaluation mode, as scored.

    python analysis/pandaset_eval_thin_check.py <frames> <twin.yaml> [<twin.yaml> ...]

Per twin, over strided val frames:
- loader stage (device-0 scan, before SHIFT_COOR): points kept / the full scan; laser lines with >= 1 point; median
  azimuth step between consecutive kept returns of a line (deg); the loader's output length must equal the mask's;
- model input: points per frame within 75 m after the processor, relative to the as-scored config on the same frames;
- random controls: their per-frame count must equal the structured twin's EXACTLY (checked when both are given).
"""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.pandaset import pandaset_rings as R
from pcdet.utils import common_utils

n = int(sys.argv[1])
twins = sys.argv[2:]
logger = common_utils.create_logger()


def build(f):
    cfg = EasyDict()
    cfg_from_yaml_file(f, cfg)
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    return cfg, ds


def in75(p):
    return int((np.hypot(p[:, 0], p[:, 1]) < 75).sum())


base_cfg, base = build('cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml')
frames = list(range(0, len(base), max(1, len(base) // n)))[:n]
base_in = {i: in75(base[i]['points']) for i in frames}
counts = {}
for f in twins:
    cfg, ds = build(f)
    thin = cfg.DATA_CONFIG_TAR.get('EVAL_RING_THIN', None)
    s = defaultdict(list)
    per_frame = []
    for i in frames:
        info = ds.pandaset_infos[i]
        pose = ds._get_pose(info)
        if thin is not None:
            import pandas as pd
            import pandaset as ps
            raw = pd.read_pickle(info['lidar_path'])
            raw = raw[raw.d == 0]
            ego = ps.geometry.lidar_points_to_ego(raw[['x', 'y', 'z']].to_numpy(), pose)
            labels = R.load_cached_labels(thin.LABEL_CACHE, info['sequence'], info['frame_idx'], len(ego))
            keep = ds._eval_ring_thin_mask(info, ego)
            out = ds._get_lidar_points(info, pose)
            assert len(out) == keep.sum(), (f, i, len(out), keep.sum())
            s['kept'].append(keep.mean())
            s['lines_full'].append(len(np.unique(labels)))
            s['lines_kept'].append(len(np.unique(labels[keep])))
            _, _, az = R.sensor_angles(ego)
            steps = []
            for c in np.unique(labels[keep]):
                a = az[keep & (labels == c)]
                if len(a) > 10:
                    d = np.abs((np.diff(a) + 180) % 360 - 180)
                    steps.append(np.median(d[d > 0]) if (d > 0).any() else 0)
            s['az_step'].append(np.median(steps) if steps else np.nan)
            per_frame.append(int(keep.sum()))
        d = ds[i]
        s['input_rel'].append(in75(d['points']) / max(base_in[i], 1))
    counts[f] = per_frame
    line = f'{f.split("/")[-1]:58s} input/as-scored {np.mean(s["input_rel"]):.3f}'
    if thin is not None:
        line += (f' | kept {np.mean(s["kept"]):.3f}  lines {np.median(s["lines_kept"]):.0f}/{np.median(s["lines_full"]):.0f}'
                 f'  az step {np.nanmedian(s["az_step"]):.3f} deg | loader = mask on {len(frames)} frames')
    print(line, flush=True)

# count matching of each random control against its structured twin
pairs = [('ringrows', 'ringrandom.'), ('ringcols.', 'ringrandomcols.'), ('ringcols4', 'ringrandomcols4'),
         ('ringazbin0p332', 'ringrandomazbin0p332')]
for a, b in pairs:
    fa = [f for f in counts if a in f]
    fb = [f for f in counts if b in f]
    if fa and fb:
        same = counts[fa[0]] == counts[fb[0]]
        print(f'count match {a.strip(".")} vs {b.strip(".")}: {"EXACT on every frame" if same else "MISMATCH"} '
              f'(mean {np.mean(counts[fa[0]]):.0f} points)')
