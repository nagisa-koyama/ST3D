"""What the detector's FIRST layer sees from a sampled source vs the target: occupied voxels per frame and
occupied neighbours per voxel (3x3x3) at the da-ieee-access voxel grid, per range ring. 20261005_01 shows
MeanVFE's first BatchNorm scales with the neighbour count; a sampler that matches POINT counts can leave
voxel occupancy unmatched (a voxel stays occupied while any of its points survive).

    python analysis/sampler_occupancy_check.py <cfg> <source key> <frames> <weights.npz> [<weights.npz> ...]
"""
import sys
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import _augmentation_off, calibration_target_config
from pcdet.datasets.processor.point_sampler import LearnedPointSampler
from pcdet.utils import common_utils
from first_layer_density_proxy import voxelize, subm  # noqa: E402  (same grid and neighbourhood as 20261005_01)

cfg_file, key, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); weights = sys.argv[4:]
cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
dc = (cfg.get('DATA_CONFIGS') or {'DATA_CONFIG': cfg.DATA_CONFIG})[key]
log = common_utils.create_logger()
src, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0, logger=log,
                             training=True, model_ontology=cfg.get('ONTOLOGY'))
tcfg, _ = calibration_target_config(cfg.DATA_CONFIG_TAR)
tgt, _, _ = build_dataloader(dataset_cfg=tcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0, logger=log,
                             training=False, model_ontology=cfg.get('ONTOLOGY'))
RINGS = [(0, 20), (20, 40), (40, 75)]

def stats(clouds):
    V, N = [], {r: [] for r in RINGS}
    for p in clouds:
        ijk, mean, _ = voxelize(p[:, :3].astype(np.float64))
        _, nn = subm(ijk, mean)
        V.append(len(ijk)); r = np.hypot(mean[:, 0], mean[:, 1])
        for lo, hi in RINGS:
            m = (r >= lo) & (r < hi); N[(lo, hi)].append(nn[m])
    return np.mean(V), {k: np.concatenate(v).mean() for k, v in N.items()}

def strided(ds, k):
    step = max(1, len(ds) // k); return [ds[i]['points'] for i in range(0, len(ds), step)[:k]]

with _augmentation_off(src):
    raw = strided(src, n)
rows = [('target ' + type(tgt).__name__, strided(tgt, n)), ('source raw', raw)]
rng = np.random.RandomState(0)
for w in weights:
    s = LearnedPointSampler.load(w)
    rows.append((w.split('/')[-1], [p[rng.rand(len(p)) < s.keep_probability(p)] for p in raw]))
print(f'{"cloud":34s} occupied voxels/frame | occupied neighbours per voxel at ' + ' / '.join(f'{a}-{b} m' for a, b in RINGS))
for name, clouds in rows:
    v, nn = stats(clouds)
    print(f'{name:34s} {v:10.0f}            | ' + ' / '.join(f'{nn[r]:.2f}' for r in RINGS))
