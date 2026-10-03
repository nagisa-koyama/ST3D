"""Shared helpers for the per-run frame figures (experiments_md/20261003_02).

Everything that has to agree between frame selection, the GPU dump and the drawing lives here:
how a frame is identified, what "raw" means per dataset, where its front image is, how a run's
config is recovered, and how the processed cloud is read back from the voxel tensor.

Definitions (design §3):
  raw              the stored single sweep - no range crop, no ego-point removal, no SHIFT_COOR
  source processed dataset[idx] in TRAINING mode, world-level augmentation off, seeded
  target processed dataset[idx] in EVAL mode
  both processed clouds are read back from voxels[:, :num_points], i.e. exactly the VFE input
"""
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS))
import _init_path  # noqa: F401,E402  (live pcdet, and the allocator guard)

from pcdet.datasets import __all__ as DATASETS  # noqa: E402

WANDB_DIRS = [Path('/home/koyama/data/wandb'), TOOLS / 'wandb']
WORLD_AUGS = ['random_world_flip', 'random_world_rotation', 'random_world_scaling',
              'random_world_translation']
CAR, PED = 'Car', 'Pedestrian'
NAME_MAP = {'car': CAR, 'vehicle': CAR, 'pedestrian': PED}


# ---------------------------------------------------------------- runs and configs
def find_run_dir(run_id):
    for root in WANDB_DIRS:
        hits = sorted(root.glob('run-*-%s' % run_id))
        if hits:
            return hits[-1]
    raise FileNotFoundError('no W&B run directory for %s under %s' % (run_id, WANDB_DIRS))


def _unwrap_wandb(d):
    return {k: v['value'] for k, v in d.items() if isinstance(v, dict) and 'value' in v}


def run_args(run_id):
    import json
    return json.load(open(find_run_dir(run_id) / 'files' / 'wandb-metadata.json')).get('args', [])


def load_run_cfg(run_id):
    """The run's RESOLVED config as W&B recorded it (uppercase keys only, i.e. the pcdet cfg).

    A run whose W&B process died before flushing config.yaml (26696: its evaluation crashed) falls
    back to the repo config named by its own --cfg_file, resolved with CURRENT code; the returned
    source string says which happened, and the figure prints it.
    """
    f = find_run_dir(run_id) / 'files' / 'config.yaml'
    if f.exists():
        raw = yaml.safe_load(open(f))
        cfg = EasyDict({k: v for k, v in _unwrap_wandb(raw).items() if k.isupper()})
        src = 'W&B-recorded config'
    else:
        from pcdet.config import cfg_from_yaml_file
        a = run_args(run_id)
        path = a[a.index('--cfg_file') + 1]
        cfg = EasyDict()
        cfg_from_yaml_file(str(TOOLS / path), cfg)
        src = 'repo config %s at HEAD (no W&B config.yaml)' % path
    cfg.LOCAL_RANK = 0
    return cfg, src


def source_configs(cfg):
    """[(name, data_cfg)] for every source, in train.py's order."""
    if cfg.get('DATA_CONFIGS', None):
        return list(cfg.DATA_CONFIGS.items())
    return [('DATA_CONFIG', cfg.DATA_CONFIG)]


def source_class_names_and_ontology(cfg):
    """Exactly train.py's rule (teacher class names; ontology dropped only for head_per_dataset)."""
    names = cfg.CLASS_NAMES
    ontology = cfg.get('ONTOLOGY', None)
    st = cfg.get('SELF_TRAIN', None)
    if st and st.get('MODEL_TEACHER', None):
        names = st.MODEL_TEACHER.get('CLASS_NAMES', None) or names
        if st.MODEL_TEACHER.get('ONTOLOGY', None) == 'head_per_dataset':
            ontology = None
    return list(names), ontology


def world_augs_off(data_cfg):
    """A copy of a dataset config whose world-level augmentation is disabled (object-level kept)."""
    dc = copy.deepcopy(data_cfg)
    aug = dc.get('DATA_AUGMENTOR', None)
    if aug is not None:
        aug.DISABLE_AUG_LIST = list(aug.get('DISABLE_AUG_LIST', [])) + WORLD_AUGS
    return dc


def quiet_logger():
    import logging
    lg = logging.getLogger('viz')
    if not lg.handlers:
        lg.addHandler(logging.NullHandler())
    return lg


def build_dataset(data_cfg, class_names, training, ontology, logger=None):
    logger = logger or quiet_logger()
    return DATASETS[data_cfg.DATASET](dataset_cfg=data_cfg, class_names=class_names, root_path=None,
                                      training=training, logger=logger, model_ontology=ontology)


# ---------------------------------------------------------------- frames
def dataset_kind(ds):
    return type(ds).__name__.replace('Dataset', '').lower()      # kitti / nuscenes / lyft / ...


def frame_id(ds, info):
    kind = dataset_kind(ds)
    if kind == 'kitti':
        return info['point_cloud']['lidar_idx']
    if kind in ('nuscenes', 'lyft'):
        return info['lidar_path']
    raise NotImplementedError('frame_id for %s (P2)' % kind)


def frame_group(ds, info):
    """Frames from different groups come from different drives (design §4)."""
    kind = dataset_kind(ds)
    if kind == 'kitti':
        return int(info['point_cloud']['lidar_idx']) // 200
    if kind == 'nuscenes':
        return Path(info['lidar_path']).name.split('__')[0]          # the log
    if kind == 'lyft':
        return '%s_%d' % (Path(info['lidar_path']).name.split('_')[0], int(info['timestamp']) // 120)
    raise NotImplementedError(kind)


def infos(ds):
    return ds.kitti_infos if hasattr(ds, 'kitti_infos') else ds.infos


def index_of(ds, fid):
    for i, info in enumerate(infos(ds)):
        if frame_id(ds, info) == fid:
            return i
    raise KeyError('frame %s not in %s (%d infos)' % (fid, type(ds).__name__, len(infos(ds))))


def raw_points(ds, info):
    """The stored single sweep, x y z intensity, sensor frame, nothing removed."""
    kind = dataset_kind(ds)
    if kind == 'kitti':
        return ds.get_lidar(info['point_cloud']['lidar_idx'])
    if kind in ('nuscenes', 'lyft'):
        p = np.fromfile(str(ds.root_path / info['lidar_path']), dtype=np.float32)
        p = p[: p.shape[0] - p.shape[0] % 5]
        return p.reshape(-1, 5)[:, :4]
    raise NotImplementedError(kind)


def raw_labels(ds, info):
    """(boxes (N,7), names in {'Car','Pedestrian'}) from the stored labels, sensor frame."""
    kind = dataset_kind(ds)
    if kind == 'kitti':
        boxes = info['annos']['gt_boxes_lidar']
        names = info['annos']['name'][:len(boxes)]
    else:
        boxes, names = info['gt_boxes'][:, :7], info['gt_names']
    names = np.array([NAME_MAP.get(str(n).lower(), '') for n in names])
    keep = names != ''
    return np.asarray(boxes, dtype=np.float32)[keep], names[keep]


def image_path(ds, info):
    kind = dataset_kind(ds)
    if kind == 'kitti':
        return str((ds.root_split_path / 'image_2' / ('%s.png' % info['point_cloud']['lidar_idx'])).resolve())
    if kind == 'nuscenes':
        return str((ds.root_path / info['cam_front_path']).resolve())
    return None          # Lyft: resolved through sample_data.json at selection time


def forward_yaw(ds, info):
    """Direction the vehicle faces, as a yaw in the LIDAR frame (degrees). KITTI's velodyne faces +x;
    nuScenes' LIDAR_TOP faces +y and Lyft's -x, read off each frame's own lidar-from-car rotation."""
    if dataset_kind(ds) == 'kitti':
        return 0.0
    f = np.asarray(info['ref_from_car'])[:3, :3] @ np.array([1.0, 0.0, 0.0])
    return float(np.degrees(np.arctan2(f[1], f[0])))


def points_per_box(points, boxes):
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    if len(boxes) == 0 or len(points) == 0:
        return np.zeros(len(boxes), dtype=np.int64)
    m = roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(np.ascontiguousarray(points[:, :3], dtype=np.float32)),
                                                  torch.from_numpy(np.ascontiguousarray(boxes[:, :7], dtype=np.float32)))
    return m.numpy().sum(axis=1)


# ---------------------------------------------------------------- processed clouds
def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random
    random.seed(seed)


def voxel_points(data_dict):
    """The points the VFE receives: voxels[:, :num_points], flattened. Asserts the count."""
    v, n = data_dict['voxels'], data_dict['voxel_num_points']
    mask = np.arange(v.shape[1])[None, :] < n[:, None]
    pts = v[mask]
    assert len(pts) == int(n.sum()), 'voxel read-back count mismatch'
    return pts


def processed(ds, idx, seed=0):
    seed_all(seed)
    dd = ds[idx]
    if isinstance(dd, (list, tuple)):          # LiDAR Distillation pair: (student, teacher)
        dd = dd[0]
    return dd, voxel_points(dd)


def shift_z(data_cfg):
    s = data_cfg.get('SHIFT_COOR', None)
    return float(s[2]) if s else 0.0
