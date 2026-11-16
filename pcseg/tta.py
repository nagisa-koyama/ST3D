"""Pointcept's semantic-segmentation test protocol (the G0 pass rule reads it; experiments_md episodes/20261116_01 §5).

Ported from Pointcept at 77f87ab: configs/nuscenes/semseg-spunet-v1m1-0-base.py `test` block, datasets/defaults.py
DefaultDataset.prepare_test_data, datasets/transform.py GridSample ('train' and 'test' modes), engines/test.py
SemSegTester. Per frame:
  1. subsample to one random point per 0.025 m cell (GridSample 'train' mode, return_inverse);
  2. ten views of that cloud: scale 0.9, 0.95, 1, 1.05, 1.1, each without and with the x and y flip (RandomFlip p=1
     negates x and y);
  3. each view is cut by GridSample 0.05 'test' mode into parts: part i takes the (i mod count)-th point of every voxel,
     so every voxel is in every part and the parts together cover every point;
  4. one forward per part (fragment batch 1, fp32, no autocast), softmax, summed per point over parts and views;
  5. argmax, mapped back to the original points through the inverse of step 1, scored against the original labels.
The split's mIoU is the per-class sum of intersections over the sum of unions, as SemSegTester's meters.

Differences, by design: the step-1 draw is seeded per frame (key_seed(frame key, 'tta')), where Pointcept's tester uses
the global numpy state; within a voxel, points are ordered by a stable sort, where Pointcept's np.argsort is not stable.
Neither changes which points are scored or how many forwards a frame gets.
"""
import numpy as np
import torch

from .grid_sample import fnv_hash_vec, grid_sample
from .losses_metrics import ConfusionMatrix
from .pandaset_seg import key_seed

TTA_VIEWS = [(s, False) for s in (0.9, 0.95, 1.0, 1.05, 1.1)] + [(s, True) for s in (0.9, 0.95, 1.0, 1.05, 1.1)]
FINE_GRID = 0.025


def grid_parts(coord, grid_size):
    """GridSample 'test' mode. Returns [(idx_part [M], grid_coord [M, 3])], one entry per part; M = occupied voxels."""
    coord = np.asarray(coord)
    grid = np.floor(coord / np.asarray(grid_size)).astype(np.int64)
    grid -= grid.min(0)
    key = fnv_hash_vec(grid)
    idx_sort = np.argsort(key, kind='stable')
    _, count = np.unique(key[idx_sort], return_counts=True)
    start = np.cumsum(np.insert(count, 0, 0)[:-1])
    parts = []
    for i in range(int(count.max())):
        idx_part = idx_sort[start + i % count]
        parts.append((idx_part, grid[idx_part]))
    return parts


def view(coord, scale, flip):
    c = coord * np.float32(scale)
    if flip:
        c[:, 0] = -c[:, 0]
        c[:, 1] = -c[:, 1]
    return c


@torch.no_grad()
def predict_frame(model, coord, strength, num_classes, device, rng, views=TTA_VIEWS, fine_grid=FINE_GRID,
                  grid_size=0.05):
    """Per-point predictions [N] for one frame under the protocol, and the number of forwards it took."""
    idx_s, _, inv_s = grid_sample(coord, fine_grid, rng)
    c_s = np.ascontiguousarray(coord[idx_s], dtype=np.float32)
    f_s = np.ascontiguousarray(strength[idx_s], dtype=np.float32)
    prob = torch.zeros((len(idx_s), num_classes), dtype=torch.float32, device=device)
    n_fwd = 0
    for scale, flip in views:
        c = view(c_s, scale, flip)
        for idx_part, g in grid_parts(c, grid_size):
            batch = {'grid_coord': torch.as_tensor(g, dtype=torch.int32, device=device),
                     'feat': torch.as_tensor(np.concatenate([c[idx_part], f_s[idx_part]], 1), device=device),
                     'offset': torch.as_tensor([len(idx_part)], dtype=torch.long, device=device)}
            logits = model(batch)
            prob.index_add_(0, torch.as_tensor(idx_part, dtype=torch.long, device=device),
                            torch.softmax(logits.float(), -1))
            n_fwd += 1
    pred_s = prob.argmax(1).cpu().numpy()
    return pred_s[inv_s], n_fwd


@torch.no_grad()
def evaluate_tta(model, dataset, class_names, device, indices=None, views=TTA_VIEWS, cone=None, shift=(0, 0, 0),
                 log_every=0, logger=None):
    """Score `dataset` frames (all, or `indices`, e.g. one rank's shard) under the protocol.

    dataset: an eval PandasetSeg / NuScenesSeg (its load(i) gives the shifted coord, strength, labels and frame key).
    Returns ({'point'[, 'cone']} confusion matrices as int64 numpy, frames scored, forwards)."""
    from .engine import cone_mask
    model.eval()
    n = len(class_names)
    mats = {'point': ConfusionMatrix(n)}
    if cone is not None:
        mats['cone'] = ConfusionMatrix(n)
    shift = np.asarray(shift, dtype=np.float32)
    indices = range(len(dataset)) if indices is None else indices
    n_frames = n_fwd = 0
    for k, i in enumerate(indices):
        coord, strength, seg, key = dataset.load(i)
        pred, f = predict_frame(model, coord, strength, n, device, np.random.default_rng(key_seed(key, 'tta')),
                                views=views)
        mats['point'].update(pred, seg)
        if cone is not None:
            m = cone_mask(coord[:, :2] - shift[:2], *cone)
            mats['cone'].update(pred[m], seg[m])
        n_frames += 1
        n_fwd += f
        if logger is not None and log_every and (k + 1) % log_every == 0:
            logger.info('tta: %d / %d frames, %d forwards' % (k + 1, len(indices), n_fwd))
    return {k: v.mat for k, v in mats.items()}, n_frames, n_fwd
