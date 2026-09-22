"""PTSN - Progressive Target Size Normalization (DALI, IEEE T-RO 2024).

A detector trained on a source domain predicts target boxes whose mean size is inherited from
the source. PTSN corrects that without ever touching the network: scale the *input* point cloud
by `s` before inference, and divide the predicted boxes by `s` afterwards. The scene the network
sees is a similarity transform of the real one, so a car of length L reads as L * s, and the
box that comes back out at 1/s carries mean size ~ (source mean) / s. Sweeping `s` and keeping
the one whose unscaled mean size matches an externally estimated target mean size therefore
moves the pseudo-label size distribution onto the target's, using only inference.

Three functions, kept here rather than inlined at their two call sites so that the forward
transform and its inverse cannot drift apart:

  scale_points   applied in the worker, by DataProcessor.forward (inference mode only)
  unscale_boxes  applied in the main process, by self_training_utils.save_pseudo_label_batch
  select_scale   the search itself, driven by tools/analysis/ptsn_search.py

UDA legality, which is inherited rather than intrinsic and must be stated wherever a PTSN
number is reported: `select_scale` needs an estimate of the target's mean object size, and the
DALI paper takes that "from SN or ROS". An SN-derived estimate IS the target statistic, so a
PTSN row built on it is not target-free. An ROS-derived estimate only perturbs the source and
keeps the row comparable with our own target-free rows - see experiments_md/20260922_03 for the
derived interval and 20260922_04 section 2.3 for why the distinction decides what the row may claim.

Upstream (xiaohulugo/T-RO2024-DALI) ships only the *result* of a search - a config constant
CAD3D_CONFIG.SCALE - and not the search; `select_scale` is ours.
"""
import numpy as np


def scale_points(points, scale):
    """Isotropically scale a point cloud's geometry, leaving every other channel alone.

    Args:
        points: (N, 3 + C) array whose first three columns are x, y, z.
        scale: positive float.

    Returns:
        A new (N, 3 + C) array. The input is never modified in place - callers hold arrays that
        may be cached in a dataset info dict and reused across epochs.
    """
    scale = float(scale)
    assert scale > 0, 'PTSN scale must be positive, got %r' % scale
    if scale == 1.0:
        return points
    out = points.copy()
    out[:, 0:3] = out[:, 0:3] * scale
    return out


def unscale_boxes(boxes, scale):
    """Undo `scale_points` on predicted boxes: divide centre and extent, leave heading.

    Args:
        boxes: (M, 7 + C) array [x, y, z, dx, dy, dz, heading, ...].
        scale: the same positive float that was given to `scale_points`.

    Returns:
        A new (M, 7 + C) array in the original (unscaled) metric frame. Heading is invariant
        under an isotropic scaling, so columns 6 onward are copied through untouched.
    """
    scale = float(scale)
    assert scale > 0, 'PTSN scale must be positive, got %r' % scale
    if scale == 1.0 or len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, 0:6] = out[:, 0:6] / scale
    return out


def select_scale(candidates, mean_sizes, target_size):
    """Pick the swept scale whose unscaled mean predicted size is closest to the target estimate.

    Args:
        candidates: (K,) the input scales that were swept.
        mean_sizes: (K, 3) the mean predicted [dx, dy, dz] at each candidate, ALREADY divided by
            that candidate - i.e. in real target metres, which is what the pipeline stores.
        target_size: (3,) E_est[Size], the externally estimated target mean object size.

    Returns:
        (best_scale, rows), rows being one (scale, mean_size, gap) tuple per candidate in the
        order given, where `gap` is the mean absolute per-dimension error in metres. Ties go to
        the first candidate, so a deliberately ordered sweep is reproducible.

    A candidate that produced no boxes arrives here as NaN. np.argmin would return that index,
    so a scale at which the detector predicted nothing would silently win the search; such
    candidates are pushed to +inf and can only be chosen if every candidate failed.
    """
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1)
    mean_sizes = np.asarray(mean_sizes, dtype=np.float64).reshape(len(candidates), 3)
    target_size = np.asarray(target_size, dtype=np.float64).reshape(3)
    assert len(candidates) > 0, 'no candidate scales to choose from'
    assert np.all(candidates > 0), 'PTSN candidate scales must be positive'

    gaps = np.abs(mean_sizes - target_size[None, :]).mean(axis=1)
    finite = np.where(np.isfinite(gaps), gaps, np.inf)
    best = int(np.argmin(finite))
    rows = [(float(candidates[k]), mean_sizes[k].copy(), float(gaps[k]))
            for k in range(len(candidates))]
    return float(candidates[best]), rows
