"""TRAIN_FACTOR_SWAP.INTENSITY_MAP (ANALYSIS, training only; absent = unchanged): the intensity factor swap of
experiments_md 20261003_05 §5 item 72 (Fig. 1's intensity row, on x, y, z, intensity oracles).

The target's TRAINING intensities are mapped onto nuScenes' distribution, per horizontal-range ring:
x' = F_ref,r^-1(F_this,r(x)), with the tables of tools/analysis/intensity_testtime_tables.py (label-free, TRAIN clouds;
`from` = this dataset, `to` = the reference). It is the evaluation-time stage DataProcessor.map_intensity_to_reference
(experiments_md 20261003_03 §6), moved into the training loader. The math is the same, and
tests/test_intensity_factor_swap.py pins that it matches the stage exactly for the same draw. There are two differences.
  - Applied to the loader's cloud BEFORE augmentation, in the frame the tables were measured in (horizontal radius about
    the loader origin, augmentation off; SHIFT_COOR moves z only).
  - The dequantization draw of integer-valued intensities (from_step > 0) is seeded by the FRAME and the arm, as
    every factor-swap arm (kitti_factor_swap.frame_rng). So the swapped training set is fixed: the same at every epoch
    and seed. The stage seeds it by the point count.
Only the intensity column changes; x, y, z, the other columns, the boxes and the labels are untouched.

    INTENSITY_MAP: {TABLES: <npz path>, INTENSITY_INDEX: 3}     # INTENSITY_INDEX: the column in the loader's cloud

Tables are read once per process (each loader worker reads its own copy on first use).
"""
import numpy as np

ARM = 'INTENSITY_MAP'
KEYS = ('TABLES', 'INTENSITY_INDEX')
_TABLES = {}


def check_cfg(sub):
    unknown = [k for k in sub.keys() if k not in KEYS]
    assert not unknown, 'INTENSITY_MAP takes only %s, got %s' % (KEYS, unknown)
    assert sub.get('TABLES', None), 'INTENSITY_MAP needs TABLES (an npz of intensity_testtime_tables.py)'


def load_tables(path):
    """The stage's table dict (edges, from_q, to_q as float64; from_step as float), cached per path."""
    path = str(path)
    if path not in _TABLES:
        t = np.load(path)
        tables = {k: np.asarray(t[k], dtype=np.float64) for k in ('edges', 'from_q', 'to_q')}
        tables['from_step'] = float(t['from_step'])
        assert tables['from_q'].shape == tables['to_q'].shape, 'INTENSITY_MAP: from_q and to_q differ in shape'
        assert tables['from_q'].shape[0] == len(tables['edges']) - 1, 'INTENSITY_MAP: one quantile row per ring'
        _TABLES[path] = tables
    return _TABLES[path]


def map_values(v, radius, tables, rng):
    """map_intensity_to_reference's math on one intensity column `v` with per-point horizontal `radius`; the
    dequantization draw (from_step > 0) is rng.uniform(-step / 2, step / 2, n), as the stage's."""
    edges, from_q, to_q = tables['edges'], tables['from_q'], tables['to_q']
    levels = np.linspace(0.0, 1.0, from_q.shape[1])
    ring = np.clip(np.searchsorted(edges, radius, side='right') - 1, 0, len(edges) - 2)
    v = np.asarray(v).astype(np.float64)
    if tables['from_step'] > 0:
        v = np.clip(v + rng.uniform(-tables['from_step'] / 2, tables['from_step'] / 2, len(v)), 0.0, 1.0)
    out = np.empty_like(v)
    for r in np.unique(ring):
        m = ring == r
        xs = np.maximum.accumulate(from_q[r]) + np.arange(len(levels)) * 1e-12
        out[m] = np.interp(np.interp(v[m], xs, levels), levels, to_q[r])
    return out


def apply_points(points, cfg, frame_id):
    """A copy of `points` with the intensity column mapped (cfg: the TRAIN_FACTOR_SWAP block, arm INTENSITY_MAP)."""
    from .kitti.kitti_factor_swap import frame_rng
    sub = cfg[ARM]
    check_cfg(sub)
    if not len(points):
        return points
    idx = int(sub.get('INTENSITY_INDEX', 3))
    out = np.array(points, copy=True)
    radius = np.linalg.norm(points[:, 0:2], axis=1)
    out[:, idx] = map_values(points[:, idx], radius, load_tables(sub['TABLES']), frame_rng(frame_id, ARM)).astype(
        points.dtype)
    return out
