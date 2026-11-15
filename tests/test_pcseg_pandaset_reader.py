"""pcseg's PandaSet reader against ST3D's (experiments_md episodes/20261116_01 §4 and §7; bug_prevention_checklist
rule 8): the points must be IDENTICAL to PandasetDataset._get_lidar_points on >= 40 frames for both devices, the labels
one per point, and every raw label a PandaSet class (1..42). Skipped when the data is absent.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
from easydict import EasyDict

from pcseg import IGNORE
from pcseg.class_map import build_class_map
from pcseg.pandaset_seg import read_frame, load_poses, has_semseg, PandasetSeg, collate

ROOT = '/home/koyama/data/pandaset/dataset'
VAL = ['045', '059', '055', '051', '020', '097', '073', '043', '003', '101', '027', '056', '011', '078', '080', '109',
       '042', '021', '094', '057']
TRAIN_HEAD = ['014', '050', '079', '048', '093', '091', '063', '104', '100', '092']

pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason='PandaSet absent')


def _frames():
    seqs = [s for s in TRAIN_HEAD + VAL if has_semseg(ROOT, s)][:16]
    return [(s, f) for s in seqs for f in (0, 41, 79)]        # 48 frames


def _st3d_points(seq, frame, device):
    import pandaset as ps
    from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset
    pose = ps.DataSet(ROOT)[seq].lidar
    pose._load_poses()
    stub = SimpleNamespace(dataset_cfg=EasyDict({'LIDAR_DEVICE': device}), training=False, ring_pattern_cfg=None,
                           ring_fov_cut_cfg=None, eval_ring_thin_cfg=None)
    info = {'lidar_path': os.path.join(ROOT, seq, 'lidar', '%02d.pkl.gz' % frame), 'sequence': seq,
            'frame_idx': frame}
    return PandasetDataset._get_lidar_points(stub, info, pose.poses[frame]), pose.poses[frame]


def test_frames_enough():
    assert len(_frames()) >= 40


@pytest.mark.parametrize('device', [0, 1])
def test_reader_identical_to_st3d(device):
    cm = build_class_map('pandaset_nus16_draft')
    n_ign, n_all, n_zero = 0, 0, 0
    for seq, frame in _frames():
        ref, pose_devkit = _st3d_points(seq, frame, device)
        pose_json = load_poses(ROOT, seq)[frame]
        assert pose_json == pose_devkit
        pts, raw, keep = read_frame(ROOT, seq, frame, device, pose_json)
        assert np.array_equal(pts, ref), '%s/%02d device %d' % (seq, frame, device)
        assert len(raw) == len(pts) == len(keep)
        assert raw.min() >= 0 and raw.max() <= 42, '%s/%02d raw labels %s' % (seq, frame, np.unique(raw))
        n_zero += int((raw == 0).sum())                     # the undocumented 0: rare (class_map.py)
        seg = cm(raw)                                       # raises on any undeclared id
        n_ign += int((seg == IGNORE).sum())
        n_all += len(seg)
    assert n_zero < 1e-4 * n_all, 'raw label 0 on %d of %d points' % (n_zero, n_all)
    assert n_ign < 0.2 * n_all, 'ignored fraction %.3f' % (n_ign / n_all)


def test_device_split_partitions_the_frame():
    seq, frame = _frames()[0]
    pose = load_poses(ROOT, seq)[frame]
    a, ra, ka = read_frame(ROOT, seq, frame, 0, pose)
    b, rb, kb = read_frame(ROOT, seq, frame, 1, pose)
    c, rc, kc = read_frame(ROOT, seq, frame, -1, pose)
    assert np.array_equal(np.sort(np.concatenate([ka, kb])), kc)
    assert np.array_equal(rc[ka], ra) and np.array_equal(rc[kb], rb)


def test_dataset_eval_sample_round_trip():
    """Target dump for one eval sample: voxel labels are the representatives' labels, and every original point's
    label is recovered from the inverse map only where its voxel is label-pure."""
    seqs = [s for s in VAL if has_semseg(ROOT, s)][:2] + ['045']     # 045 has no semseg: must be skipped, not read
    cfg = {'ROOT': ROOT, 'SEQUENCES': seqs, 'LIDAR_DEVICE': 0, 'SHIFT_COOR': [0.0, 0.0, 1.8],
           'CLASS_MAP': 'pandaset_nus16_draft', 'GRID_SIZE': 0.05}
    ds = PandasetSeg(cfg, training=False)
    assert ds.skipped == ['045'] and len(ds.sequences) == 2
    assert len(ds) == 80 * len(ds.sequences)
    s1, s2 = ds[0], ds[0]
    assert np.array_equal(s1['segment'], s2['segment'])          # frame-keyed voxel draw
    b = collate([s1])
    vox_lab = b['segment'][b['inverse']]
    agree = float((vox_lab == b['origin_segment']).float().mean())
    assert agree > 0.95, 'voxel-label agreement %.3f' % agree
    coord, _, seg, _ = ds.load(0)
    assert np.allclose(b['origin_coord'].numpy(), coord)
    assert np.array_equal(b['origin_segment'].numpy(), seg)
