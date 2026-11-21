"""Shared helpers of the S3b render study (experiments_md episodes/20261116_01 §8.9): the TRAIN split, raw frame access
with per-point times, and the two sensors' fitted models (written by pcseg_render_sensor_model.py).

Points here come straight from lidar/NN.pkl.gz in WORLD coordinates with their own time `t`; pcseg.render.PoseTrack
moves them into the normative ego frame (at the frame's pose time, which equals pcseg.pandaset_seg.read_frame, or at any
time). Labels are read only by the fidelity script, and only to measure the renderer.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from pcseg.pandaset_seg import has_semseg                            # noqa: E402

ROOT = '/home/koyama/data/pandaset/dataset'
RING_CACHE = '/home/koyama/data/pandaset_rings/v1'
OUT = '/home/koyama/data/pcseg/render_study'
CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cfgs/pcseg/spunet-pandaset64.yaml')


def split(name='train'):
    seqs = yaml.safe_load(open(CFG))['DATA']['SPLITS'][name]
    return [s for s in seqs if has_semseg(ROOT, s)]


def raw_frame(seq, frame, device, labels=False):
    """World xyz [N, 3] float64, time t [N] (s), stored-order indices into the frame, and (optionally) raw semseg ids
    [N], of one device's points in STORED order."""
    lidar = pd.read_pickle(os.path.join(ROOT, seq, 'lidar', '%02d.pkl.gz' % frame))
    keep = np.flatnonzero(lidar['d'].to_numpy() == device)
    xyz = lidar[['x', 'y', 'z']].to_numpy()[keep].astype(np.float64)
    t = lidar['t'].to_numpy()[keep].astype(np.float64)
    if not labels:
        return xyz, t, keep
    sem = pd.read_pickle(os.path.join(ROOT, seq, 'annotations', 'semseg', '%02d.pkl.gz' % frame))
    assert len(sem) == len(lidar), (seq, frame)
    return xyz, t, keep, sem['class'].to_numpy()[keep].astype(np.int64)


def ring_channels(seq, frame, n_points):
    path = os.path.join(RING_CACHE, seq, '%02d.npy' % frame)
    lab = np.load(path)
    assert len(lab) == n_points, (path, len(lab), n_points)
    return lab.astype(np.int64)


def rot_from_roll_pitch(roll_deg, pitch_deg):
    """Sensor -> ego rotation: roll about x (forward), then pitch about y (left)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('xy', [roll_deg, pitch_deg], degrees=True).as_matrix()


def load_models(out=OUT):
    with open(os.path.join(out, 'sensor_models.json')) as f:
        m = json.load(f)
    tpl = np.load(os.path.join(out, 'pandargt_template.npz'))
    m['pandargt']['template_az_el'] = tpl['az_el']
    m['pandargt']['template_count'] = tpl['count']
    return m


def rings_module():
    """pcdet/datasets/pandaset/pandaset_rings.py loaded by path (read only), without importing the pcdet package."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../pcdet/datasets/pandaset/pandaset_rings.py')
    spec = importlib.util.spec_from_file_location('pcseg_render_pandaset_rings', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
