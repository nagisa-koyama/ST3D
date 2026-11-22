"""A PandaSet sequence's scans in world coordinates, with capture time, device and label per point, plus the two data
files pcocc takes from the semseg session (experiments_md 20261121_02 §3 and §5.1).

Points are read from the same files as pcseg.pandaset_seg.read_frame (lidar/<f>.pkl.gz, annotations/semseg/<f>.pkl.gz),
kept in WORLD coordinates with their per-point time `t`, which read_frame drops; tests/test_occ_pandaset.py checks that
the world -> ego move of these points equals read_frame on >= 40 frames.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

from pcseg.pandaset_seg import N_FRAMES

SENSOR_MODELS = '/home/koyama/data/pcseg/render_study/sensor_models.json'
SENSOR_MODELS_SHA256 = 'd372056be3f4a8f86b13dffa6fad7179fb12a5b552da6e66949134eccbc3a84e'
CENSUS = '/home/koyama/data/pcseg/pandaset_duplicate_scans/v1/duplicate_scans.json'
CENSUS_SHA256 = 'e6ea7b3b70dd48509da6b236c2243646553657bafb9c408b8a27dad29e8e2eeb'

PANDAR64, PANDARGT = 0, 1
RAW_REFLECTION = 4                    # never occupies, never casts a ray (a mirror return behind a real surface)
RAW_TRANSIENT = (1, 2, 3)             # Smoke, Exhaust, Spray or rain: rays from every scan, occupancy only from frame t


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _checked_json(path, expect):
    got = sha256(path)
    if got != expect:
        raise RuntimeError('%s changed: sha256 %s, pinned %s. A refit belongs in a new file; re-pin deliberately.'
                           % (path, got, expect))
    with open(path) as f:
        return json.load(f)


def sensor_origins(path=SENSOR_MODELS, expect=SENSOR_MODELS_SHA256):
    """{device: origin [3]} in the normative ego frame: the semseg session's fits (20261116_01 §8.9)."""
    m = _checked_json(path, expect)
    return {PANDAR64: np.asarray(m['pandar64']['origin'], dtype=np.float64),
            PANDARGT: np.asarray(m['pandargt']['origin'], dtype=np.float64)}


def stale_frames(path=CENSUS, expect=CENSUS_SHA256):
    """{seq: {device: set of STALE frames}} from the repeated-scan census. Of a repeated pair the stale copy is frame 0
    (of 0 / 1) or frame 79 (of 78 / 79); any other pair raises, as the census found none."""
    c = _checked_json(path, expect)
    out = {}
    for s in c['sequences']:
        per = {}
        for dev, d in s['devices'].items():
            st = set()
            for p in d['pairs']:
                fr = sorted(p['frames'])
                if fr == [0, 1]:
                    st.add(0)
                elif fr == [N_FRAMES - 2, N_FRAMES - 1]:
                    st.add(N_FRAMES - 1)
                else:
                    raise ValueError('%s device %s: repeated pair %s is not 0/1 or 78/79' % (s['seq'], dev, fr))
            per[int(dev)] = st
        out[s['seq']] = per
    return out


class SequenceScans:
    """All frames of one sequence, both devices. Arrays are per point, in stored order frame by frame:
    xyz world [N, 3] float64, t [N] float64 (s), dev [N] int8, frame [N] int16, raw [N] int16 (PandaSet class id),
    cls [N] int16 (class-map index or pcseg.IGNORE), stale [N] bool (a stale repeated copy, from the census)."""

    def __init__(self, root, seq, class_map, stale=None, frames=None):
        frames = range(N_FRAMES) if frames is None else frames
        xyz, t, dev, fr, raw = [], [], [], [], []
        self.n_per_frame = {}
        for f in frames:
            lidar = pd.read_pickle(os.path.join(root, seq, 'lidar', '%02d.pkl.gz' % f))
            sem = pd.read_pickle(os.path.join(root, seq, 'annotations', 'semseg', '%02d.pkl.gz' % f))
            if len(lidar) != len(sem):
                raise ValueError('%s/%02d: %d points but %d labels' % (seq, f, len(lidar), len(sem)))
            xyz.append(lidar[['x', 'y', 'z']].to_numpy(dtype=np.float64))
            t.append(lidar['t'].to_numpy(dtype=np.float64))
            dev.append(lidar['d'].to_numpy().astype(np.int8))
            raw.append(sem['class'].to_numpy().astype(np.int16))
            fr.append(np.full(len(lidar), f, dtype=np.int16))
            self.n_per_frame[f] = len(lidar)
        self.seq = seq
        self.xyz = np.concatenate(xyz)
        self.t = np.concatenate(t)
        self.dev = np.concatenate(dev)
        self.frame = np.concatenate(fr)
        self.raw = np.concatenate(raw)
        if not set(np.unique(self.dev).tolist()) <= {PANDAR64, PANDARGT}:
            raise ValueError('%s: unexpected device ids %s' % (seq, np.unique(self.dev).tolist()))
        self.cls = class_map(self.raw).astype(np.int16)
        self.stale = np.zeros(len(self.t), dtype=bool)
        self.stale_frames = {PANDAR64: set(), PANDARGT: set()}
        if stale is not None:
            for d, fs in stale.get(seq, {}).items():
                self.stale_frames[d] = set(fs)
                for f in fs:
                    self.stale |= (self.dev == d) & (self.frame == f)
        self.reflection = self.raw == RAW_REFLECTION
        self.transient = np.isin(self.raw, RAW_TRANSIENT)

    def __len__(self):
        return len(self.t)
