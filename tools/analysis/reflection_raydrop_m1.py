"""M1: per-sensor ray-drop inside surfaces and its intensity selectivity, label-free (experiments_md 20261003_05 §5 item
35b / 35c; design fixed 2026-11-07 10:08 JST before any number). TRAIN clouds only; no label is read or used.

Sensors and ring sources (35b):
    nuscenes  LIDAR_TOP (HDL-32E), stored ring index (column 5 of the .bin), sensor frame. Frames: sample.json timestamps
              matched to samples/LIDAR_TOP file names, scenes of the devkit TRAIN split (no annotation table is read).
    kitti     HDL-64E, `kitti_rings.rings_from_order` (stored order), velodyne frame. ImageSets/train.txt.
    waymo     TOP native range image, first return (`waymo_reextract_bench` reader, as `waymo_raydrop_rate.py`): row =
              beam, column = firing; z from the per-pixel-pose reconstruction, about the sensor height.
    pandar64  PandaSet device 0, channels from the `pandaset_rings` cache v1, points in ego axes about the fitted sensor
              origin `pandaset_rings.ORIGIN` (PandaSet stores WORLD points). Sequences of the config's TRAIN split.
    lyft40    Lyft 40-beam LIDAR_TOP, rings from the `lyft_rings` cache v1_40beam (stored-order counts, crc-checked),
              kept frames only, sensor frame. Scenes from sample.json.
    Lyft-64 is not measured (no ring recovery on that platform).

Frames: 200 per sensor, numpy.random.default_rng(20261107) per sensor. With >= 200 TRAIN scenes: 200 random scenes, one
uniformly drawn frame each (KITTI: 200 random TRAIN frames). With fewer scenes (PandaSet 61, Lyft-40 123) one frame per
scene cannot give 200; `spread` (proposed amendment, not in 35b as fixed): every scene gets floor(200 / n) frames and
200 mod n random scenes one more, drawn without replacement inside a scene. `--draw_short one_each` takes one frame per
scene instead.

Holes (35b): per ring, returns sorted by azimuth about the sensor; Δ0 = the median consecutive azimuth step within rings,
pooled over the sensor's frames; a step < 0.5 Δ0 is a duplicate (the later return is dropped); m = round(step / Δ0) - 1
missing slots between consecutive kept returns. A HOLE is a missing slot in a run of m = 1-4 whose bracketing returns agree
in range within max(0.5 m, 0.02 r); an ELIGIBLE RETURN has both ring neighbours one slot away, agreeing with each other.
h = holes / (holes + eligible returns); the single-hole rate uses runs of m = 1 only. A slot is binned by its bracketing
returns' mean 3D range (0-10, ..., 60-75 m) and classed by their mean height above the frame's ground (ground = mode of
sensor-frame z over returns at 5-30 m horizontal range, 0.05 m bins): ground |dz| <= 0.3 m, above-ground 0.3-3.0 m.
Returns within 3 m (ego) take part in no slot: a slot is dropped when any of its bracketing returns is that close.
nuScenes stores every firing, a non-return as a zero point (r = 0): those fall out as duplicates at azimuth 0 / ego
returns, leaving the slot missing, as a non-return should.

Proposed amendments found while building (2026-11-07; not in 35b as fixed, each behind an option whose default is 35b):
- `--draw_short spread` (see Frames above) for PandaSet and Lyft-40;
- `--dup_frac 0.25`: KITTI's HDL-64E fires on a grid with a regular half step (about every 6-7th azimuth step is
  0.46-0.56 d0, in every ring and sector, on continuous surfaces). Under the 0.5 d0 rule the half steps below 0.5 d0
  (about half of them) drop a real return as a duplicate, and wherever the half step plus the next step reach 1.5 d0
  that leaves a fake one-slot hole; at 0.25 d0 a half step is a link (m = 0) and a 1.5 d0 gap one missing slot. (m
  rounds half up in both settings.)
- `--ground plane`: the nuScenes (+1.35 / +2.29 deg) and Lyft (-1.1 deg) sensor frames are pitched against the road, so
  a single z mode misclasses ground beyond ~8-15 m along the driving direction; the plane is
  `ground_tilt.lowest_per_cell` + `fit_plane` (4-40 m).

Selectivity: relative intensity = the bracketing returns' mean intensity / the sensor's median return intensity in that
bin and class (pooled); h per quartile of it (edges from that bin's slots, holes weighted by run length) and
S = h(lowest quartile) / h(highest). Intensity floor: p1 / p5 / p50 of return intensity per bin and class, divided by the
class's p50 at 10-20 m.

Readout (35c): e_T(b) = (h_nuS(b) - h_T(b)) / (1 - h_T(b)) per bin and class. ROOM on T if some bin of 10-75 m (either
class) has >= 2,000 slots on both sensors, e_T >= 0.05 and nuScenes' S >= 2. NOT MAKEABLE if e_T < 0 in every bin and
class where it is defined. Otherwise "no room at M1's resolution".

Run as a Slurm CPU job from ST3D/tools (memory/skills/launch-job §3b):
    python3 analysis/reflection_raydrop_m1.py --out_dir /home/koyama/data/reflection_m1 --workers 8
    python3 analysis/reflection_raydrop_m1.py --readout /home/koyama/data/reflection_m1/m1.json     # re-print only
    python3 analysis/reflection_raydrop_m1.py --smoke 2 --out_dir <scratch>                       # 2 frames per sensor
"""
import argparse
import json
import os
import struct
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data')
sys.path.insert(0, REPO)          # the live pcdet, not the copy baked into the image
sys.path.append(HERE)             # waymo_reextract_bench, ground_tilt

SEED = 20261107
N_FRAMES = 200
RANGE_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0])
CLASSES = ('ground', 'above')
MAX_RUN = 4
TOL_ABS, TOL_REL = 0.5, 0.02
EGO_M = 3.0
GROUND_WIN = (5.0, 30.0)
GROUND_BIN = 0.05
GROUND_DZ, ABOVE_HI = 0.3, 3.0
DUP_FRAC = 0.5
MIN_SLOTS, E_MIN, S_MIN = 2000, 0.05, 2.0
ROOM_BINS = (1, 2, 3, 4, 5, 6)          # 10-75 m
SOURCE = 'nuscenes'
TARGETS = ('kitti', 'waymo', 'pandar64', 'lyft40')
SENSORS = (SOURCE,) + TARGETS


# ----------------------------------------------------------------------------------------------------------- core

def frame_ground(z, rh):
    """Mode of z (0.05 m bins) over returns at 5-30 m horizontal range; nan if there are none."""
    m = (rh >= GROUND_WIN[0]) & (rh <= GROUND_WIN[1])
    if not np.any(m):
        return np.nan
    b = np.floor(z[m] / GROUND_BIN).astype(np.int64)
    u, c = np.unique(b, return_counts=True)
    return (u[np.argmax(c)] + 0.5) * GROUND_BIN


def ring_steps(ring, az):
    """Consecutive azimuth steps within rings (deg), returns sorted by (ring, azimuth)."""
    o = np.lexsort((az, ring))
    r, a = ring[o], az[o]
    same = r[1:] == r[:-1]
    return np.diff(a)[same]


def range_bin(r):
    """Bin index of RANGE_EDGES, -1 outside [0, 75) m."""
    b = np.searchsorted(RANGE_EDGES, r, side='right') - 1
    b[(r < RANGE_EDGES[0]) | (r >= RANGE_EDGES[-1])] = -1
    return b


def height_class(dz):
    """0 = ground (|dz| <= 0.3 m), 1 = above-ground (0.3 < dz <= 3.0 m), -1 = not counted."""
    c = np.full(len(dz), -1, dtype=np.int64)
    c[np.abs(dz) <= GROUND_DZ] = 0
    c[(dz > GROUND_DZ) & (dz <= ABOVE_HI)] = 1
    return c


def tolerance(r_mean):
    return np.maximum(TOL_ABS, TOL_REL * r_mean)


def frame_slots(ring, az, r, dz, inten, d0, dup_frac=DUP_FRAC):
    """Holes and eligible returns of one frame.

    ring, az (deg), r (3D range about the sensor, m), dz (height above the frame's ground, m), inten: one entry per
    return. Returns (slots, n_dup, n_returns): slots is a dict of equal-length arrays, one row per HOLE RUN or ELIGIBLE
    RETURN - `hole` (bool), `count` (missing slots of the run, 1 for an eligible return), `r`, `dz`, `inten` (the
    bracketing returns' means)."""
    ring = np.asarray(ring, dtype=np.int64); az = np.asarray(az, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64); dz = np.asarray(dz, dtype=np.float64)
    inten = np.asarray(inten, dtype=np.float64)
    o = np.lexsort((az, ring))
    ring, az, r, dz, inten = ring[o], az[o], r[o], dz[o], inten[o]
    n0 = len(r)
    dup = np.zeros(n0, dtype=bool)
    if n0 > 1:
        dup[1:] = (ring[1:] == ring[:-1]) & (np.diff(az) < dup_frac * d0)
    keep = ~dup
    ring, az, r, dz, inten = ring[keep], az[keep], r[keep], dz[keep], inten[keep]
    empty = dict(hole=np.zeros(0, bool), count=np.zeros(0, np.int64), r=np.zeros(0), dz=np.zeros(0), inten=np.zeros(0))
    if len(r) < 2:
        return empty, int(dup.sum()), n0
    same = ring[1:] == ring[:-1]
    # round half up (1.5 d0 is one missing slot); a kept step below 0.5 d0 (dup_frac < 0.5) is a link, m = 0
    m = np.maximum(np.floor(np.diff(az) / d0 + 0.5).astype(np.int64) - 1, 0)
    m[~same] = -1                                             # no link across rings
    near = r < EGO_M
    # gaps g between returns g and g + 1
    rl, rr = r[:-1], r[1:]
    rm = 0.5 * (rl + rr)
    gap_ok = same & ~near[:-1] & ~near[1:] & (np.abs(rl - rr) <= tolerance(rm))
    hg = gap_ok & (m >= 1) & (m <= MAX_RUN)
    # eligible returns i (1 .. n-2): gaps i-1 and i both one slot (m == 0), neighbours i-1 and i+1 agree
    rp, rn = r[:-2], r[2:]
    rmm = 0.5 * (rp + rn)
    el = ((m[:-1] == 0) & (m[1:] == 0) & ~near[:-2] & ~near[1:-1] & ~near[2:]
          & (np.abs(rp - rn) <= tolerance(rmm)))
    hole = np.concatenate([np.ones(int(hg.sum()), bool), np.zeros(int(el.sum()), bool)])
    out = dict(
        hole=hole,
        count=np.concatenate([m[hg], np.ones(int(el.sum()), np.int64)]),
        r=np.concatenate([rm[hg], rmm[el]]),
        dz=np.concatenate([0.5 * (dz[:-1] + dz[1:])[hg], 0.5 * (dz[:-2] + dz[2:])[el]]),
        inten=np.concatenate([0.5 * (inten[:-1] + inten[1:])[hg], 0.5 * (inten[:-2] + inten[2:])[el]]),
    )
    return out, int(dup.sum()), n0


def quartile_rates(rel, hole, count):
    """Hole rate per quartile of relative intensity. Edges from the slots (a hole run weighs its slot count).
    Returns (edges[5], slots[4], holes[4], rates[4], S)."""
    w = np.where(hole, count, 1).astype(np.int64)
    if w.sum() == 0:
        return [np.nan] * 5, [0] * 4, [0] * 4, [np.nan] * 4, np.nan
    rep = np.repeat(rel, w)
    edges = np.quantile(rep, [0.0, 0.25, 0.5, 0.75, 1.0])
    q = np.clip(np.searchsorted(edges[1:4], rel, side='right'), 0, 3)
    slots = np.bincount(q, weights=w, minlength=4).astype(np.int64)
    holes = np.bincount(q[hole], weights=w[hole], minlength=4).astype(np.int64)
    rates = np.where(slots > 0, holes / np.maximum(slots, 1), np.nan)
    if not (np.isfinite(rates[0]) and np.isfinite(rates[3])):
        s = np.nan
    else:
        s = rates[0] / rates[3] if rates[3] > 0 else (np.inf if rates[0] > 0 else np.nan)
    return edges.tolist(), slots.tolist(), holes.tolist(), rates.tolist(), float(s)


def excess_drop(h_src, h_tgt):
    """e_T = (h_src - h_tgt) / (1 - h_tgt): the share of the target's returns a source-like sensor would lose."""
    return (h_src - h_tgt) / (1.0 - h_tgt)


def verdict(cells_src, cells_tgt):
    """35c on one target. cells_*: {(bin, cls): dict(slots, h, S)}. Returns (verdict, qualifying cells, e table)."""
    e = {}
    for key, ct in cells_tgt.items():
        cs = cells_src.get(key)
        if cs is None or cs['slots'] == 0 or ct['slots'] == 0 or not (ct['h'] < 1.0):
            continue
        e[key] = excess_drop(cs['h'], ct['h'])
    qual = []
    for (b, c), v in sorted(e.items()):
        cs, ct = cells_src[(b, c)], cells_tgt[(b, c)]
        if (b in ROOM_BINS and cs['slots'] >= MIN_SLOTS and ct['slots'] >= MIN_SLOTS and v >= E_MIN
                and cs['S'] >= S_MIN):
            qual.append((b, c))
    if qual:
        return 'room', qual, e
    if e and all(v < 0 for v in e.values()):
        return 'not makeable in the degrading direction', [], e
    return "no room at M1's resolution", [], e


# ----------------------------------------------------------------------------------------------------------- draws

def spread_draw(rng, scene_frames, n_total=N_FRAMES, mode='spread'):
    """scene_frames: list (sorted scene order) of frame-id lists. >= n_total scenes: n_total random scenes, one frame
    each (a uniform draw per chosen scene). Fewer: `spread` or `one_each` (see the module docstring)."""
    n = len(scene_frames)
    if n >= n_total:
        pick = rng.choice(n, n_total, replace=False)
        u = rng.random(n_total)
        return [scene_frames[s][int(np.floor(x * len(scene_frames[s])))] for s, x in zip(pick, u)], 'one per scene'
    if mode == 'one_each':
        u = rng.random(n)
        return [f[int(np.floor(x * len(f)))] for f, x in zip(scene_frames, u)], 'one_each (short)'
    base, extra = divmod(n_total, n)
    plus = set(rng.choice(n, extra, replace=False).tolist())
    out = []
    for s, f in enumerate(scene_frames):
        k = base + (s in plus)
        out += [f[i] for i in rng.choice(len(f), min(k, len(f)), replace=False)]
    return out, 'spread (short: %d scenes)' % n


def draw_nuscenes(rng, mode):
    from nuscenes.utils import splits
    root = os.path.join(DATA, 'nuscenes', 'v1.0-trainval')
    meta = os.path.join(root, 'v1.0-trainval')
    smp = json.load(open(os.path.join(meta, 'sample.json')))
    scenes = {s['token']: s for s in json.load(open(os.path.join(meta, 'scene.json')))}
    logs = {l['token']: l for l in json.load(open(os.path.join(meta, 'log.json')))}
    files = {int(f.split('__')[-1].split('.')[0]): f for f in os.listdir(os.path.join(root, 'samples', 'LIDAR_TOP'))}
    train = set(splits.train)
    by_scene = {}
    for s in smp:
        sc = scenes[s['scene_token']]
        if sc['name'] in train:
            by_scene.setdefault(sc['name'], []).append(s)
    names = sorted(by_scene)
    lists = []
    for nm in names:
        ss = sorted(by_scene[nm], key=lambda s: s['timestamp'])
        loc = logs[scenes[ss[0]['scene_token']]['log_token']]['location']
        lists.append([dict(id=files[s['timestamp']], path=os.path.join(root, 'samples', 'LIDAR_TOP', files[s['timestamp']]),
                           scene=nm, location='boston' if loc.startswith('boston') else 'singapore') for s in ss])
    return spread_draw(rng, lists, mode=mode), len(names)


def draw_kitti(rng, mode):
    ids = [s.strip() for s in open(os.path.join(DATA, 'kitti', 'ImageSets', 'train.txt')) if s.strip()]
    pick = rng.choice(len(ids), N_FRAMES, replace=False)
    return ([dict(id=ids[i], path=os.path.join(DATA, 'kitti', 'training', 'velodyne', ids[i] + '.bin')) for i in pick],
            'random frames'), len(ids)


def draw_waymo(rng, mode):
    segs = [s.strip().replace('.tfrecord', '') for s in open(os.path.join(DATA, 'waymo', 'ImageSets', 'train.txt'))
            if s.strip()]
    pick = rng.choice(len(segs), N_FRAMES, replace=False)
    u = rng.random(N_FRAMES)
    return ([dict(id=segs[s], path=os.path.join(DATA, 'waymo', 'raw_data', segs[s] + '.tfrecord'), u=float(x))
             for s, x in zip(pick, u)], 'one per segment'), len(segs)


def draw_pandar64(rng, mode):
    import yaml
    cfg = os.path.join(REPO, 'tools', 'cfgs', 'da-ieee-access', 'da_pandaset_dataset.yaml')
    seqs = yaml.safe_load(open(cfg))['SEQUENCES']['train']
    cache = '/home/koyama/data/pandaset_rings/v1'
    lists = []
    for s in seqs:
        n = len([f for f in os.listdir(os.path.join(cache, s)) if f.endswith('.npy')])
        lists.append([dict(id='%s/%02d' % (s, fi), seq=s, frame=fi) for fi in range(n)])
    return spread_draw(rng, lists, mode=mode), len(seqs)


LYFT_CACHE = '/home/koyama/data/lyft_rings/v1_40beam'


def draw_lyft40(rng, mode):
    frames = json.load(open(os.path.join(LYFT_CACHE, 'frames_v1_40beam.json')))
    kept = set(s.strip() for s in open(os.path.join(LYFT_CACHE, 'kept_frames_v1_40beam.txt')) if s.strip())
    smp = {s['token']: s for s in json.load(open(os.path.join(DATA, 'lyft', 'trainval', 'data', 'sample.json')))}
    by_scene = {}
    for f in frames:
        if f['stem'] in kept:
            s = smp[f['token']]
            by_scene.setdefault(s['scene_token'], []).append((s['timestamp'], f['stem']))
    lists = [[dict(id=st, path=os.path.join(DATA, 'lyft', 'trainval', 'lidar', st + '.bin')) for _, st in sorted(v)]
             for _, v in sorted(by_scene.items())]
    return spread_draw(rng, lists, mode=mode), len(lists)


DRAWS = dict(nuscenes=draw_nuscenes, kitti=draw_kitti, waymo=draw_waymo, pandar64=draw_pandar64, lyft40=draw_lyft40)


# ----------------------------------------------------------------------------------------------------------- readers
# each returns dict(ring, az, x, y, z, r, inten): x, y, z about the sensor (z up), r the 3D range, az in degrees

def _polar(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    return dict(az=np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])), x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                r=np.linalg.norm(xyz, axis=1))


def read_nuscenes(fr):
    p = np.fromfile(fr['path'], dtype=np.float32).reshape(-1, 5)
    return dict(_polar(p[:, :3]), ring=p[:, 4].astype(np.int64), inten=p[:, 3])


def read_kitti(fr):
    from pcdet.datasets.kitti.kitti_rings import rings_from_order
    p = np.fromfile(fr['path'], dtype=np.float32).reshape(-1, 4)
    ring, _ = rings_from_order(p[:, :3])
    return dict(_polar(p[:, :3]), ring=ring, inten=p[:, 3])


def _waymo_record(path, u):
    """(n_frames, k, the record of frame k = floor(u * n)), skipping record bodies (no decode)."""
    offs = []
    with open(path, 'rb') as f:
        while True:
            pos = f.tell()
            h = f.read(8)
            if len(h) < 8:
                break
            (ln,) = struct.unpack('<Q', h)
            offs.append((pos, ln))
            f.seek(4 + ln + 4, 1)
        k = int(np.floor(u * len(offs)))
        f.seek(offs[k][0] + 12)
        return len(offs), k, f.read(offs[k][1])


def read_waymo(fr):
    from waymo_reextract_bench import parse_frame, reconstruct
    n, k, buf = _waymo_record(fr['path'], fr['u'])
    fr['frame'], fr['n_frames'] = k, n
    pose, cal, lasers = parse_frame(buf)
    ri, rpose = lasers[1]
    W = ri.shape[1]
    P, _, _, _, valid = reconstruct(ri, cal[1], rpose, pose)
    rows, cols = np.nonzero(valid)                     # the order of reconstruct's boolean indexing
    d = P.astype(np.float64) - cal[1][3][:3, 3]         # vehicle frame, about the TOP's position
    return dict(ring=rows.astype(np.int64), az=-cols * (360.0 / W), x=d[:, 0], y=d[:, 1], z=d[:, 2],
                r=ri[..., 0][valid].astype(np.float64), inten=ri[..., 1][valid].astype(np.float64))


def read_pandar64(fr):
    from pcdet.datasets.pandaset import pandaset_rings as R
    import pandas as pd
    import pandaset as ps
    root = '/home/koyama/data/pandaset/dataset'
    poses = json.load(open(os.path.join(root, fr['seq'], 'lidar', 'poses.json')))
    df = pd.read_pickle(os.path.join(root, fr['seq'], 'lidar', '%02d.pkl.gz' % fr['frame']))
    d0 = df[df.d == 0]
    ego = ps.geometry.lidar_points_to_ego(d0[['x', 'y', 'z']].to_numpy(), poses[fr['frame']])
    lab = R.load_cached_labels('/home/koyama/data/pandaset_rings/v1', fr['seq'], fr['frame'], len(ego))
    return dict(_polar(np.asarray(ego, dtype=np.float64) - R.ORIGIN), ring=lab,
                inten=d0['i'].to_numpy().astype(np.float64))


_LYFT = {}


def read_lyft40(fr):
    from pcdet.datasets.lyft import lyft_ring_pattern as LRP
    if 'frames' not in _LYFT:
        _LYFT['frames'] = LRP.load_ring_cache(os.path.join(LYFT_CACHE, 'rings_v1_40beam.pkl'))
    entry = _LYFT['frames'].get(fr['id'])
    p = np.fromfile(fr['path'], dtype=np.float32)
    p = p[: len(p) - len(p) % 5].reshape(-1, 5)[:, :4]
    LRP.check_entry(fr['id'], entry, p, 40)
    return dict(_polar(p[:, :3]), ring=LRP.ring_ids(entry['counts']), inten=p[:, 3])


def ground_height(d, how):
    """Height above the frame's ground. `mode` (35b as fixed): z - the mode of z over returns at 5-30 m horizontal range.
    `plane` (proposed amendment): z - the frame's ground plane (ground_tilt.lowest_per_cell + fit_plane, 4-40 m),
    because the nuScenes and Lyft sensor frames are pitched 1.1-2.3 deg against the road."""
    if how == 'mode':
        g = frame_ground(d['z'], np.hypot(d['x'], d['y']))
        return d['z'] - g, dict(ground=float(g))
    from ground_tilt import lowest_per_cell, fit_plane
    xyz = np.c_[d['x'], d['y'], d['z']]
    fit = fit_plane(lowest_per_cell(xyz), rng=np.random.default_rng(SEED))
    if fit is None:
        g = frame_ground(d['z'], np.hypot(d['x'], d['y']))
        return d['z'] - g, dict(ground=float(g), plane=None)
    a, b, c = fit[0]
    return d['z'] - (a * d['x'] + b * d['y'] + c), dict(plane=[float(a), float(b), float(c)], inliers=fit[1])


READERS = dict(nuscenes=read_nuscenes, kitti=read_kitti, waymo=read_waymo, pandar64=read_pandar64, lyft40=read_lyft40)


# ----------------------------------------------------------------------------------------------------------- passes

def pass1(args):
    """Read one frame; return its compact arrays (dz above the frame's ground) and its within-ring steps."""
    sensor, fr, how = args
    t = time.time()
    d = READERS[sensor](fr)
    dz, gi = ground_height(d, how)
    arr = dict(ring=d['ring'].astype(np.int32), az=d['az'].astype(np.float64), r=d['r'].astype(np.float32),
               dz=dz.astype(np.float32), inten=np.asarray(d['inten'], dtype=np.float32))
    steps = ring_steps(arr['ring'], arr['az']).astype(np.float32)
    info = {k: v for k, v in fr.items() if k != 'path'}
    info.update(gi)
    info.update(returns=int(len(arr['r'])), rings=int(len(np.unique(arr['ring']))), sec=round(time.time() - t, 2))
    return info, arr, steps


def pass2(args):
    arr, d0, dup_frac = args
    slots, n_dup, n0 = frame_slots(arr['ring'], arr['az'], arr['r'], arr['dz'], arr['inten'], d0, dup_frac)
    rb, cl = range_bin(np.asarray(arr['r'], np.float64)), height_class(np.asarray(arr['dz'], np.float64))
    ok = (rb >= 0) & (cl >= 0) & (arr['r'] >= EGO_M)
    ret = dict(rb=rb[ok].astype(np.int8), cl=cl[ok].astype(np.int8), inten=arr['inten'][ok])
    srb, scl = range_bin(slots['r']), height_class(slots['dz'])
    k = (srb >= 0) & (scl >= 0)
    sl = dict(rb=srb[k].astype(np.int8), cl=scl[k].astype(np.int8), hole=slots['hole'][k],
              count=slots['count'][k].astype(np.int8), inten=slots['inten'][k].astype(np.float32))
    return sl, ret, n_dup, n0


def cell_stats(sl, ret):
    """Per (bin, class): slots, holes, h, single-hole rate, quartiles and S, intensity floor."""
    cells = {}
    p50_ref = {}
    for c in range(len(CLASSES)):
        m = (ret['rb'] == 1) & (ret['cl'] == c)
        p50_ref[c] = float(np.median(ret['inten'][m])) if np.any(m) else np.nan
    for b in range(len(RANGE_EDGES) - 1):
        for c in range(len(CLASSES)):
            ms = (sl['rb'] == b) & (sl['cl'] == c)
            mr = (ret['rb'] == b) & (ret['cl'] == c)
            hole, cnt = sl['hole'][ms], sl['count'][ms].astype(np.int64)
            holes = int(cnt[hole].sum()); elig = int((~hole).sum())
            slots = holes + elig
            h1_holes = int(np.sum(hole & (cnt == 1)))
            ri = ret['inten'][mr].astype(np.float64)
            med = float(np.median(ri)) if len(ri) else np.nan
            cell = dict(slots=slots, holes=holes, eligible_returns=elig,
                        h=holes / slots if slots else np.nan,
                        h_single=h1_holes / (h1_holes + elig) if (h1_holes + elig) else np.nan,
                        returns=int(mr.sum()), median_intensity=med)
            if slots and med > 0:
                rel = sl['inten'][ms].astype(np.float64) / med
                e, qs, qh, qr, s = quartile_rates(rel, hole, cnt)
                cell.update(q_edges=e, q_slots=qs, q_holes=qh, q_h=qr, S=s)
            else:
                cell.update(q_edges=None, q_slots=None, q_holes=None, q_h=None, S=np.nan)
            if len(ri) and p50_ref[c] > 0:
                cell['floor'] = [float(np.percentile(ri, p)) / p50_ref[c] for p in (1, 5, 50)]
            else:
                cell['floor'] = None
            cells['%d,%d' % (b, c)] = cell
    return cells


def concat(parts, keys):
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def run_sensor(sensor, a, pool):
    rng = np.random.default_rng(SEED)
    (frames, how), n_scenes = DRAWS[sensor](rng, a.draw_short)
    if a.smoke:
        frames = frames[:a.smoke]
    t0 = time.time()
    res = pool.map(pass1, [(sensor, f, a.ground) for f in frames], chunksize=1)
    infos = [r[0] for r in res]
    d0 = float(np.median(np.concatenate([r[2] for r in res])))
    out2 = pool.map(pass2, [(r[1], d0, a.dup_frac) for r in res], chunksize=1)
    groups = {'all': list(range(len(res)))}
    if sensor == 'nuscenes':
        for loc in ('boston', 'singapore'):
            groups[loc] = [i for i, f in enumerate(infos) if f.get('location') == loc]
    tables = {}
    for g, idx in groups.items():
        if not idx:
            continue
        sl = concat([out2[i][0] for i in idx], ('rb', 'cl', 'hole', 'count', 'inten'))
        ret = concat([out2[i][1] for i in idx], ('rb', 'cl', 'inten'))
        tables[g] = cell_stats(sl, ret)
    n_dup = sum(o[2] for o in out2); n0 = sum(o[3] for o in out2)
    print('[%s] %d frames (%s; %d scenes), d0 %.4f deg, duplicates %d of %d returns, %.0f s' % (
        sensor, len(frames), how, n_scenes, d0, n_dup, n0, time.time() - t0), flush=True)
    return dict(frames=infos, draw=how, n_scenes=n_scenes, d0_deg=d0, duplicates=n_dup, returns=n0, tables=tables)


# ----------------------------------------------------------------------------------------------------------- readout

def _cells(table):
    out = {}
    for k, v in table.items():
        b, c = map(int, k.split(','))
        out[(b, c)] = dict(slots=v['slots'], h=v['h'], S=v['S'] if v['S'] is not None else np.nan)
    return out


def readout(res):
    lines = []
    bins = ['%g-%g' % (RANGE_EDGES[i], RANGE_EDGES[i + 1]) for i in range(len(RANGE_EDGES) - 1)]
    for s in SENSORS:
        if s not in res['sensors']:
            continue
        R = res['sensors'][s]
        lines.append('\n## %s: %d frames (%s), d0 %.4f deg, duplicates %d / %d' % (
            s, len(R['frames']), R['draw'], R['d0_deg'], R['duplicates'], R['returns']))
        for g, T in R['tables'].items():
            for c, cname in enumerate(CLASSES):
                lines.append('%s / %s | bin | slots | h | h single | h by rel. intensity quartile (low..high) | S | '
                             'floor p1 / p5 / p50' % (g, cname))
                for b in range(len(bins)):
                    v = T['%d,%d' % (b, c)]
                    q = '-' if not v['q_h'] else ' / '.join('%.3f' % x for x in v['q_h'])
                    fl = '-' if not v['floor'] else ' / '.join('%.2f' % x for x in v['floor'])
                    lines.append('  %s m | %d | %.4f | %.4f | %s | %.2f | %s' % (
                        bins[b], v['slots'], v['h'], v['h_single'], q, v['S'], fl))
    if SOURCE in res['sensors']:
        src = _cells(res['sensors'][SOURCE]['tables']['all'])
        res['readout'] = {}
        lines.append('\n## 35c readout (e_T = (h_nuS - h_T) / (1 - h_T); room: 10-75 m, >= %d slots both, e >= %.2f, '
                     'nuScenes S >= %g)' % (MIN_SLOTS, E_MIN, S_MIN))
        for t in TARGETS:
            if t not in res['sensors']:
                continue
            tgt = _cells(res['sensors'][t]['tables']['all'])
            v, qual, e = verdict(src, tgt)
            res['readout'][t] = dict(verdict=v, qualifying=[[bins[b], CLASSES[c]] for b, c in qual],
                                     e={'%s %s' % (bins[b], CLASSES[c]): float(x) for (b, c), x in sorted(e.items())})
            lines.append('%s: %s%s' % (t, v, (' at ' + ', '.join('%s m %s' % (bins[b], CLASSES[c]) for b, c in qual))
                                       if qual else ''))
            for c, cname in enumerate(CLASSES):
                lines.append('  %s e: %s' % (cname, ' | '.join(
                    '%s %s' % (bins[b], ('%+.3f' % e[(b, c)]) if (b, c) in e else '-') for b in range(len(bins)))))
    return '\n'.join(lines)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='/home/koyama/data/reflection_m1')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--sensors', nargs='+', default=list(SENSORS))
    ap.add_argument('--draw_short', default='spread', choices=['spread', 'one_each'])
    ap.add_argument('--ground', default='mode', choices=['mode', 'plane'])
    ap.add_argument('--dup_frac', type=float, default=DUP_FRAC, help='duplicate below this many d0 (35b: 0.5)')
    ap.add_argument('--smoke', type=int, default=0, help='first N drawn frames per sensor only')
    ap.add_argument('--readout', default=None, help='re-print the readout of an existing m1.json')
    a = ap.parse_args()
    if a.readout:
        res = json.load(open(a.readout))
        print(readout(res))
        return
    os.makedirs(a.out_dir, exist_ok=True)
    res = dict(design='experiments_md 20261003_05 §5 item 35b / 35c', seed=SEED, draw_short=a.draw_short,
               ground=a.ground, dup_frac=a.dup_frac, smoke=a.smoke, sensors={})
    with Pool(a.workers) as pool:
        for s in a.sensors:
            res['sensors'][s] = run_sensor(s, a, pool)
    txt = readout(res)
    print(txt)
    json.dump(res, open(os.path.join(a.out_dir, 'm1.json'), 'w'), indent=1, default=_json_default)
    open(os.path.join(a.out_dir, 'm1.txt'), 'w').write(txt + '\n')
    print('saved %s/m1.json, m1.txt' % a.out_dir)


if __name__ == '__main__':
    main()
