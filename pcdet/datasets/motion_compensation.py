"""Per-object motion compensation for sweep accumulation.

Accumulating sweeps with each sweep's stored `transform_matrix` compensates EGO motion only.
Anything that moves in the world is left smeared along its own trajectory, and the points it
contributed in earlier sweeps land outside its box in the anchor frame. Measured on nuScenes, that
costs a moving object essentially the entire benefit of accumulating: over 30 frames it gains
x1.01 where a static object gains x2.74, so deep accumulation teaches a detector that
densely-sampled cars are parked ones. See
experiments_md/20260922_05_accumulation_depth.md section 4.

The fix is a rigid box-to-box transform per tracked object: take the points inside an object's box
at the time a sweep was captured and move them onto that object's box in the anchor frame. It is
strictly stronger than extrapolating a velocity, since it captures yaw change and non-constant
motion for free, and it consumes only SOURCE labels, so it stays UDA-legal.

Boxes exist only at annotated keyframes while points arrive between them, so a sweep's boxes are
interpolated between the two keyframes that bracket it in time.
"""
from pathlib import Path

import numpy as np


def _timestamp_seconds(ts):
    """A keyframe timestamp in SECONDS.

    Both info builders already divide by 1e6 (`ref_time = 1e-6 * sd_rec['timestamp']`), so the
    stored value is seconds, and `time_lag` is seconds too. Dividing again is not a small error:
    it compresses a 20-minute scene into 1.2 ms, every keyframe of a scene lands on the same
    instant, and the bracketing search then returns an arbitrary keyframe - which is exactly how a
    parked car appeared to move 136 m in 0.3 s. Epoch seconds are ~1.5e9, epoch microseconds
    ~1.5e15, so the two are never ambiguous; accept either.
    """
    t = float(ts)
    return t * 1e-6 if abs(t) > 1e12 else t


def _yaw_to_R(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def interpolate_boxes(boxes_a, boxes_b, w):
    """Boxes at fraction `w` from `boxes_a` toward `boxes_b`, matched by track id.

    Only tracks present in BOTH are returned: a track that appears or disappears between the two
    keyframes has no defined position in between, and guessing one would sweep up whatever points
    happen to lie there.
    """
    out = {}
    for tid, a in boxes_a.items():
        b = boxes_b.get(tid)
        if b is None:
            continue
        box = a.copy()
        box[:3] = a[:3] + w * (b[:3] - a[:3])
        # shortest angular path, so a wrap across +-pi does not spin the box the long way round
        d = (b[6] - a[6] + np.pi) % (2 * np.pi) - np.pi
        box[6] = a[6] + w * d
        out[tid] = box
    return out


def move_points_between_boxes(points, boxes_then, boxes_now):
    """Move points inside each `boxes_then` box onto the same track's `boxes_now` box.

    `points` are already in the anchor frame (ego-compensated); so are both box sets. Returns a
    new array - the input is not modified.

    A point is claimed by at most one track. Containment is tested against the ORIGINAL positions
    for every track before anything moves, so two overlapping boxes cannot move the same point
    twice, and the order tracks are visited in does not change the result.
    """
    if not boxes_then or not boxes_now or len(points) == 0:
        return points
    out = points.copy()
    src = points[:, :3]
    claimed = np.zeros(len(points), dtype=bool)
    for tid, then in boxes_then.items():
        now = boxes_now.get(tid)
        if now is None:
            continue
        d = src - then[:3]
        R = _yaw_to_R(-then[6])
        local_x = d[:, 0] * R[0, 0] + d[:, 1] * R[0, 1]
        local_y = d[:, 0] * R[1, 0] + d[:, 1] * R[1, 1]
        inside = ((np.abs(local_x) <= then[3] / 2) & (np.abs(local_y) <= then[4] / 2)
                  & (np.abs(d[:, 2]) <= then[5] / 2) & ~claimed)
        if not inside.any():
            continue
        dyaw = now[6] - then[6]
        Rf = _yaw_to_R(dyaw)
        rel = d[inside]
        moved = np.empty_like(rel)
        moved[:, 0] = rel[:, 0] * Rf[0, 0] + rel[:, 1] * Rf[0, 1]
        moved[:, 1] = rel[:, 0] * Rf[1, 0] + rel[:, 1] * Rf[1, 1]
        moved[:, 2] = rel[:, 2]
        out[inside, :3] = moved + now[:3]
        claimed |= inside
    return out


def boxes_to_frame(boxes, names, track_ids, S_from, S_to, classes=None):
    """Boxes given in the `S_from` ego frame, expressed in the `S_to` ego frame, keyed by track.

    `S_*` map GLOBAL -> that frame's ego coordinates, which is what nuScenes-style infos store as
    `ref_from_car @ car_from_global`. Going between two frames is therefore S_to @ inv(S_from).
    """
    if boxes is None or len(boxes) == 0:
        return {}
    T = S_to @ np.linalg.inv(S_from)
    R, t = T[:3, :3], T[:3, 3]
    dyaw = np.arctan2(T[1, 0], T[0, 0])
    out = {}
    for i, tid in enumerate(track_ids):
        if tid is None:
            continue
        if classes is not None and names is not None and names[i] not in classes:
            continue
        b = np.asarray(boxes[i], dtype=np.float64)
        centre = R @ b[:3] + t
        out[tid] = np.concatenate([centre, b[3:6], [b[6] + dyaw]])
    return out


def find_devkit_meta_dir(root_path, version=None):
    """Directory holding the devkit json tables, given a dataset root.

    The two devkit-format datasets nest their tables differently, relative to the `root_path`
    each loader has already built (`DATA_PATH / VERSION` in both):

      nuScenes  <root_path>/v1.0-trainval/sample.json   (the version dir, repeated)
      Lyft      <root_path>/data/sample.json            (a dir literally called `data`)

    Each loader knows its own answer, so hard-coding it works - but it puts a silent layout
    assumption in two places, and a wrong DATA_PATH then surfaces as a bare FileNotFoundError on
    sample_annotation.json with nothing said about what was expected. Probe the candidates
    instead and name every one that was tried.
    """
    root = Path(root_path)
    cands = []
    if version:
        cands += [root / version / version, root / version / 'data', root / version]
    cands += [root / 'data', root]
    for c in cands:
        if (c / 'sample_annotation.json').exists() and (c / 'sample.json').exists():
            return c
    raise FileNotFoundError(
        'GT_BOXES_MOTION_COMPENSATION needs the nuScenes-devkit json tables (sample.json, '
        'sample_annotation.json) - the track ids live there and nowhere in the infos. None of '
        'these holds them:\n  %s' % '\n  '.join(str(c) for c in cands))


class DevkitSweepCompensator:
    """Per-object compensation for nuScenes-devkit-format datasets (nuScenes and Lyft).

    Both store, per keyframe, `ref_from_car @ car_from_global` as the global -> ego transform, a
    `timestamp`, and `gt_boxes_token` holding one PER-FRAME annotation token per box. The track id
    is not in the infos: it is `instance_token` in the devkit's `sample_annotation.json`, so that
    file is read once and the annotation -> instance mapping cached.

    Boxes exist only at keyframes. A sweep is located in time by `anchor_timestamp - time_lag`,
    the two keyframes bracketing it are found within the same scene, and each track's box is
    interpolated between them.
    """

    def __init__(self, infos, meta_dir, classes=None, logger=None):
        import json
        self.classes = set(classes) if classes else None
        meta = Path(meta_dir)
        ann2inst = {r['token']: r['instance_token']
                    for r in json.load(open(meta / 'sample_annotation.json'))}
        tok2scene = {r['token']: r['scene_token'] for r in json.load(open(meta / 'sample.json'))}

        # One timeline per scene, ordered in time. Lyft's infos are randomly ordered on disk, so
        # index adjacency cannot be used to find a keyframe's neighbours - group by scene instead.
        self.frames = []
        self.tok2frame = {}
        by_scene = {}
        for info in infos:
            scene = tok2scene.get(info['token'])
            if scene is None:
                continue
            tokens = np.asarray(info.get('gt_boxes_token', [])).ravel()
            tids = [ann2inst.get(str(t)) for t in tokens]
            idx = len(self.frames)
            self.frames.append({
                't': _timestamp_seconds(info['timestamp']),
                'S': np.asarray(info['ref_from_car']) @ np.asarray(info['car_from_global']),
                'boxes': np.asarray(info.get('gt_boxes', np.zeros((0, 7))))[:, :7],
                'names': np.asarray(info.get('gt_names', [])),
                'tids': tids,
            })
            by_scene.setdefault(scene, []).append(idx)
            self.frames[idx]['scene'] = scene
            # keyed here, not from enumerate(infos): an info whose scene is unknown is skipped
            # above, after which an infos index and a frames index no longer agree.
            self.tok2frame[info['token']] = idx
        self.scene_order = {s: sorted(ix, key=lambda i: self.frames[i]['t'])
                            for s, ix in by_scene.items()}
        self.tok2scene = tok2scene
        if logger is not None:
            logger.info('motion compensation: %d keyframes over %d scenes, %d annotation tokens'
                        % (len(self.frames), len(self.scene_order), len(ann2inst)))

    def _boxes_in(self, frame_idx, S_anchor):
        f = self.frames[frame_idx]
        return boxes_to_frame(f['boxes'], f['names'], f['tids'], f['S'], S_anchor, self.classes)

    def boxes_at(self, token, t, S_anchor):
        """Every track's box at absolute time `t`, expressed in the anchor's ego frame."""
        i = self.tok2frame.get(token)
        if i is None:
            return {}
        order = self.scene_order.get(self.frames[i]['scene'], [])
        if not order:
            return {}
        times = [self.frames[j]['t'] for j in order]
        # the two keyframes bracketing t; clamped at the ends, where a sweep before the first
        # keyframe or after the last simply uses that keyframe's boxes
        k = int(np.searchsorted(times, t))
        if k <= 0:
            return self._boxes_in(order[0], S_anchor)
        if k >= len(order):
            return self._boxes_in(order[-1], S_anchor)
        lo, hi = order[k - 1], order[k]
        t0, t1 = self.frames[lo]['t'], self.frames[hi]['t']
        if t1 - t0 < 1e-6:
            return self._boxes_in(lo, S_anchor)
        w = float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        return interpolate_boxes(self._boxes_in(lo, S_anchor),
                                 self._boxes_in(hi, S_anchor), w)

    def compensate_sweep(self, info, sweep, points):
        """Points of one sweep, already ego-transformed into the anchor frame."""
        S = np.asarray(info['ref_from_car']) @ np.asarray(info['car_from_global'])
        anchor_t = _timestamp_seconds(info['timestamp'])
        now = self.boxes_at(info['token'], anchor_t, S)
        if not now:
            return points
        then = self.boxes_at(info['token'], anchor_t - float(sweep.get('time_lag', 0.0)), S)
        return move_points_between_boxes(points, then, now)


def assert_not_supported(dataset_cfg, dataset_name):
    """Refuse `GT_BOXES_MOTION_COMPENSATION` on a dataset that cannot honour it.

    A config key that a loader silently ignores is worse than one that fails: the run looks like
    it did what was asked and the result is quietly a different experiment. That failure mode has
    already cost this project several times over - a correction configured but never installed, an
    N=15 that was really N=10 - so an unsupported dataset raises here rather than accepting the key
    and doing nothing with it.

    Only nuScenes and Lyft implement it. Both carry nuScenes-devkit metadata, which is where the
    track ids live: `gt_boxes_token` in the infos is a per-FRAME annotation token, and the track is
    `instance_token` in `sample_annotation.json`.
    """
    if not dataset_cfg.get('GT_BOXES_MOTION_COMPENSATION', False):
        return
    raise NotImplementedError(
        'GT_BOXES_MOTION_COMPENSATION is set but %s does not implement it. Only NuScenesDataset '
        'and LyftDataset do, both via nuScenes-devkit track ids (instance_token in '
        'sample_annotation.json).\n'
        '  KITTI    cannot: it has no sequences, so there is nothing to accumulate or compensate.\n'
        '  PandaSet could: every frame is annotated and cuboids carry a persistent uuid - but its\n'
        '           loader has no sweep accumulation at all, so there is nothing to compensate\n'
        '           until one exists.\n'
        '  Waymo    could: annos carry obj_ids per frame and infos carry pose - same caveat, no\n'
        '           accumulation path in the loader.\n'
        'Remove the key, or implement accumulation for this dataset first.' % dataset_name)
