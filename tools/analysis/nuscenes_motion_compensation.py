"""Accumulation with per-object motion compensation: move points inside a GT box to where that
object is in the anchor frame.

Ego compensation alone leaves moving objects smeared, because the world-static transform does not
follow them. The fix is a per-object rigid transform, which is strictly stronger than a velocity
correction: it captures yaw change and non-constant motion, both of which a linear extrapolation
of gt_boxes_velocity drops.

nuScenes' gt_boxes_token is a per-FRAME annotation token, so tracks come from instance_token in
sample_annotation.json. Each instance's motion in the anchor frame is derived box-to-box from the
anchor and its preceding keyframe, then applied linearly in dt - boxes only exist at 2 Hz keyframes
while points arrive at 20 Hz, so the sweeps in between have to be interpolated.
"""
import json, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nuscenes_chain import INFOS, SCENE, _ld, _rm, _S, NS
from domain_gap_analysis import build_platforms, mask_range, points_in_boxes, MAX_DIST

_meta = NS / 'v1.0-trainval'
print('building annotation -> instance map...', flush=True)
_t = time.time()
ANN2INST = {r['token']: r['instance_token'] for r in json.load(open(_meta / 'sample_annotation.json'))}
print('  %.1f s' % (time.time() - _t), flush=True)


def _yaw_of(T):
    return np.arctan2(T[1, 0], T[0, 0])


def _boxes_in_anchor(idx, anchor_S):
    """That keyframe's Car boxes, mapped into the anchor frame, keyed by instance token."""
    info = INFOS[idx]
    g = np.asarray(info['gt_boxes'])
    nm = np.asarray(info['gt_names'])
    tok = np.asarray(info['gt_boxes_token']).ravel()
    m = nm == 'car'
    if not m.sum():
        return {}
    T = anchor_S @ np.linalg.inv(_S(info))
    dyaw = _yaw_of(T)
    out, vel = {}, {}
    for row, tk in zip(g[m], tok[m]):
        inst = ANN2INST.get(str(tk))
        if inst is None:
            continue
        c = T[:3, :3] @ row[:3] + T[:3, 3]
        v = np.zeros(3)
        if len(row) >= 9:
            v = T[:3, :3] @ np.nan_to_num(np.array([row[7], row[8], 0.0]))
        # columns 7:10 carry the velocity in the anchor frame; id()-keyed side tables are unsafe
        # because CPython reuses ids after garbage collection.
        out[inst] = np.concatenate([c, row[3:6], [row[6] + dyaw], v])
    return out


def exact_box_transform(points, boxes_then, boxes_now, shrink=1.0):
    """Move points inside each object's box AT THAT TIME onto its box in the anchor frame.

    Exact for keyframes: both boxes are annotated, so the rigid transform between them is the
    object's true displacement and yaw change over the interval - no velocity model, no
    extrapolation, and non-constant motion is handled for free.
    """
    if not boxes_then:
        return points
    p = points.copy()
    src = points[:, :3].copy()      # containment is tested against the ORIGINAL positions:
    claimed = np.zeros(len(p), bool)  # mutating in place lets one point be moved twice
    for inst, then in boxes_then.items():
        now = boxes_now.get(inst)
        if now is None:
            continue
        q = src - then[:3]
        ca, sa = np.cos(-then[6]), np.sin(-then[6])
        x = q[:, 0] * ca - q[:, 1] * sa
        y = q[:, 0] * sa + q[:, 1] * ca
        inside = ((np.abs(x) <= then[3] * shrink / 2) & (np.abs(y) <= then[4] * shrink / 2) &
                  (np.abs(q[:, 2]) <= then[5] * shrink / 2) & ~claimed)
        if not inside.any():
            continue
        claimed |= inside
        dyaw = now[6] - then[6]
        cr, sr = np.cos(dyaw), np.sin(dyaw)
        rel = src[inside] - then[:3]
        p[inside, :3] = np.column_stack([rel[:, 0] * cr - rel[:, 1] * sr,
                                         rel[:, 0] * sr + rel[:, 1] * cr, rel[:, 2]]) + now[:3]
    return p


def interp_boxes(boxes_a, boxes_b, w, dt=None, hermite=False):
    """Boxes w of the way from a to b, for sweeps between two annotated keyframes.

    Linear placement is wrong for a turning or accelerating object, and a misplaced box sweeps up
    neighbouring points which then get teleported onto the anchor box. Cubic Hermite uses the
    annotated velocity at both ends as the tangent, which is exact under constant acceleration.
    """
    out = {}
    for inst, ba in boxes_a.items():
        bb = boxes_b.get(inst)
        if bb is None:
            out[inst] = ba
            continue
        if hermite and dt and len(ba) >= 10 and len(bb) >= 10:
            # u runs 0 at b (older) to 1 at a (newer); tangents are velocity * interval
            u = 1.0 - w
            h00 = 2*u**3 - 3*u**2 + 1
            h10 = u**3 - 2*u**2 + u
            h01 = -2*u**3 + 3*u**2
            h11 = u**3 - u**2
            c = (h00 * bb[:3] + h10 * (bb[7:10] * dt) +
                 h01 * ba[:3] + h11 * (ba[7:10] * dt))
        else:
            c = ba[:3] * (1 - w) + bb[:3] * w
        dy = np.arctan2(np.sin(bb[6] - ba[6]), np.cos(bb[6] - ba[6]))
        out[inst] = np.concatenate([c, ba[3:6], [ba[6] + dy * w], ba[7:10] if len(ba) >= 10 else np.zeros(3)])
    return out


def accumulate_exact(anchor_idx, n_frames, compensate_motion=True, interpolate_sweeps=True,
                     shrink=1.0, hermite=False):
    """Chain keyframes, compensating each object with its OWN annotated box at that keyframe."""
    anchor = INFOS[anchor_idx]
    Sa = _S(anchor)
    boxes_anchor = _boxes_in_anchor(anchor_idx, Sa)
    out, used, k = [], 0, anchor_idx
    while used < n_frames and k >= 0 and SCENE[k] == SCENE[anchor_idx]:
        info = INFOS[k]
        T = Sa @ np.linalg.inv(_S(info))
        R, t = T[:3, :3], T[:3, 3]
        boxes_k = _boxes_in_anchor(k, Sa) if compensate_motion else {}
        # A sweep PRECEDES its keyframe by time_lag, so it sits between keyframe k-1 and k.
        # Interpolating toward k+1 would extrapolate the wrong way and double the error.
        has_prev = compensate_motion and k - 1 >= 0 and SCENE[k - 1] == SCENE[anchor_idx]
        boxes_prev = _boxes_in_anchor(k - 1, Sa) if has_prev else boxes_k
        p = _rm(_ld(info['lidar_path']), 1.5)
        p = p if k == anchor_idx else (R @ p.T).T + t
        out.append(exact_box_transform(p, boxes_k, boxes_anchor, shrink) if compensate_motion else p)
        used += 1
        gap = (info['timestamp'] - INFOS[k - 1]['timestamp']) if has_prev else 1.0
        for s in info['sweeps']:
            if used >= n_frames:
                break
            q = _rm(_ld(s['lidar_path']), 1.0)
            tm = s['transform_matrix']
            if tm is not None:
                q = (tm[:3, :3] @ q.T).T + tm[:3, 3]
            q = (R @ q.T).T + t
            if compensate_motion:
                w = float(s['time_lag']) / gap if (gap > 1e-3 and interpolate_sweeps) else 0.0
                w = float(np.clip(w, 0.0, 1.0))
                q = exact_box_transform(q, interp_boxes(boxes_k, boxes_prev, w, gap, hermite),
                                        boxes_anchor, shrink)
            out.append(q)
            used += 1
        k -= 1
    return np.concatenate(out), used


def object_motion(anchor_idx):
    """Per-instance (velocity, yaw-rate) in the anchor frame, from the box-to-box displacement."""
    Sa = _S(INFOS[anchor_idx])
    cur = _boxes_in_anchor(anchor_idx, Sa)
    prev_idx = anchor_idx - 1
    if prev_idx < 0 or SCENE[prev_idx] != SCENE[anchor_idx]:
        return cur, {k: (np.zeros(3), 0.0) for k in cur}
    prev = _boxes_in_anchor(prev_idx, Sa)
    dt = INFOS[anchor_idx]['timestamp'] - INFOS[prev_idx]['timestamp']
    motion = {}
    for inst, box in cur.items():
        if inst in prev and dt > 1e-3:
            v = (box[:3] - prev[inst][:3]) / dt
            w = float(np.arctan2(np.sin(box[6] - prev[inst][6]), np.cos(box[6] - prev[inst][6])) / dt)
        else:
            v, w = np.zeros(3), 0.0
        motion[inst] = (v, w)
    return cur, motion


def compensate(points, boxes, motion, dt):
    """Move points that were inside each object's box dt seconds ago onto its anchor-frame box."""
    if dt <= 0 or not boxes:
        return points
    p = points.copy()
    for inst, box in boxes.items():
        v, w = motion[inst]
        if not np.any(v) and abs(w) < 1e-6:
            continue
        c_then = box[:3] - v * dt                       # where the object was
        yaw_then = box[6] - w * dt
        q = p[:, :3] - c_then
        ca, sa = np.cos(-yaw_then), np.sin(-yaw_then)
        x = q[:, 0] * ca - q[:, 1] * sa
        y = q[:, 0] * sa + q[:, 1] * ca
        inside = ((np.abs(x) <= box[3] / 2) & (np.abs(y) <= box[4] / 2) &
                  (np.abs(q[:, 2]) <= box[5] / 2))
        if not inside.any():
            continue
        dyaw = box[6] - yaw_then
        cr, sr = np.cos(dyaw), np.sin(dyaw)
        rel = p[inside, :3] - c_then
        rot = np.column_stack([rel[:, 0] * cr - rel[:, 1] * sr,
                               rel[:, 0] * sr + rel[:, 1] * cr, rel[:, 2]])
        p[inside, :3] = rot + box[:3]
    return p


def accumulate_compensated(anchor_idx, n_frames, compensate_motion=True):
    anchor = INFOS[anchor_idx]
    Sa = _S(anchor)
    boxes, motion = object_motion(anchor_idx)
    t_anchor = anchor['timestamp']
    out, used, k = [], 0, anchor_idx
    while used < n_frames and k >= 0 and SCENE[k] == SCENE[anchor_idx]:
        info = INFOS[k]
        T = Sa @ np.linalg.inv(_S(info))
        R, t = T[:3, :3], T[:3, 3]
        dt_kf = t_anchor - info['timestamp']
        p = _rm(_ld(info['lidar_path']), 1.5)
        p = p if k == anchor_idx else (R @ p.T).T + t
        out.append(compensate(p, boxes, motion, dt_kf) if compensate_motion else p)
        used += 1
        for s in info['sweeps']:
            if used >= n_frames:
                break
            q = _rm(_ld(s['lidar_path']), 1.0)
            tm = s['transform_matrix']
            if tm is not None:
                q = (tm[:3, :3] @ q.T).T + tm[:3, 3]
            q = (R @ q.T).T + t
            dt = dt_kf + float(s['time_lag'])
            out.append(compensate(q, boxes, motion, dt) if compensate_motion else q)
            used += 1
        k -= 1
    return np.concatenate(out), used
