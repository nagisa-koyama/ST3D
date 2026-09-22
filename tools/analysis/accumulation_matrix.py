"""Accumulation depth N needed for each source to reach each target's density.

N counts frames INCLUDING the anchor, so it maps directly to `MAX_SWEEPS: N`.

Two criteria, because they give different answers and only the second is what a detector sees:
  global  - the source's mean radial histogram meets the target's in EVERY 5-70 m bin
  object  - the source's median points per Car box meets the target's

nuScenes and Lyft ship 9 stored sweeps, so N>10 is unreachable through MAX_SWEEPS and is reported
as such rather than extrapolated. PandaSet and Waymo are chained over sequence frames by ego pose.
KITTI has no sequences at all and cannot be accumulated by any mechanism.
"""
import json, pickle, sys, collections
import numpy as np
sys.path.insert(0, '/home/koyama/code/ST3D/tools/analysis'); sys.path.insert(0, '/home/koyama/code/ST3D/tools')
from domain_gap_analysis import (build_platforms, mask_range, radial_hist, points_in_boxes,
                                 quat_to_rot, Platform, MAX_DIST, DATA)

EDGES = np.linspace(0, MAX_DIST, 16)
BAND = slice(1, 14)
BOX_LO, BOX_HI = 5.0, 70.0          # Car boxes are scored on the SAME band as the histogram,
                                    # else PandaSet (55% of its Car labels sit outside
                                    # POINT_CLOUD_RANGE) has a median of literally zero.
NMAX, ANCHORS, TGT_FRAMES = 10, 20, 40

P = build_platforms()

# ---- accumulation for the two platforms that have none in the registry -----------------
pd_infos = pickle.load(open(DATA / 'pandaset/pandaset_infos_train.pkl', 'rb'))
PD_BY_SEQ = collections.defaultdict(dict)
for _i in pd_infos:
    PD_BY_SEQ[_i['sequence']][_i['frame_idx']] = _i
_pdpose = {}
fix = lambda p: str(p).replace('/root/ST3D/data/pandaset', str(DATA / 'pandaset'))

def pd_pose(seq, idx):
    if seq not in _pdpose:
        _pdpose[seq] = json.load(open(DATA / f'pandaset/dataset/{seq}/lidar/poses.json'))
    p = _pdpose[seq][idx]
    return quat_to_rot(*[p['heading'][k] for k in 'wxyz']), np.array([p['position'][k] for k in 'xyz'])

def pd_sweeps_factory(device):
    def sweeps(info, n):
        import pandas as pd
        seq, a = info['sequence'], info['frame_idx']
        R, t = pd_pose(seq, a)                      # points are WORLD coords: one transform serves all
        out = []
        for k in range(a, max(-1, a - n), -1):
            if k not in PD_BY_SEQ[seq]:
                break
            df = pd.read_pickle(fix(PD_BY_SEQ[seq][k]['lidar_path']))
            if device != -1:
                df = df[df.d == device]
            ego = (R.T @ (df[['x', 'y', 'z']].to_numpy() - t).T).T
            out.append(np.column_stack([ego[:, 1], -ego[:, 0], ego[:, 2], df['i'].to_numpy() / 255.0]))
        return np.concatenate(out)
    return sweeps

wa_infos = pickle.load(open(DATA / 'waymo/waymo_infos_train.pkl', 'rb'))
WA_BY_SEQ = collections.defaultdict(dict)
for _i in wa_infos:
    WA_BY_SEQ[_i['point_cloud']['lidar_sequence']][_i['point_cloud']['sample_idx']] = _i

def wa_raw(info):
    pc = info['point_cloud']
    f = np.load(DATA / 'waymo/waymo_processed_data' / pc['lidar_sequence'] / ('%04d.npy' % pc['sample_idx']))
    f = f[f[:, 5] == -1]
    return np.column_stack([f[:, :3], np.tanh(f[:, 3])])

def wa_sweeps(info, n):
    pc = info['point_cloud']; seq, a = pc['lidar_sequence'], pc['sample_idx']
    Pa = np.linalg.inv(np.asarray(info['pose']).reshape(4, 4))
    out = []
    for k in range(a, max(-1, a - n), -1):
        if k not in WA_BY_SEQ[seq]:
            break
        ik = WA_BY_SEQ[seq][k]
        p = wa_raw(ik)
        T = Pa @ np.asarray(ik['pose']).reshape(4, 4)
        out.append(np.column_stack([(T[:3, :3] @ p[:, :3].T).T + T[:3, 3], p[:, 3]]))
    return np.concatenate(out)

P['PandaSet Pandar64']._sweeps = pd_sweeps_factory(0)
P['PandaSet PandarGT']._sweeps = pd_sweeps_factory(1)
P['PandaSet Pandar64'].infos = [i for i in pd_infos if i['frame_idx'] % 4 == 0]
P['PandaSet PandarGT'].infos = [i for i in pd_infos if i['frame_idx'] % 4 == 0]
P['Waymo']._sweeps = wa_sweeps

# ---- dataset-level aggregates (the mix you would actually train on) --------------------
def agg(name, parts, car, rate):
    a, b = P[parts[0]], P[parts[1]]
    return Platform(name, car, list(a.infos) + list(b.infos), a._load, rate_hz=rate, sweeps=a._sweeps)

P['nuScenes (all)'] = agg('nuScenes (all)', ['nuScenes n008 Boston', 'nuScenes n015 Singapore'], 'car', 20)
P['Lyft (all)'] = agg('Lyft (all)', ['Lyft 40-beam', 'Lyft 64-beam'], 'car', 5)

TARGETS = ['KITTI', 'nuScenes (all)', 'nuScenes n008 Boston', 'nuScenes n015 Singapore',
           'Lyft (all)', 'Lyft 40-beam', 'Lyft 64-beam']
SOURCES = ['KITTI', 'nuScenes (all)', 'nuScenes n008 Boston', 'nuScenes n015 Singapore',
           'Lyft (all)', 'Lyft 40-beam', 'Lyft 64-beam',
           'PandaSet Pandar64', 'PandaSet PandarGT', 'Waymo']
NO_OBJ = {'Waymo'}                     # points-in-box is broken in this checkout

# ---- target references (single frame) --------------------------------------------------
print('target references (single frame)', flush=True)
TH, TB = {}, {}
for t in TARGETS:
    plat = P[t]; hs, bc = [], []
    for info in plat.sample(TGT_FRAMES):
        fr = plat.frame(info); p = mask_range(fr.points)
        hs.append(radial_hist(p, EDGES))
        m = fr.names == plat.car_class
        if m.sum():
            b = fr.boxes[m]
            r = np.linalg.norm(b[:, :2], axis=1)
            b = b[(r >= BOX_LO) & (r < BOX_HI)]
            if len(b):
                bc += list(points_in_boxes(p, b))
    TH[t] = np.mean(hs, axis=0)
    TB[t] = float(np.median(bc)) if bc else float('nan')
    print('  %-26s %9.0f pts/frame  %5.0f pts/Car box' % (t, TH[t].sum(), TB[t]), flush=True)

# ---- source curves ---------------------------------------------------------------------
print('\nsource accumulation curves', flush=True)
SH, SB = {}, {}
for s in SOURCES:
    plat = P[s]
    if plat._sweeps is None:
        SH[s] = {1: TH[s] if s in TH else None}; SB[s] = {1: TB.get(s)}
        if SH[s][1] is None:                       # KITTI is also a target, so already measured
            hs = [radial_hist(mask_range(plat.frame(i).points), EDGES) for i in plat.sample(TGT_FRAMES)]
            SH[s] = {1: np.mean(hs, axis=0)}
        print('  %-26s no sequences - cannot accumulate' % s, flush=True)
        continue
    hs = {n: [] for n in range(1, NMAX + 1)}
    bc = {n: [] for n in range(1, NMAX + 1)}
    for info in plat.sample(ANCHORS):
        fr = plat.frame(info)
        m = fr.names == plat.car_class
        boxes = None
        if m.sum():
            b = fr.boxes[m]
            r = np.linalg.norm(b[:, :2], axis=1)
            b = b[(r >= BOX_LO) & (r < BOX_HI)]
            boxes = b if len(b) else None
        for n in range(1, NMAX + 1):
            p = mask_range(plat.accumulate(info, n))
            hs[n].append(radial_hist(p, EDGES))
            if boxes is not None and s not in NO_OBJ:
                bc[n] += list(points_in_boxes(p, boxes))
    SH[s] = {n: np.mean(v, axis=0) for n, v in hs.items()}
    SB[s] = {n: (float(np.median(bc[n])) if bc[n] else float('nan')) for n in bc}
    print('  %-26s N=1 %8.0f -> N=%d %9.0f pts/frame   Car box %4.0f -> %4.0f'
          % (s, SH[s][1].sum(), NMAX, SH[s][NMAX].sum(), SB[s][1], SB[s][NMAX]), flush=True)

np.save('/tmp/claude-2010/-home-koyama/ef638020-c693-4617-878d-83ed11a08bf9/scratchpad/accmatrix2.npy',
        dict(TH=TH, TB=TB, SH=SH, SB=SB, TARGETS=TARGETS, SOURCES=SOURCES), allow_pickle=True)

# ---- the matrices -----------------------------------------------------------------------
UNUSABLE = {'PandaSet PandarGT'}   # over half its in-band Car boxes lie outside the +-29 deg flash
                                   # cone, so the median box is never illuminated at ANY depth and
                                   # the object criterion is undefined rather than merely unmet

def need_global(s, t):
    if len(SH[s]) == 1: return None
    for n in sorted(SH[s]):
        if np.all(SH[s][n][BAND] >= TH[t][BAND]): return n
    return 99

def need_object(s, t):
    if s in NO_OBJ or s in UNUSABLE or len(SB[s]) == 1: return None
    for n in sorted(SB[s]):
        if SB[s][n] >= TB[t]: return n
    return 99

fmt = lambda v: 'n/a' if v is None else ('>%d' % NMAX if v == 99 else str(v))
hdr = lambda: ('%-24s' % 'source \\ target' +
               ''.join('%12s' % t.replace('nuScenes', 'nuSc').replace('PandaSet', 'PS')
                       .replace(' Singapore', '').replace(' Boston', '') for t in TARGETS))

for title, fn in [('GLOBAL parity - mean radial histogram meets the target in every 5-70 m bin', need_global),
                  ('OBJECT parity - median points per Car box meets the target (same 5-70 m band)', need_object)]:
    print('\n\n%s\n' % title); print(hdr())
    for s in SOURCES:
        print('%-24s%s' % (s.replace('PandaSet', 'PS'),
              ''.join('%12s' % ('--' if s == t else fmt(fn(s, t))) for t in TARGETS)))

print('\n\nRECOMMENDED N - the binding constraint, max(global, object)\n'); print(hdr())
for s in SOURCES:
    row = ''
    for t in TARGETS:
        if s == t: row += '%12s' % '--'; continue
        a, b = need_global(s, t), need_object(s, t)
        if a is None: row += '%12s' % 'n/a'; continue
        row += '%12s' % (fmt(max(a, b) if b is not None else a) + ('' if b is None or b <= a else '*'))
    print('%-24s%s' % (s.replace('PandaSet', 'PS'), row))

print("""
n/a  source has no sequences and cannot be accumulated by any mechanism (KITTI).
>%d  not reachable from the 9 stored sweeps, i.e. not settable through MAX_SWEEPS.
*    the OBJECT criterion binds - global parity alone would under-size N.
Waymo has no object column: its processed points and boxes are geometrically inconsistent
in this checkout. PandarGT has none either, for the structural reason noted in the source.
Points per box is NOT size-normalised; Lyft cars are 4.73 m against KITTI's 3.83 m, so part
of any per-box gap is box size, which ROS/SN address separately.""" % NMAX)

print('\n\npoints per Car box, 5-70 m band\n')
print('%-24s %10s %10s' % ('', 'N=1', 'N=%d' % NMAX))
for s in SOURCES:
    if s in NO_OBJ or len(SB[s]) == 1: print('%-24s %10s %10s' % (s, '-', '-')); continue
    print('%-24s %10.0f %10.0f' % (s, SB[s][1], SB[s][NMAX]))
print('\ntarget references (single frame, same band)')
for t in TARGETS:
    print('  %-24s %6.0f pts/Car box   %9.0f pts/frame' % (t, TB[t], TH[t].sum()))
