"""Calibration of the temporal pseudo-label propagation rule on the labelled SOURCE (experiments_md/20261017_02 §4).

Reads only nuScenes VAL (labels, devkit track ids, poses) and teacher 27471's predictions on it (job 27608 step 2). No
Waymo file is opened. Three stages, each its own Slurm CPU job:

  prep    R1 s_conf, R3 P_ref, R2 d_assoc, R5 gates d_init / d_cv; static clusters, CV tracks and their candidates at
          H_MAX on the two 1 Hz phases of every val scene. No points.            -> <out>/prep.pkl
  points  per platform (n008 / n015): the cloud the teacher predicted on, rebuilt by the generation config through the
          same loader in the same mode, and the ground-relative point count (§3.6) of every direct Car pseudo-label and
          every candidate of that platform's frames.                            -> <out>/counts_<platform>.pkl
  search  R4 (q*, H*) for variant S, then R5 (q_D, H_D) for variant D.           -> <out>/calibration.json + .txt

    python analysis/pseudo_label_propagation_source.py prep   --out_dir <dir>
    python analysis/pseudo_label_propagation_source.py points --out_dir <dir> --platform n008 --workers 6
    python analysis/pseudo_label_propagation_source.py search --out_dir <dir>
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pseudo_label_propagation as P  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1]
CFG_27704 = TOOLS / 'cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2waymo.yaml'
PLATFORMS = {
    'n008': {'ps': '/home/koyama/data/pseudo_labels/legaldepth27471gb_nuscenes_val_n008/ps_label_e0.pkl',
             'cfg': TOOLS / 'analysis/pseudo_label_threshold/cfgs/teacher_s2gblobs_on_nuscenes_val_n008_15sweeps.yaml'},
    'n015': {'ps': '/home/koyama/data/pseudo_labels/legaldepth27471gb_nuscenes_val_n015/ps_label_e0.pkl',
             'cfg': TOOLS / 'analysis/pseudo_label_threshold/cfgs/teacher_s2gblobs_on_nuscenes_val_n015_10sweeps.yaml'},
}
INFOS = TOOLS / '../data/nuscenes/v1.0-trainval/nuscenes_infos_200sweeps_val.pkl'
META = TOOLS / '../data/nuscenes/v1.0-trainval/v1.0-trainval'

# Pre-declared constants (20261017_02 §4)
S_GRID = np.round(np.arange(0.18, 0.951, 0.01), 2)
P_SEED, MIN_KEPT = 0.90, 50
STATIC_GT_M = 0.2
PAIR_DT = (0.5, 10.5)
H_MAX = 10
H_GRID = list(range(1, H_MAX + 1))
Q_GRID = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
MIN_PER_DELTA = 50
CAR_CUT = 0.18


def ceil_to(x, step):
    return float(math.ceil(x / step - 1e-9) * step)


def load_cuts():
    import yaml
    c = yaml.safe_load(open(CFG_27704))
    return np.array(c['SELF_TRAIN']['SCORE_THRESH'], float), np.array(c['SELF_TRAIN']['NEG_THRESH'], float)


def apply_cuts(rows, score_th, neg_th):
    """27704's cuts on floor-0.0001 predictions: drop below NEG, negate below SCORE (as save_pseudo_label_batch)."""
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 9).copy()
    if len(rows) == 0:
        return rows
    cls = np.abs(rows[:, 7]).astype(np.int64)
    assert cls.min() >= 1 and cls.max() <= len(score_th), 'unexpected class index in the source predictions'
    keep = rows[:, 8] >= neg_th[cls - 1]
    rows, cls = rows[keep], cls[keep]
    ign = rows[:, 8] < score_th[cls - 1]
    rows[ign, 7] = -np.abs(rows[ign, 7])
    return rows


def shift_of(cfg_path):
    import yaml
    c = yaml.safe_load(open(cfg_path))
    return np.array(c['DATA_CONFIG_TAR'].get('SHIFT_COOR', [0, 0, 0]), float)


# ----------------------------------------------------------------------------------------------- prep

def stage_prep(args):
    t0 = time.time()
    score_th, neg_th = load_cuts()
    shifts = {p: shift_of(v['cfg']) for p, v in PLATFORMS.items()}
    assert np.allclose(shifts['n008'], shifts['n015']), shifts
    shift = shifts['n008']
    ps = {}
    plat_of = {}
    for p, v in PLATFORMS.items():
        d = pickle.load(open(v['ps'], 'rb'))
        for k in d:
            plat_of[k] = p
        assert not set(d) & set(ps)
        ps.update(d)
    infos = pickle.load(open(INFOS, 'rb'))
    ann2inst = {r['token']: r['instance_token'] for r in json.load(open(META / 'sample_annotation.json'))}
    tok2scene = {r['token']: r['scene_token'] for r in json.load(open(META / 'sample.json'))}
    print(f'cuts SCORE {score_th.tolist()} NEG {neg_th.tolist()}; SHIFT_COOR {shift.tolist()}; '
          f'{len(infos)} val infos, {len(ps)} predicted frames', flush=True)

    Tz = np.eye(4); Tz[:3, 3] = -shift
    frames = []  # flat list of frame records
    by_scene = {}
    for info in infos:
        fid = info['lidar_path'].split('/')[-1][:-4]
        assert fid in ps, f'no teacher prediction for {fid}'
        S = np.asarray(info['ref_from_car']) @ np.asarray(info['car_from_global'])
        M = np.linalg.inv(S) @ Tz
        raw = np.asarray(ps[fid]['gt_boxes'], dtype=np.float64).reshape(-1, 9)
        names = np.asarray(info['gt_names'])
        gtb = np.asarray(info['gt_boxes'])[:, :7] if len(names) else np.zeros((0, 7))
        npts = np.asarray(info['num_lidar_pts'])
        toks = np.asarray(info['gt_boxes_token']).ravel()
        car = names == 'car'
        gglob = P.transform_boxes(gtb, np.linalg.inv(S)) if len(gtb) else np.zeros((0, 7))
        rec = {
            'fid': fid, 'platform': plat_of[fid], 't': float(info['timestamp']) * (1e-6 if info['timestamp'] > 1e12 else 1),
            'M': M, 'raw': raw, 'rows': apply_cuts(raw, score_th, neg_th),
            # GT for matching: car, >= 1 lidar point, centre in range (as rule B)
            'gt_xy': gtb[car & (npts >= 1) & P.in_pc_range(gtb)][:, :2] if len(gtb) else np.zeros((0, 2)),
            'gt_inst': np.array([ann2inst.get(str(t)) for t in toks[car & (npts >= 1) & P.in_pc_range(gtb)]], object)
            if len(gtb) else np.zeros(0, object),
            # every annotated car with >= 1 point, global BEV, for the static / moving annotation
            'car_inst_all': np.array([ann2inst.get(str(t)) for t in toks[car & (npts >= 1)]], object) if len(gtb) else np.zeros(0, object),
            'car_glob_all': gglob[car & (npts >= 1)][:, :2] if len(gtb) else np.zeros((0, 2)),
        }
        by_scene.setdefault(tok2scene[info['token']], []).append(len(frames))
        frames.append(rec)
    del infos, ann2inst

    # R1 / R3: precision of the direct Car predictions by score threshold, over all val keyframes (rule B matching)
    sc, hit = [], []
    for f in frames:
        r = f['raw']
        p = r[(r[:, 7] == P.CAR_LABEL) & P.in_pc_range(r)]
        h, _ = P.greedy_match(p[:, :2], p[:, 8], f['gt_xy'])
        sc.append(p[:, 8]); hit.append(h)
    sc, hit = np.concatenate(sc), np.concatenate(hit)
    prec = {float(t): (float(hit[sc >= t].mean()) if (sc >= t).any() else float('nan'), int((sc >= t).sum()))
            for t in S_GRID}
    s_conf = None
    for t in S_GRID:
        later = [prec[float(u)] for u in S_GRID if u >= t and prec[float(u)][1] >= MIN_KEPT]
        if prec[float(t)][1] >= MIN_KEPT and all(pp >= P_SEED for pp, _ in later):
            s_conf = float(t)
            break
    P_ref = float(hit[sc >= CAR_CUT].mean())
    print(f'R1 s_conf = {s_conf}; R3 P_ref (Car >= {CAR_CUT}) = {P_ref:.4f} over {(sc >= CAR_CUT).sum()} boxes', flush=True)
    assert s_conf is not None, 'R1: no confident score - no seeds, no propagation (null result)'

    # phases: each scene's keyframes in time order, split into even / odd index
    phases = []
    for scene, ix in sorted(by_scene.items()):
        ix = sorted(ix, key=lambda i: frames[i]['t'])
        phases += [ix[0::2], ix[1::2]]
    phases = [ph for ph in phases if ph]

    # seeds matched to GT instance, per frame
    for f in frames:
        r = f['raw']
        sidx = np.nonzero((r[:, 7] == P.CAR_LABEL) & (r[:, 8] >= s_conf))[0]
        m = P.match_to_gt_index(r[sidx, :2], r[sidx, 8], f['gt_xy'])
        f['seed_idx'] = sidx
        f['seed_inst'] = np.array([f['gt_inst'][j] if j >= 0 else None for j in m], object)
        f['seed_glob'] = P.transform_boxes(r[sidx, :7], f['M'])[:, :2] if len(sidx) else np.zeros((0, 2))

    def inst_pos(f):
        return {i: xy for i, xy in zip(f['car_inst_all'], f['car_glob_all']) if i is not None}

    # R2: pairs of seeds on the same annotated-static instance, 1-10 s apart within a phase
    pair_d = []
    for ph in phases:
        pos = [inst_pos(frames[i]) for i in ph]
        sd = [{inst: frames[i]['seed_glob'][k] for k, inst in enumerate(frames[i]['seed_inst']) if inst is not None}
              for i in ph]
        for a in range(len(ph)):
            for b in range(a + 1, len(ph)):
                dt = frames[ph[b]]['t'] - frames[ph[a]]['t']
                if dt < PAIR_DT[0] or dt > PAIR_DT[1]:
                    continue
                for inst in set(sd[a]) & set(sd[b]):
                    if inst in pos[a] and inst in pos[b] and np.hypot(*(pos[a][inst] - pos[b][inst])) <= STATIC_GT_M:
                        pair_d.append(float(np.hypot(*(sd[a][inst] - sd[b][inst]))))
    pair_d = np.array(pair_d)
    d_assoc = ceil_to(np.percentile(pair_d, 95), 0.05)
    print(f'R2 d_assoc = {d_assoc} m (p95 {np.percentile(pair_d, 95):.3f} of {len(pair_d)} static pairs; '
          f'median {np.median(pair_d):.3f})', flush=True)

    # R5 gates
    disp, cv_err = [], []
    for ph in phases:
        pos = [inst_pos(frames[i]) for i in ph]
        for a in range(len(ph) - 1):
            for inst in set(pos[a]) & set(pos[a + 1]):
                d = float(np.hypot(*(pos[a][inst] - pos[a + 1][inst])))
                if d > d_assoc:
                    disp.append(d)
        sd = [{inst: frames[i]['seed_glob'][k] for k, inst in enumerate(frames[i]['seed_inst']) if inst is not None}
              for i in ph]
        for a in range(len(ph) - 2):
            t1, t2, t3 = (frames[ph[a + k]]['t'] for k in range(3))
            for inst in set(sd[a]) & set(sd[a + 1]) & set(sd[a + 2]):
                if inst in pos[a] and inst in pos[a + 2] and np.hypot(*(pos[a][inst] - pos[a + 2][inst])) > d_assoc:
                    p1, p2, p3 = sd[a][inst], sd[a + 1][inst], sd[a + 2][inst]
                    pred = p2 + (p2 - p1) * (t3 - t2) / max(t2 - t1, 1e-6)
                    cv_err.append(float(np.hypot(*(p3 - pred))))
    disp, cv_err = np.array(disp), np.array(cv_err)
    d_init = ceil_to(np.percentile(disp, 95), 0.5)
    d_cv = ceil_to(np.percentile(cv_err, 95), 0.1)
    print(f'R5 d_init = {d_init} m (p95 {np.percentile(disp, 95):.2f} of {len(disp)} moving 1-s displacements); '
          f'd_cv = {d_cv} m (p95 {np.percentile(cv_err, 95):.2f} of {len(cv_err)} triplets)', flush=True)

    # clusters, tracks, candidates at H_MAX, per phase
    n_static = n_clusters = n_tracks = 0
    for f in frames:
        f['S_cands'], f['D_cands'] = [], []
    for ph in phases:
        seq = [P.Frame(frames[i]['fid'], frames[i]['t'], frames[i]['M'], frames[i]['rows']) for i in ph]
        clusters, _ = P.static_clusters(seq, s_conf, d_assoc)
        n_clusters += len(clusters)
        n_static += sum(c['static'] for c in clusters)
        members = {m for c in clusters if c['static'] for m in c['members']}
        tracks = P.cv_tracks(seq, s_conf, members, d_init, d_cv)
        n_tracks += len(tracks)
        for g, cl in P.propagate_static(seq, clusters, H_MAX).items():
            frames[ph[g]]['S_cands'] += cl
        for g, cl in P.propagate_tracks(seq, tracks, H_MAX).items():
            frames[ph[g]]['D_cands'] += cl
    print(f'{len(phases)} phase sequences; {n_clusters} clusters ({n_static} static), {n_tracks} CV tracks; '
          f'S candidates {sum(len(f["S_cands"]) for f in frames)}, D candidates {sum(len(f["D_cands"]) for f in frames)} '
          f'at H_MAX {H_MAX}', flush=True)

    calib = {'s_conf': s_conf, 'P_ref': P_ref, 'd_assoc': d_assoc, 'd_init': d_init, 'd_cv': d_cv,
             'precision_curve': {f'{t:.2f}': prec[float(t)] for t in S_GRID},
             'n_static_pairs': int(len(pair_d)), 'pair_d_median': float(np.median(pair_d)),
             'n_moving_disp': int(len(disp)), 'n_cv_triplets': int(len(cv_err)),
             'n_phases': len(phases), 'n_clusters': n_clusters, 'n_static_clusters': n_static, 'n_tracks': n_tracks,
             'shift': shift.tolist(), 'cuts': {'SCORE': score_th.tolist(), 'NEG': neg_th.tolist()}}
    keep = ('fid', 'platform', 't', 'rows', 'gt_xy', 'S_cands', 'D_cands')
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pickle.dump({'calib': calib, 'frames': [{k: f[k] for k in keep} for f in frames]}, open(out / 'prep.pkl', 'wb'))
    print(f'wrote {out / "prep.pkl"} in {time.time() - t0:.0f} s', flush=True)


# ----------------------------------------------------------------------------------------------- points

_DS = None
_JOBS = None


def boxes_to_count(f):
    r = f['rows']
    direct = r[(r[:, 7] == P.CAR_LABEL) & (r[:, 8] >= CAR_CUT)][:, :7]
    s = np.array([c['box'] for c in f['S_cands']]).reshape(-1, 7)
    d = np.array([c['box'] for c in f['D_cands']]).reshape(-1, 7)
    return direct, s, d


def _count_one(job):
    fid, ds_idx, direct, s, d = job
    data = _DS[ds_idx]
    assert data['frame_id'] == fid, (data['frame_id'], fid)
    pts = np.asarray(data['points'])[:, :3]
    pts = pts[(pts[:, 2] >= -2) & (pts[:, 2] <= 4)]  # the voxeliser's z range (POINT_CLOUD_RANGE)
    from scipy.spatial import cKDTree
    tree = cKDTree(pts[:, :2])
    return fid, (P.count_nonground(pts, direct, tree), P.count_nonground(pts, s, tree),
                 P.count_nonground(pts, d, tree), len(pts))


def stage_points(args):
    global _DS
    t0 = time.time()
    out = Path(args.out_dir)
    prep = pickle.load(open(out / 'prep.pkl', 'rb'))
    frames = [f for f in prep['frames'] if f['platform'] == args.platform]
    del prep
    os.chdir(TOOLS)  # the configs' relative DATA_PATH
    sys.path.insert(0, str(TOOLS))
    import _init_path  # noqa: F401
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    logger = common_utils.create_logger(out / f'points_{args.platform}.log')
    cfg_from_yaml_file(str(PLATFORMS[args.platform]['cfg']), cfg)
    cfg.LOCAL_RANK = 0
    # Exactly as generate_pseudo_labels.py: built as a training set, then switched to eval mode before any worker forks.
    ds, _, _ = build_dataloader(cfg.DATA_CONFIG_TAR, cfg.CLASS_NAMES, 1, dist=False, workers=0, logger=logger,
                                training=True, model_ontology=cfg.get('ONTOLOGY', None))
    ds.eval()
    _DS = ds
    index = {Path(i['lidar_path']).name[:-4]: k for k, i in enumerate(ds.infos)}
    jobs = []
    for f in frames:
        direct, s, d = boxes_to_count(f)
        jobs.append((f['fid'], index[f['fid']], direct, s, d))
    print(f'{args.platform}: {len(jobs)} frames, {sum(len(j[2]) for j in jobs)} direct Car, '
          f'{sum(len(j[3]) for j in jobs)} S and {sum(len(j[4]) for j in jobs)} D candidates', flush=True)
    res = {}
    import multiprocessing as mp
    with mp.get_context('fork').Pool(args.workers) as pool:
        for n, (fid, c) in enumerate(pool.imap_unordered(_count_one, jobs, chunksize=4)):
            res[fid] = c
            if n % 500 == 0:
                print(f'  {n}/{len(jobs)} frames, {time.time() - t0:.0f} s', flush=True)
    pickle.dump(res, open(out / f'counts_{args.platform}.pkl', 'wb'))
    print(f'wrote counts_{args.platform}.pkl ({len(res)} frames) in {time.time() - t0:.0f} s', flush=True)


# ----------------------------------------------------------------------------------------------- search

def evaluate(frames, added_by_frame, counts_by_frame, ktab, claimed_by_frame):
    """Per Delta bin: (positives, matched). Positives matched to GT not claimed by the direct (or earlier) labels."""
    n = np.zeros(H_MAX + 1, int)
    tp = np.zeros(H_MAX + 1, int)
    pos_used = {}
    for fi, f in enumerate(frames):
        added = added_by_frame.get(fi, [])
        if not added:
            continue
        cnt = counts_by_frame[fi][added_idx_key(added)]
        boxes = np.array([a['box'] for a in added])
        cls = P.classify(cnt, P.range_of(boxes), ktab)
        pos = np.nonzero(cls == 'pos')[0]
        if len(pos) == 0:
            continue
        hit, used = P.greedy_match(boxes[pos, :2], np.array([added[i]['score'] for i in pos]), f['gt_xy'],
                                   used=claimed_by_frame[fi])
        pos_used[fi] = (pos, hit, used)
        for i, h in zip(pos, hit):
            b = P.delta_bin(added[i]['delta'])
            n[b] += 1
            tp[b] += int(h)
    return n, tp, pos_used


def added_idx_key(added):
    return np.array([a['cid'] for a in added], dtype=np.int64)


def search(frames, cand_key, counts_key, existing_of, claimed_of, direct_counts, direct_ranges, P_ref, label):
    """R4 over (q, H). Returns (choice, table, added sets per H)."""
    added_H = {}
    dups = {}
    for H in H_GRID:
        a_by, nde, ndp = {}, 0, 0
        for fi, f in enumerate(frames):
            cands = f[cand_key]
            if not cands:
                continue
            added, de, dp = P.dedup(cands, existing_of(fi), H)
            nde += de; ndp += dp
            if added:
                a_by[fi] = added
        added_H[H] = a_by
        dups[H] = (nde, ndp)
    counts_by_frame = [f[counts_key] for f in frames]
    table = {}
    best = None
    for q in Q_GRID:
        ktab = P.k_table(direct_counts, direct_ranges, q)
        for H in H_GRID:
            n, tp, _ = evaluate(frames, added_H[H], counts_by_frame, ktab, [claimed_of(fi) for fi in range(len(frames))])
            pr = [tp[d] / n[d] if n[d] else float('nan') for d in range(1, H + 1)]
            ok = all(n[d] >= MIN_PER_DELTA and tp[d] / n[d] >= P_ref for d in range(1, H + 1))
            table[f'q{q:.2f}_H{H}'] = {'q': q, 'H': H, 'k': ktab.tolist(), 'n': n[1:H + 1].tolist(),
                                       'tp': tp[1:H + 1].tolist(), 'prec': pr, 'admissible': bool(ok),
                                       'tp_total': int(tp.sum()), 'pos_total': int(n.sum()),
                                       'added_total': int(sum(len(v) for v in added_H[H].values())),
                                       'dup_direct': dups[H][0], 'dup_prop': dups[H][1]}
            if ok:
                key = (int(tp.sum()), q, -H)
                if best is None or key > best[0]:
                    best = (key, q, H)
    choice = {'q': best[1], 'H': best[2]} if best else {'q': None, 'H': 0}
    print(f'{label}: choice {choice}', flush=True)
    return choice, table, added_H


def stage_search(args):
    out = Path(args.out_dir)
    prep = pickle.load(open(out / 'prep.pkl', 'rb'))
    calib, frames = prep['calib'], prep['frames']
    counts = {}
    for p in PLATFORMS:
        counts.update(pickle.load(open(out / f'counts_{p}.pkl', 'rb')))
    assert len(counts) == len(frames), (len(counts), len(frames))
    direct_counts, direct_ranges = [], []
    for f in frames:
        cd, cs, cdd, npts = counts[f['fid']]
        f['n_points'] = npts
        for i, c in enumerate(f['S_cands']):
            c['cid'] = i
        for i, c in enumerate(f['D_cands']):
            c['cid'] = i
        f['S_counts'] = np.asarray(cs)
        f['D_counts'] = np.asarray(cdd)
        r = f['rows']
        direct = r[(r[:, 7] == P.CAR_LABEL) & (r[:, 8] >= CAR_CUT)]
        assert len(direct) == len(cd)
        direct_counts.append(np.asarray(cd)); direct_ranges.append(P.range_of(direct))
        # GT claimed by the direct Car positives (in range), matched first by score
        dp = direct[P.in_pc_range(direct)]
        _, used = P.greedy_match(dp[:, :2], dp[:, 8], f['gt_xy'])
        f['claimed_direct'] = used
    direct_counts, direct_ranges = np.concatenate(direct_counts), np.concatenate(direct_ranges)
    P_ref = calib['P_ref']

    # variant S
    chS, tabS, addS = search(frames, 'S_cands', 'S_counts', lambda fi: frames[fi]['rows'][:, :7],
                             lambda fi: frames[fi]['claimed_direct'], direct_counts, direct_ranges, P_ref, 'S')
    # GT claimed after S at its choice; the S-added boxes join the existing set for D
    S_added = addS.get(chS['H'], {}) if chS['H'] else {}
    if chS['H']:
        ktabS = P.k_table(direct_counts, direct_ranges, chS['q'])
        _, _, used = evaluate(frames, S_added, [f['S_counts'] for f in frames], ktabS,
                              [f['claimed_direct'] for f in frames])
    else:
        used = {}
    claimed_S = [used[fi][2] if fi in used else f['claimed_direct'] for fi, f in enumerate(frames)]

    def existing_D(fi):
        ex = frames[fi]['rows'][:, :7]
        if fi in S_added:
            ex = np.concatenate([ex, np.array([a['box'] for a in S_added[fi]])], axis=0)
        return ex
    chD, tabD, _ = search(frames, 'D_cands', 'D_counts', existing_D, lambda fi: claimed_S[fi],
                          direct_counts, direct_ranges, P_ref, 'D')

    k_src = {f'{q:.2f}': P.k_table(direct_counts, direct_ranges, q).tolist() for q in Q_GRID}
    result = dict(calib)
    result.update({'S': chS, 'D': chD, 'k_source': k_src, 'table_S': tabS, 'table_D': tabD,
                   'direct_car_boxes': int(len(direct_counts)),
                   'direct_car_count_median_by_bin': [float(np.median(direct_counts[P.range_bin(direct_ranges) == b]))
                                                      if (P.range_bin(direct_ranges) == b).any() else None
                                                      for b in range(len(P.RANGE_NAMES))],
                   'created': time.strftime('%Y-%m-%d %H:%M %Z')})
    json.dump(result, open(out / 'calibration.json', 'w'), indent=1)
    lines = [f"s_conf {calib['s_conf']}  P_ref {P_ref:.4f}  d_assoc {calib['d_assoc']}  d_init {calib['d_init']}  "
             f"d_cv {calib['d_cv']}", f"S choice {chS}   D choice {chD}", '']
    for name, tab in (('S', tabS), ('D', tabD)):
        lines.append(f'== variant {name}: marginal precision of added positives per Delta (n), P_ref {P_ref:.3f}')
        for q in Q_GRID:
            r = tab[f'q{q:.2f}_H{H_MAX}']
            cells = '  '.join(f'{d + 1}:{(p if p == p else 0):.2f}({nn})' for d, (p, nn) in enumerate(zip(r['prec'], r['n'])))
            adm = max([H for H in H_GRID if tab[f'q{q:.2f}_H{H}']['admissible']], default=0)
            lines.append(f'q {q:.2f} k {[int(x) for x in r["k"]]} | {cells} | largest admissible H {adm}, '
                         f'TP at it {tab[f"q{q:.2f}_H{adm}"]["tp_total"] if adm else 0}')
        lines.append('')
    open(out / 'calibration.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['prep', 'points', 'search'])
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--platform', choices=list(PLATFORMS))
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    {'prep': stage_prep, 'points': stage_points, 'search': stage_search}[args.stage](args)


if __name__ == '__main__':
    main()
