"""Temporal pseudo-label propagation on Waymo TRAIN with the source-calibrated rule (experiments_md/20261017_02 §5-§7).

  count     (LABEL-FREE) one shard of the train sequences: 27704's pseudo-labels, the poses and the clouds only. Every
            info is reduced to frame_id / pose / point_cloud as it is loaded (`annos` never reaches this code). Static
            clusters at (s_conf, d_assoc), S candidates at H*, CV tracks, D candidates at H_D, deduplication, and the
            ground-relative point count of every direct Car pseudo-label and every added box.
                                                                    -> <out>/shard_<i>.pkl
  merge     (LABEL-FREE) k(q, bin) from the Waymo direct Car counts, positive / ignore / empty per added box, the
            label-free tables, and the augmented pseudo-label files (27704 + S, 27704 + S + D).
                                                                    -> <out>/merged.pkl, label_free.json/.txt,
                                                                       <ps_out>/{S,SD}/ps_label_e0.pkl
  diagnose  (ANALYSIS ONLY - reads Waymo TRAIN labels; never feeds the rule or the GPU-row decision) precision of the
            direct and added labels, recall by range, static-call error.
                                                                    -> <out>/diagnosis.json/.txt
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pseudo_label_propagation as P  # noqa: E402

TOOLS = Path(__file__).resolve().parents[1]
WAYMO = TOOLS / '../data/waymo/waymo_processed_data'
PS_27704 = '/home/koyama/data/wandb/run-20261008_030315-5hzcs4e4/files/ps_label/ps_label_e0.pkl'
CAR_CUT = 0.18
FEATURE_CELL = 0.4  # m: voxel 0.1 x feature-map stride 4
GRID_SIDE = 150.4


def seq_of(fid):
    return fid[:-4]


def sample_idx_of(fid):
    return int(fid[-3:])


def load_label_free_infos(seq):
    """The sequence's infos with only frame_id, pose and point_cloud: labels are dropped on load (20261017_02 §2)."""
    with open(WAYMO / seq / f'{seq}.pkl', 'rb') as f:
        infos = pickle.load(f)
    out = {}
    for info in infos:
        out[info['frame_id']] = P.strip_labels(info)
    del infos
    return out


def process_sequence(seq, fids, ps, calib, with_points=True):
    """Label-free propagation for one sequence. Returns (frame records, static clusters)."""
    infos = load_label_free_infos(seq)
    fids = sorted(fids, key=sample_idx_of)
    frames = [P.Frame(fid, sample_idx_of(fid) * 0.1, infos[fid]['pose'], ps[fid]['gt_boxes']) for fid in fids]
    s_conf, d_assoc = calib['s_conf'], calib['d_assoc']
    H_S, H_D = calib['S']['H'] or 0, calib['D']['H'] or 0
    clusters, _ = P.static_clusters(frames, s_conf, d_assoc)
    S_c = P.propagate_static(frames, clusters, H_S) if H_S >= 1 else {}
    members = {m for c in clusters if c['static'] for m in c['members']}
    tracks = P.cv_tracks(frames, s_conf, members, calib['d_init'], calib['d_cv'])
    D_c = P.propagate_tracks(frames, tracks, H_D) if H_D >= 1 else {}
    recs = []
    for g, f in enumerate(frames):
        rows = f.rows
        sa, s_de, s_dp = P.dedup(S_c.get(g, []), rows[:, :7], H_S)
        ex = np.concatenate([rows[:, :7], np.array([a['box'] for a in sa]).reshape(-1, 7)], axis=0)
        da, d_de, d_dp = P.dedup(D_c.get(g, []), ex, H_D)
        direct = rows[(rows[:, 7] == P.CAR_LABEL) & (rows[:, 8] >= CAR_CUT)]
        sb = np.array([a['box'] for a in sa]).reshape(-1, 7)
        db = np.array([a['box'] for a in da]).reshape(-1, 7)
        rec = {'fid': f.fid, 'direct_car': direct[:, :7].copy(), 'direct_score': direct[:, 8].copy(),
               'S_box': sb, 'S_score': np.array([a['score'] for a in sa]), 'S_delta': np.array([a['delta'] for a in sa]),
               'S_src': np.array([a['src'] for a in sa], dtype=np.int64),
               'D_box': db, 'D_score': np.array([a['score'] for a in da]), 'D_delta': np.array([a['delta'] for a in da]),
               'n_cand_S': len(S_c.get(g, [])), 'dup_direct_S': s_de, 'dup_prop_S': s_dp,
               'n_cand_D': len(D_c.get(g, [])), 'dup_direct_D': d_de, 'dup_prop_D': d_dp}
        if with_points:
            pts = P.waymo_get_lidar(WAYMO / seq / ('%04d.npy' % sample_idx_of(f.fid)))
            from scipy.spatial import cKDTree
            tree = cKDTree(pts[:, :2])
            rec['direct_n'] = P.count_nonground(pts, direct[:, :7], tree)
            rec['S_n'] = P.count_nonground(pts, sb, tree)
            rec['D_n'] = P.count_nonground(pts, db, tree)
            rec['n_points'] = len(pts)
        recs.append(rec)
    cl = [{'members': [(frames[fi].fid, int(ri), frames[fi].rows[ri, :2].copy()) for fi, ri in c['members']],
           'anchor': c['anchor'], 'score': c['score']}
          for c in clusters if c['static']]
    return recs, cl, len(tracks)


def load_calib(args):
    """The source calibration, or - with --context S_H,D_H,q - a stated NON-RULE setting for label-free context counts.

    A context run is tagged CONTEXT_NOT_RULE in every output, writes no augmented pseudo-label file, and its records
    may only be diagnosed with --direct_only: a Waymo-label diagnosis of non-rule propagated boxes would make any later
    redesign of this idea label-influenced (experiments_md 20261017_02 §11)."""
    calib = json.load(open(args.calib))
    if getattr(args, 'context', None):
        s_h, d_h, q = args.context.split(',')
        calib = dict(calib)
        calib['S'] = {'q': float(q), 'H': int(s_h)}
        calib['D'] = {'q': float(q), 'H': int(d_h)}
        calib['CONTEXT_NOT_RULE'] = True
    return calib


def stage_inspect(args):
    """(LABEL-FREE) structure of 27704's pseudo-label file and of the per-sequence infos it needs; no propagation."""
    ps = pickle.load(open(args.ps, 'rb'))
    fid0 = sorted(ps)[0]
    e = ps[fid0]
    print(f'{len(ps)} frames; entry keys {sorted(e)}', flush=True)
    for k, v in e.items():
        a = np.asarray(v) if v is not None else None
        print(f'  {k}: {None if a is None else (a.dtype, a.shape)}', flush=True)
    labels = np.concatenate([np.asarray(v['gt_boxes'])[:, 7] for v in ps.values()])
    vals, cnt = np.unique(labels, return_counts=True)
    print('label values:', dict(zip(vals.tolist(), (cnt / len(ps)).round(3).tolist())), '(per frame)', flush=True)
    car = np.concatenate([np.asarray(v['gt_boxes'])[np.asarray(v['gt_boxes'])[:, 7] == 1, 8] for v in ps.values()])
    print(f'Car score min {car.min():.4f}, max {car.max():.4f}', flush=True)
    by_seq = {}
    for fid in ps:
        by_seq.setdefault(seq_of(fid), []).append(sample_idx_of(fid))
    steps = np.concatenate([np.diff(sorted(v)) for v in by_seq.values() if len(v) > 1])
    n = np.array([len(v) for v in by_seq.values()])
    print(f'{len(by_seq)} sequences; frames per sequence min {n.min()} median {np.median(n)} max {n.max()}; '
          f'sample_idx steps {dict(zip(*[x.tolist() for x in np.unique(steps, return_counts=True)]))}', flush=True)
    missing = 0
    for seq, idx in sorted(by_seq.items()):
        lf = load_label_free_infos(seq)
        missing += sum(f'{seq}_{i:03d}' not in lf for i in idx)
    print(f'frames without an info: {missing}', flush=True)


def stage_count(args):
    t0 = time.time()
    calib = load_calib(args)
    ps = pickle.load(open(args.ps, 'rb'))
    by_seq = {}
    for fid in ps:
        by_seq.setdefault(seq_of(fid), []).append(fid)
    seqs = sorted(by_seq)[args.shard::args.num_shards]
    print(f'shard {args.shard}/{args.num_shards}: {len(seqs)} of {len(by_seq)} sequences, '
          f'{sum(len(by_seq[s]) for s in seqs)} frames; s_conf {calib["s_conf"]} d_assoc {calib["d_assoc"]} '
          f'H_S {calib["S"]["H"]} d_init {calib["d_init"]} d_cv {calib["d_cv"]} H_D {calib["D"]["H"]}', flush=True)
    out = {'records': [], 'clusters': {}, 'n_tracks': 0}
    for n, seq in enumerate(seqs):
        recs, cl, nt = process_sequence(seq, by_seq[seq], ps, calib)
        out['records'] += recs
        out['clusters'][seq] = cl
        out['n_tracks'] += nt
        if n % 10 == 0:
            print(f'  {n}/{len(seqs)} sequences, {time.time() - t0:.0f} s', flush=True)
    d = Path(args.out_dir); d.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(d / f'shard_{args.shard}.pkl', 'wb'))
    print(f'wrote shard_{args.shard}.pkl in {time.time() - t0:.0f} s', flush=True)


def stage_merge(args):
    t0 = time.time()
    calib = load_calib(args)
    d = Path(args.out_dir)
    recs, clusters, n_tracks = [], {}, 0
    for i in range(args.num_shards):
        s = pickle.load(open(d / f'shard_{i}.pkl', 'rb'))
        recs += s['records']; clusters.update(s['clusters']); n_tracks += s['n_tracks']
    ps = pickle.load(open(args.ps, 'rb'))
    assert len(recs) == len(ps), (len(recs), len(ps))
    dn = np.concatenate([r['direct_n'] for r in recs])
    dr = np.concatenate([P.range_of(r['direct_car']) for r in recs])
    qS, qD = calib['S']['q'], calib['D']['q']
    kS = P.k_table(dn, dr, qS) if qS is not None else None
    kD = P.k_table(dn, dr, qD) if qD is not None else None
    for r in recs:
        r['S_cls'] = P.classify(r['S_n'], P.range_of(r['S_box']), kS) if kS is not None else np.zeros(0, object)
        r['D_cls'] = P.classify(r['D_n'], P.range_of(r['D_box']), kD) if kD is not None else np.zeros(0, object)
    nb = len(P.RANGE_NAMES)
    F = len(recs)

    def per_bin(boxes_list, mask_list=None):
        c = np.zeros(nb, int)
        for i, b in enumerate(boxes_list):
            if len(b) == 0:
                continue
            m = np.ones(len(b), bool) if mask_list is None else mask_list[i]
            c += np.bincount(P.range_bin(P.range_of(b[m])), minlength=nb)[:nb]
        return c

    direct_bin = per_bin([r['direct_car'] for r in recs])
    tab = {'frames': F, 'range_bins': list(P.RANGE_NAMES), 'direct_car_per_frame': (direct_bin / F).tolist(),
           'k_S': kS.tolist() if kS is not None else None, 'k_D': kD.tolist() if kD is not None else None,
           'static_clusters': int(sum(len(v) for v in clusters.values())), 'cv_tracks': int(n_tracks),
           'direct_car_count_median_by_bin': [float(np.median(dn[P.range_bin(dr) == b])) if (P.range_bin(dr) == b).any()
                                              else None for b in range(nb)]}
    for V in ('S', 'D'):
        boxes = [r[f'{V}_box'] for r in recs]
        cls = [r[f'{V}_cls'] for r in recs]
        tab[V] = {
            'candidates_per_frame': float(sum(r[f'n_cand_{V}'] for r in recs) / F),
            'dup_direct_per_frame': float(sum(r[f'dup_direct_{V}'] for r in recs) / F),
            'dup_prop_per_frame': float(sum(r[f'dup_prop_{V}'] for r in recs) / F),
            'added_per_frame_by_bin': (per_bin(boxes) / F).tolist(),
            'pos_per_frame_by_bin': (per_bin(boxes, [c == 'pos' for c in cls]) / F).tolist(),
            'ign_per_frame_by_bin': (per_bin(boxes, [c == 'ign' for c in cls]) / F).tolist(),
            'empty_per_frame_by_bin': (per_bin(boxes, [c == 'empty' for c in cls]) / F).tolist(),
            'delta_hist': np.bincount(np.concatenate([np.round(r[f'{V}_delta']).astype(int) for r in recs] + [np.zeros(0, int)]),
                                      minlength=11).tolist(),
            'ignored_area_share': float(np.mean([
                (np.prod(r[f'{V}_box'][:, 3:5], axis=1)[r[f'{V}_cls'] != 'pos'].sum() if len(r[f'{V}_box']) else 0.0)
                / GRID_SIDE ** 2 for r in recs])),
        }
        pos = np.array(tab[V]['pos_per_frame_by_bin'])
        far = slice(3, nb)  # >= 30 m
        tab[V]['pos_gain_ge30_rel'] = float(pos[far].sum() / max(direct_bin[far].sum() / F, 1e-9))
        tab[V]['pos_gain_50_75_rel'] = float(pos[5] / max(direct_bin[5] / F, 1e-9))
    # J1 / J2 (20261017_02 §7)
    tab['J1'] = bool((calib['S']['H'] or 0) >= 1)
    tab['J2'] = bool(tab['S']['pos_gain_ge30_rel'] >= 0.10)
    if calib.get('CONTEXT_NOT_RULE'):
        tab['CONTEXT_NOT_RULE'] = f"S H={calib['S']['H']} D H={calib['D']['H']} q={calib['S']['q']}: not the rule's values"
        tab['J1'] = tab['J2'] = 'not applicable (context run)'
    tab['created'] = time.strftime('%Y-%m-%d %H:%M %Z')
    json.dump(tab, open(d / 'label_free.json', 'w'), indent=1)

    lines = ([f'CONTEXT, NOT THE RULE: {tab["CONTEXT_NOT_RULE"]}'] if 'CONTEXT_NOT_RULE' in tab else []) + [
             f'LABEL-FREE, Waymo TRAIN, {F} frames (27704 pseudo-labels); per frame by range bin {list(P.RANGE_NAMES)}',
             f'direct Car positives  ' + '  '.join(f'{x:.2f}' for x in tab['direct_car_per_frame'])]
    for V in ('S', 'D'):
        t = tab[V]
        lines += [f'-- variant {V}: candidates {t["candidates_per_frame"]:.2f}/frame, dup of direct {t["dup_direct_per_frame"]:.2f}, '
                  f'dup among propagated {t["dup_prop_per_frame"]:.2f}',
                  f'   added     ' + '  '.join(f'{x:.3f}' for x in t['added_per_frame_by_bin']),
                  f'   positive  ' + '  '.join(f'{x:.3f}' for x in t['pos_per_frame_by_bin']),
                  f'   ignore    ' + '  '.join(f'{x:.3f}' for x in t['ign_per_frame_by_bin']),
                  f'   empty     ' + '  '.join(f'{x:.3f}' for x in t['empty_per_frame_by_bin']),
                  f'   positives over direct: >=30 m +{100 * t["pos_gain_ge30_rel"]:.1f}%, 50-75 m +{100 * t["pos_gain_50_75_rel"]:.1f}%; '
                  f'Delta histogram {t["delta_hist"]}; ignored area share {100 * t["ignored_area_share"]:.3f}%']
    lines += [f'k_S {tab["k_S"]}  k_D {tab["k_D"]}; static clusters {tab["static_clusters"]}, CV tracks {tab["cv_tracks"]}',
              f'J1 {tab["J1"]}  J2 {tab["J2"]}']
    open(d / 'label_free.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)

    # augmented files (never for a context run)
    by_fid = {r['fid']: r for r in recs}
    for name, variants in ((() if calib.get('CONTEXT_NOT_RULE') else (('S', ('S',)), ('SD', ('S', 'D'))))):
        new = {}
        for fid, entry in ps.items():
            r = by_fid[fid]
            boxes, labels, scores = [], [], []
            for V in variants:
                if len(r[f'{V}_box']):
                    boxes.append(r[f'{V}_box'])
                    labels.append(np.where(r[f'{V}_cls'] == 'pos', 1.0, -1.0))
                    scores.append(r[f'{V}_score'])
            if boxes:
                new[fid] = P.merge_entry(entry, np.concatenate(boxes), np.concatenate(labels), np.concatenate(scores))
            else:
                new[fid] = entry
        od = Path(args.ps_out) / name
        od.mkdir(parents=True, exist_ok=True)
        pickle.dump(new, open(od / 'ps_label_e0.pkl', 'wb'))
        n_add = sum(len(v['gt_boxes']) for v in new.values()) - sum(len(v['gt_boxes']) for v in ps.values())
        print(f'wrote {od / "ps_label_e0.pkl"}: {len(new)} frames, {n_add} added rows', flush=True)
    pickle.dump({'records': recs, 'clusters': clusters, 'context': bool(calib.get('CONTEXT_NOT_RULE'))},
                open(d / 'merged.pkl', 'wb'))
    print(f'merge done in {time.time() - t0:.0f} s', flush=True)


def stage_diagnose(args):
    """ANALYSIS ONLY: reads Waymo TRAIN labels."""
    t0 = time.time()
    d = Path(args.out_dir)
    m = pickle.load(open(d / 'merged.pkl', 'rb'))
    recs, clusters = m['records'], m['clusters']
    assert args.direct_only or not m.get('context', False), \
        'a context (non-rule) run may only be diagnosed with --direct_only (20261017_02 §11)'
    variants = () if args.direct_only else ('S', 'D')
    nb = len(P.RANGE_NAMES)
    gt = {}
    for seq in sorted({seq_of(r['fid']) for r in recs}):
        with open(WAYMO / seq / f'{seq}.pkl', 'rb') as f:
            for info in pickle.load(f):
                a = info.get('annos', None)
                if a is None:
                    continue
                names = np.asarray(a['name'])
                b = np.asarray(a['gt_boxes_lidar']).reshape(-1, 7) if len(names) else np.zeros((0, 7))
                gt[info['frame_id']] = {'veh': names == 'Vehicle', 'npts': np.asarray(a['num_points_in_gt']),
                                        'box': b, 'ids': np.asarray(a['obj_ids']), 'pose': np.asarray(info['pose'])}
    stats = {k: np.zeros(nb, int) for k in ('gt', 'rec_direct', 'rec_S', 'rec_SD', 'cov_SD')}
    pr = {k: np.zeros((2, nb), int) for k in ('direct', 'S_pos', 'D_pos')}
    pr_delta = {k: np.zeros((2, 11), int) for k in ('S_pos', 'D_pos')}
    ign = {k: np.zeros(2, int) for k in ('S', 'D')}
    for r in recs:
        g = gt[r['fid']]
        sel = g['veh'] & (g['npts'] >= 1) & P.in_pc_range(g['box'])
        gxy = g['box'][sel, :2]
        gb = P.range_bin(P.range_of(g['box'][sel]))
        stats['gt'] += np.bincount(gb, minlength=nb)[:nb]
        dsel = P.in_pc_range(r['direct_car'])
        dbox, dsc = r['direct_car'][dsel], r['direct_score'][dsel]
        hit, used = P.greedy_match(dbox[:, :2], dsc, gxy)
        pr['direct'][0] += np.bincount(P.range_bin(P.range_of(dbox)), minlength=nb)[:nb]
        pr['direct'][1] += np.bincount(P.range_bin(P.range_of(dbox[hit])), minlength=nb)[:nb]
        stats['rec_direct'] += np.bincount(gb[used], minlength=nb)[:nb]
        for V in variants:
            pos = r[f'{V}_cls'] == 'pos'
            b = r[f'{V}_box'][pos]
            h, used = P.greedy_match(b[:, :2], r[f'{V}_score'][pos], gxy, used=used)
            pr[f'{V}_pos'][0] += np.bincount(P.range_bin(P.range_of(b)), minlength=nb)[:nb]
            pr[f'{V}_pos'][1] += np.bincount(P.range_bin(P.range_of(b[h])), minlength=nb)[:nb]
            db = np.clip(np.round(r[f'{V}_delta'][pos]).astype(int), 0, 10)
            pr_delta[f'{V}_pos'][0] += np.bincount(db, minlength=11)[:11]
            pr_delta[f'{V}_pos'][1] += np.bincount(db[h], minlength=11)[:11]
            stats['rec_S' if V == 'S' else 'rec_SD'] += np.bincount(gb[used], minlength=nb)[:nb]
            # ignore regions on any Vehicle GT (any point count)
            ib = r[f'{V}_box'][~pos]
            vxy = g['box'][g['veh'], :2]
            if len(ib) and len(vxy):
                dmin = np.min(np.hypot(ib[:, None, 0] - vxy[None, :, 0], ib[:, None, 1] - vxy[None, :, 1]), axis=1)
                ign[V] += [len(ib), int((dmin <= 1.0).sum())]
            elif len(ib):
                ign[V][0] += len(ib)
        # coverage: GT inside the union of positives and ignore regions (centre within the box footprint)
        allb = np.concatenate([dbox] + [r[f'{V}_box'] for V in variants]).reshape(-1, 7)
        if len(allb) and len(gxy):
            cov = np.zeros(len(gxy), bool)
            for b in allb:
                c, s = np.cos(b[6]), np.sin(b[6])
                u = (gxy[:, 0] - b[0]) * c + (gxy[:, 1] - b[1]) * s
                v = -(gxy[:, 0] - b[0]) * s + (gxy[:, 1] - b[1]) * c
                cov |= (np.abs(u) <= b[3] / 2) & (np.abs(v) <= b[4] / 2)
            stats['cov_SD'] += np.bincount(gb[cov], minlength=nb)[:nb]
    # static-call error: seeds of static clusters matched to GT, whose GT moved > 0.2 m over the cluster's frames
    calls = {'one_static_object': 0, 'one_object_moved': 0, 'several_objects': 0, 'unmatched_member': 0}
    for seq, cl in (() if args.direct_only else clusters.items()):
        for c in cl:
            ids, gglob = [], {}
            for fid, ri, xy in c['members']:
                g = gt[fid]
                v = g['veh']
                if not v.any():
                    ids.append(None); continue
                dd = np.hypot(g['box'][v, 0] - xy[0], g['box'][v, 1] - xy[1])
                j = int(np.argmin(dd))
                if dd[j] > 1.0:
                    ids.append(None); continue
                oid = g['ids'][v][j]
                ids.append(oid)
            if any(i is None for i in ids):
                calls['unmatched_member'] += 1
                continue
            if len(set(ids)) > 1:
                calls['several_objects'] += 1
                continue
            oid = ids[0]
            pos = []
            for fid, _, _ in c['members']:
                g = gt[fid]
                k = np.nonzero(g['ids'] == oid)[0]
                if len(k):
                    pos.append(P.transform_boxes(g['box'][k[:1]], g['pose'])[0, :2])
            pos = np.array(pos)
            moved = len(pos) >= 2 and np.max(np.hypot(*(pos[:, None, :] - pos[None, :, :]).transpose(2, 0, 1))) > 0.2
            calls['one_object_moved' if moved else 'one_static_object'] += 1
    out = {'ANALYSIS_ONLY': 'reads Waymo TRAIN labels; never feeds the rule or the GPU-row decision',
           'static_cluster_calls': calls,
           'range_bins': list(P.RANGE_NAMES), 'gt': stats['gt'].tolist(),
           'recall_direct': (stats['rec_direct'] / np.maximum(stats['gt'], 1)).tolist(),
           'recall_direct_S': (stats['rec_S'] / np.maximum(stats['gt'], 1)).tolist(),
           'recall_direct_S_D': (stats['rec_SD'] / np.maximum(stats['gt'], 1)).tolist(),
           'coverage_all_labels': (stats['cov_SD'] / np.maximum(stats['gt'], 1)).tolist(),
           'precision': {k: {'n': v[0].tolist(), 'tp': v[1].tolist(),
                             'prec': (v[1] / np.maximum(v[0], 1)).tolist(),
                             'pooled': float(v[1].sum() / max(v[0].sum(), 1))} for k, v in pr.items()},
           'precision_by_delta': {k: {'n': v[0].tolist(), 'tp': v[1].tolist()} for k, v in pr_delta.items()},
           'ignore_on_vehicle': {k: {'n': int(v[0]), 'on_vehicle': int(v[1])} for k, v in ign.items()},
           'created': time.strftime('%Y-%m-%d %H:%M %Z')}
    if args.direct_only:
        out = {k: v for k, v in out.items() if k not in ('recall_direct_S', 'recall_direct_S_D', 'precision_by_delta',
                                                         'ignore_on_vehicle', 'static_cluster_calls')}
        out['precision'] = {'direct': out['precision']['direct']}
        out['coverage_all_labels'] = out.pop('coverage_all_labels')  # direct labels only in this mode
        out['DIRECT_ONLY'] = 'only 27704 direct pseudo-labels were read against the labels'
    json.dump(out, open(d / 'diagnosis.json', 'w'), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print(f'diagnosis done in {time.time() - t0:.0f} s', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['inspect', 'count', 'merge', 'diagnose'])
    ap.add_argument('--calib', help='calibration.json of the source stage')
    ap.add_argument('--ps', default=PS_27704)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ps_out', default='/home/koyama/data/pseudo_labels/prop27704_v1')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=8)
    ap.add_argument('--context', default=None, metavar='S_H,D_H,q',
                    help='LABEL-FREE context counts at a stated NON-RULE setting (tagged; no augmented file)')
    ap.add_argument('--direct_only', action='store_true',
                    help='diagnose: only 27704\'s direct labels (the pre-declared target-side reference)')
    args = ap.parse_args()
    {'inspect': stage_inspect, 'count': stage_count, 'merge': stage_merge,
     'diagnose': stage_diagnose}[args.stage](args)


if __name__ == '__main__':
    main()
