"""Item 63 (experiments_md 20261003_05 §5 item 63, design fixed 2026-11-27 13:02 JST, before any number): rule A per class and
x-y range bin on 59b's P0 floor pass (label-free), and the per-seed leaf configs. ANALYSIS.

    python3 analysis/item63_range_cuts.py cuts       # CPU job: /home/koyama/data/item63/range_cuts.json
    python3 analysis/item63_range_cuts.py configs    # cfgs/da-ieee-access-analysis/item63/centerpoint-item63-b-bn-rr-s<seed>.yaml
    python3 analysis/item63_range_cuts.py countbal   # item 63′, CPU job: /home/koyama/data/item63/count_balance_cuts.json
    python3 analysis/item63_range_cuts.py configs_cb # item 63′ leaves centerpoint-item63-b-bn-cb-s<seed>.yaml

Bins: item 48's rings by box-centre x-y range, 0-10 / 10-20 / 20-30 / 30-40 / 40-75.2 m, inside rule A's own |x|, |y| <= 75.2
window. A bin with fewer than MIN_BOXES floor-pass boxes of a class takes the class's global rule-A cut (59b's), flagged.
Tests: tests/test_item63_range_cuts.py.
"""
import json
import pickle
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import _init_path  # noqa: F401,E402  (before any pcdet import: the repo's pcdet, never the image's stale copy)

OUT = Path('/home/koyama/data/item63')
P0 = Path('/home/koyama/data/item59')
CUTS_59B = P0 / 'cuts.json'
EDGES = (0.0, 10.0, 20.0, 30.0, 40.0)
CLASSES = ('Car', 'Pedestrian', 'Cyclist')
SEEDS = (666, 667)
MIN_BOXES = 1000
CFG_DIR = 'cfgs/da-ieee-access-analysis/item63'
LEAF_59B = 'cfgs/da-ieee-access-analysis/item59/centerpoint-item59-b-bn-s%d.yaml'


def bins_of(rho):
    import numpy as np
    return np.searchsorted(np.asarray(EDGES), rho, side='right') - 1


def range_rule_a(boxes, global_cuts):
    """{class: dict(cuts per bin, boxes per bin, fallback flags)} from floor-pass boxes (N, 9: ..., label, score)."""
    import numpy as np
    sys.path.insert(0, str(TOOLS / 'analysis' / 'pseudo_label_threshold'))
    from plateau_neg_thresh import RANGE, plateau_neg
    b = boxes[(np.abs(boxes[:, 0]) <= RANGE) & (np.abs(boxes[:, 1]) <= RANGE)]
    rb = bins_of(np.hypot(b[:, 0], b[:, 1]))
    out = {}
    for ci, name in enumerate(CLASSES, start=1):
        cuts, counts, fb = [], [], []
        for k in range(len(EDGES)):
            s = b[(np.abs(b[:, 7]) == ci) & (rb == k), 8]
            counts.append(int(len(s)))
            if len(s) < MIN_BOXES:
                cuts.append(float(global_cuts[name])); fb.append(True)
                continue
            neg, _, _, _ = plateau_neg(s)
            cuts.append(round(float(neg), 3)); fb.append(False)
        out[name] = dict(cuts=cuts, boxes=counts, fallback=fb)
    return out


def cmd_cuts():
    import numpy as np
    g = json.load(open(CUTS_59B))
    res = {}
    for seed in SEEDS:
        ps = pickle.load(open(P0 / ('p0_s%d' % seed) / 'ps_label_e0.pkl', 'rb'))
        boxes = np.concatenate([np.asarray(v['gt_boxes']).reshape(-1, 9) for v in ps.values()])
        gc = {c: g[str(seed)]['rule_a'][c]['neg'] for c in CLASSES}
        res[str(seed)] = dict(frames=len(ps), edges=list(EDGES), rule_a=range_rule_a(boxes, gc), global_cuts=gc)
        # descriptive: kept pseudo-labels per frame by bin at the per-range and at the global cut
        sys.path.insert(0, str(TOOLS / 'analysis' / 'pseudo_label_threshold'))
        from plateau_neg_thresh import RANGE
        b = boxes[(np.abs(boxes[:, 0]) <= RANGE) & (np.abs(boxes[:, 1]) <= RANGE)]
        rb = bins_of(np.hypot(b[:, 0], b[:, 1]))
        car = np.abs(b[:, 7]) == 1
        table = np.asarray(res[str(seed)]['rule_a']['Car']['cuts'])
        res[str(seed)]['car_per_frame'] = dict(
            range_cut=[float(((rb == k) & car & (b[:, 8] >= table[k])).sum() / len(ps)) for k in range(len(EDGES))],
            global_cut=[float(((rb == k) & car & (b[:, 8] >= gc['Car'])).sum() / len(ps)) for k in range(len(EDGES))])
        print(seed, json.dumps(res[str(seed)]['rule_a']), res[str(seed)]['car_per_frame'])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'range_cuts.json', 'w'), indent=1)


def leaf_text(seed, res):
    r = res[str(seed)]['rule_a']
    cuts = [r[c]['cuts'] for c in CLASSES]
    fb = {c: [EDGES[k] for k, f in enumerate(r[c]['fallback']) if f] for c in CLASSES}
    return ('# experiments_md 20261003_05 §5 item 63 (design fixed 2026-11-27 13:02 JST, before any number), seed %d. ANALYSIS:\n'
            '# 59b\'s b_bn (the base leaf, unchanged) with one rule-A cut per class and x-y range bin (item 48\'s rings) from\n'
            '# 59b\'s P0 floor pass (label-free), replacing NEG_THRESH / SCORE_THRESH (no ignore band). Written by\n'
            '# analysis/item63_range_cuts.py configs from /home/koyama/data/item63/range_cuts.json; do not edit by hand.\n'
            '# Bins that fell back to the global cut (fewer than %d boxes): %s\n'
            '_BASE_CONFIG_: %s\n'
            'SELF_TRAIN:\n'
            '    THRESH_BY_RANGE:\n'
            '        EDGES: %s\n'
            '        CUTS: %s\n') % (seed, MIN_BOXES, json.dumps(fb), LEAF_59B % seed, json.dumps(list(EDGES)), json.dumps(cuts))


def leaf_path(seed):
    return TOOLS / CFG_DIR / ('centerpoint-item63-b-bn-rr-s%d.yaml' % seed)


def cmd_configs():
    res = json.load(open(OUT / 'range_cuts.json'))
    for seed in SEEDS:
        p = leaf_path(seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(leaf_text(seed, res))
        print('wrote', p)


# ---------------------------------------------------------------- item 63′: count balance by range (design 13:46 JST)
KITTI = Path('/home/koyama/data/kitti')
SHIFT_Z = 1.70                                   # da_kitti_dataset.yaml SHIFT_COOR [0, 0, 1.70]: the floor pass is in the loader frame
FOV_MARGIN = 5                                   # the evaluation's (kitti_dataset.py, get_fov_flag(..., margin=5))
FLOOR = 0.0001
CB_DIR = CFG_DIR
LEAF_CB = 'centerpoint-item63-b-bn-cb-s%d.yaml'


def fov_mask(centres, calib, img_shape):
    """The evaluation's camera-FOV test on box centres (raw LiDAR frame)."""
    import numpy as np
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    if len(centres) == 0:
        return np.zeros(0, dtype=bool)
    return KittiDataset.get_fov_flag(calib.lidar_to_rect(centres), img_shape, calib, margin=FOV_MARGIN)


def calib_of(split, idx):
    from pcdet.utils import calibration_kitti
    return calibration_kitti.Calibration(KITTI / split / 'calib' / ('%s.txt' % idx))


def per_frame_counts(boxes_by_frame, frames):
    """{class: [mean boxes per frame in each bin]} from {frame: (M, >= 8) boxes with labels in column 7}."""
    import numpy as np
    tot = {c: np.zeros(len(EDGES)) for c in CLASSES}
    for b in boxes_by_frame.values():
        if len(b) == 0:
            continue
        rb = bins_of(np.hypot(b[:, 0], b[:, 1]))
        for ci, c in enumerate(CLASSES, start=1):
            sel = np.abs(b[:, 7]) == ci
            tot[c] += np.bincount(rb[sel], minlength=len(EDGES))[:len(EDGES)]
    return {c: (tot[c] / frames).tolist() for c in CLASSES}


def count_cut(scores, k):
    """The score keeping the k highest of `scores` (ties may keep a few more); every one if k exceeds them; none if k is 0."""
    import numpy as np
    if k <= 0:
        return 1.01
    if k >= len(scores):
        return FLOOR
    return float(np.sort(np.asarray(scores))[::-1][k - 1])


def source_density():
    """Mean annotated boxes per KITTI train frame, per class and bin, inside the camera FOV (the labels P_K trained with)."""
    import pickle
    import numpy as np
    infos = pickle.load(open(KITTI / 'kitti_infos_train.pkl', 'rb'))
    by = {}
    for info in infos:
        a = info['annos']
        names = np.asarray(a['name'])
        keep = np.isin(names, CLASSES)
        g = np.asarray(a['gt_boxes_lidar']).reshape(-1, 7)
        if len(g) != len(names):                        # gt_boxes_lidar holds no DontCare rows
            names = names[names != 'DontCare']
            keep = np.isin(names, CLASSES)
        g = g[keep]
        lab = np.array([CLASSES.index(n) + 1 for n in names[keep]], dtype=np.float64)
        idx = info['point_cloud']['lidar_idx']
        m = fov_mask(g[:, :3], calib_of('training', idx), info['image']['image_shape'])
        by[idx] = np.concatenate([g[m], lab[m, None]], axis=1) if m.any() else np.zeros((0, 8))
    return per_frame_counts(by, len(infos)), len(infos)


def target_fov(ps, infos_test):
    """{frame: floor-pass boxes inside the camera FOV}, raw LiDAR frame (SHIFT_COOR removed), and all boxes."""
    import numpy as np
    shapes = {i['point_cloud']['lidar_idx']: i['image']['image_shape'] for i in infos_test}
    fov, allb = {}, {}
    for fid, v in ps.items():
        b = np.asarray(v['gt_boxes']).reshape(-1, 9).copy()
        b[:, 2] -= SHIFT_Z
        allb[fid] = b
        m = fov_mask(b[:, :3], calib_of('testing', fid), shapes[fid])
        fov[fid] = b[m]
    return fov, allb


def count_balance_cuts(fov, frames, density):
    """{class: cuts per bin}: keep, in each bin, round(density x frames) FOV boxes of the class."""
    import numpy as np
    out = {}
    for ci, c in enumerate(CLASSES, start=1):
        cuts, ks, avail = [], [], []
        sc = {k: [] for k in range(len(EDGES))}
        for b in fov.values():
            if len(b) == 0:
                continue
            sel = np.abs(b[:, 7]) == ci
            rb = bins_of(np.hypot(b[sel, 0], b[sel, 1]))
            for k in range(len(EDGES)):
                sc[k].append(b[sel][rb == k, 8])
        for k in range(len(EDGES)):
            s = np.concatenate(sc[k]) if sc[k] else np.zeros(0)
            n = int(round(density[c][k] * frames))
            cuts.append(round(count_cut(s, n), 4)); ks.append(n); avail.append(int(len(s)))
        out[c] = dict(cuts=cuts, target_boxes=ks, fov_boxes=avail)
    return out


def kept_per_frame(boxes_by_frame, frames, cuts):
    import numpy as np
    out = {c: np.zeros(len(EDGES)) for c in CLASSES}
    for b in boxes_by_frame.values():
        if len(b) == 0:
            continue
        rb = bins_of(np.hypot(b[:, 0], b[:, 1]))
        for ci, c in enumerate(CLASSES, start=1):
            sel = (np.abs(b[:, 7]) == ci) & (b[:, 8] >= np.asarray(cuts[c]['cuts'])[rb])
            out[c] += np.bincount(rb[sel], minlength=len(EDGES))[:len(EDGES)]
    return {c: (out[c] / frames).round(3).tolist() for c in CLASSES}


def cmd_countbal():
    import pickle
    dens, nsrc = source_density()
    infos_test = pickle.load(open(KITTI / 'kitti_infos_test.pkl', 'rb'))
    res = dict(edges=list(EDGES), source_frames=nsrc, source_density=dens)
    for seed in SEEDS:
        ps = pickle.load(open(P0 / ('p0_s%d' % seed) / 'ps_label_e0.pkl', 'rb'))
        fov, allb = target_fov(ps, infos_test)
        cuts = count_balance_cuts(fov, len(ps), dens)
        res[str(seed)] = dict(frames=len(ps), cuts=cuts, kept_fov=kept_per_frame(fov, len(ps), cuts),
                              kept_360=kept_per_frame(allb, len(ps), cuts))
        print(seed, json.dumps({c: cuts[c]['cuts'] for c in CLASSES}), 'kept/frame FOV', res[str(seed)]['kept_fov']['Car'],
              '360', res[str(seed)]['kept_360']['Car'])
    print('source density (FOV, per frame):', json.dumps(dens))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'count_balance_cuts.json', 'w'), indent=1)


def leaf_text_cb(seed, res):
    cuts = [res[str(seed)]['cuts'][c]['cuts'] for c in CLASSES]
    return ('# experiments_md 20261003_05 §5 item 63′ (design fixed 2026-11-27 13:46 JST, before any number), seed %d. ANALYSIS:\n'
            '# 59b\'s b_bn (the base leaf, unchanged) with one cut per class and x-y range bin (item 48\'s rings) at which the\n'
            '# teacher\'s floor-pass pseudo-labels in the camera FOV match the source labels\' density per frame (count balance\n'
            '# by range; source labels and unlabelled target clouds only). Replaces NEG_THRESH / SCORE_THRESH (no ignore band).\n'
            '# Written by analysis/item63_range_cuts.py configs_cb from /home/koyama/data/item63/count_balance_cuts.json.\n'
            '_BASE_CONFIG_: %s\n'
            'SELF_TRAIN:\n'
            '    THRESH_BY_RANGE:\n'
            '        EDGES: %s\n'
            '        CUTS: %s\n') % (seed, LEAF_59B % seed, json.dumps(list(EDGES)), json.dumps(cuts))


def cmd_configs_cb():
    res = json.load(open(OUT / 'count_balance_cuts.json'))
    for seed in SEEDS:
        p = TOOLS / CB_DIR / (LEAF_CB % seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(leaf_text_cb(seed, res))
        print('wrote', p)


# ---------------------------------------------------------------- item 63′: rows, check, readings
ROW_CB = 'P_PL_BN_CB'
PK = {666: 32283, 667: 32284}
REF_57 = '/home/koyama/data/item57/item57b_manifest.json'
REF_58 = '/home/koyama/data/item58/item58_manifest.json'
REF_59B = '/home/koyama/data/item59/item59b_manifest.json'


def parse_jobs(args):
    return {int(x.split('=')[0].lstrip('s')): int(x.split('=')[1]) for x in args}


def manifest_cb(jobs):
    import item38_kitti_rings as i38
    rows = []
    for s_, j in sorted(jobs.items()):
        r = i38.resolve('item63', ROW_CB, s_, j, 'epoch_10')
        if r['status'] == 'ok' and not Path(r['result']).exists():
            r['status'] = 'result.pkl missing'
        rows.append(r)
    for path, keep in ((REF_57, ('P_RS1', 'L_ada', 'O_ada')), (REF_58, ('P_BN_G',)), (REF_59B, ('P_PL_BN',))):
        for r in json.load(open(path)):
            if r['row'] in keep:
                rows.append(dict(r, group='item63'))
    return rows


def readings_cb(rows):
    import item58_bn_bound as m58
    seeds = [m58.per_seed(rows, s_, ROW_CB) for s_ in SEEDS]
    word = m58.reading(seeds)
    d = {}
    for s_ in SEEDS:
        a, b = m58.rings(rows, ROW_CB, s_), m58.rings(rows, 'P_PL_BN', s_)
        d[s_] = None if a is None or b is None else a[3] - b[3]          # 30-40 m
    v = list(d.values())
    if any(x is None for x in v):
        w2 = 'not read'
    elif all(x >= 2.0 for x in v):
        w2 = 'count balance by range helps where 59b fell short'
    elif all(abs(x) < 1.0 for x in v):
        w2 = 'no difference'
    else:
        w2 = 'partly'
    return seeds, word, d, w2


def cmd_read_cb(path):
    rows = json.load(open(path))['rows']
    for r in rows:
        print('%-11s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    seeds, word, d, w2 = readings_cb(rows)
    print('\n== %s against P_RS1: %s' % (ROW_CB, word))
    for s_, x in zip(SEEDS, seeds):
        print('  seed %d: %s' % (s_, 'not read' if x is None else 's_x far %.2f / %.2f, delta %+.2f, R %.2f' % (
            x['s_x'][2], x['s_x'][3], x['delta'], x['R'])))
    print('== %s - P_PL_BN at 30-40 m: %s %s' % (ROW_CB, w2, {k: (None if v is None else round(v, 2)) for k, v in d.items()}))


def cmd_check_cb(dst, jobs):
    import torch
    import item58_bn_bound as m58
    import item59_bn_split as m59
    out = {}
    for s_, j in sorted(jobs.items()):
        c = m58.compare(torch.load(m59.ckpt_of(PK[s_], 152), map_location='cpu')['model_state'],
                        torch.load(m59.ckpt_of(j, 10), map_location='cpu')['model_state'])
        c['ok'] = not c['non_bn_differ'] and not c['missing']
        out['%s s%d' % (ROW_CB, s_)] = dict(c, job=j)
        print('%s s%d (job %d): non-BN tensors differing %d, BN tensors changed %d, key mismatch %d, counters %s -> %s' % (
            ROW_CB, s_, j, len(c['non_bn_differ']), c['bn_changed'], len(c['missing']), c['counters'],
            'ok' if c['ok'] else 'FAILED'))
    json.dump(out, open(dst, 'w'), indent=1)
    sys.exit(0 if all(c['ok'] for c in out.values()) else 1)


def main():
    a = sys.argv[1:]
    if a[:1] == ['manifest_cb'] and len(a) >= 4 and a[2] == '--jobs':
        rows = manifest_cb(parse_jobs(a[3:]))
        json.dump(rows, open(a[1], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r.get('job'), r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['read_cb'] and len(a) == 2:
        return cmd_read_cb(a[1])
    if a[:1] == ['check_cb'] and len(a) >= 4 and a[2] == '--jobs':
        return cmd_check_cb(a[1], parse_jobs(a[3:]))
    if a == ['cuts']:
        return cmd_cuts()
    if a == ['configs']:
        return cmd_configs()
    if a == ['countbal']:
        return cmd_countbal()
    if a == ['configs_cb']:
        return cmd_configs_cb()
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
