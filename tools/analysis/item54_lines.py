"""Item 54 (experiments_md 20261003_05 §5 item 54, design fixed 2026-11-26 22:53 JST, experiments_md commit 86a0173, before any
number): is the dense-to-sparse loss on S' set by the lines on the box rather than by range? ANALYSIS (KITTI labels score only).

    python3 analysis/item54_lines.py count OUT_DIR [--workers 8]                 # stage A, CPU: lines per box, train distributions
    python3 analysis/item54_lines.py score OUT_DIR [--real-kernel] [--workers 4]  # stage B, GPU node for --real-kernel: the cells
    python3 analysis/item54_lines.py read OUT_DIR/item54_ap.json                  # the fixed readings

S' is the KITTI val scan thinned exactly as item 47's eval loader thins it (kitti_factor_swap.apply_points with the item 47 P
config's EVAL_FACTOR_SWAP and its frame-seeded draw; checked point for point on every frame). The lines on a box are the distinct
laser rings (kitti_rings.rings_from_order on the raw scan) among the S' points inside it that lie >= MIN_HEIGHT above its bottom.
A detection takes the line count and range of the Car GT it overlaps most in BEV (IoU > 0), otherwise its own. A cell's view is
item 38's: a Car GT outside the cell becomes Van (ignored), a detection outside it is dropped. Rows: item 48's 16 reverse rows
(/home/koyama/data/item47/item48_manifest_rev.json); a row whose pooled value differs from its logged one by more than 0.02 is
not read. Tests: tests/test_item54_lines.py.
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

MANIFEST = '/home/koyama/data/item47/item48_manifest_rev.json'
EVAL_CFG = 'cfgs/da-ieee-access-analysis/item47/centerpoint-item47-kitti-p-on-nuscenes-lattice.yaml'
TRAIN_INFOS = TOOLS.parent / 'data/kitti/kitti_infos_train.pkl'
MIN_HEIGHT = 0.3
TOL = 0.02
INF = float('inf')
RINGS = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 70.4)]   # item 48's rings
LINE_BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, INF)]        # 0, 1, 2, 3, 4-5, 6-9, >= 10
CLASSES = {'few': (1, 3), 'many': (3, INF)}                                     # 1-2 lines, >= 3 lines
MIN_GT, MIN_P = 100, 10.0
D1, D1_ABSENT, D2, D3 = 0.20, 0.05, 0.10, 0.15


def row_key(r):
    return '%s_%s_s%d' % (r['row'], r['stats'], r['seed'])


def cells():
    """(name, line range, distance range) of every view scored per row."""
    out = [('pooled', (0, INF), (0.0, INF))]
    out += [('lines %s' % _lname(b), b, (0.0, INF)) for b in LINE_BINS]
    out += [('class %s' % c, b, (0.0, INF)) for c, b in CLASSES.items()]
    out += [('ring %g-%g' % g, (0, INF), g) for g in RINGS]
    out += [('ring %g-%g %s' % (g + (c,)), b, g) for g in RINGS for c, b in CLASSES.items()]
    return out


def _lname(b):
    lo, hi = b
    return ('>=%d' % lo) if hi == INF else ('%d' % lo if hi == lo + 1 else '%d-%d' % (lo, hi - 1))


# ---------------------------------------------------------------- lines on a box (stage A)

def box_lines(points, ring, boxes, min_height=MIN_HEIGHT, inside_fn=None):
    """Distinct rings among `points` (n, >=3, LiDAR frame) inside each box (m, 7: x y z dx dy dz heading, z the centre) that lie at
    least `min_height` above the box bottom."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    m = len(boxes)
    if m == 0 or len(points) == 0:
        return np.zeros(m, dtype=np.int64)
    if inside_fn is None:
        from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu as inside_fn
    inside = np.asarray(inside_fn(np.ascontiguousarray(points[:, :3], dtype=np.float32), boxes)) > 0   # (m, n)
    above = points[None, :, 2] >= (boxes[:, 2] - boxes[:, 5] / 2 + min_height)[:, None]
    hit = inside & above
    ring = np.asarray(ring)
    return np.array([len(np.unique(ring[h])) for h in hit], dtype=np.int64)


def assign(own_lines, own_range, iou, gt_lines, gt_range):
    """A detection takes the line count and range of the Car GT it overlaps most in BEV (IoU > 0), otherwise its own."""
    own_lines, own_range = np.asarray(own_lines), np.asarray(own_range, dtype=np.float64)
    if len(own_lines) == 0 or len(gt_lines) == 0:
        return own_lines.copy(), own_range.copy()
    iou = np.asarray(iou).reshape(len(own_lines), len(gt_lines))
    j = iou.argmax(1)
    has = iou.max(1) > 0
    return np.where(has, np.asarray(gt_lines)[j], own_lines), np.where(has, np.asarray(gt_range, dtype=np.float64)[j], own_range)


def _sprime(points, fid, swap):
    from pcdet.datasets.kitti import kitti_factor_swap
    from pcdet.datasets.kitti.kitti_rings import eval_ring_thin_mask, rings_from_order
    ring, _ = rings_from_order(points[:, :3])
    keep = eval_ring_thin_mask(points, swap['RING_THIN'], kitti_factor_swap.frame_rng(fid, 'RING_THIN'))
    return ring, keep


def _bev_iou(a, b):
    import torch
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return boxes_bev_iou_cpu(torch.from_numpy(np.asarray(a, np.float32)[:, :7]),
                             torch.from_numpy(np.asarray(b, np.float32)[:, :7])).numpy()


_A = {}


def _count_init(swap, train_swap):
    import _init_path  # noqa: F401
    _A['swap'], _A['train_swap'] = swap, train_swap


def _count_val(task):
    from pcdet.datasets.kitti import kitti_factor_swap
    import kitti_eval_cpu
    fid, a, dets = task
    pts = np.fromfile(kitti_eval_cpu.VELODYNE / ('%s.bin' % fid), dtype=np.float32).reshape(-1, 4)
    ring, keep = _sprime(pts, fid, _A['swap'])
    ref = kitti_factor_swap.apply_points(pts, _A['swap'], fid)
    assert np.array_equal(pts[keep], ref), '%s: S\' differs from the eval loader\'s cut' % fid
    sp, sr = pts[keep], ring[keep]
    n = len(a['gt_boxes_lidar'])
    names = np.asarray(a['name'][:n])
    boxes = np.asarray(a['gt_boxes_lidar'], dtype=np.float32).reshape(-1, 7)
    car = names == 'Car'
    gt_lines = np.full(n, -1, dtype=np.int64)
    gt_lines[car] = box_lines(sp, sr, boxes[car])
    gt_lines_full = np.full(n, -1, dtype=np.int64)
    gt_lines_full[car] = box_lines(pts, ring, boxes[car])
    gt_range = np.hypot(boxes[:, 0], boxes[:, 1]) if n else np.zeros(0)
    out = {'fid': fid, 'gt_lines': gt_lines, 'gt_lines_full': gt_lines_full, 'gt_range': gt_range, 'kept': int(keep.sum()),
           'dets': {}}
    for k, db in dets.items():
        db = np.asarray(db, dtype=np.float32)
        db = db.reshape(len(db), -1)[:, :7] if len(db) else np.zeros((0, 7), np.float32)
        own = box_lines(sp, sr, db)
        own_r = np.hypot(db[:, 0], db[:, 1]) if len(db) else np.zeros(0)
        iou = _bev_iou(db, boxes[car])
        lines, rng = assign(own, own_r, iou, gt_lines[car], gt_range[car])
        out['dets'][k] = {'lines': lines, 'range': rng, 'own_lines': own}
    return out


def _count_train(info):
    fid = info['point_cloud']['lidar_idx']
    a = info['annos']
    n = len(a['gt_boxes_lidar'])
    car = np.asarray(a['name'][:n]) == 'Car'
    if not car.any():
        return None
    import kitti_eval_cpu
    pts = np.fromfile(kitti_eval_cpu.VELODYNE / ('%s.bin' % fid), dtype=np.float32).reshape(-1, 4)
    ring, keep = _sprime(pts, fid, _A['train_swap'])
    boxes = np.asarray(a['gt_boxes_lidar'], dtype=np.float32).reshape(-1, 7)[car]
    return {'range': np.hypot(boxes[:, 0], boxes[:, 1]), 'full': box_lines(pts, ring, boxes),
            'thinned': box_lines(pts[keep], ring[keep], boxes)}


def load_swaps():
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    cfg = cfg_from_yaml_file(EVAL_CFG, EasyDict())
    swap = cfg.DATA_CONFIG_TAR.EVAL_FACTOR_SWAP
    train_swap = cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP
    assert dict(swap['RING_THIN']) == dict(train_swap['RING_THIN']), (swap, train_swap)   # item 47: P's own training cut
    return {'RING_THIN': dict(swap['RING_THIN'])}, {'RING_THIN': dict(train_swap['RING_THIN'])}


def distribution(lines, rng):
    """Per ring of item 48, the share of boxes per line bin (and the count)."""
    lines, rng = np.asarray(lines), np.asarray(rng)
    out = []
    for lo, hi in RINGS:
        m = (rng >= lo) & (rng < hi)
        out.append({'ring': [lo, hi], 'n': int(m.sum()),
                    'share': {_lname(b): (float(((lines[m] >= b[0]) & (lines[m] < b[1])).mean()) if m.any() else None)
                              for b in LINE_BINS}})
    return out


def cmd_count(args):
    import _init_path  # noqa: F401
    import kitti_eval_cpu
    os.chdir(TOOLS)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    swap, train_swap = load_swaps()
    rows = [r for r in json.load(open(MANIFEST)) if r['status'] == 'ok']
    assert len(rows) == 16, len(rows)
    dets = {row_key(r): {d['frame_id']: d['boxes_lidar'] for d in pickle.load(open(r['result'], 'rb'))} for r in rows}
    infos = pickle.load(open(kitti_eval_cpu.INFOS, 'rb'))
    tasks = [(i['point_cloud']['lidar_idx'], i['annos'], {k: v[i['point_cloud']['lidar_idx']] for k, v in dets.items()})
             for i in infos]
    with mp.get_context('spawn').Pool(args.workers, initializer=_count_init, initargs=(swap, train_swap)) as pool:
        val = pool.map(_count_val, tasks, chunksize=8)
        train = [t for t in pool.map(_count_train, pickle.load(open(TRAIN_INFOS, 'rb')), chunksize=8) if t is not None]
    pickle.dump({'rows': rows, 'swap': swap, 'min_height': MIN_HEIGHT, 'val': val}, open(out_dir / 'item54_counts.pkl', 'wb'))
    official = kitti_eval_cpu.load_official_eval()
    mod_lines, mod_range = [], []
    for info, v in zip(infos, val):
        a = info['annos']
        n = len(a['gt_boxes_lidar'])
        empty = {'name': np.array([]), 'bbox': np.zeros((0, 4)), 'boxes_lidar': np.zeros((0, 7))}
        _, ignored, _, _ = official.clean_data(a, empty, 0, 1)
        valid = (np.array(ignored[:n]) == 0) & (v['gt_lines'] >= 0)
        mod_lines.append(v['gt_lines'][valid]); mod_range.append(v['gt_range'][valid])
    tr = lambda k: np.concatenate([t[k] for t in train])  # noqa: E731
    desc = {'train O (full scan)': distribution(tr('full'), tr('range')),
            'train P (thinned)': distribution(tr('thinned'), tr('range')),
            'val moderate on S\'': distribution(np.concatenate(mod_lines), np.concatenate(mod_range))}
    json.dump(desc, open(out_dir / 'item54_distributions.json', 'w'), indent=1)
    print('frames %d, S\' check passed on every frame; train frames with Car %d' % (len(val), len(train)))
    for name, d in desc.items():
        print(name)
        for g in d:
            print('  %g-%g m n %d | %s' % (g['ring'][0], g['ring'][1], g['n'], ' '.join(
                '%s:%.2f' % (k, s) for k, s in g['share'].items() if s is not None)))


# ---------------------------------------------------------------- cells (stage B)

def cell_view(gt_anno, dt_anno, gt_lines, gt_range, dt_lines, dt_range, lines_b, range_b):
    """Copies of one frame's GT and detections restricted to the cell (item 38's view; DontCare rows past gt_boxes_lidar kept)."""
    gt, dt = copy.deepcopy(gt_anno), copy.deepcopy(dt_anno)
    n = len(gt['gt_boxes_lidar'])
    names = np.array(gt['name'], dtype=object)
    assert all(nm == 'DontCare' for nm in names[n:]), 'GT rows without a LiDAR box must be DontCare'
    (llo, lhi), (rlo, rhi) = lines_b, range_b
    gl, gr = np.asarray(gt_lines), np.asarray(gt_range)
    inside = (gl >= llo) & (gl < lhi) & (gr >= rlo) & (gr < rhi)
    names[:n][(names[:n] == 'Car') & ~inside] = 'Van'
    gt['name'] = names.astype(str)
    nd = len(dt['name'])
    dl, dr = np.asarray(dt_lines), np.asarray(dt_range)
    keep = (dl >= llo) & (dl < lhi) & (dr >= rlo) & (dr < rhi) if nd else np.zeros(0, bool)
    for k, v in dt.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == nd:
            dt[k] = v[keep]
    return gt, dt


_B = {}


def _score_init(real_kernel, counts_path):
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    import _init_path  # noqa: F401
    import kitti_eval_cpu
    _B['official'] = kitti_eval_cpu.load_official_eval(real_kernel=real_kernel)
    _B['counts'] = pickle.load(open(counts_path, 'rb'))
    _B['infos'] = pickle.load(open(kitti_eval_cpu.INFOS, 'rb'))


def _score(row):
    official, counts, infos = _B['official'], _B['counts'], _B['infos']
    k = row_key(row)
    dets = {d['frame_id']: d for d in pickle.load(open(row['result'], 'rb'))}
    val = {v['fid']: v for v in counts['val']}
    out = []
    for name, lb, rb in cells():
        gt, dt = [], []
        for info in infos:
            fid = info['point_cloud']['lidar_idx']
            v, d = val[fid], dets[fid]
            g, e = cell_view(info['annos'], d, v['gt_lines'], v['gt_range'], v['dets'][k]['lines'], v['dets'][k]['range'], lb, rb)
            gt.append(g); dt.append(e)
        n_gt = int(sum(official.clean_data(g, e, 0, 1)[0] for g, e in zip(gt, dt)))
        cell = {'cell': name, 'lines': [lb[0], lb[1]], 'range': [rb[0], rb[1]], 'n_gt': n_gt, 'bev': None}
        if n_gt:
            _, ap = official.get_official_eval_result(gt, dt, ['Car'])
            cell['bev'] = float(ap['Car_bev/moderate_R40'])
        out.append(cell)
    pooled = out[0]['bev']
    diff = None if row['logged_bev'] is None else pooled - row['logged_bev']
    print('%s pooled %.4f logged %s' % (k, pooled, row['logged_bev']), flush=True)
    return dict(row, cells=out, diff=diff, reproduced=bool(diff is not None and abs(diff) <= TOL))


def cmd_score(args):
    out_dir = Path(args.out_dir)
    counts = out_dir / 'item54_counts.pkl'
    rows = pickle.load(open(counts, 'rb'))['rows']
    with mp.get_context('spawn').Pool(args.workers, initializer=_score_init, initargs=(args.real_kernel, str(counts))) as pool:
        res = pool.map(_score, rows, chunksize=1)
    json.dump({'kernel': 'cuda' if args.real_kernel else 'cpu', 'cells': [c[0] for c in cells()], 'rows': res},
              open(out_dir / 'item54_ap.json', 'w'), indent=1, default=str)
    print('wrote', out_dir / 'item54_ap.json')


# ---------------------------------------------------------------- readings

def table(rows, stats):
    """{cell: {seed: {'O':, 'P':, 'n_gt':}}} for reproduced rows of `stats`."""
    t = {}
    for r in rows:
        if r['stats'] != stats or not r.get('reproduced'):
            continue
        for c in r['cells']:
            e = t.setdefault(c['cell'], {}).setdefault(r['seed'], {})
            e[r['row']] = c['bev']
            e['n_gt'] = c['n_gt']
    return t


def q_of(t, cell, seeds=(666, 667)):
    """O / P at each seed if the cell is read (>= MIN_GT moderate GT, P >= MIN_P at both seeds, O and P present), else None."""
    e = t.get(cell)
    if e is None or any(s not in e for s in seeds):
        return None
    for s in seeds:
        x = e[s]
        if x.get('O') is None or x.get('P') is None or x['n_gt'] < MIN_GT or x['P'] < MIN_P:
            return None
    return [e[s]['O'] / e[s]['P'] for s in seeds]


def reading1(t):
    few, many = q_of(t, 'class few'), q_of(t, 'class many')
    if few is None or many is None:
        return 'not read', None
    d = [m - f for m, f in zip(many, few)]
    if all(x >= D1 for x in d):
        return 'present', d
    if all(x <= D1_ABSENT for x in d):
        return 'absent', d
    return 'unclear', d


def reading2(t):
    per = {}
    for g in RINGS:
        name = 'ring %g-%g' % g
        few, many = q_of(t, name + ' few'), q_of(t, name + ' many')
        if few is None or many is None:
            continue
        d = [m - f for m, f in zip(many, few)]
        per[name] = (all(x >= D2 for x in d), d)
    if not per:
        return 'not testable', per
    ok = [v[0] for v in per.values()]
    return ('present' if all(ok) else 'absent' if not any(ok) else 'mixed'), per


def reading3(t):
    out = {}
    for c in CLASSES:
        qs = {('%g-%g' % g): q_of(t, 'ring %g-%g %s' % (g + (c,))) for g in RINGS}
        qs = {k: v for k, v in qs.items() if v is not None}
        if len(qs) < 2:
            out[c] = ('not testable', qs)
            continue
        spread = [max(v[i] for v in qs.values()) - min(v[i] for v in qs.values()) for i in range(2)]
        out[c] = ('range-free' if all(x <= D3 for x in spread) else 'range also matters', dict(qs, spread=spread))
    return out


def combined(r1, r2, r3):
    testable = [v[0] for v in r3.values() if v[0] != 'not testable']
    if r1 == 'present' and r2 == 'present' and testable and all(x == 'range-free' for x in testable):
        return 'line count, not range'
    if r1 == 'present' and any(x == 'range also matters' for x in testable):
        return 'lines and range'
    if r1 == 'absent':
        return 'not lines'
    return 'as measured (1 %s, 2 %s, 3 %s)' % (r1, r2, ', '.join('%s %s' % (c, v[0]) for c, v in r3.items()))


def cmd_read(path):
    d = json.load(open(path))
    rows = d['rows']
    for r in rows:
        if not r.get('reproduced'):
            print('NOT REPRODUCED (not read): %s pooled %.4f logged %s' % (row_key(r), r['cells'][0]['bev'], r['logged_bev']))
    names = d['cells']
    for stats in ('none', 'target'):
        print('\n== %s' % stats)
        for r in sorted([r for r in rows if r['stats'] == stats], key=lambda r: ('OPLC'.index(r['row'][0]), r['seed'])):
            print('%-3s s%d %s' % (r['row'], r['seed'], ' | '.join(
                '%s %s (%d)' % (c['cell'], '-' if c['bev'] is None else '%.2f' % c['bev'], c['n_gt']) for c in r['cells'])))
        t = table(rows, stats)
        for name in names:
            q = q_of(t, name)
            if q is not None:
                print('  q %-22s %.3f / %.3f' % (name, q[0], q[1]))
        r1, d1 = reading1(t)
        r2, per2 = reading2(t)
        r3 = reading3(t)
        print('reading 1 (drop below 3 lines): %s %s' % (r1, d1))
        print('reading 2 (lines within a ring): %s %s' % (r2, {k: v[1] for k, v in per2.items()}))
        print('reading 3 (range at fixed lines): %s' % {c: v for c, v in r3.items()})
        print('combined: %s%s' % (combined(r1, r2, r3), '  <- the reading' if stats == 'none' else ''))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('count'); c.add_argument('out_dir'); c.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('score'); s.add_argument('out_dir'); s.add_argument('--workers', type=int, default=4)
    s.add_argument('--real-kernel', action='store_true')
    r = sub.add_parser('read'); r.add_argument('path')
    args = p.parse_args()
    if args.cmd == 'count':
        cmd_count(args)
    elif args.cmd == 'score':
        cmd_score(args)
    else:
        cmd_read(args.path)


if __name__ == '__main__':
    main()
