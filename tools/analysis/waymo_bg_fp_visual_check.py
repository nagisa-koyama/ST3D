"""Blinded visual check of the confident background false positives on Waymo val (CPU, ANALYSIS).

experiments_md 20261017_13 §6 holds the set, the checks, the categories and the readings, pre-declared (§6.0-6.3)
before this ran. ANALYSIS ONLY: it reads Waymo val labels to diagnose, never to choose a setting.

The set: detections that `waymo_ranking_loss_fp_taxonomy.py` classed as background (with or without points) and that
score above the model's own IoU-0.7 TP median, read from its `pass.pkl` (the matching is not recomputed); the boxes come
from each model's `result.pkl` in the pass's order, and every selected score must equal the pass's.

`render` (one Slurm CPU job):
- §6.1 checks for every box: (a) temporal labels, in the world frame (each frame's `pose`), any GT of any class at BEV
  IoU >= 0.3 in the same sequence within +-10 frames, Vehicle hits flagged static if the hit object's world centre
  spreads < 1 m over the window; (b) the other model's best box at BEV IoU >= 0.5 in the same frame, any score;
  (c) l / w / h, points in the box and in the box enlarged 1.5x, and the bottom height above the local ground;
- blinded contact sheets (6 panels each) for the visual set: BEV +-12 m coloured by height above the local ground with
  every GT box and the FP box, side and end views of the points inside the 1.5x box with the ground line. A panel shows
  its blind ID, range, points in the box and l / w / h only;
- `key.csv` and `checks.csv` are NOT to be read until `classification.csv` is complete. The job prints counts only.

`unblind`: joins key, checks and `classification.csv` (blind_id, category, note) into `tables.md`.

    python analysis/waymo_bg_fp_visual_check.py render --pass_pkl <.../full/pass.pkl> --out <dir> [--workers 6]
    python analysis/waymo_bg_fp_visual_check.py unblind --out <dir>
"""
import argparse
import csv
import math
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import waymo_ranking_loss_fp_taxonomy as T  # noqa: E402  (sets sys.path, imports _init_path)

VISUAL = ('ours27596', 'oracle')            # blinded together
CROSS = {'ours27596': 'oracle', 'oracle': 'ours27596', 'pl27704': 'oracle'}
CATEGORIES = ('V', 'Vpart', 'S', 'G', 'P', 'U')
TEMPORAL_HALF, TEMPORAL_IOU, STATIC_M = 10, 0.3, 1.0
CROSS_IOU, ENLARGE, BAND_M, LOW_FRAC, MIN_BAND_PTS = 0.5, 1.5, 2.0, 0.2, 5
BEV_HALF, PANELS_PER_SHEET, SEED = 12.0, 6, 20261018
GT_COLOURS = {'Vehicle': 'tab:blue', 'Pedestrian': 'tab:green', 'Cyclist': 'tab:orange', 'Sign': 'tab:purple'}


# ----------------------------------------------------------------------------------------------- numbers (unit-tested)
def select_confident_background(det):
    """Indices of background FPs (with points or empty) scoring above the model's own IoU-0.7 TP median; the median."""
    tp = det['gt7'] >= 0
    med = float(np.median(det['score'][tp]))
    sel = np.isin(det['code'], (T.BG_PTS, T.BG_EMPTY)) & (det['score'] > med)
    return np.where(sel)[0], med


def to_world(boxes, pose):
    """(x, y, z, dx, dy, dz, heading) boxes from a frame's vehicle frame to the world frame with its 4x4 pose."""
    b = np.array(boxes, np.float64).reshape(-1, 7)
    c = np.c_[b[:, :3], np.ones(len(b))] @ np.asarray(pose, np.float64).T
    b[:, :3] = c[:, :3]
    b[:, 6] = b[:, 6] + math.atan2(pose[1][0], pose[0][0])
    return b


def window_frames(seq_frames, sample_idx, half=TEMPORAL_HALF):
    """[(offset, frame record)] for the stored frames of one sequence within +-half of sample_idx (itself included)."""
    return [(k - sample_idx, seq_frames[k]) for k in range(sample_idx - half, sample_idx + half + 1) if k in seq_frames]


def temporal_check(box, pose, window, thr=TEMPORAL_IOU, static_m=STATIC_M):
    """§6.1(a). box: one vehicle-frame box of the frame with `pose`; window: window_frames output, each record a dict
    with pose, names, boxes (vehicle frame) and ids. IoU in the world frame, about the FP box's world centre."""
    fw = to_world(box, pose)[0]
    ref = fw[:3].copy()
    fw[:3] -= ref
    veh_hit_ids, other_hit, nearest = set(), False, None
    centres = {}  # Vehicle id -> world centres over the window
    for off, rec in window:
        if len(rec['boxes']) == 0:
            continue
        gw = to_world(rec['boxes'], rec['pose'])
        gw[:, :3] -= ref
        for k, n in enumerate(rec['names']):
            if n == 'Vehicle':
                centres.setdefault(rec['ids'][k], []).append(gw[k, :2])
        iou = T.bev_iou(fw[None].astype(np.float32), gw.astype(np.float32))[0]
        for k in np.where(iou >= thr)[0]:
            if rec['names'][k] == 'Vehicle':
                veh_hit_ids.add(rec['ids'][k])
                nearest = abs(off) if nearest is None else min(nearest, abs(off))
            else:
                other_hit = True
    spreads = [float(np.hypot(*np.ptp(np.array(centres[i]), axis=0))) for i in veh_hit_ids]
    labelled = [len(centres[i]) for i in veh_hit_ids]
    return dict(veh_hit=bool(veh_hit_ids), veh_nearest=-1 if nearest is None else int(nearest),
                veh_static=bool(spreads) and min(spreads) < static_m, other_hit=other_hit,
                veh_min_spread=min(spreads) if spreads else float('nan'),
                veh_labelled_frames=max(labelled) if labelled else 0)


def cross_check(box, other_boxes, other_scores, thr=CROSS_IOU):
    """§6.1(b): the other model's best BEV IoU with the box, and its best score among boxes at IoU >= thr."""
    if len(other_boxes) == 0:
        return dict(cross_iou=0.0, cross_hit=False, cross_score=float('nan'))
    iou = T.bev_iou(np.asarray(box, np.float32)[None], np.asarray(other_boxes, np.float32))[0]
    hit = iou >= thr
    return dict(cross_iou=float(iou.max()), cross_hit=bool(hit.any()),
                cross_score=float(np.max(other_scores[hit])) if hit.any() else float('nan'))


def local_frame(points, box):
    dx, dy = points[:, 0] - box[0], points[:, 1] - box[1]
    c, s = math.cos(box[6]), math.sin(box[6])
    return dx * c + dy * s, -dx * s + dy * c


def local_ground(points, box, band=BAND_M, low_frac=LOW_FRAC, min_pts=MIN_BAND_PTS):
    """Median z of the lowest `low_frac` (at least min_pts) of the points in the `band`-m BEV band around the box
    footprint (inside the footprint widened by `band` on each side, outside the footprint); NaN below min_pts."""
    if len(points) == 0:
        return float('nan')
    lx, ly = local_frame(points, box)
    foot = (np.abs(lx) <= box[3] / 2) & (np.abs(ly) <= box[4] / 2)
    wide = (np.abs(lx) <= box[3] / 2 + band) & (np.abs(ly) <= box[4] / 2 + band)
    z = np.sort(points[wide & ~foot, 2])
    if len(z) < min_pts:
        return float('nan')
    return float(np.median(z[:max(min_pts, int(math.ceil(low_frac * len(z))))]))


def enlarged(box, f=ENLARGE):
    b = np.array(box, np.float64).copy()
    b[3:6] *= f
    return b


def geometry(points, tree, box):
    """§6.1(c)."""
    g = local_ground(points, box)
    n = int(T.points_in_boxes_count(points, np.asarray(box)[None], tree)[0])
    n15 = int(T.points_in_boxes_count(points, enlarged(box)[None], tree)[0])
    return dict(l=float(box[3]), w=float(box[4]), h=float(box[5]), ground=g,
                bottom_above_ground=float(box[2] - box[5] / 2 - g) if np.isfinite(g) else float('nan'),
                pts=n, pts15=n15, range=float(np.hypot(box[0], box[1])))


# --------------------------------------------------------------------------------------------------------- inputs
ITEMS, SEQ, SHEETS = None, None, None  # filled in the parent before the pool forks


def load_sequences():
    """{sequence: {sample_idx: {pose, names, boxes, ids}}} for every stored val frame (labels read for diagnosis)."""
    infos = pickle.load(open(T.INFOS, 'rb'))
    seq = {}
    for inf in infos:
        a = inf['annos']
        names = np.asarray(a['name'])
        boxes = np.asarray(a['gt_boxes_lidar'], np.float32).reshape(len(names), -1)[:, :7] if len(names) \
            else np.zeros((0, 7), np.float32)
        seq.setdefault(inf['point_cloud']['lidar_sequence'], {})[int(inf['point_cloud']['sample_idx'])] = dict(
            pose=np.asarray(inf['pose'], np.float64), names=names, boxes=boxes, ids=np.asarray(a['obj_ids']))
    return seq


def build_items(pass_pkl):
    passed = pickle.load(open(pass_pkl, 'rb'))
    runs = [tuple(r) for r in passed['runs']]
    frames, models, _ = T.load_inputs(runs, passed.get('max_frames'))
    labels = passed['labels']
    items = []
    for mi, label in enumerate(labels):
        det = passed['models'][label]['det']
        idx, med = select_confident_background(det)
        start = np.concatenate([[0], np.cumsum([len(s) for _, s in models[mi]])])
        for g in idx:
            fi = int(det['frame'][g])
            j = int(g - start[fi])
            b, s = models[mi][fi]
            assert abs(s[j] - det['score'][g]) < 1e-9, f'{label}: score mismatch at {g}'
            o = labels.index(CROSS[label])
            items.append(dict(model=label, fi=fi, j=j, score=float(s[j]), box=b[j].astype(np.float64),
                              fid=frames[fi]['fid'], seq=frames[fi]['seq'], idx=frames[fi]['idx'], tp_median=med,
                              **cross_check(b[j], models[o][fi][0], models[o][fi][1])))
        print(f'{label}: {len(idx)} confident background FPs (TP median {med:.3f})', flush=True)
    return items, len(frames)


# -------------------------------------------------------------------------------------------------------- drawing
def _corners(box):
    c, s = math.cos(box[6]), math.sin(box[6])
    hl, hw = box[3] / 2, box[4] / 2
    pts = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]])
    return np.c_[box[0] + pts[:, 0] * c - pts[:, 1] * s, box[1] + pts[:, 0] * s + pts[:, 1] * c]


def draw_panel(fig, spec, item, points, gt):
    import matplotlib.pyplot as plt  # noqa: F401
    box, geo = item['box'], item['geo']
    sub = spec.subgridspec(2, 2, width_ratios=[1.25, 1], wspace=0.25, hspace=0.35)
    ax = fig.add_subplot(sub[:, 0])
    x0, y0 = box[0], box[1]
    win = (np.abs(points[:, 0] - x0) <= BEV_HALF) & (np.abs(points[:, 1] - y0) <= BEV_HALF)
    p = points[win]
    g = geo['ground'] if np.isfinite(geo['ground']) else 0.0
    ax.scatter(p[:, 0] - x0, p[:, 1] - y0, c=np.clip(p[:, 2] - g, -0.5, 3.0), s=0.3, cmap='viridis', vmin=-0.5,
               vmax=3.0, linewidths=0)
    for n, b in zip(gt['names'], gt['boxes']):
        if abs(b[0] - x0) < BEV_HALF + 5 and abs(b[1] - y0) < BEV_HALF + 5:
            cc = _corners(b)
            ax.plot(cc[:, 0] - x0, cc[:, 1] - y0, color=GT_COLOURS.get(str(n), 'grey'), lw=1.0)
    cc = _corners(box)
    ax.plot(cc[:, 0] - x0, cc[:, 1] - y0, color='red', lw=1.8)
    d = -np.array([x0, y0]) / max(np.hypot(x0, y0), 1e-6)  # towards the sensor
    ax.annotate('', xy=(d[0] * 3, d[1] * 3), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.set_xlim(-BEV_HALF, BEV_HALF)
    ax.set_ylim(-BEV_HALF, BEV_HALF)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=6)
    ax.set_title(f"{item['bid']}  r {geo['range']:.1f} m  pts {geo['pts']}  l/w/h {geo['l']:.1f}/{geo['w']:.1f}/"
                 f"{geo['h']:.1f}", fontsize=8, loc='left')
    big = enlarged(box)
    lx, ly = local_frame(points, box)  # display selection only; the counts in geo use pcdet's rule
    m = (np.abs(lx) < big[3] / 2) & (np.abs(ly) < big[4] / 2) & (np.abs(points[:, 2] - box[2]) <= big[5] / 2)
    zb, zt = box[2] - box[5] / 2, box[2] + box[5] / 2
    for k, (coord, half, name) in enumerate(((lx, box[3] / 2, 'side (along length)'), (ly, box[4] / 2, 'end'))):
        a = fig.add_subplot(sub[k, 1])
        a.scatter(coord[m], points[m, 2], s=1.5, c='k', linewidths=0)
        a.plot([-half, half, half, -half, -half], [zb, zb, zt, zt, zb], color='red', lw=1.2)
        if np.isfinite(geo['ground']):
            a.axhline(geo['ground'], color='tab:brown', ls='--', lw=1.0)
        lo = min(box[2] - big[5] / 2, (geo['ground'] if np.isfinite(geo['ground']) else zb) - 0.3)
        a.set_xlim(-half * ENLARGE, half * ENLARGE)
        a.set_ylim(lo, box[2] + big[5] / 2)
        a.set_aspect('equal')
        a.tick_params(labelsize=6)
        a.set_title(name, fontsize=7, loc='left')


def _render_sheet(si):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_png, ids = SHEETS[si]
    fig = plt.figure(figsize=(18, 15), dpi=100)
    grid = fig.add_gridspec(3, 2, wspace=0.12, hspace=0.25)
    geos = {}
    for k, bid in enumerate(ids):
        item = ITEMS[bid]
        points, tree = T.load_points(item['seq'], item['idx'])
        item['geo'] = geometry(points, tree, item['box'])
        geos[bid] = item['geo']
        rec = SEQ[item['seq']][item['idx']]
        draw_panel(fig, grid[k // 2, k % 2], item, points, rec)
    fig.savefig(out_png)
    plt.close(fig)
    return si, geos


def _geometry_only(bid):
    item = ITEMS[bid]
    points, tree = T.load_points(item['seq'], item['idx'])
    return bid, geometry(points, tree, item['box'])


def render(pass_pkl, out, workers):
    global ITEMS, SEQ, SHEETS
    out = Path(out)
    (out / 'sheets').mkdir(parents=True, exist_ok=True)
    items, n_frames = build_items(pass_pkl)
    SEQ = load_sequences()
    for it in items:
        it.update(temporal_check(it['box'], SEQ[it['seq']][it['idx']]['pose'],
                                 window_frames(SEQ[it['seq']], it['idx'])))
    rng = np.random.default_rng(SEED)
    vis = [it for it in items if it['model'] in VISUAL]
    order = rng.permutation(len(vis))
    ITEMS = {}
    for n, k in enumerate(order):
        vis[k]['bid'] = 'B%03d' % (n + 1)
        ITEMS[vis[k]['bid']] = vis[k]
    rest = [it for it in items if it['model'] not in VISUAL]
    for n, it in enumerate(rest):
        it['bid'] = 'P%03d' % (n + 1)
        ITEMS[it['bid']] = it
    bids = ['B%03d' % (n + 1) for n in range(len(vis))]
    SHEETS = [(str(out / 'sheets' / ('sheet_%02d.png' % (s + 1))), bids[s * PANELS_PER_SHEET:(s + 1) * PANELS_PER_SHEET])
              for s in range(math.ceil(len(bids) / PANELS_PER_SHEET))]
    with Pool(workers) as pool:
        for si, geos in pool.imap_unordered(_render_sheet, range(len(SHEETS))):
            for bid, g in geos.items():
                ITEMS[bid]['geo'] = g
        for bid, g in pool.imap_unordered(_geometry_only, [it['bid'] for it in rest]):
            ITEMS[bid]['geo'] = g
    with open(out / 'sheets' / 'index.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sheet', 'blind_ids'])
        for png, ids in SHEETS:
            w.writerow([Path(png).name, ' '.join(ids)])
    with open(out / 'key.csv', 'w', newline='') as f:  # NOT to be read before classification.csv is complete
        w = csv.writer(f)
        w.writerow(['blind_id', 'model', 'frame_id', 'det_index_in_frame', 'score', 'tp_median'])
        for bid, it in sorted(ITEMS.items()):
            w.writerow([bid, it['model'], it['fid'], it['j'], '%.6f' % it['score'], '%.6f' % it['tp_median']])
    (out / 'n_frames.txt').write_text(str(n_frames))
    cols = ['veh_hit', 'veh_nearest', 'veh_static', 'veh_min_spread', 'veh_labelled_frames', 'other_hit', 'cross_iou',
            'cross_hit', 'cross_score']
    gcols = ['range', 'l', 'w', 'h', 'pts', 'pts15', 'ground', 'bottom_above_ground']
    with open(out / 'checks.csv', 'w', newline='') as f:  # NOT to be read before classification.csv is complete
        w = csv.writer(f)
        w.writerow(['blind_id'] + cols + gcols)
        for bid, it in sorted(ITEMS.items()):
            w.writerow([bid] + [it[c] for c in cols] + [it['geo'][c] for c in gcols])
    template = out / 'classification_template.csv'
    with open(template, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['blind_id', 'category', 'note'])
        for bid in bids:
            w.writerow([bid, '', ''])
    print(f'{n_frames} frames; {len(vis)} blinded panels on {len(SHEETS)} sheets; {len(rest)} boxes checked only',
          flush=True)


# -------------------------------------------------------------------------------------------------------- unblind
def unblind(out):
    out = Path(out)
    rd = lambda p: list(csv.DictReader(open(p)))  # noqa: E731
    key = {r['blind_id']: r for r in rd(out / 'key.csv')}
    chk = {r['blind_id']: r for r in rd(out / 'checks.csv')}
    cls = {r['blind_id']: r for r in rd(out / 'classification.csv')}
    blind = sorted(b for b in key if b.startswith('B'))
    missing = [b for b in blind if cls.get(b, {}).get('category') not in CATEGORIES]
    assert not missing, f'unclassified or unknown category: {missing[:10]}'
    n_frames = int((out / 'n_frames.txt').read_text().strip())
    lines = []
    T_ = lambda v: v == 'True'  # noqa: E731
    models = list(VISUAL)
    counts = {m: {c: 0 for c in CATEGORIES} for m in models}
    for b in blind:
        counts[key[b]['model']][cls[b]['category']] += 1
    lines.append(f'frames {n_frames}')
    lines.append('| model | n | ' + ' | '.join(CATEGORIES) + ' |')
    lines.append('|---|---|' + '---|' * len(CATEGORIES))
    for m in models:
        n = sum(counts[m].values())
        lines.append(f'| {m} | {n} | ' + ' | '.join('%d (%.1f%%, %.5f/frame)' % (counts[m][c], 100 * counts[m][c] / n,
                                                                                  counts[m][c] / n_frames)
                                                    for c in CATEGORIES) + ' |')
    sg = {m: (counts[m]['S'] + counts[m]['G']) / n_frames for m in models}
    vv = {m: (counts[m]['V'] + counts[m]['Vpart']) / sum(counts[m].values()) for m in models}
    lines.append('')
    lines.append('(ii) (S+G) per frame: ours %.5f, oracle %.5f, ratio %s' % (
        sg['ours27596'], sg['oracle'], 'n/a' if sg['oracle'] == 0 else '%.2f' % (sg['ours27596'] / sg['oracle'])))
    lines.append('(iii) V+Vpart share: ' + ', '.join('%s %.1f%%' % (m, 100 * vv[m]) for m in models))
    lines.append('')
    lines.append('### checks by category: n | Vehicle hit (static) | other-class hit | cross-model hit | median l / w / h | '
                 'median bottom above ground | median pts / pts 1.5x')
    lines.append('| model | category | n | Vehicle hit | static | other-class hit | cross hit | l / w / h | bottom | pts / pts15 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')

    def row(m, c, ids):
        if not ids:
            return f'| {m} | {c} | 0 | | | | | | | |'
        f = lambda k: np.array([float(chk[b][k]) for b in ids])  # noqa: E731
        return '| %s | %s | %d | %d | %d | %d | %d | %.1f / %.1f / %.1f | %.2f | %d / %d |' % (
            m, c, len(ids), sum(T_(chk[b]['veh_hit']) for b in ids), sum(T_(chk[b]['veh_static']) for b in ids),
            sum(T_(chk[b]['other_hit']) for b in ids), sum(T_(chk[b]['cross_hit']) for b in ids),
            np.median(f('l')), np.median(f('w')), np.median(f('h')), np.nanmedian(f('bottom_above_ground')),
            np.median(f('pts')), np.median(f('pts15')))
    for m in models:
        for c in CATEGORIES + ('all',):
            ids = [b for b in blind if key[b]['model'] == m and (c == 'all' or cls[b]['category'] == c)]
            lines.append(row(m, c, ids))
    pl = [b for b in key if key[b]['model'] == 'pl27704']
    if pl:
        lines.append(row('pl27704', 'all (not classified)', pl))
    (out / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('render', 'unblind'))
    ap.add_argument('--pass_pkl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    if a.stage == 'render':
        render(a.pass_pkl, a.out, a.workers)
    else:
        unblind(a.out)


if __name__ == '__main__':
    main()
