"""Item 61c, the label-content control (experiments_md 20261003_05 §5 item 61c, design fixed 2026-11-27 15:47 JST, before
any number). ANALYSIS on item 48's forward pair; no target label is read here or in the rows it feeds.

59b's own epoch-0 pseudo-labels of a seed, each box rotated about the sensor's vertical axis (x = y = 0 of the loader
frame) by its own random angle: centre x-y rotated, heading + the angle; z, size, class and score kept. A draw is accepted
when the centre is inside the x-y range, at least MIN_POINTS points of the frame's cloud (as the trainer's loader serves
it, augmentation off) lie inside the box (by >= 5 mm, so the trainer's recount agrees), and the box's footprint circle
(radius half the footprint diagonal) clears every original box's circle and every box already moved. Boxes in stored
order, up to TRIES draws each (numpy RNG seeded by (seed, frame id)); a box with no accepted draw is dropped. If more than
2 % of the boxes are dropped at 100 draws, the build is redone once at 1,000 (fixed in the design). The file is served
through the frozen-file path PL-MATCH (DATA_CONFIG_TAR.ANALYSIS_TARGET_PL_MATCH_PSEUDO_LABELS), every box recounted.

    python analysis/item61c_moved.py build SEED            # identity + moved files, build_stats_s<SEED>.json
    python analysis/item61c_moved.py check SEED            # identity path control, loader on every frame, target dump
    python analysis/item61c_moved.py bitwise OUT --jobs moved666=J moved667=J
    python analysis/item61c_moved.py manifest OUT --jobs moved666=J moved667=J
    python analysis/item61c_moved.py read AP_JSON

Tests: tests/test_item61c_moved.py.
"""
import json
import os
import pickle
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS.parent, TOOLS, TOOLS / 'analysis'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _init_path  # noqa: F401,E402  (the checkout's pcdet, not the image's copy)

OUT = Path('/home/koyama/data/item61/moved')
PS_E0 = {666: '/home/koyama/data/wandb/run-20261127_085027-qv08jln7/files/ps_label/ps_label_e0.pkl',     # 59b b_bn 37209
         667: '/home/koyama/data/wandb/run-20261127_085027-q3z6o5zb/files/ps_label/ps_label_e0.pkl'}     # 59b b_bn 37210
BBN_CFG = 'cfgs/da-ieee-access-analysis/item59/centerpoint-item59-b-bn-s%d.yaml'
ROW_CFG = 'cfgs/da-ieee-access-analysis/item61/centerpoint-item61c-b-bn-moved-s%d.yaml'
NAMES = ['Car', 'Pedestrian', 'Cyclist']
MIN_POINTS = 1
TRIES = (100, 1000)
DROP_MAX = 0.02
INSET = 0.005                       # a point counts only when inside the box by >= 5 mm on every face
RINGS = (0.0, 10.0, 20.0, 30.0, 40.0)
EPOCH = 10
ROW = 'P_PL_MOVED'
SEEDS = (666, 667)
REF_57 = '/home/koyama/data/item57/item57b_manifest.json'
REF_58 = '/home/koyama/data/item58/item58_manifest.json'
REF_59B = '/home/koyama/data/item59/item59b_manifest.json'
PK_CKPT = {666: '/home/koyama/data/wandb/run-20261108_020021-0i5ai4hn/files/ckpt/checkpoint_epoch_152.pth',
           667: '/home/koyama/data/wandb/run-20261108_020023-1ya9lozy/files/ckpt/checkpoint_epoch_152.pth'}


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def frame_key(fid):
    s = str(fid)
    return int(s) if s.isdigit() else zlib.crc32(s.encode())


def points_inside(points, x, y, z, dx, dy, dz, heading, inset=INSET):
    """(N,) bool: points (N, >= 3) inside the box by at least `inset` on every face."""
    p = np.asarray(points, dtype=np.float64)
    sx, sy = p[:, 0] - x, p[:, 1] - y
    c, s = np.cos(heading), np.sin(heading)
    lx = sx * c + sy * s
    ly = -sx * s + sy * c
    return ((np.abs(lx) <= dx / 2 - inset) & (np.abs(ly) <= dy / 2 - inset) & (np.abs(p[:, 2] - z) <= dz / 2 - inset))


def move_frame(boxes, points, rng, xy_range, min_points=MIN_POINTS, tries=TRIES[0]):
    """59b boxes (N, 9: x, y, z, dx, dy, dz, heading, label, score) of one frame -> (moved (M, 9), keep (N,), draws (N,)).

    Each box is rotated about (0, 0) by its own angle ~ U[-pi, pi); see the module doc for the acceptance rule. Labels must
    be positive integers: 59b has no ignore band, and a 0, a -99 or a negative (ignore) label is refused, not moved."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 9)
    lab = b[:, 7]
    if len(b) and not (np.all(lab == np.round(lab)) and np.all(lab >= 1)):
        raise ValueError('item61c: labels must be positive integers (59b has no ignore band), got %s'
                         % np.unique(lab).tolist())
    pts = np.asarray(points, dtype=np.float64).reshape(-1, np.asarray(points).shape[-1] if len(points) else 3)[:, :3]
    rho = np.hypot(pts[:, 0], pts[:, 1])
    half = 0.5 * np.hypot(b[:, 3], b[:, 4])
    x0, y0, x1, y1 = [float(v) for v in xy_range]
    cxy, cr = [b[:, :2]], [half]
    out = b.copy()
    keep = np.zeros(len(b), dtype=bool)
    draws = np.zeros(len(b), dtype=np.int64)
    for i in range(len(b)):
        r = float(np.hypot(b[i, 0], b[i, 1]))
        phi = float(np.arctan2(b[i, 1], b[i, 0]))
        z, dz = b[i, 2], b[i, 5]
        cand = pts[(np.abs(rho - r) <= half[i]) & (np.abs(pts[:, 2] - z) <= dz / 2)]   # every point a placed box can hold
        oxy, orr = np.concatenate(cxy), np.concatenate(cr)
        for t in range(tries):
            d = float(rng.uniform(-np.pi, np.pi))
            draws[i] = t + 1
            x, y = r * np.cos(phi + d), r * np.sin(phi + d)
            if not (x0 < x < x1 and y0 < y < y1):
                continue
            if np.any(np.hypot(oxy[:, 0] - x, oxy[:, 1] - y) < orr + half[i]):
                continue
            h = float(wrap(b[i, 6] + d))
            if len(cand) == 0 or int(points_inside(cand, x, y, z, b[i, 3], b[i, 4], dz, h).sum()) < min_points:
                continue
            out[i, 0], out[i, 1], out[i, 6] = x, y, h
            keep[i] = True
            cxy.append(np.array([[x, y]]))
            cr.append(half[i:i + 1])
            break
    return out[keep], keep, draws


def pl_entry(entry, boxes=None, keep=None):
    """A PL-MATCH entry from one stored 59b entry: `boxes` (default: unmoved) and the side arrays sliced by `keep`; every
    box recounted on the loaded cloud (PL_MATCH_RECOUNT)."""
    from pcdet.utils import self_training_utils as stu
    g = np.asarray(entry['gt_boxes'])
    keep = np.ones(len(g), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    boxes = g if boxes is None else np.asarray(boxes, dtype=g.dtype)
    assert len(boxes) == int(keep.sum()), (len(boxes), int(keep.sum()))

    def sel(k):
        v = entry.get(k, None)
        if v is None:
            return None
        v = np.asarray(v)
        assert len(v) == len(g), (k, len(v), len(g))
        return v[keep]
    return {'gt_boxes': boxes, 'cls_scores': sel('cls_scores'), 'iou_scores': sel('iou_scores'),
            'memory_counter': np.zeros(len(boxes)), 'teacher_classes': sel('teacher_classes'),
            'num_points_in_gt': np.full(len(boxes), stu.PL_MATCH_RECOUNT, dtype=np.int64)}


def ring_of(xy):
    return np.digitize(np.hypot(xy[:, 0], xy[:, 1]), RINGS[1:])


def out_path(kind, seed):
    return OUT / ('pl%s_59bbn_s%d.pkl' % (kind, seed))


def write(path, labels, meta):
    from pcdet.utils import self_training_utils as stu
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump({'format': stu.ANALYSIS_PL_MATCH_FORMAT, 'meta': meta, 'labels': labels}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def head_sha():
    import subprocess
    try:
        return subprocess.check_output(['git', '-C', str(TOOLS.parent), 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


def load_cfg(path):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    c = EasyDict()
    cfg_from_yaml_file(path, c)
    return c


def train_dataset(cfg):
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                workers=0, logger=common_utils.create_logger(), training=True,
                                model_ontology=cfg.get('ONTOLOGY'))
    return ds


def served(ds, workers=6):
    """Every frame the dataset serves, in order (a plain DataLoader over it; the caller sets the augmentation)."""
    import torch
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers, collate_fn=lambda x: x[0])
    for i, d in enumerate(dl):
        yield i, d


def build(seed):
    os.chdir(TOOLS)
    ps = pickle.load(open(PS_E0[seed], 'rb'))
    cfg = load_cfg(BBN_CFG % seed)                   # 59b's own config: its loader, its clouds
    pcr = list(cfg.DATA_CONFIG_TAR.POINT_CLOUD_RANGE)
    xy_range = (pcr[0], pcr[1], pcr[3], pcr[4])
    ds = train_dataset(cfg)
    ds.set_pseudo_labels(dict(ps))
    ds.data_augmentor.data_augmentor_queue = []      # augmentation off: the cloud as loaded
    stats = {}
    for tries in TRIES:                              # one streaming pass per setting (the clouds do not fit in memory)
        moved, keep_all, draws_all, orig_pts, moved_pts = {}, [], [], [], []
        cls_o = np.zeros((len(NAMES), len(RINGS)), dtype=np.int64)
        cls_k = np.zeros_like(cls_o)
        for i, d in served(ds):
            fid = d['frame_id']
            assert fid not in moved, ('frame %s served twice' % fid)
            pts = np.asarray(d['points'][:, :3], dtype=np.float64)
            e = ps[fid]
            g = np.asarray(e['gt_boxes'])
            rng = np.random.default_rng([seed, frame_key(fid)])
            out, keep, draws = move_frame(g, pts, rng, xy_range, MIN_POINTS, tries)
            moved[fid] = pl_entry(e, out, keep)
            keep_all.append(keep)
            draws_all.append(draws[keep])
            lab = g[:, 7].astype(np.int64)
            rb = ring_of(g[:, :2]) if len(g) else np.zeros(0, dtype=np.int64)
            np.add.at(cls_o, (lab - 1, rb), 1)
            np.add.at(cls_k, (lab[keep] - 1, rb[keep]), 1)
            for j in np.flatnonzero(keep)[:5]:                       # points per box, first 5 kept boxes per frame
                orig_pts.append(int(points_inside(pts, *g[j, :7]).sum()))
                k = int(keep[:j].sum())
                moved_pts.append(int(points_inside(pts, *out[k, :7]).sum()))
            if i % 1000 == 0:
                print('[61c] %d draws: frame %d / %d' % (tries, i, len(ds)), flush=True)
        missing = [f for f in ps if f not in moved]
        assert not missing, ('%d stored frames were not served (first %s)' % (len(missing), missing[:3]))
        keep_all = np.concatenate(keep_all) if keep_all else np.zeros(0, dtype=bool)
        dropped = float(1 - keep_all.mean()) if len(keep_all) else 0.0
        stats[tries] = dict(tries=tries, boxes=int(len(keep_all)), kept=int(keep_all.sum()), dropped_share=round(dropped, 5),
                            per_class_ring_original={n: cls_o[c].tolist() for c, n in enumerate(NAMES)},
                            per_class_ring_kept={n: cls_k[c].tolist() for c, n in enumerate(NAMES)},
                            draws_used=dict(zip(('p50', 'p90', 'p99', 'max'), np.percentile(
                                np.concatenate(draws_all), [50, 90, 99, 100]).round(1).tolist())),
                            points_per_box_median=dict(original=float(np.median(orig_pts)), moved=float(np.median(moved_pts))))
        print('[61c] seed %d, %d draws: %d boxes, kept %d, dropped share %.4f' % (
            seed, tries, stats[tries]['boxes'], stats[tries]['kept'], dropped), flush=True)
        if dropped <= DROP_MAX:
            break
    meta_common = dict(what='ANALYSIS item 61c (experiments_md 20261003_05 §5, design 2026-11-27 15:47 JST): 59b b_bn '
                            'pseudo-labels of seed %d; LABEL-FREE (no target label read). PL-MATCH is reused as the '
                            'frozen-file loader only.' % seed,
                       source_pseudo_labels=PS_E0[seed], class_names=NAMES, n_frames=len(ps), st3d_head=head_sha(),
                       written=time.strftime('%Y-%m-%d %H:%M %Z'))
    write(out_path('ident', seed), {f: pl_entry(e) for f, e in ps.items()}, dict(meta_common, row='IDENTITY'))
    write(out_path('move', seed), moved, dict(meta_common, row='MOVED', tries=tries, min_points=MIN_POINTS,
                                              rotation='about x = y = 0, U[-pi, pi) per box, seeded (seed, frame id)'))
    res = dict(meta_common, used_tries=tries, builds=list(stats.values()))
    json.dump(res, open(OUT / ('build_stats_s%d.json' % seed), 'w'), indent=1)
    print(json.dumps(res, indent=1))


def check(seed, frames=60):
    """1. identity path control; 2. the moved file served on every frame keeps every stored box; 3. target dump."""
    os.chdir(TOOLS)
    import plmatch_labels as pm
    from pcdet.utils import common_utils, self_training_utils as stu
    key = common_utils.ANALYSIS_TARGET_PL_MATCH_KEY
    ps = pickle.load(open(PS_E0[seed], 'rb'))
    res, ok = {'seed': seed, 'st3d_head': head_sha()}, True
    # 1. identity: 59b's labels unmoved through PL-MATCH against 59b's own path, augmentation off and on (fixed seed)
    cfg_o = load_cfg(BBN_CFG % seed)
    cfg_n = load_cfg(ROW_CFG % seed)
    assert cfg_n.DATA_CONFIG_TAR[key] == str(out_path('move', seed)), cfg_n.DATA_CONFIG_TAR[key]
    cfg_n.DATA_CONFIG_TAR[key] = str(out_path('ident', seed))
    ds_o, ds_n = train_dataset(cfg_o), train_dataset(cfg_n)
    ident, _, _ = stu.load_analysis_pl_match_pseudo_labels(str(out_path('ident', seed)), NAMES, min_points=MIN_POINTS)
    ds_o.set_pseudo_labels(dict(ps))
    ds_n.set_pseudo_labels(dict(ident))
    queue = {id(d): list(d.data_augmentor.data_augmentor_queue) for d in (ds_o, ds_n)}
    res['1_identity'] = {}
    for mode in ('aug_off', 'aug_on_fixed_seed'):
        for d in (ds_o, ds_n):
            d.data_augmentor.data_augmentor_queue = [] if mode == 'aug_off' else list(queue[id(d)])
        step = max(1, len(ds_o) // frames)
        idx = list(range(0 if mode == 'aug_off' else step // 2, len(ds_o), step))[:frames]
        st = {'frames': 0, 'equal': 0, 'boxes': 0, 'failures': []}
        for i in idx:
            a, b = pm.seeded(ds_o, i, 20261127 + i), pm.seeded(ds_n, i, 20261127 + i)
            dk = pm.same_dict(a, b)
            st['frames'] += 1
            st['equal'] += int(not dk)
            st['boxes'] += len(b['gt_boxes'])
            if dk and len(st['failures']) < 5:
                st['failures'].append({'frame': a['frame_id'], 'keys': dk})
        ok &= st['equal'] == st['frames'] and st['boxes'] > 0
        res['1_identity'][mode] = st
    print(json.dumps(res['1_identity'], indent=1), flush=True)
    del ds_o, ds_n
    # 2. the moved file, augmentation off, every frame: every stored box served, unchanged
    cfg_m = load_cfg(ROW_CFG % seed)
    ds_m = train_dataset(cfg_m)
    moved, meta, counts = stu.load_analysis_pl_match_pseudo_labels(str(out_path('move', seed)), NAMES,
                                                                   min_points=MIN_POINTS)
    assert meta['row'] == 'MOVED', meta
    ds_m.set_pseudo_labels(dict(moved))
    q = list(ds_m.data_augmentor.data_augmentor_queue)
    ds_m.data_augmentor.data_augmentor_queue = []
    st = {'frames': 0, 'frames_all_kept': 0, 'stored': 0, 'served': 0, 'failures': []}
    for i, d in served(ds_m):
        e = moved[d['frame_id']]
        g = np.asarray(e['gt_boxes'])
        s = np.asarray(d['gt_boxes'])
        same = (len(s) == len(g) and np.allclose(s[:, :7], g[:, :7].astype(np.float32), atol=1e-5)
                and np.array_equal(s[:, 7].astype(np.int64), g[:, 7].astype(np.int64)))
        st['frames'] += 1
        st['frames_all_kept'] += int(same)
        st['stored'] += len(g)
        st['served'] += len(s)
        if not same and len(st['failures']) < 5:
            st['failures'].append({'frame': d['frame_id'], 'stored': len(g), 'served': len(s)})
    ok &= st['frames_all_kept'] == st['frames'] == len(ps)
    res['2_loader_moved'] = dict(st, counts={k: v.tolist() for k, v in counts.items()})
    print(json.dumps(res['2_loader_moved'], indent=1), flush=True)
    # 3. one-batch target dump (4 frames, augmentation off, then on): heatmap peaks against positives
    head, side, _, _ = pm.make_head(cfg_m)
    res['3_targets'] = {}
    for mode in ('aug_off', 'aug_on'):
        ds_m.data_augmentor.data_augmentor_queue = [] if mode == 'aug_off' else list(q)
        pos, peaks = np.zeros(len(NAMES), dtype=np.int64), np.zeros(len(NAMES), dtype=np.int64)
        for i in range(0, 4 * 997, 997):
            d = pm.seeded(ds_m, i, 61 + i)
            t = pm.targets(head, side, d['gt_boxes'])
            lab = np.asarray(d['gt_boxes'])[:, 7].astype(np.int64)
            for c in range(len(NAMES)):
                pos[c] += int((lab == c + 1).sum())
                peaks[c] += int(t['heatmaps'][0][0][c].eq(1).sum())
        res['3_targets'][mode] = dict(positives=pos.tolist(), peaks=peaks.tolist())
        if mode == 'aug_off':
            ok &= bool(np.array_equal(pos, peaks)) and pos.sum() > 0
    print(json.dumps(res['3_targets'], indent=1), flush=True)
    res['ok'] = bool(ok)
    json.dump(res, open(OUT / ('check_s%d.json' % seed), 'w'), indent=1)
    print('[61c] check seed %d: %s' % (seed, 'PASS' if ok else 'FAIL'), flush=True)
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- after training
def parse_jobs(args):
    out = {}
    for a in args:
        k, v = a.split('=')
        m = re.match(r'moved(\d+)$', k)
        out[int(m.group(1))] = int(v)
    return out


def bitwise(jobs):
    import torch
    import item58_bn_bound as m58
    import item59_bn_split as m59
    out = {}
    for s, j in sorted(jobs.items()):
        c = m58.compare(torch.load(PK_CKPT[s], map_location='cpu')['model_state'],
                        torch.load(m59.ckpt_of(j, EPOCH), map_location='cpu')['model_state'])
        c['ok'] = not c['non_bn_differ'] and not c['missing']
        out['%s s%d' % (ROW, s)] = dict(c, job=j)
    return out


def manifest(jobs):
    import item38_kitti_rings as i38
    rows = []
    for s, j in sorted(jobs.items()):
        r = i38.resolve('item61c', ROW, s, j, 'epoch_%d' % EPOCH)
        if r['status'] == 'ok' and not Path(r['result']).exists():
            r['status'] = 'result.pkl missing'
        rows.append(r)
    for path, keep in ((REF_57, ('P_RS1', 'L_ada', 'O_ada')), (REF_58, ('P_BN_G',)), (REF_59B, ('P_PL_BN',))):
        for r in json.load(open(path)):
            if r['row'] in keep:
                rows.append(dict(r, group='item61c'))
    return rows


WORDS = {'a normalization-level correction can close most of the far residual':
         "the gain does not need the labels' content: training gamma / beta on the target clouds under boxes of the right "
         "count, size and range gives it",
         'it cannot': "the gain needs the labels' content", 'partly': 'partly'}


def reading(rows):
    import item58_bn_bound as m58
    seeds = [m58.per_seed(rows, s, ROW) for s in SEEDS]
    return seeds, WORDS.get(m58.reading(seeds), m58.reading(seeds))


def cmd_read(path):
    import item58_bn_bound as m58
    rows = json.load(open(path))['rows']
    for r in rows:
        print('%-11s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    seeds, word = reading(rows)
    g = {s: m58.per_seed(rows, s, 'P_BN_G') for s in SEEDS}
    print('\n== %s: %s' % (ROW, word))
    for s, d in zip(SEEDS, seeds):
        if d is None:
            print('  seed %d: not read' % s)
            continue
        share = d['delta'] / g[s]['delta'] if g[s] and g[s]['delta'] else None
        print("  seed %d: s_x far %.2f / %.2f, delta %+.2f, R %.2f, share of G's delta %s" % (
            s, d['s_x'][2], d['s_x'][3], d['delta'], d['R'], '%.2f' % share if share is not None else '-'))
    for s in SEEDS:                                       # descriptive: against 59b's b_bn per ring
        a = [r for r in rows if r['row'] == ROW and r['seed'] == s]
        b = [r for r in rows if r['row'] == 'P_PL_BN' and r['seed'] == s]
        if a and b:
            print('  %s - P_PL_BN s%d: rings %s, pooled %+.2f, 3D %+.2f' % (
                ROW, s, ' '.join('%+.2f' % (x['bev'] - y['bev']) for x, y in zip(a[0]['rings'], b[0]['rings'])),
                a[0]['pooled']['bev'] - b[0]['pooled']['bev'], a[0]['pooled']['3d'] - b[0]['pooled']['3d']))


def main():
    a = sys.argv[1:]
    if a[:1] == ['build'] and len(a) == 2:
        return build(int(a[1]))
    if a[:1] == ['check'] and len(a) == 2:
        return check(int(a[1]))
    if a[:1] == ['bitwise'] and len(a) >= 4 and a[2] == '--jobs':
        out = bitwise(parse_jobs(a[3:]))
        json.dump(out, open(a[1], 'w'), indent=1)
        for k, c in out.items():
            print('%s (job %d): non-BN tensors differing %d, BN tensors changed %d, key mismatch %d, counters %s -> %s' % (
                k, c['job'], len(c['non_bn_differ']), c['bn_changed'], len(c['missing']), c['counters'],
                'ok' if c['ok'] else 'FAILED'))
        sys.exit(0 if all(c['ok'] for c in out.values()) else 1)
    if a[:1] == ['manifest'] and len(a) >= 4 and a[2] == '--jobs':
        rows = manifest(parse_jobs(a[3:]))
        json.dump(rows, open(a[1], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r.get('job'), r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['read'] and len(a) == 2:
        return cmd_read(a[1])
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
