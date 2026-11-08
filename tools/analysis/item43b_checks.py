"""Pre-GPU checks of the item 43b row L'_K (experiments_md 20261003_05 §5 item 43; bug_prevention rules 1, 4, 8, 9) and the
info check behind the 43a own-input twins. CPU, ANALYSIS; nuScenes TRAIN / val labels are read only as the loaders read them.

    python analysis/item43b_checks.py identity <frames> <out.json>
        Rule 8: per platform block, L'_K's block with the calibration installed at RATE 1 in every bin (src = tgt) against
        27468's block (no arm), training mode, the same seed per frame: points, voxels, voxel coords and points per voxel
        must be EXACTLY equal on every frame (>= 40 per block), and gt_boxes too, except that the arm's recount may drop a
        positive box the base itself holds with no point (listed as recount_only; anything else is a failure).
    python analysis/item43b_checks.py inert <frames> <out.json>
        The evaluation view of L'_K (DATA_CONFIG_TAR, eval mode) against the nuScenes oracle's: exactly equal frames.
    python analysis/item43b_checks.py emptied <n008|n015> <frames> <out.json>
        The calibration installed as train.py installs it (calibration_reference + link_point_calibration, the block's
        HIST_DIST_* keys, 1,000 frames each side); the per-bin rates are saved. Per frame, same seed, twice: without the
        recount (drop_empty_gt_boxes removed from the queue) the share of positive boxes holding >= 1 point before the
        processor and 0 after (analysis/emptied_boxes.py's count; that script never installs an on-the-fly calibration, so
        on this row it would read a no-op), and with the recount: no positive box left empty, the dropped boxes = the
        positives the thinning left empty.
    python analysis/item43b_checks.py infos <out.json>
        The 200-sweep val infos (own-input twins) hold the same 6,019 frames, in the same order, with the same boxes (to
        1e-6; the two builds differ by float noise, first run 32416: 6,012 frames unequal under exact comparison), names
        and lidar point counts as the 10-sweep val infos the oracle is scored on.
    python analysis/item43b_checks.py pooled <gate_n008.json> <gate_n015.json> <oracle.json>
        The per-platform arm_point_count_bins measurements of L'_K pooled by platform TRAIN frames (n008 15,695 / n015
        12,435), read against the single-sweep nuScenes source measured on the oracle config, by arm_point_count_bins.reading.
Run inside the container from ST3D/tools.
"""
import copy
import json
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'analysis')
import _init_path  # noqa: F401,E402
import make_item43_configs as G  # noqa: E402

FRAMES = {'n008': 15695, 'n015': 12435}


def logger():
    from pcdet.utils import common_utils
    return common_utils.create_logger()


def cfg_of(path):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    return cfg_from_yaml_file(path, EasyDict())


def build(block, cfg, training):
    from pcdet.datasets import build_dataloader
    ds, _, _ = build_dataloader(dataset_cfg=block, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger(), training=training, model_ontology=cfg.get('ONTOLOGY'))
    return ds


def strided(n_total, n):
    return list(range(0, n_total, max(1, n_total // n)))[:n]


def same(a, b):
    keys = ('points', 'gt_boxes', 'voxels', 'voxel_coords', 'voxel_num_points')
    return {k: bool(np.array_equal(a[k], b[k])) for k in keys if k in a or k in b}


def identity(n, out):
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    lp, tc = cfg_of(G.LPRIME), cfg_of(G.TEACHER)
    res = {}
    for v, b in G.BLOCK.items():
        arm, base = build(lp.DATA_CONFIGS[b], lp, True), build(tc.DATA_CONFIGS[b], tc, True)
        names = [p.NAME for p in lp.DATA_CONFIGS[b].DATA_PROCESSOR]
        assert 'sample_points_hist_based' in names and 'drop_empty_gt_boxes' in names, names
        ones = np.ones(lp.DATA_CONFIGS[b].HIST_DIST_BINS)
        arm.data_processor.set_hist_dist(ones, ones, max_dist=lp.DATA_CONFIGS[b].HIST_DIST_MAX_DIST)
        frames, bad = strided(len(arm), n), []
        recount_only = []
        for j, i in enumerate(frames):
            np.random.seed(7000 + i)
            a = arm[i]
            np.random.seed(7000 + i)
            c = base[i]
            s = same(a, c)
            if not all(s.values()):
                # The arm's recount (amendment A5) drops a POSITIVE box holding no point. At rate 1 that can only be a
                # box the base row itself trains empty: accepted iff the arm's boxes are exactly the base's minus those.
                held = roiaware_pool3d_utils.points_in_boxes_cpu(c['points'][:, :3], c['gt_boxes'][:, :7]).sum(1)
                want = c['gt_boxes'][~((c['gt_boxes'][:, 7] > 0) & (held == 0))]
                if all(v for k, v in s.items() if k != 'gt_boxes') and np.array_equal(a['gt_boxes'], want):
                    recount_only.append([int(i), int(len(c['gt_boxes']) - len(want))])
                else:
                    bad.append([int(i), s])
            if j % 10 == 0:
                print('identity %s frame %d / %d, %d points, %d boxes, equal %s' % (v, j, len(frames), len(a['points']),
                                                                                    len(a['gt_boxes']), s), flush=True)
        res[v] = dict(frames=len(frames), unequal=bad, exact=not bad, recount_only=recount_only)
    res['verdict'] = 'EXACT' if all(res[v]['exact'] for v in G.BLOCK) else 'DIFFERS'
    json.dump(res, open(out, 'w'), indent=1)
    print('identity at rate 1:', res['verdict'], {v: (res[v]['frames'], len(res[v]['unequal'])) for v in G.BLOCK},
          'frames differing only by the recount of an empty positive the base trains:',
          {v: res[v]['recount_only'] for v in G.BLOCK})


def inert(n, out):
    lp, oc = cfg_of(G.LPRIME), cfg_of(G.ORACLE)
    a_ds, o_ds = build(lp.DATA_CONFIG_TAR, lp, False), build(oc.DATA_CONFIG_TAR, oc, False)
    assert len(a_ds) == len(o_ds) == 6019, (len(a_ds), len(o_ds))
    bad = []
    for i in strided(len(a_ds), n):
        s = same(a_ds[i], o_ds[i])
        if not all(s.values()):
            bad.append([int(i), s])
    res = dict(frames=n, val_frames=len(a_ds), unequal=bad, verdict='INERT' if not bad else 'DIFFERS')
    json.dump(res, open(out, 'w'), indent=1)
    print('eval view of L\'_K vs the oracle:', res['verdict'], '(%d frames of %d)' % (n, len(a_ds)))


def emptied(vehicle, n, out):
    from pcdet.datasets.point_calibration import calibration_reference, calibration_target_config, link_point_calibration
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
    cfg = cfg_of(G.LPRIME)
    dc = cfg.DATA_CONFIGS[G.BLOCK[vehicle]]
    src = build(dc, cfg, True)
    tcfg, _ = calibration_target_config(cfg.DATA_CONFIG_TAR)
    tgt = build(tcfg, cfg, False)
    reference, pc_range = calibration_reference(dc, src, tgt, logger=logger())        # as train.py
    h_src, h_ref = link_point_calibration(src, reference, num_frames=dc.HIST_DIST_FRAMES, num_bins=dc.HIST_DIST_BINS,
                                          max_dist=dc.HIST_DIST_MAX_DIST, logger=logger(), pc_range=pc_range)
    rate = src.data_processor.per_bin_sample_rate([p for p in dc.DATA_PROCESSOR
                                                   if p.NAME == 'sample_points_hist_based'][0])
    proc = src.data_processor
    full = list(proc.data_processor_queue)
    no_recount = [f for f in full if getattr(getattr(f, 'func', None), '__name__', '') != 'drop_empty_gt_boxes']
    assert len(no_recount) == len(full) - 1
    captured = {}
    orig = proc.forward

    def spy(data_dict):
        captured['before'] = data_dict['points'][:, :3].copy()
        return orig(data_dict=data_dict)
    proc.forward = spy
    tot, emp, dropped, left_empty, zero_after, kept = {}, {}, 0, 0, 0, []
    for j, i in enumerate(strided(len(src), n)):
        np.random.seed(9000 + i)
        proc.data_processor_queue = no_recount
        d = src[i]
        b = d['gt_boxes']
        b = b[b[:, 7] > 0] if len(b) else b
        kept.append(len(d['points']) / max(len(captured['before']), 1))
        if len(b):
            cb = roiaware_pool3d_utils.points_in_boxes_cpu(captured['before'], b[:, :7]).sum(1)
            ca = roiaware_pool3d_utils.points_in_boxes_cpu(d['points'][:, :3], b[:, :7]).sum(1)
            for c in np.unique(b[:, 7]).astype(int):
                m = (b[:, 7] == c) & (cb > 0)
                tot[c] = tot.get(c, 0) + int(m.sum())
                emp[c] = emp.get(c, 0) + int((m & (ca == 0)).sum())
            n_emptied = int(((cb > 0) & (ca == 0)).sum())
        else:
            n_emptied = 0
        np.random.seed(9000 + i)
        proc.data_processor_queue = full
        e = src[i]
        eb = e['gt_boxes']
        eb = eb[eb[:, 7] > 0] if len(eb) else eb
        held = roiaware_pool3d_utils.points_in_boxes_cpu(e['points'][:, :3], eb[:, :7]).sum(1) if len(eb) else np.zeros(0)
        left_empty += int((held == 0).sum())
        dropped += len(b) - len(eb)
        zero_after += int((ca == 0).sum()) if len(b) else 0
        if j % 50 == 0:
            print('emptied %s frame %d / %d: kept %.3f of points, %d boxes emptied' % (vehicle, j, n, kept[-1], n_emptied),
                  flush=True)
    names = list(cfg.CLASS_NAMES)
    res = dict(vehicle=vehicle, frames=n, points_kept=float(np.mean(kept)),
               rate=np.asarray(rate).tolist(), hist_src=np.asarray(h_src).tolist(), hist_ref=np.asarray(h_ref).tolist(),
               src_per_frame=float(np.sum(h_src)), ref_per_frame=float(np.sum(h_ref)),
               emptied={names[c - 1]: [emp[c], tot[c], emp[c] / max(tot[c], 1)] for c in sorted(tot)},
               recount_dropped=int(dropped), recount_left_empty=int(left_empty),
               recount_ok=bool(left_empty == 0 and dropped == zero_after), positives_empty_after_thinning=int(zero_after))
    json.dump(res, open(out, 'w'), indent=1)
    print('%s: calibration %.0f -> %.0f in-grid pts/frame, rates %.3f..%.3f; points kept %.3f' % (
        vehicle, res['src_per_frame'], res['ref_per_frame'], min(res['rate']), max(res['rate']), res['points_kept']))
    for k, (e_, t_, s_) in res['emptied'].items():
        print('  %-10s boxes with points before %6d, emptied by the thinning %5d (%.1f%%)' % (k, t_, e_, 100 * s_))
    print('  recount: dropped %d, positives left empty %d -> %s' % (dropped, left_empty, 'OK' if res['recount_ok'] else 'FAIL'))


def infos(out):
    import pickle
    root = '../data/nuscenes/v1.0-trainval/'
    a = pickle.load(open(root + 'nuscenes_infos_10sweeps_val.pkl', 'rb'))
    b = pickle.load(open(root + 'nuscenes_infos_200sweeps_val.pkl', 'rb'))
    bad, dmax = [], 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        g, h = np.asarray(x['gt_boxes']), np.asarray(y['gt_boxes'])
        ok = (x['token'] == y['token'] and x['lidar_path'] == y['lidar_path'] and g.shape == h.shape
              and np.allclose(g, h, rtol=0, atol=1e-6, equal_nan=True) and list(x['gt_names']) == list(y['gt_names'])
              and np.array_equal(x['num_lidar_pts'], y['num_lidar_pts']))
        if g.shape == h.shape and len(g):
            dmax = max(dmax, float(np.nanmax(np.abs(g - h))))
        if not ok:
            bad.append(i)
    res = dict(n10=len(a), n200=len(b), unequal=bad[:50], n_unequal=len(bad), max_abs_box_diff=dmax, tolerance=1e-6,
               sweeps10=int(np.median([len(x['sweeps']) for x in a])), sweeps200=int(np.median([len(y['sweeps']) for y in b])),
               verdict='SAME' if len(a) == len(b) and not bad else 'DIFFERS')
    json.dump(res, open(out, 'w'), indent=1)
    print('val infos 10-sweep vs 200-sweep:', res['verdict'], res['n10'], res['n200'], 'max |box diff| %.2e' % dmax,
          'median stored sweeps', res['sweeps10'], res['sweeps200'])


def pooled(g8, g15, oracle):
    import arm_point_count_bins as A
    j8, j15, jo = (json.load(open(p)) for p in (g8, g15, oracle))
    w8, w15 = FRAMES['n008'], FRAMES['n015']
    mix = {u: dict(mean=((w8 * np.asarray(j8['source'][u]['mean']) + w15 * np.asarray(j15['source'][u]['mean']))
                         / (w8 + w15)).tolist()) for u in ('points', 'points_all', 'voxels')}
    for u in ('points', 'points_all', 'voxels'):
        r = A.reading(mix[u], jo['source'][u], jo['target'][u])
        m, s = np.asarray(mix[u]['mean']), np.asarray(jo['source'][u]['mean'])
        print("%s: L'_K pooled / single-sweep nuScenes per bin %s; per frame (0-70 m) %.2f, bins outside %s -> %s" % (
            u, ' '.join('%.2f' % (x / max(y, 1e-9)) for x, y in zip(m[:14], s[:14])), r['frame_ratio'],
            [f'{5 * i}-{5 * i + 5}' for i in r['bins_outside']], 'MATCHED' if r['matched'] else 'DIFFER'))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'identity':
        identity(int(sys.argv[2]), sys.argv[3])
    elif cmd == 'inert':
        inert(int(sys.argv[2]), sys.argv[3])
    elif cmd == 'emptied':
        emptied(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif cmd == 'infos':
        infos(sys.argv[2])
    elif cmd == 'pooled':
        pooled(*sys.argv[2:5])
    else:
        sys.exit(__doc__)
