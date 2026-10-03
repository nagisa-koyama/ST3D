"""Dump everything one run's figure needs (experiments_md/20261003_02 §3, §5). GPU, from tools/:

    python analysis/viz/dump_frames.py --rows 25835,25781 [--out /storage/viz/dump]

Per manifest row: the run's recorded config, its scored checkpoint, the common frames of its target
(val) and of each of its sources (train). For every frame: raw cloud (sources), processed cloud read
back from the voxel tensor, labels, predictions, and - for a self-training row - what the row's
pseudo-label rule (teacher + SCORE_THRESH / NEG_THRESH) gives on that frame. Plus the closeness
statistics over CLOSE_FRAMES frames per cloud. One pickle per row.

The density correction is RECOMPUTED (runs logged its summary, not its rates) through the same
link_point_calibration call train.py makes, measuring the target on the split the run used, and
cached per (source config, target config, split). Its summary is printed for comparison with the
run's own log.
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz_common as vc  # noqa: E402

from pcdet.datasets.point_calibration import (calibration_target_config, link_foreground_calibration,  # noqa: E402
                                              link_point_calibration)
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402

HERE = Path(__file__).resolve().parent
CLOSE_FRAMES, RING_EDGES = 30, np.arange(0, 80, 5.0)
BOX_RINGS = np.arange(0, 80, 10.0)
MIN_R = 3.0          # drop self-returns (raw nuScenes keeps its roof) from every curve
SAVE_SCORE = 0.1     # predictions saved from here; drawn from 0.3


def resolve_ckpt(spec):
    if spec.startswith('/'):
        return spec
    run, ep = spec.split(':')
    return str(vc.find_run_dir(run) / 'files' / 'ckpt' / ('checkpoint_epoch_%s.pth' % ep))


def teacher_ckpt_from_args(run_id):
    args = vc.run_args(run_id)
    for i, a in enumerate(args):
        if a == '--pretrained_model_teacher':
            return args[i + 1].replace('/storage/', '/home/koyama/data/')
    return None


def frame_sets_for(ds, data_cfg, frames, split):
    kind = vc.dataset_kind(ds)
    keys = [k for k in frames if k.startswith('%s/%s' % (kind, split))]
    plat = vc.platform(data_cfg)
    if plat is not None:
        keys = [k for k in keys if k.endswith('/%s' % plat)]
    return [f for k in keys for f in frames[k]['frames']]


def in_cone(xy, cone):
    if cone is None:
        return np.ones(len(xy), bool)
    return np.abs(np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))) <= cone / 2.0


def ring_counts(points, cone=None):
    r = np.linalg.norm(points[:, :2], axis=1)
    keep = (r >= MIN_R) & in_cone(points[:, :2], cone)
    return np.histogram(r[keep], bins=RING_EDGES)[0]


def car_box_points(points, boxes, names, cone=None):
    car = (names == vc.CAR) & in_cone(boxes[:, :2], cone) if len(boxes) else names == vc.CAR
    b = boxes[car]
    n = vc.points_per_box(points, b)
    return np.stack([np.linalg.norm(b[:, :2], axis=1), n], 1) if len(b) else np.zeros((0, 2))


def gt_from_dict(dd, class_names):
    g = dd.get('gt_boxes', np.zeros((0, 8)))
    # the class index is the LAST column: some configs keep two velocity columns in between
    names = vc.canon([class_names[int(c) - 1] if c >= 1 else '' for c in g[:, -1]]) if len(g) else np.zeros(0, str)
    return g[:, :7].astype(np.float32), names


def infer(model, ds, dd):
    batch = ds.collate_batch([dd])
    load_data_to_gpu(batch)
    with torch.no_grad():
        pred, _ = model(batch)
    return batch, pred


def closeness_frames(n, shown, seed=1):
    rest = [i for i in np.random.RandomState(seed).permutation(n) if i not in shown]
    return list(shown) + rest[:CLOSE_FRAMES - len(shown)]


def calibrate(sset, cfg, dc, split, logger, cache_dir, cone_aug_on=False):
    cone = dc.get('HIST_DIST_FOV_DEGREE', None) is not None
    aug_on = cone and cone_aug_on
    key = hashlib.md5(json.dumps([dc, vc.eval_config(cfg), split, aug_on], sort_keys=True, default=str).encode()).hexdigest()[:12]
    f = cache_dir / ('calib_%s.npz' % key)
    if f.exists():
        z = np.load(f)
        sset.data_processor.set_hist_dist(z['src'], z['tgt'], max_dist=float(z['max_dist']))
    elif aug_on:
        # A cone calibration from before ST3D 2e5ae50 measured the source THROUGH its training
        # augmentation. Reproduce that: measure a twin source set with the run's own augmentor,
        # then install the pair on the display set (whose world augmentation is off).
        from pcdet.datasets.point_calibration import compute_range_histogram
        md = dc.get('HIST_DIST_MAX_DIST', 75.0)
        nb, nf = dc.get('HIST_DIST_BINS', 50), dc.get('HIST_DIST_FRAMES', 1000)
        fov, head = dc.HIST_DIST_FOV_DEGREE, dc.get('HIST_DIST_FOV_HEADING', 0.0)
        twin = vc.build_dataset(dc, list(sset.class_names), True, sset.model_ontology_name if hasattr(sset, 'model_ontology_name') else None, logger)
        vc.seed_all(0)
        src = compute_range_histogram(twin, nf, nb, md, logger=logger, fov_degree=fov, fov_heading=head)
        ccfg, _ = calibration_target_config(vc.eval_config(cfg), split=split)
        tset = vc.build_dataset(ccfg, cfg.CLASS_NAMES, False, cfg.get('ONTOLOGY', None))
        tgt = compute_range_histogram(tset, nf, nb, md, logger=logger, fov_degree=fov, fov_heading=head)
        sset.data_processor.set_hist_dist(src, tgt, max_dist=md)
        np.savez(f, src=src, tgt=tgt, max_dist=md)
        del twin, tset
    else:
        ccfg, _ = calibration_target_config(vc.eval_config(cfg), split=split)
        tset = vc.build_dataset(ccfg, cfg.CLASS_NAMES, False, cfg.get('ONTOLOGY', None))
        md = dc.get('HIST_DIST_MAX_DIST', 75.0)
        src, tgt = link_point_calibration(sset, tset, num_frames=dc.get('HIST_DIST_FRAMES', 1000),
                                          num_bins=dc.get('HIST_DIST_BINS', 50), max_dist=md, logger=logger,
                                          fov_degree=dc.get('HIST_DIST_FOV_DEGREE', None),
                                          fov_heading=dc.get('HIST_DIST_FOV_HEADING', 0.0))
        np.savez(f, src=src, tgt=tgt, max_dist=md)
    src, tgt = sset.data_processor.hist_dist_src, sset.data_processor.hist_dist_tgt
    rate = sset.data_processor.per_bin_sample_rate(processor_cfg(dc))
    return dict(src_pts=float(np.sum(src)), tgt_pts=float(np.sum(tgt)), rate_min=float(rate.min()),
                rate_max=float(rate.max()), bins_below_1=int((rate < 1).sum()), cache=f.name,
                uniform=bool(processor_cfg(dc).get('UNIFORM_RATE', False)), cone=cone, cone_aug_on=aug_on)


def processor_cfg(dc):
    return next((p for p in dc.get('DATA_PROCESSOR', []) if p.get('NAME') == 'sample_points_hist_based'), {})


def fg_class_ids(dc, class_names):
    """train_st_utils._foreground_class_ids, restated (importing train_st_utils pulls the whole loop)."""
    names = dc.get('HIST_DIST_FOREGROUND_CLASSES', None)
    return None if names is None else [list(class_names).index(n) + 1 for n in names]


def calibrate_foreground(sset, cfg, dc, row, logger, cache_dir):
    """The foreground-aware correction as train_model_st installs it after the FIRST pseudo-label
    pass: the run's own ps_label_e0.pkl on the training-mode target, then link_foreground_calibration.
    (An unfrozen-teacher run re-calibrates after every pass; this reproduces the first one only.)"""
    crun = row['ckpt'].split(':')[0]
    ps = vc.find_run_dir(crun) / 'files' / 'ps_label' / 'ps_label_e0.pkl'
    if not ps.exists():
        return dict(error='no ps_label_e0.pkl in %s; processed cloud shown UNCORRECTED' % crun)
    cids = fg_class_ids(dc, sset.class_names)
    key = hashlib.md5(json.dumps([dc, cfg.DATA_CONFIG_TAR, str(ps), cids], sort_keys=True, default=str).encode()).hexdigest()[:12]
    f = cache_dir / ('fgcalib_%s.npz' % key)
    md = dc.get('HIST_DIST_MAX_DIST', 75.0)
    if f.exists():
        z = np.load(f, allow_pickle=True)
        sset.data_processor.set_hist_dist(z['tot_s'], z['tot_t'], max_dist=md)
        sset.data_processor.set_foreground_hist(z['fg_s'], z['bg_s'], z['fg_t'], z['bg_t'], class_ids=cids)
    else:
        tset = vc.build_dataset(cfg.DATA_CONFIG_TAR, list(cfg.CLASS_NAMES), True, cfg.get('ONTOLOGY', None), logger)
        tset.set_pseudo_labels(pickle.load(open(ps, 'rb')))
        (fg_s, bg_s, tot_s), (fg_t, bg_t, tot_t) = link_foreground_calibration(
            sset, tset, num_frames=dc.get('HIST_DIST_FRAMES', 1000), num_bins=dc.get('HIST_DIST_BINS', 50),
            max_dist=md, logger=logger, min_points_in_box=dc.get('HIST_DIST_MIN_POINTS_IN_BOX', 1), class_ids=cids)
        np.savez(f, fg_s=fg_s, bg_s=bg_s, tot_s=tot_s, fg_t=fg_t, bg_t=bg_t, tot_t=tot_t)
        del tset
    p = sset.data_processor
    rf, rb = p.per_bin_sample_rate(None, 'fg'), p.per_bin_sample_rate(None, 'bg')
    return dict(foreground=True, fg_rate=(float(rf.min()), float(rf.max())), bg_rate=(float(rb.min()), float(rb.max())),
                classes=cids, pseudo_labels=str(ps), cache=f.name,
                src_pts=float(np.sum(p.hist_dist_src)), tgt_pts=float(np.sum(p.hist_dist_tgt)))


def check_against_result_pkl(eval_dir, idx, fid, anno):
    """Inference vs the scored result.pkl on one frame: box count >= 0.3 and worst centre offset.

    Matched by DATASET INDEX (evaluation keeps dataset order, DDP merge included), with the frame id
    verified: PandaSet's frame_id is only the frame number inside a sequence, so matching on it alone
    picked the same frame number from another sequence (26510: 38 vs 18 boxes)."""
    f = eval_dir / 'result.pkl'
    if not f.exists():
        return None
    res = pickle.load(open(f, 'rb'))
    if idx >= len(res) or str(res[idx]['frame_id']) != str(fid):
        return dict(n_ours=None, n_ref=None, max_offset=None, note='result.pkl order differs from the dataset')
    ref = res[idx]
    a, b = anno['boxes_lidar'][anno['score'] >= 0.3], ref['boxes_lidar'][ref['score'] >= 0.3]
    if len(a) != len(b):
        return dict(n_ours=len(a), n_ref=len(b), max_offset=None)
    if not len(a):
        return dict(n_ours=0, n_ref=0, max_offset=0.0)
    d = np.linalg.norm(a[:, None, :3] - b[None, :, :3], axis=2).min(1)
    return dict(n_ours=len(a), n_ref=len(b), max_offset=float(d.max()))


def run_row(row, frames, out_dir, cache_dir, logger):
    t0 = time.time()
    cfg, cfg_src = vc.load_run_cfg(row['run'])
    classes = list(cfg.CLASS_NAMES)
    eval_ont = cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None)
    out = dict(row=row, classes=classes, target=dict(frames=[]), sources=[], calib=[], cfg_source=cfg_src)
    crun, cep = row['ckpt'].split(':') if not row['ckpt'].startswith('/') else (None, None)
    if not row.get('eval_dir') and crun == row['run']:
        row['eval_dir'] = str(vc.find_run_dir(crun) / 'files' / 'eval' / 'eval_with_train' / ('epoch_%s' % cep) / 'val')

    # ------------------------------------------------ target: eval mode, the run's scoring path
    tcfg = vc.eval_config(cfg)
    tset = vc.build_dataset(tcfg, classes, False, eval_ont, logger)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(classes), dataset=tset)
    model.load_params_from_file(filename=resolve_ckpt(row['ckpt']), to_cpu=False, logger=logger)
    model.cuda().eval()
    teacher = None
    tckpt = row.get('teacher') or (teacher_ckpt_from_args(row['run']) if cfg.get('SELF_TRAIN', None) else None)
    if tckpt:
        tc = cfg.SELF_TRAIN.MODEL_TEACHER
        teacher = build_network(model_cfg=tc, num_class=len(tc.get('CLASS_NAMES', None) or classes), dataset=tset)
        teacher.load_params_from_file(filename=tckpt, to_cpu=False, logger=logger)
        teacher.cuda().eval()
        out['pseudo_rule'] = dict(teacher=tckpt, score=list(cfg.SELF_TRAIN.SCORE_THRESH), neg=list(cfg.SELF_TRAIN.NEG_THRESH))
    zs = vc.shift_z(tcfg)
    tframes = frame_sets_for(tset, tcfg, frames, 'val')
    out['target'].update(kind=vc.dataset_kind(tset), shift_z=zs, fov_filter=bool(tcfg.get('TEST', {}).get('BOX_FILTER', {}).get('FOV_FILTER', False)))
    shown = []
    for f in tframes:
        idx = vc.index_of(tset, f['fid']); shown.append(idx)
        dd, pts = vc.processed(tset, idx)
        gtb, gtn = gt_from_dict(dd, classes)
        batch, pred = infer(model, tset, dd)
        anno = tset.generate_prediction_dicts(batch, pred, classes)[0]      # un-shifted; KITTI FOV filter as scored
        keep = anno['score'] >= SAVE_SCORE
        boxes = anno['boxes_lidar'][keep].astype(np.float32); boxes[:, 2] += zs   # back to the processed frame
        rec = dict(fid=f['fid'], image=f['image'], points=pts[:, :4].astype(np.float32), gt=(gtb, gtn),
                   pred=(boxes, anno['score'][keep], vc.canon(np.asarray(anno['name'])[keep])),
                   check=check_against_result_pkl(Path(row['eval_dir']), idx, dd['frame_id'], anno) if row.get('eval_dir') else None)
        if teacher is not None:
            with torch.no_grad():
                tp, _ = teacher(batch)
            tp = tp[0]
            s, lab, b = tp['pred_scores'].cpu().numpy(), tp['pred_labels'].cpu().numpy(), tp['pred_boxes'].cpu().numpy()
            pos = s >= np.array(cfg.SELF_TRAIN.SCORE_THRESH)[lab - 1]
            ign = (~pos) & (s >= np.array(cfg.SELF_TRAIN.NEG_THRESH)[lab - 1])
            rec['pseudo'] = (b[pos | ign, :7].astype(np.float32), s[pos | ign], vc.canon(np.array(classes)[lab[pos | ign] - 1]), pos[pos | ign])
        out['target']['frames'].append(rec)
    # a flash (PandarGT) target or source: measure every cloud inside its 60 deg cone (20260928_02)
    cone = 60.0 if any(int(dc.get('LIDAR_DEVICE', 0)) == 1 and dc.get('DATASET') == 'PandasetDataset'
                       for dc in [tcfg] + [d for _, d in vc.source_configs(cfg)]) else None
    out['close_cone'] = cone
    out['target']['close'] = closeness(tset, shown, lambda i: vc.processed(tset, i)[0], classes, cone=cone)
    del tset

    # ------------------------------------------------ sources: training mode, world augmentation off
    snames, sont = vc.source_class_names_and_ontology(cfg)
    for name, dc in vc.source_configs(cfg):
        sset = vc.build_dataset(vc.world_augs_off(dc), snames, True, sont, logger)
        calib = None
        if dc.get('HIST_DIST_FOREGROUND_FROM_PSEUDO_LABELS', False):
            calib = calibrate_foreground(sset, cfg, dc, row, logger, cache_dir)
        elif not dc.get('HIST_DIST_ON_THE_FLY', False) and sset.data_processor.hist_dist_src is not None:
            # the MIRU2025-era configs install SHIPPED histogram files at dataset init (HIST_DIST_*_PATH)
            rate = sset.data_processor.per_bin_sample_rate(processor_cfg(dc))
            calib = dict(shipped=True, rate_min=float(rate.min()), rate_max=float(rate.max()),
                         bins_below_1=int((rate < 1).sum()), files=sorted({str(v) for k, v in dc.items() if k.startswith('HIST_DIST_') and k.endswith('_PATH')}))
        elif dc.get('HIST_DIST_ON_THE_FLY', False):
            calib = calibrate(sset, cfg, dc, row.get('calib_split', 'train'), logger, cache_dir,
                              cone_aug_on=row.get('cone_aug_on', False))
        out['calib'].append((name, calib))
        sframes = frame_sets_for(sset, dc, frames, 'train')
        src = dict(name=name, kind=vc.dataset_kind(sset), shift_z=vc.shift_z(dc), frames=[], calib=calib,
                   max_sweeps=dc.get('MAX_SWEEPS', 1), motion_comp=bool(dc.get('GT_BOXES_MOTION_COMPENSATION', False)))
        shown = []
        for f in sframes:
            idx = vc.index_of(sset, f['fid']); shown.append(idx)
            info = vc.infos(sset)[idx]
            raw = vc.raw_points(sset, info)
            rb, rn = vc.raw_labels(sset, info)
            dd, pts = vc.processed(sset, idx)
            gtb, gtn = gt_from_dict(dd, snames)
            batch, pred = infer(model, sset, dd)
            p = pred[0]
            s = p['pred_scores'].cpu().numpy(); keep = s >= SAVE_SCORE
            src['frames'].append(dict(
                fid=f['fid'], image=f['image'], raw=raw.astype(np.float32), raw_gt=(rb, rn),
                points=pts[:, :4].astype(np.float32), gt=(gtb, gtn),
                pred=(p['pred_boxes'].cpu().numpy()[keep, :7], s[keep], vc.canon(np.array(classes)[p['pred_labels'].cpu().numpy()[keep] - 1]))))
        src['close_raw'] = closeness(sset, shown, None, snames, raw=True, cone=cone)
        src['close'] = closeness(sset, shown, lambda i: vc.processed(sset, i)[0], snames, cone=cone)
        out['sources'].append(src)
        del sset
    out['seconds'] = time.time() - t0
    pickle.dump(out, open(out_dir / ('%s.pkl' % row['job']), 'wb'))
    return out


def closeness(ds, shown, get_dd, classes, raw=False, cone=None):
    """Per-ring points/frame and (range, points) per Car box over CLOSE_FRAMES frames. With `cone`, both
    counted inside that forward cone (full angle): a limited-FOV sensor must be compared inside its FOV."""
    I = vc.infos(ds)
    rings, boxes = [], []
    for i in closeness_frames(len(I), shown):
        if raw:
            pts = vc.raw_points(ds, I[i]); b, n = vc.raw_labels(ds, I[i])
        else:
            dd = get_dd(i); pts = vc.voxel_points(dd); b, n = gt_from_dict(dd, classes)
        rings.append(ring_counts(pts, cone))
        boxes.append(car_box_points(pts, b, n, cone))
    return dict(rings=np.array(rings), car_boxes=np.concatenate(boxes), n_frames=len(rings), cone=cone)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True)
    ap.add_argument('--manifest', default=str(HERE / 'manifest.yaml'))
    ap.add_argument('--out', default='/home/koyama/data/viz/dump')
    args = ap.parse_args()
    from pcdet.utils import common_utils
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / 'calib_cache'; cache.mkdir(exist_ok=True)
    logger = common_utils.create_logger(out_dir / 'dump.log')
    frames = json.load(open(HERE / 'viz_frames.json'))
    rows = {str(r['job']): r for r in yaml.safe_load(open(args.manifest))}
    import traceback
    failed = []
    jobs = list(rows) if args.rows == 'all' else args.rows.split(',')
    for job in jobs:
        row = rows[job]
        logger.info('==== %s %s', job, row['row'])
        try:
            o = run_row(row, frames, out_dir, cache, logger)
        except Exception:  # noqa: BLE001 - one row's failure must not cost the rest of the batch
            tb = traceback.format_exc()
            logger.error('%s FAILED\n%s', job, tb)
            (out_dir / ('%s.error.txt' % job)).write_text(tb)
            failed.append(job)
            torch.cuda.empty_cache()
            continue
        (out_dir / ('%s.error.txt' % job)).unlink(missing_ok=True)     # a stale failure from an earlier batch
        logger.info('%s done in %.0f s; calib %s; result.pkl check %s', job, o['seconds'], o['calib'],
                    [fr['check'] for fr in o['target']['frames']])
    logger.info('batch finished; failed: %s', failed or 'none')


if __name__ == '__main__':
    main()
