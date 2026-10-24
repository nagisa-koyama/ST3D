"""OSENS-2 pre-GPU check (experiments_md 20261025_03): every twin through the REAL test loader (full data processor, as
evaluation runs it), one dumped frame per arm with figures, and per-arm statistics over strided val frames. CPU only,
ANALYSIS (it reads target val labels only to place the boxes the arms are defined by, as the evaluation does).

Checks, per oracle (Waymo 26398, nuScenes 25743, KITTI 26843), on the dumped frame:
  - height twins: points and GT z move by exactly dz, x / y unchanged; a perfect prediction made in the twin's frame
    comes back to the base's infos-frame boxes (generate_prediction_dicts);
  - pitch twins: GT boxes identical to the base; points = pitch_points(base points) with the twin's origin and axis;
    prediction frames identical;
  - C twins: GT boxes identical; in-box / outside counts as the arm says;
  - nuScenes keepego: the extra points are exactly the |x|, |y| < 1.5 m square;
  - Waymo second return: the stored TOP block never exceeds the 64 x 2650 range image (first return only).
Per arm over --frames strided frames: points before / after, in-box share, ground fit failures and plane, and the share
of points in the shell between the two readings of "enlarged by 0.25 m" (+0.25 m in each dimension, the convention used,
against +0.25 m on every side) as a bound on that reading.

    python analysis/osens2_dump.py --out /home/koyama/data/osens2/dump_<date> [--frames 60] [--oracles waymo,nuscenes,kitti]
"""
import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
import _init_path  # noqa: F401,E402
from analysis.make_osens2_configs import ORACLES, arms, twin_path  # noqa: E402
from pcdet.datasets import osens2_utils as O  # noqa: E402


def cfg_of(path):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    return cfg_from_yaml_file(str(path), EasyDict())


def dataset_of(cfg):
    from pcdet.datasets import build_dataloader
    log = logging.getLogger('osens2_dump'); log.setLevel(logging.WARNING)
    ds, _, _ = build_dataloader(dataset_cfg=copy.deepcopy(cfg.DATA_CONFIG_TAR), class_names=cfg.CLASS_NAMES, batch_size=1,
                                dist=False, workers=0, logger=log, training=False,
                                model_ontology=cfg.get('ONTOLOGY', None))
    return ds


def unshifted_predictions(ds, frame):
    """generate_prediction_dicts on a 'perfect' prediction = this frame's own GT boxes (model frame)."""
    import torch
    boxes = frame['gt_boxes'][:, :7].astype(np.float32)
    pred = [{'pred_boxes': torch.from_numpy(boxes.copy()), 'pred_scores': torch.ones(len(boxes)),
             'pred_labels': torch.ones(len(boxes), dtype=torch.long)}]
    batch = {'frame_id': [frame['frame_id']]}
    if 'calib' in frame:
        batch.update(calib=[frame['calib']], image_shape=[frame['image_shape']])
    return ds.generate_prediction_dicts(batch, pred, ['Car', 'Pedestrian', 'Cyclist'])[0]['boxes_lidar']


class Recorder:
    """Wraps osens2_utils.perturb to keep the stats of every call, plus the inputs of the first."""

    def __init__(self):
        self.stats, self.first = [], None
        self._orig = O.perturb

    def __enter__(self):
        def wrapped(points, gt_boxes, frame_id, cfg):
            out, st = self._orig(points, gt_boxes, frame_id, cfg)
            if self.first is None:
                self.first = dict(points=np.array(points), gt_boxes=None if gt_boxes is None else np.array(gt_boxes),
                                  out=np.array(out), frame_id=frame_id)
            if gt_boxes is not None and len(gt_boxes) and cfg.get('BOX_MARGIN', None) is not None:
                # shell between +m in each dimension (used) and +m on every side (= +2m per dimension)
                a = O.in_any_box(points, O.enlarge_boxes(np.asarray(gt_boxes), cfg['BOX_MARGIN']))
                b = O.in_any_box(points, O.enlarge_boxes(np.asarray(gt_boxes), 2 * cfg['BOX_MARGIN']))
                st['n_shell'] = int((b & ~a).sum())
            self.stats.append(st)
            return out, st
        O.perturb = wrapped
        return self

    def __exit__(self, *a):
        O.perturb = self._orig


def figure(path, rec, title, kind):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p, q, b = rec['points'], rec['out'], rec['gt_boxes']
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    if kind == 'pitch':
        sel = np.abs(p[:, 1 if title.endswith('y') else 0]) < 60
        for a_, (pts, nm) in zip(ax, ((p, 'as read'), (q, 'perturbed'))):
            f = 1 if 'fwd=y' in title else 0
            lat = 1 - f
            s = np.abs(pts[:, lat]) < 3.0
            a_.scatter(pts[s, f], pts[s, 2], s=0.3, c='k')
            a_.set_xlabel('forward (m)'); a_.set_ylabel('z (m)'); a_.set_title(f'{nm}: |lateral| < 3 m'); a_.grid(True)
            a_.set_ylim(-3, 6)
    else:
        inside = O.in_any_box(p, O.enlarge_boxes(b, 0.25)) if b is not None and len(b) else np.zeros(len(p), bool)
        inside_q = O.in_any_box(q, O.enlarge_boxes(b, 0.25)) if b is not None and len(b) else np.zeros(len(q), bool)
        for a_, (pts, ins, nm) in zip(ax, ((p, inside, 'as read'), (q, inside_q, 'perturbed'))):
            s = (np.abs(pts[:, 0]) < 50) & (np.abs(pts[:, 1]) < 50)
            a_.scatter(pts[s & ~ins, 0], pts[s & ~ins, 1], s=0.2, c=pts[s & ~ins, 2], cmap='viridis', vmin=-0.5, vmax=3)
            a_.scatter(pts[s & ins, 0], pts[s & ins, 1], s=0.6, c='r')
            a_.set_aspect('equal'); a_.set_title(f'{nm}: {s.sum()} pts within 50 m (red = in an enlarged box)')
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def waymo_first_return(ds, n=300):
    idx = np.linspace(0, len(ds.infos) - 1, n).astype(int)
    tops = np.array([ds.infos[i]['num_points_of_each_lidar'][0] for i in idx])
    return {'frames': int(n), 'top_points_max': int(tops.max()), 'top_points_median': float(np.median(tops)),
            'range_image_pixels': 64 * 2650, 'exceeds_one_return': bool(tops.max() > 64 * 2650)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--oracles', default='waymo,nuscenes,kitti')
    ap.add_argument('--index', type=int, default=100, help='val index of the dumped frame')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {}
    for oracle in args.oracles.split(','):
        t0 = time.time()
        name = ORACLES[oracle][0]
        base_cfg = cfg_of(Path('cfgs/da-ieee-access') / f'{name}.yaml')
        db = dataset_of(base_cfg)
        n_val = len(db)
        strided = np.linspace(0, n_val - 1, args.frames).astype(int)
        fb = db[args.index]
        pred_b = unshifted_predictions(db, fb)
        R = report[oracle] = {'n_val': n_val, 'index': args.index, 'frame_id': str(fb['frame_id']),
                              'base_points': int(len(fb['points'])), 'arms': {}}
        if oracle == 'waymo':
            R['first_return_check'] = waymo_first_return(db)
        for tag, tar in arms(oracle):
            tcfg = cfg_of(twin_path(oracle, tag))
            dt = dataset_of(tcfg)
            A = R['arms'][tag] = {}
            with Recorder() as rec:
                ft = dt[args.index]
            A['points'] = int(len(ft['points']))
            same_gt = np.array_equal(ft['gt_boxes'], fb['gt_boxes'])
            if 'SHIFT_COOR' in tar:
                dz = tar['SHIFT_COOR'][2] - float(base_cfg.DATA_CONFIG_TAR.SHIFT_COOR[2])
                A['check'] = bool(len(ft['points']) == len(fb['points'])
                                  and np.allclose(ft['points'][:, 2] - fb['points'][:, 2], dz, atol=1e-4)
                                  and np.array_equal(ft['points'][:, :2], fb['points'][:, :2])
                                  and np.allclose(ft['gt_boxes'][:, 2] - fb['gt_boxes'][:, 2], dz, atol=1e-4)
                                  and np.allclose(unshifted_predictions(dt, ft), pred_b, atol=1e-4))
                A['what'] = 'points and GT z + %.2f; predictions back to the infos frame' % dz
            elif 'EVAL_KEEP_EGO_POINTS' in tar:
                raw_b = db.get_lidar_with_sweeps(args.index, max_sweeps=1)
                raw_t = dt.get_lidar_with_sweeps(args.index, max_sweeps=1)
                ego = (np.abs(raw_t[:, 0]) < 1.5) & (np.abs(raw_t[:, 1]) < 1.5)
                A['check'] = bool(same_gt and np.array_equal(raw_t[~ego], raw_b))
                A['what'] = 'ego points added: %d raw (%d -> %d processed)' % (ego.sum(), len(fb['points']), len(ft['points']))
            else:
                cfg = tcfg.DATA_CONFIG_TAR.EVAL_OSENS2
                A['hook'] = rec.stats[0] if rec.stats else None
                if cfg.get('PITCH_DEG', None) is not None:
                    pitched = O.pitch_points(rec.first['points'], cfg.PITCH_DEG, cfg.SENSOR_ORIGIN, cfg.FORWARD_AXIS)
                    A['check'] = bool(same_gt and np.array_equal(rec.first['out'], pitched)
                                      and np.array_equal(unshifted_predictions(dt, fb), pred_b))
                    A['what'] = 'points pitched %+.0f deg about %s (forward %s); GT and prediction frames unchanged' % (
                        cfg.PITCH_DEG, list(cfg.SENSOR_ORIGIN), cfg.FORWARD_AXIS)
                    kind = 'pitch'
                else:
                    st = rec.stats[0]
                    if cfg.get('REGION_THIN', None) is not None:
                        side, keep = cfg.REGION_THIN.SIDE, cfg.REGION_THIN.KEEP
                        exp_in = st['n_inside'] if side == 'background' else int(round(st['n_inside'] * keep))
                        exp_out = int(round(st['n_outside'] * keep)) if side == 'background' else st['n_outside']
                        A['check'] = bool(same_gt and st['n_out'] == exp_in + exp_out)
                        A['what'] = '%s thinned to %.2f: in-box %d -> %d, outside %d -> %d' % (
                            side, keep, st['n_inside'], exp_in, st['n_outside'], exp_out)
                    else:
                        A['check'] = bool(same_gt and st['fit_failed'] == 0 and st['n_out'] == st['n_in'] - st['n_dropped'])
                        A['what'] = 'ground removed outside boxes: %d of %d points dropped, plane %s' % (
                            st['n_dropped'], st['n_in'], st.get('plane'))
                    kind = 'region'
                figure(out / f'{oracle}_{tag}.png', rec.first,
                       f'{oracle} {tag} frame {fb["frame_id"]} fwd={cfg.get("FORWARD_AXIS", "")}', kind)
                # statistics over strided frames (the hook only; the data processor's voxelisation is skipped)
                with Recorder() as rec2:
                    for i in strided:
                        dt[int(i)]
                S = rec2.stats
                agg = {'frames': len(S), 'points_in_mean': float(np.mean([s['n_in'] for s in S])),
                       'points_out_mean': float(np.mean([s['n_out'] for s in S]))}
                if 'n_inside' in S[0]:
                    agg['inside_share'] = float(np.sum([s['n_inside'] for s in S]) / max(1, np.sum([s['n_in'] for s in S])))
                    agg['shell_share_of_all'] = float(np.sum([s.get('n_shell', 0) for s in S]) / max(1, np.sum([s['n_in'] for s in S])))
                    agg['shell_share_of_inside'] = float(np.sum([s.get('n_shell', 0) for s in S]) / max(1, np.sum([s['n_inside'] for s in S])))
                if 'fit_failed' in S[0]:
                    agg['fit_failed'] = int(np.sum([s['fit_failed'] for s in S]))
                    agg['dropped_share'] = float(np.sum([s['n_dropped'] for s in S]) / max(1, np.sum([s['n_in'] for s in S])))
                    planes = np.array([s['plane'] for s in S if s.get('plane') is not None])
                    if len(planes):
                        agg['plane_c_median'] = float(np.median(planes[:, 2]))
                        agg['slope_x_deg_median'] = float(np.degrees(np.arctan(np.median(planes[:, 0]))))
                        agg['slope_y_deg_median'] = float(np.degrees(np.arctan(np.median(planes[:, 1]))))
                A['strided'] = agg
            print(f'{oracle} {tag}: check={A.get("check")} {A.get("what", "")}', flush=True)
        R['seconds'] = round(time.time() - t0, 1)
        (out / 'dump.json').write_text(json.dumps(report, indent=1, default=lambda o: o.tolist()
                                                  if hasattr(o, 'tolist') else str(o)))
    ok = all(a.get('check') for r in report.values() for a in r['arms'].values())
    print('ALL CHECKS PASS' if ok else 'SOME CHECK FAILED', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
