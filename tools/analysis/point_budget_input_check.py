"""Pre-launch check of a FIXED-POINT-BUDGET source row (IA-SSD's sample_points): what each processor step does to the
points and to the Car training boxes, against the target as scored.

Built for the IA-SSD replication of 27807 (experiments_md 20261005_01 §15.8): control / density-matched C1 / scan
pattern, all three ending in sample_points 16,384 + drop_empty_gt_boxes. Installs the density calibration exactly as
train.py does (HIST_DIST_BEFORE_POINT_BUDGET included) BEFORE instrumenting the processor. Reports, over strided
source TRAIN frames (training mode, as trained) and target frames of the EVALUATION split (eval mode, as scored):
- points per frame within 75 m after each step; the budget's padding (duplicate share) and the smallest pre-budget frame;
- the radial profile, source / target, before the budget and after it;
- Car training boxes: 0 points entering the processor, emptied by each point-dropping step, and positives left with
  0 points at the end (must be 0);
- median points per Car box by ring after the budget, source vs target. ANALYSIS: the target side reads nuScenes val
  labels to describe the input, never to choose anything.

    python analysis/point_budget_input_check.py <cfg> <calibration frames> <source frames> <target frames>
"""
import sys
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.point_calibration import calibration_target_config, link_point_calibration
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

cfg_file, n_calib, n_src, n_tgt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cfg = EasyDict(); cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger()
dc = cfg.DATA_CONFIG
src, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                             logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
if dc.get('HIST_DIST_ON_THE_FLY', False):
    tcfg, _ = calibration_target_config(cfg.DATA_CONFIG_TAR)
    tcal, _, _ = build_dataloader(dataset_cfg=tcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                  logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    link_point_calibration(src, tcal, num_frames=n_calib, num_bins=dc.get('HIST_DIST_BINS', 50),
                           max_dist=dc.get('HIST_DIST_MAX_DIST', 75.0), logger=logger,
                           skip_point_budget=dc.get('HIST_DIST_BEFORE_POINT_BUDGET', False))
    del tcal
tgt, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                             workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))

RINGS = [0, 10, 20, 30, 40, 50, 75]


def instrument(ds):
    """Record (points, gt_boxes) entering the processor and after every step. Call AFTER the calibration."""
    proc, rec = ds.data_processor, {}
    orig_forward = proc.forward

    def forward(data_dict):
        rec.clear()
        rec['input'] = (data_dict['points'][:, :3].copy(), None if data_dict.get('gt_boxes') is None
                        else np.asarray(data_dict['gt_boxes']).copy())
        return orig_forward(data_dict=data_dict)

    def wrap(p, name):
        def step(data_dict=None):
            out = p(data_dict=data_dict)
            rec[name] = (out['points'][:, :3].copy(), None if out.get('gt_boxes') is None
                         else np.asarray(out['gt_boxes']).copy())
            return out
        return step

    names = [p.func.__name__ for p in proc.data_processor_queue]
    proc.data_processor_queue = [wrap(p, n) for p, n in zip(proc.data_processor_queue, names)]
    proc.forward = forward
    return rec, names


def in75(p):
    return np.hypot(p[:, 0], p[:, 1]) < 75


def ring_hist(p):
    r = np.hypot(p[:, 0], p[:, 1])
    return np.histogram(r[r < 75], bins=RINGS)[0].astype(float)


def car_counts(points, boxes):
    if boxes is None or not len(boxes):
        return np.zeros(0), np.zeros(0)
    b = boxes[boxes[:, 7] == 1]
    if not len(b):
        return np.zeros(0), np.zeros(0)
    c = roiaware_pool3d_utils.points_in_boxes_cpu(np.ascontiguousarray(points, dtype=np.float32),
                                                  np.ascontiguousarray(b[:, :7], dtype=np.float32)).sum(1) \
        if len(points) else np.zeros(len(b))
    return np.hypot(b[:, 0], b[:, 1]), c


def walk(ds, n, rec, names):
    budget_at = names.index('sample_points')
    pre = names[budget_at - 1] if budget_at > 0 else 'input'
    out = dict(n_pre=[], n_final=[], dup=[], h_pre=np.zeros(len(RINGS) - 1), h_final=np.zeros(len(RINGS) - 1),
               steps={k: [] for k in ['input'] + names}, car_r=[], car_c=[],
               car=dict(entering=0, zero_entering=0, emptied={k: 0 for k in names}, final_zero=0, final=0))
    for i in range(0, len(ds), max(1, len(ds) // n))[:n]:
        d = ds[i]
        for k in out['steps']:
            if k in rec:
                out['steps'][k].append(in75(rec[k][0]).sum())
        p_pre, p_fin = rec[pre][0], rec['sample_points'][0]
        out['n_pre'].append(in75(p_pre).sum()); out['n_final'].append(in75(p_fin).sum())
        out['dup'].append(1 - len(np.unique(p_fin, axis=0)) / max(len(p_fin), 1))
        out['h_pre'] += ring_hist(p_pre); out['h_final'] += ring_hist(np.unique(p_fin, axis=0))
        # Car boxes along the queue (training only: eval keeps every box)
        boxes_in = rec['input'][1]
        r0, c0 = car_counts(rec['input'][0], boxes_in)
        out['car']['entering'] += len(c0); out['car']['zero_entering'] += int((c0 == 0).sum())
        prev = 'input'
        for k in names:
            if k in ('sample_points_hist_based', 'sample_points'):
                _, cb = car_counts(rec[prev][0], rec[prev][1])
                _, ca = car_counts(rec[k][0], rec[prev][1])
                out['car']['emptied'][k] += int(((cb > 0) & (ca == 0)).sum())
            prev = k
        rf, cf = car_counts(d['points'][:, :3], d.get('gt_boxes'))
        out['car']['final'] += len(cf); out['car']['final_zero'] += int((cf == 0).sum())
        # points per Car box after the budget, unique positions
        ru, cu = car_counts(np.unique(p_fin, axis=0), rec['sample_points'][1])
        out['car_r'].append(ru); out['car_c'].append(cu)
    out['car_r'] = np.concatenate(out['car_r']) if out['car_r'] else np.zeros(0)
    out['car_c'] = np.concatenate(out['car_c']) if out['car_c'] else np.zeros(0)
    return out


srec, snames = instrument(src)
trec, tnames = instrument(tgt)
S = walk(src, n_src, srec, snames)
T = walk(tgt, n_tgt, trec, tnames)
ns, nt = len(S['n_pre']), len(T['n_pre'])

print(f'\n{cfg_file}: calibration {n_calib}, {ns} source TRAIN frames (training mode), {nt} target EVAL frames')
print('source points per frame within 75 m after each step: ' +
      ', '.join(f'{k} {np.mean(v):.0f}' for k, v in S['steps'].items() if v))
print(f"source pre-budget: mean {np.mean(S['n_pre']):.0f}, min {np.min(S['n_pre'])}, p1 {np.percentile(S['n_pre'], 1):.0f}; "
      f"frames under the budget {np.mean(np.array(S['n_pre']) < 16384):.1%}, under half of it "
      f"{np.mean(np.array(S['n_pre']) < 8192):.1%}; duplicate share after the budget {np.mean(S['dup']):.1%}")
print(f"target pre-budget: mean {np.mean(T['n_pre']):.0f}, min {np.min(T['n_pre'])}; duplicate share {np.mean(T['dup']):.1%}")
print(f"source / target pre-budget points: {np.mean(S['n_pre']) / np.mean(T['n_pre']):.2f}")
print('| ring (m) | ' + ' | '.join(f'{RINGS[k]}-{RINGS[k + 1]}' for k in range(len(RINGS) - 1)) + ' |')
print('|---' * len(RINGS) + '|')
print('| pre-budget src / tgt | ' + ' | '.join(f"{(S['h_pre'][k] / ns) / max(T['h_pre'][k] / nt, 1e-9):.2f}"
                                             for k in range(len(RINGS) - 1)) + ' |')
print('| after budget (unique) src / tgt | ' + ' | '.join(f"{(S['h_final'][k] / ns) / max(T['h_final'][k] / nt, 1e-9):.2f}"
                                                        for k in range(len(RINGS) - 1)) + ' |')
c = S['car']; e = max(c['entering'], 1)
print(f"source Car boxes entering the processor {c['entering']}: 0 points already {c['zero_entering'] / e:.1%}; "
      + ', '.join(f"emptied by {k} {v / e:.1%}" for k, v in c['emptied'].items() if k.startswith('sample_points'))
      + f"; positives left after the processor {c['final']}, of which 0 points: {c['final_zero']}")
print('median points per Car box after the budget (unique positions; boxes holding >= 1), by ring:')
print('| ring (m) | ' + ' | '.join(f'{RINGS[k]}-{RINGS[k + 1]}' for k in range(len(RINGS) - 1)) + ' |')
print('|---' * len(RINGS) + '|')
for name, X in (('source', S), ('target (val labels, analysis)', T)):
    cells = []
    for k in range(len(RINGS) - 1):
        m = (X['car_r'] >= RINGS[k]) & (X['car_r'] < RINGS[k + 1]) & (X['car_c'] > 0)
        cells.append(f"{np.median(X['car_c'][m]):.0f} (n={m.sum()})" if m.any() else '-')
    print(f'| {name} | ' + ' | '.join(cells) + ' |')
