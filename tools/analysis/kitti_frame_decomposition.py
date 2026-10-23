"""Frame-level bookkeeping of the KITTI metric (Car, BEV, AP_R40 at IoU 0.7) on a non-KITTI target.

Used by `scene_error_decomposition.py` (ENV E1) and `nuscenes_condition_ap.py` (ENV E2), experiments_md 20261024_02.
CPU only. Nothing here re-derives the protocol:

- `capture_eval_annos` builds the eval dataset exactly as test.py does (DATA_CONFIG_TAR, training=False), checks that
  result.pkl is in the dataset's frame order, and runs the dataset's OWN `evaluation(..., eval_metric='kitti')`. The
  official evaluator it calls is `kitti_eval_cpu.load_official_eval()` (eval.py unmodified, only the CUDA IoU kernel
  swapped for pcdet's CPU op), wrapped so that the KITTI-format annos the dataset built (class map, fabricated 2D
  boxes, every GT box as stored) are captured as they enter `get_official_eval_result`. The Car BEV AP the sanity
  gate compares with the logged value is the official `eval_class` on exactly those annos (`official_car_bev_ap`),
  with the IoU computed one frame at a time: a pair's IoU does not depend on the batch it is computed in, and the
  default 100-part batching would cost ~1e9 cross-frame pairs per city on the CPU op (the bbox and 3D metrics, which
  the decomposition does not use, are skipped for the same reason).
- `CarBevFrames` takes those annos and, per frame, the official `calculate_iou_partly` (BEV overlaps), `_prepare_data`
  (Car, difficulty 0: on a non-KITTI target the three columns are identical, "all objects") and the official
  `compute_statistics_jit` first pass (the TP scores `get_thresholds` reads). The second pass is recorded per frame
  as a step function of the score threshold: a frame's (tp, fp, fn) change only at its own Car detection scores, so
  the counts are taken at every distinct score with `_stats_detail` (a copy of `compute_statistics_jit`'s
  compute_fp branch for metric 1 that also returns which detections were assigned) and asserted equal to the
  official function at every one of those thresholds. AP over any multiset of frames (a city, a condition cell, a
  bootstrap resample) is then the official `get_thresholds` plus the summed step functions, and over all frames it is
  asserted equal to the official `eval_class`.

Categories (T05 §5 item 4, E1; the metric's own matching at BEV IoU 0.7 is never changed):
- background FP: a false positive whose max BEV IoU with any GT car is < 0.1;
- localisation FP: a false positive with max BEV IoU >= 0.1 (mislocalised, 0.1 <= IoU <= 0.7, and duplicates: IoU
  above 0.7 with a GT another detection was matched to);
- missed GT: a GT car with no Car detection (any saved score) at BEV IoU >= 0.1.
"AP with background (localisation) FPs removed" keeps every TP, FN and threshold and drops that category from the FP
count at each threshold; "AP with missed GT removed" drops those GT from the GT set (they never match, so only FN and
the GT count change). Background FPs and missed GT can also be removed by deleting them from the annos and re-running
the official `eval_class`; `official_car_bev_ap` does that as a cross-check.
"""
import copy
import pickle
import sys
import types
from pathlib import Path

import numba
import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for _p in (str(TOOLS), str(TOOLS / 'analysis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _init_path  # noqa: F401,E402  (the live repo's pcdet, not the image's stale copy)
from kitti_eval_cpu import load_official_eval  # noqa: E402

CAR = 0                # class index in the official evaluator
DIFFICULTY = 0         # identical columns on a non-KITTI target: "all objects", never "moderate"
METRIC_BEV = 1
MIN_OVERLAP = 0.7      # Car BEV, the metric's own threshold (strict >)
CATEGORY_IOU = 0.1     # background / localisation / missed boundary (design)
N_SAMPLE_PTS = 41
_EVAL_MODULE = 'pcdet.datasets.kitti.kitti_object_eval_python.eval'
_PACKAGE = 'pcdet.datasets.kitti.kitti_object_eval_python'
DT_KEYS = ('name', 'score', 'bbox', 'truncated', 'occluded', 'location', 'dimensions', 'rotation_y', 'alpha')
GT_KEYS = ('name', 'bbox', 'truncated', 'occluded', 'location', 'dimensions', 'rotation_y', 'alpha')


# ---------------------------------------------------------------------------------------------------------------------
# The eval set and its KITTI-format annos, built by the dataset's own code
# ---------------------------------------------------------------------------------------------------------------------
def eval_data_config(cfg):
    """test.py's get_eval_configs: DATA_CONFIG_TAR when present, else DATA_CONFIG."""
    return cfg.DATA_CONFIG_TAR if cfg.get('DATA_CONFIG_TAR', None) else cfg.DATA_CONFIG


def capture_eval_annos(cfg_file, result_pkl, official=None, logger=None):
    """Run the dataset's own KITTI-metric evaluation on a saved result.pkl and capture its annos.

    Returns dict(car_bev_r40=<official Car BEV AP_R40 @0.7>, gt=[...], dt=[...], infos=dataset.infos, class_names,
    eval_cfg).
    `gt[i]` is the KITTI-format GT of `infos[i]` (every box as stored, in the info's order), `dt[i]` the detections.
    """
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils

    official = official or load_official_eval()
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    eval_cfg = eval_data_config(cfg)
    assert eval_cfg.DATASET == 'NuScenesDataset', eval_cfg.DATASET
    assert not eval_cfg.get('GT_FILTER', None), 'a GT filter would break the info <-> KITTI-anno box alignment'
    assert cfg.MODEL.POST_PROCESSING.EVAL_METRIC == 'kitti'
    class_names = list(cfg.CLASS_NAMES)
    assert ':' not in class_names[0], 'multi-dataset class names: eval_utils remaps them; not handled here'
    ontology = cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None)
    logger = logger or common_utils.create_logger()
    dataset, _, _ = build_dataloader(dataset_cfg=eval_cfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                     workers=0, logger=logger, training=False, model_ontology=ontology)
    with open(result_pkl, 'rb') as f:
        det_annos = pickle.load(f)
    assert len(det_annos) == len(dataset.infos), (len(det_annos), len(dataset.infos))
    for d, info in zip(det_annos, dataset.infos):
        assert d['metadata']['token'] == info['token'], 'result.pkl is not in the eval dataset order'

    captured = {}

    def capture(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
        captured.update(gt=gt_annos, dt=dt_annos, classes=list(current_classes))
        return '', {}

    shim = types.ModuleType(_EVAL_MODULE)
    shim.get_official_eval_result = capture
    with _module_swapped(shim):
        dataset.evaluation(copy.deepcopy(det_annos), class_names, eval_metric='kitti')
    assert 'Car' in captured['classes'], captured['classes']
    for g, info in zip(captured['gt'], dataset.infos):
        assert g['token'] == info['token'] and len(g['name']) == len(info['gt_boxes']) == len(g['num_lidar_pts'])
    ap = official_car_bev_ap(captured['gt'], captured['dt'], official)
    return dict(car_bev_r40=ap, gt=captured['gt'], dt=captured['dt'], infos=dataset.infos,
                class_names=class_names, eval_cfg=eval_cfg)


class _module_swapped:
    """Point `from ..kitti.kitti_object_eval_python import eval` at `shim` (sys.modules AND the package attribute:
    a submodule already bound on its package ignores sys.modules), restoring both afterwards."""

    def __init__(self, shim):
        self.shim = shim

    def __enter__(self):
        import importlib
        self.pkg = importlib.import_module(_PACKAGE)
        self.saved_mod = sys.modules.get(_EVAL_MODULE)
        self.had_attr = hasattr(self.pkg, 'eval')
        self.saved_attr = getattr(self.pkg, 'eval', None)
        sys.modules[_EVAL_MODULE] = self.shim
        self.pkg.eval = self.shim
        return self

    def __exit__(self, *exc):
        if self.saved_mod is None:
            sys.modules.pop(_EVAL_MODULE, None)
        else:
            sys.modules[_EVAL_MODULE] = self.saved_mod
        if self.had_attr:
            self.pkg.eval = self.saved_attr
        else:
            delattr(self.pkg, 'eval')
        return False


# ---------------------------------------------------------------------------------------------------------------------
# Official AP on (possibly edited) annos
# ---------------------------------------------------------------------------------------------------------------------
def official_car_bev_ap(gt_annos, dt_annos, official):
    """Car BEV AP_R40 at IoU 0.7, difficulty 0, through the official eval_class (as do_eval runs it)."""
    min_overlaps = np.full((1, 3, 1), MIN_OVERLAP)
    ret = official.eval_class(gt_annos, dt_annos, [CAR], [DIFFICULTY], METRIC_BEV, min_overlaps,
                              num_parts=len(gt_annos))  # one frame per part: same pair IoUs, no cross-frame pairs
    return float(official.get_mAP_R40(ret['precision'])[0, 0, 0])


def subset_anno(anno, keep, keys):
    """A copy holding only the evaluator's per-box keys, restricted to boxes where `keep` is True."""
    out = {k: np.asarray(anno[k])[keep] for k in keys}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Per-frame step functions of the official second pass
# ---------------------------------------------------------------------------------------------------------------------
@numba.jit(nopython=True)
def _stats_detail(overlaps, ignored_gt, ignored_det, dt_scores, min_overlap, thresh):
    """compute_statistics_jit(..., compute_fp=True) for metric 1 (no DontCare step), plus the assignment.

    Same loop order, same tie-breaking. Returns tp, fp, fn, assigned[det] (bool), matched_det[gt] (-1 = none).
    """
    det_size = overlaps.shape[0]
    gt_size = overlaps.shape[1]
    assigned = np.zeros(det_size, dtype=np.bool_)
    below = np.zeros(det_size, dtype=np.bool_)
    for j in range(det_size):
        if dt_scores[j] < thresh:
            below[j] = True
    matched = np.full(gt_size, -1, dtype=np.int64)
    NO_DETECTION = -10000000
    tp, fp, fn = 0, 0, 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0.0
        assigned_ignored_det = False
        for j in range(det_size):
            if ignored_det[j] == -1:
                continue
            if assigned[j]:
                continue
            if below[j]:
                continue
            overlap = overlaps[j, i]
            if (overlap > min_overlap) and (overlap > max_overlap or assigned_ignored_det) and ignored_det[j] == 0:
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (overlap > min_overlap) and (valid_detection == NO_DETECTION) and ignored_det[j] == 1:
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif (valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1):
            assigned[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            assigned[det_idx] = True
            matched[i] = det_idx
    for j in range(det_size):
        if not (assigned[j] or ignored_det[j] == -1 or ignored_det[j] == 1 or below[j]):
            fp += 1
    return tp, fp, fn, assigned, matched


class CarBevFrames:
    """Per-frame records of the official Car BEV evaluation; AP over any multiset of frames.

    Per GT car (`self.gt`, frame-contiguous, `gt_offsets`): frame, box index in the anno, best BEV IoU with any Car
    detection, matched at IoU 0.7 by the metric with every saved detection (the end of the PR curve); missed = best
    < 0.1. `gt_meta` = {key: per-frame arrays aligned with the GT anno boxes} adds e.g. range and points per box.
    Per Car detection (`self.dt`, `dt_offsets`): frame, score, max BEV IoU with any GT car, category with every saved
    detection participating (0 tp, 1 background FP, 2 localisation FP).
    """

    def __init__(self, gt_annos, dt_annos, official, gt_meta=None, verify=True):
        assert len(gt_annos) == len(dt_annos)
        self.official = official
        self.n_frames = len(gt_annos)
        overlaps, _, _, _ = official.calculate_iou_partly(dt_annos, gt_annos, METRIC_BEV, max(self.n_frames, 1))
        prep = official._prepare_data(gt_annos, dt_annos, CAR, DIFFICULTY)
        gt_datas, dt_datas, ignored_gts, ignored_dets, dontcares, _, total_valid = prep
        self.total_valid_gt = int(total_valid)

        tp_scores, n_valid, n_missed = [], np.zeros(self.n_frames, np.int64), np.zeros(self.n_frames, np.int64)
        ev = {k: [] for k in ('score', 'tp', 'fp', 'fn', 'fpb', 'fpl')}
        ev_count = np.zeros(self.n_frames, np.int64)
        gt_rec = {k: [] for k in ('frame', 'best_iou', 'matched07', 'index')}
        meta_rec = {k: [] for k in (gt_meta or {})}
        dt_rec = {k: [] for k in ('frame', 'score', 'max_iou', 'category')}
        n_checked = 0
        for f in range(self.n_frames):
            ov = np.asarray(overlaps[f], np.float32)
            ig_gt, ig_dt = ignored_gts[f], ignored_dets[f]
            # the decomposition is exact only without ignored-but-matchable boxes (none on a fabricated-2D target)
            assert not np.any(ig_gt == 1) and not np.any(ig_dt == 1), 'ignored (=1) GT or detections present'
            scores = dt_datas[f][:, -1]
            _, _, _, _, thr = official.compute_statistics_jit(
                ov, gt_datas[f], dt_datas[f], ig_gt, ig_dt, dontcares[f], METRIC_BEV,
                min_overlap=MIN_OVERLAP, thresh=0.0, compute_fp=False)
            tp_scores.append(np.asarray(thr, np.float64))
            car_gt = ig_gt == 0
            car_dt = ig_dt == 0
            n_valid[f] = int(car_gt.sum())
            ov_car = ov[:, car_gt] if car_gt.any() else np.zeros((len(scores), 0), np.float32)
            det_max = ov_car.max(axis=1) if ov_car.shape[1] else np.zeros(len(scores), np.float32)
            ov_det = ov[car_dt] if car_dt.any() else np.zeros((0, ov.shape[1]), np.float32)
            gt_best = ov_det.max(axis=0) if ov_det.shape[0] else np.zeros(ov.shape[1], np.float32)
            missed = car_gt & (gt_best < CATEGORY_IOU)
            n_missed[f] = int(missed.sum())
            bkg = car_dt & (det_max < CATEGORY_IOU)

            uniq = np.unique(scores[car_dt])[::-1]  # descending; state k participates dets with score >= uniq[k-1]
            prev = (0, 0, int(n_valid[f]), 0, 0)
            if verify:
                o = official.compute_statistics_jit(ov, gt_datas[f], dt_datas[f], ig_gt, ig_dt, dontcares[f],
                                                    METRIC_BEV, min_overlap=MIN_OVERLAP, thresh=np.inf,
                                                    compute_fp=True)
                assert (o[0], o[1], o[2]) == prev[:3], (f, 'state 0', o[:3], prev)
            last_assigned, last_matched = np.zeros(len(scores), np.bool_), np.full(ov.shape[1], -1, np.int64)
            for t in uniq:
                tp, fp, fn, assigned, matched = _stats_detail(ov, ig_gt, ig_dt, scores, MIN_OVERLAP, t)
                part = car_dt & (scores >= t) & ~assigned
                fpb = int((part & bkg).sum())
                fpl = int((part & ~bkg).sum())
                assert fpb + fpl == fp
                if verify:
                    o = official.compute_statistics_jit(ov, gt_datas[f], dt_datas[f], ig_gt, ig_dt, dontcares[f],
                                                        METRIC_BEV, min_overlap=MIN_OVERLAP, thresh=t,
                                                        compute_fp=True)
                    assert (o[0], o[1], o[2]) == (tp, fp, fn), (f, t, o[:3], (tp, fp, fn))
                    n_checked += 1
                cur = (tp, fp, fn, fpb, fpl)
                ev['score'].append(t)
                for key, a, b in zip(('tp', 'fp', 'fn', 'fpb', 'fpl'), cur, prev):
                    ev[key].append(a - b)
                prev = cur
                last_assigned, last_matched = assigned, matched
            ev_count[f] = len(uniq)
            assert n_missed[f] <= prev[2], 'a missed GT must be a false negative at every threshold'
            gi = np.nonzero(car_gt)[0]
            gt_rec['frame'].append(np.full(len(gi), f))
            gt_rec['index'].append(gi)
            gt_rec['best_iou'].append(gt_best[gi])
            gt_rec['matched07'].append(last_matched[gi] >= 0)
            for key in meta_rec:
                meta = np.asarray(gt_meta[key][f])
                assert len(meta) == len(car_gt), (key, f, len(meta), len(car_gt))
                meta_rec[key].append(meta[gi])
            di = np.nonzero(car_dt)[0]
            cat = np.where(last_assigned[di], 0, np.where(bkg[di], 1, 2))  # 0 tp, 1 bkg, 2 loc (all saved dets)
            dt_rec['frame'].append(np.full(len(di), f))
            dt_rec['score'].append(scores[di])
            dt_rec['max_iou'].append(det_max[di])
            dt_rec['category'].append(cat)

        self.n_state_checks = n_checked
        self.tp_offsets = np.concatenate([[0], np.cumsum([len(t) for t in tp_scores])]).astype(np.int64)
        self.tp_flat = np.concatenate(tp_scores) if tp_scores else np.zeros(0)
        self.n_valid = n_valid
        self.n_missed = n_missed
        self.ev_offsets = np.concatenate([[0], np.cumsum(ev_count)]).astype(np.int64)
        self.ev = {k: np.asarray(v, np.float64 if k == 'score' else np.int64) for k, v in ev.items()}
        gt_rec.update(meta_rec)
        self.gt = {k: np.concatenate(v) if v else np.zeros(0) for k, v in gt_rec.items()}
        self.dt = {k: np.concatenate(v) if v else np.zeros(0) for k, v in dt_rec.items()}
        self.gt_offsets = np.concatenate([[0], np.cumsum(np.bincount(self.gt['frame'].astype(np.int64),
                                                                     minlength=self.n_frames))])
        self.dt_offsets = np.concatenate([[0], np.cumsum(np.bincount(self.dt['frame'].astype(np.int64),
                                                                     minlength=self.n_frames))])

    # -----------------------------------------------------------------------------------------------------------------
    def ap(self, frames=None, drop=None, drop_missed=False):
        """Car BEV AP_R40 at 0.7 over `frames` (indices, repeats allowed; None = all frames).

        drop: None, 'bkg' or 'loc' (that FP category is not counted); drop_missed: missed GT leave the GT set.
        """
        frames = np.arange(self.n_frames) if frames is None else np.asarray(frames, np.int64)
        n_gt = int(self.n_valid[frames].sum())
        n_miss = int(self.n_missed[frames].sum()) if drop_missed else 0
        n_gt_eff = n_gt - n_miss
        tp_scores = self.tp_flat[_gather(self.tp_offsets, frames)]
        if n_gt_eff <= 0:
            return 0.0
        thresholds = np.asarray(self.official.get_thresholds(tp_scores.copy(), n_gt_eff), np.float64)
        prec = np.zeros(N_SAMPLE_PTS)
        if len(thresholds):
            idx = _gather(self.ev_offsets, frames)
            score = self.ev['score'][idx]
            order = np.argsort(-score, kind='stable')
            s_sorted = -score[order]  # ascending in -score
            cum = {k: np.cumsum(self.ev[k][idx][order]) for k in ('tp', 'fp', 'fn', 'fpb', 'fpl')}
            for i, t in enumerate(thresholds):
                n = int(np.searchsorted(s_sorted, -t, side='right'))  # events with score >= t
                if n == 0:
                    tp = fp = fpb = fpl = 0
                else:
                    tp, fp, fpb, fpl = (int(cum[k][n - 1]) for k in ('tp', 'fp', 'fpb', 'fpl'))
                if drop == 'bkg':
                    fp -= fpb
                elif drop == 'loc':
                    fp -= fpl
                elif drop is not None:
                    raise ValueError(drop)
                prec[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            for i in range(len(thresholds)):
                prec[i] = np.max(prec[i:])
        return float(np.sum(prec[1:]) / 40 * 100)


def _gather(offsets, frames):
    """Concatenated index ranges offsets[f]:offsets[f+1] for f in frames (repeats allowed), vectorised."""
    frames = np.asarray(frames, np.int64)
    starts = offsets[frames]
    lens = offsets[frames + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    ends = np.cumsum(lens)
    return np.arange(total, dtype=np.int64) - np.repeat(ends - lens, lens) + np.repeat(starts, lens)


# ---------------------------------------------------------------------------------------------------------------------
# Edited annos for the official cross-checks
# ---------------------------------------------------------------------------------------------------------------------
def annos_without(gt_annos, dt_annos, frames_obj, what):
    """Copies of the annos with background-FP Car detections ('bkg') or missed GT cars ('miss') deleted."""
    gt_out, dt_out = [], []
    names_ok = lambda a: np.array([str(n).lower() == 'car' for n in a['name']], bool)  # noqa: E731
    for f, (g, d) in enumerate(zip(gt_annos, dt_annos)):
        keep_g = np.ones(len(g['name']), bool)
        keep_d = np.ones(len(d['name']), bool)
        sel_g = slice(frames_obj.gt_offsets[f], frames_obj.gt_offsets[f + 1])
        sel_d = slice(frames_obj.dt_offsets[f], frames_obj.dt_offsets[f + 1])
        if what == 'bkg':
            car_d = np.nonzero(names_ok(d))[0]
            assert len(car_d) == sel_d.stop - sel_d.start
            keep_d[car_d[frames_obj.dt['max_iou'][sel_d] < CATEGORY_IOU]] = False
        elif what == 'miss':
            gi = frames_obj.gt['index'][sel_g]
            keep_g[gi[frames_obj.gt['best_iou'][sel_g] < CATEGORY_IOU]] = False
        else:
            raise ValueError(what)
        gt_out.append(subset_anno(g, keep_g, GT_KEYS))
        dt_out.append(subset_anno(d, keep_d, DT_KEYS))
    return gt_out, dt_out
