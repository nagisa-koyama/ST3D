"""Pre-launch smoke of the Lyft 40-beam rows P (RING_PATTERN) and C1-analog (per-bin thinning + drop_empty_gt_boxes);
experiments_md 20261024_01 §5. CPU. Source TRAIN frames, training mode, exactly as train.py builds them: the C1-analog's
histograms are installed with train.py's own link_point_calibration (1,000 kept Lyft frames, 1,000 nuScenes TRAIN).

    python3 analysis/lyft_ring_pattern_check.py <P cfg> <C1-analog cfg> <frames> <out.json>

Reports, over strided kept frames (same frames and seeds for both rows):
- loader stage (P): share of raw points RING_PATTERN keeps and the rings it keeps (gate diagnostic 25.7%, 17 rings);
  the loader's own thinned cloud must equal the mask recomputed here from the cache with the same seed;
- boxes per class at the loader stage (raw sensor frame): holding >= 1 point in the full scan, and of those emptied by
  the pattern (prepare_data's zero-point filter drops them: the recount);
- C1-analog: boxes per class (centre inside POINT_CLOUD_RANGE) holding >= 1 point when entering the data processor and
  0 after it, i.e. emptied by sample_points_hist_based (drop_empty_gt_boxes drops them: the processor-stage recount);
- what reaches the model, per row: positives per class per frame, points per box (median) by box range, boxes with 0
  points (must be 0 for the C1-analog; for P the range mask's residual, as 27807 / 28034, with where they sit), points
  per frame per radial bin beside nuScenes TRAIN's (the C1-analog's installed target histogram) and the source's
  unthinned histogram.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pathlib import Path
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.lyft import lyft_ring_pattern as RP
from pcdet.datasets.point_calibration import calibration_target_config, link_point_calibration
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

cfg_p, cfg_c, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
logger = common_utils.create_logger()
KEY = 'LYFT_40BEAM'
BOX_R = [0, 20, 40, 60, 75]
COARSE = [0, 7.5, 15, 30, 45, 60, 75]


def build(f):
    cfg = EasyDict()
    cfg_from_yaml_file(f, cfg)
    dc = cfg.DATA_CONFIGS[KEY]
    ds, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
    return cfg, dc, ds


cfgP, dcP, P = build(cfg_p)
cfgC, dcC, C = build(cfg_c)
assert [i['lidar_path'] for i in P.infos] == [i['lidar_path'] for i in C.infos], 'the two rows must train one frame set'
# the C1-analog's calibration, as train.py installs it (HIST_DIST_ON_THE_FLY): kept Lyft frames vs nuScenes TRAIN
tcfg, split = calibration_target_config(cfgC.DATA_CONFIG_TAR)
T, _, _ = build_dataloader(dataset_cfg=tcfg, class_names=cfgC.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                           logger=logger, training=False, model_ontology=cfgC.get('ONTOLOGY'))
src_hist, tgt_hist = link_point_calibration(C, T, num_frames=dcC.HIST_DIST_FRAMES, num_bins=dcC.HIST_DIST_BINS,
                                            max_dist=dcC.HIST_DIST_MAX_DIST, logger=logger)
rate = C.data_processor.per_bin_sample_rate()
edges = np.linspace(0, dcC.HIST_DIST_MAX_DIST, dcC.HIST_DIST_BINS + 1)
names = list(cfgP.CLASS_NAMES)
to_model = P.map_ontology_dataset_to_model


def in_boxes(pts, boxes):
    if len(boxes) == 0 or len(pts) == 0:
        return np.zeros((len(boxes), len(pts)), dtype=bool)
    return roiaware_pool3d_utils.points_in_boxes_cpu(torch.from_numpy(np.ascontiguousarray(pts[:, :3])).float(),
                                                     torch.from_numpy(np.ascontiguousarray(boxes[:, :7])).float()
                                                     ).numpy() > 0


def radial_hist(pts, e):
    r = np.linalg.norm(pts[:, :2], axis=1)
    return np.histogram(np.clip(r, 0, e[-1] - 1e-4), bins=e)[0]


# spies: the cloud entering prepare_data (P) and the boxes entering / points leaving the processor (C)
cap = {}
for ds, tag in ((P, 'P'), (C, 'C')):
    orig_prep, orig_proc = ds.prepare_data, ds.data_processor.forward

    def prep(data_dict, _o=orig_prep, _t=tag):
        cap[_t + '_in'] = len(data_dict['points'])
        return _o(data_dict=data_dict)

    def proc(data_dict, _o=orig_proc, _t=tag):
        cap[_t + '_boxes_before'] = data_dict['gt_boxes'].copy()
        cap[_t + '_pts_before'] = data_dict['points'][:, :3].copy()
        return _o(data_dict=data_dict)
    ds.prepare_data, ds.data_processor.forward = prep, proc

S = {t: dict(frames=0, pos=np.zeros(len(names)), zero=np.zeros(len(names)), hist=np.zeros(len(edges) - 1),
             ppb={c: {b: [] for b in range(len(BOX_R) - 1)} for c in names}) for t in ('P', 'C')}
L = dict(kept_share=[], kept_rings=[], raw_hist=np.zeros(len(edges) - 1),
         hit={c: 0 for c in names}, emptied={c: 0 for c in names}, hit70={c: 0 for c in names},
         emptied70={c: 0 for c in names})
CE = dict(hit={c: 0 for c in names}, emptied={c: 0 for c in names})
mismatch = 0
step = max(1, len(P) // n)
idx = list(range(0, len(P), step))[:n]
for i in idx:
    info = P.infos[i]
    stem = Path(info['lidar_path']).stem
    raw = P.get_lidar_with_sweeps(i, max_sweeps=1)
    np.random.seed(1000 + i)
    keep = RP.ring_pattern_keep(stem, raw, P._ring_cache[stem], dcP.RING_PATTERN)
    ring = RP.ring_ids(P._ring_cache[stem]['counts'])
    L['kept_share'].append(float(keep.mean()))
    L['kept_rings'].append(int(len(np.unique(ring[keep]))))
    shifted = raw[:, :3] + np.array(dcP.SHIFT_COOR, np.float32)
    inr = (shifted[:, 2] >= -2) & (shifted[:, 2] <= 4)
    L['raw_hist'] += radial_hist(raw[inr], edges)
    # loader-stage boxes, raw sensor frame (the infos' boxes and the stored cloud share it)
    gtn = np.array([to_model.get(nm, nm) for nm in info['gt_names']])
    inb = in_boxes(raw, info['gt_boxes'])
    hit = inb.sum(1) > 0
    emp = hit & (inb[:, keep].sum(1) == 0)
    rr = np.hypot(info['gt_boxes'][:, 0], info['gt_boxes'][:, 1]) if len(info['gt_boxes']) else np.zeros(0)
    for c in names:
        m = gtn == c
        L['hit'][c] += int((hit & m).sum()); L['emptied'][c] += int((emp & m).sum())
        L['hit70'][c] += int((hit & m & (rr < 70)).sum()); L['emptied70'][c] += int((emp & m & (rr < 70)).sum())
    # what reaches the model
    for t, ds in (('P', P), ('C', C)):
        np.random.seed(1000 + i)
        d = ds[i]
        if t == 'P':
            mismatch += int(cap['P_in'] != int(keep.sum()))
        if t == 'C':
            bb = cap['C_boxes_before']
            pcr = np.array(dcC.POINT_CLOUD_RANGE)
            bb = bb[(np.abs(bb[:, 0]) < pcr[3]) & (np.abs(bb[:, 1]) < pcr[4]) & (bb[:, 2] > pcr[2]) & (bb[:, 2] < pcr[5])]
            b_in = in_boxes(cap['C_pts_before'], bb).sum(1) > 0
            b_out = in_boxes(d['points'], bb).sum(1) > 0
            for k, c in enumerate(names):
                m = bb[:, 7] == k + 1
                CE['hit'][c] += int((m & b_in).sum()); CE['emptied'][c] += int((m & b_in & ~b_out).sum())
        st = S[t]
        st['frames'] += 1
        st['hist'] += radial_hist(d['points'], edges)
        gb = d['gt_boxes']
        if len(gb):
            cnt = in_boxes(d['points'], gb).sum(1)
            br = np.hypot(gb[:, 0], gb[:, 1])
            for k, c in enumerate(names):
                m = gb[:, 7] == k + 1
                st['pos'][k] += m.sum(); st['zero'][k] += int((m & (cnt == 0)).sum())
                for b in range(len(BOX_R) - 1):
                    mb = m & (br >= BOX_R[b]) & (br < BOX_R[b + 1])
                    st['ppb'][c][b].extend(cnt[mb].tolist())
            # where the boxes reaching the model with 0 points sit: |x|, |y| against POINT_CLOUD_RANGE's 75.2 m edge
            st.setdefault('zero_where', []).extend(
                [[round(float(max(abs(b[0]), abs(b[1]))), 2), round(float(b[2]), 2), int(b[7])] for b in gb[cnt == 0]])


def coarse(h):
    c = np.zeros(len(COARSE) - 1)
    mids = (edges[:-1] + edges[1:]) / 2
    for j in range(len(COARSE) - 1):
        c[j] = h[(mids >= COARSE[j]) & (mids < COARSE[j + 1])].sum()
    return c


f = len(idx)
res = {
    'frames': f, 'stride': step, 'kept_frames_in_row': len(P), 'loader_mask_mismatches': mismatch,
    'pattern_kept_point_share': {'mean': float(np.mean(L['kept_share'])),
                                 'p5_p50_p95': [float(x) for x in np.percentile(L['kept_share'], [5, 50, 95])]},
    'pattern_kept_rings': {str(k): int(v) for k, v in zip(*np.unique(L['kept_rings'], return_counts=True))},
    'loader_stage_boxes_P': {c: {'with_points_full_scan': L['hit'][c], 'emptied_by_pattern': L['emptied'][c],
                                 'share': L['emptied'][c] / max(L['hit'][c], 1),
                                 'within_70m': [L['hit70'][c], L['emptied70'][c]]} for c in names},
    'processor_stage_boxes_C': {c: {'with_points_entering': CE['hit'][c], 'emptied_by_thinning': CE['emptied'][c],
                                    'share': CE['emptied'][c] / max(CE['hit'][c], 1)} for c in names},
    'reaching_model': {t: {'positives_per_frame': {c: float(S[t]['pos'][k] / f) for k, c in enumerate(names)},
                           'boxes_with_0_points': {c: int(S[t]['zero'][k]) for k, c in enumerate(names)},
                           'zero_point_boxes_max_abs_xy_z_label': S[t].get('zero_where', []),
                           'points_per_box_median_by_range': {
                               c: {'%d-%d m' % (BOX_R[b], BOX_R[b + 1]):
                                   [float(np.median(v)) if v else None, len(v)] for b, v in S[t]['ppb'][c].items()}
                               for c in names}} for t in ('P', 'C')},
    'radial_bins_m': COARSE,
    'points_per_frame_by_bin': {'P': (coarse(S['P']['hist']) / f).round(1).tolist(),
                                'C1_analog': (coarse(S['C']['hist']) / f).round(1).tolist(),
                                'nuscenes_train_target': coarse(tgt_hist).round(1).tolist(),
                                'lyft_kept_unthinned_C_calibration': coarse(src_hist).round(1).tolist(),
                                'lyft_raw_loader_stage_z_in_range': (coarse(L['raw_hist']) / f).round(1).tolist()},
    'calibration': {'target_split': split, 'rate_min_max': [float(rate.min()), float(rate.max())],
                    'bins_below_1': int((rate < 1).sum()), 'bins': int(len(rate))},
}
for k in ('P', 'C1_analog'):
    a = np.array(res['points_per_frame_by_bin'][k])
    res['points_per_frame_by_bin'][k + '_over_target'] = (a / np.maximum(coarse(tgt_hist), 1e-9)).round(3).tolist()
res['points_per_frame_by_bin']['P_over_C1_analog'] = (
    np.array(res['points_per_frame_by_bin']['P']) / np.maximum(res['points_per_frame_by_bin']['C1_analog'], 1e-9)
).round(3).tolist()
json.dump(res, open(out, 'w'), indent=1)
print('SMOKE', json.dumps(res))
assert mismatch == 0, 'the loader thinned a different cloud than the cache mask'
# C1-analog's recount runs after the processor's range mask, so nothing empty may reach the model. P recounts at the
# loader (prepare_data's zero-point filter, as 27807 / 28034), so a box the RANGE MASK empties later still reaches it:
# the precedents' disclosed residual (20261010_05 §4.1-4.2), reported here, with where those boxes sit.
assert all(v == 0 for v in res['reaching_model']['C']['boxes_with_0_points'].values()), \
    'a box with 0 points reached the model in the C1-analog row'
