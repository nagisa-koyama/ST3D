"""Is a foreground-aware calibration's `points per box tgt/src` real, or an artefact of its inputs?

Job 26220 (`centerpoint-foreground-lyft2nuscenes`) logged `points per box tgt/src = 0.05` (Lyft
40-beam) and `0.02` (64-beam), so its foreground channel thins source objects by up to 100x. Its
teacher also accepted ~75 boxes/frame on nuScenes, where the real Car density is 6.77-16.30. This
separates the candidate causes by re-measuring the same 1000 target frames three ways:

  * with the PSEUDO-LABELS the run actually used (reproduces the log);
  * with nuScenes' REAL GT, same frames, same points - what the channel would read with a perfect
    teacher;
  * with the pseudo-labels split into those that MATCH a real box (TP) and those that do not (FP).

and reports both the calibration's own per-bin quantity (foreground POINTS binned by point radius,
divided by BOXES binned by box-centre radius) and a box-attributed one (each box's own count, binned
by its centre), per 10 m ring.

DIAGNOSIS ONLY. Target GT is read here to explain a number, never to choose a parameter - using it
for that would make the row supervised on the target (see 20260925_02).

CPU only; run from tools/ inside the container:
    python analysis/pseudo_label_foreground_audit.py \
        --cfg_file cfgs/da-ieee-access/centerpoint-foreground-lyft2nuscenes.yaml \
        --ps_label /storage/wandb/run-20260926_102950-uq83obp7/files/ps_label/ps_label_e0.pkl
"""
import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _init_path  # noqa: F401  - the live repo's pcdet, not the image's stale copy
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.utils import common_utils, self_training_utils

RINGS = np.array([0, 10, 20, 30, 40, 50, 60, 70, 1e9])
RING_NAMES = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
MATCH_DIST = 2.0          # m, BEV centre distance, same class - the nuScenes-style loose match


def strided(n, num_frames):
    """The calibration's own frame selection (point_calibration.compute_foreground_histograms)."""
    step = max(1, n // num_frames)
    return list(range(0, n, step))[:num_frames]


def disable_augmentation(dataset):
    # forward() then only applies gt_boxes_mask and limit_period: points and boxes stay in the
    # frame the teacher predicted in, so the pseudo and real passes line up box for box.
    dataset.data_augmentor.data_augmentor_queue = []


def ring_of(r):
    return np.clip(np.searchsorted(RINGS, r, side='right') - 1, 0, len(RING_NAMES) - 1)


class Channel:
    """Accumulates one box population in both definitions."""

    def __init__(self, num_bins, max_dist, num_classes):
        self.edges = np.linspace(0, max_dist, num_bins + 1)
        self.max_dist = max_dist
        self.fg = np.zeros(num_bins)      # calibration: fg points by POINT radius
        self.nbox = np.zeros(num_bins)    # calibration: occupied boxes by CENTRE radius
        self.records = []                 # (ring, class, npts, centre radius)
        self.frames = 0
        self.nc = num_classes

    def add(self, points, boxes, classes):
        self.frames += 1
        if boxes is None or len(boxes) == 0:
            return
        mask, per_box, kept = DataProcessor.box_occupancy(points, boxes)
        # box_occupancy drops degenerate boxes; carry the class through the same filter
        keep = (np.asarray(boxes, dtype=np.float32)[:, 3:6] > 1e-3).all(axis=1)
        classes = np.asarray(classes)[keep]
        dist = np.clip(np.linalg.norm(points[:, 0:2], axis=1), 0, self.max_dist - 1e-4)
        self.fg += np.histogram(dist[mask], bins=self.edges)[0]
        occ = per_box >= 1
        bdist = np.linalg.norm(kept[:, 0:2], axis=1)
        self.nbox += np.histogram(np.clip(bdist[occ], 0, self.max_dist - 1e-4), bins=self.edges)[0]
        for r, c, n in zip(bdist, classes, per_box):
            self.records.append((int(ring_of(r)), int(c), int(n), float(r)))

    def per_bin(self):
        return np.divide(self.fg, self.nbox, out=np.zeros_like(self.fg), where=self.nbox > 0)

    def table(self):
        """{(ring, class or -1 for all): (occupied boxes/frame, mean pts/box over occupied)}"""
        rec = np.array(self.records, dtype=np.float64).reshape(-1, 4)
        out = {}
        for ri in range(len(RING_NAMES)):
            for c in [-1] + list(range(1, self.nc + 1)):
                sel = (rec[:, 0] == ri) & (rec[:, 2] >= 1)
                if c > 0:
                    sel &= rec[:, 1] == c
                n = sel.sum()
                out[(ri, c)] = (n / max(self.frames, 1), rec[sel, 2].mean() if n else np.nan)
        return out


def calib_ratio(tgt, src):
    """The number the run logged: sum of per-bin pts/box over bins both sides populate."""
    ft, fs = tgt.per_bin(), src.per_bin()
    live = (fs > 0) & (ft > 0)
    return ft[live].sum() / max(fs[live].sum(), 1e-9)


def attributed_ratio(tgt, src, weight_by='tgt'):
    """Box-attributed alternative: per-ring mean pts/box, ratio averaged over rings weighted by the
    TARGET's occupied-box share, so the two domains' range mixes do not enter (see 20260922_08 on
    why a pooled ratio is not a density comparison)."""
    tt, ts = tgt.table(), src.table()
    num = den = 0.0
    for ri in range(len(RING_NAMES) - 1):          # 70+ is clipped into the last bin; leave it out
        (nt, mt), (_, ms) = tt[(ri, -1)], ts[(ri, -1)]
        if np.isfinite(mt) and np.isfinite(ms) and ms > 0:
            num += nt * mt / ms
            den += nt
    return num / den if den else np.nan


def match(ps_boxes, ps_cls, ps_score, gt_boxes, gt_cls):
    """Greedy by score, same class, BEV centre distance <= MATCH_DIST. Returns TP mask over ps and
    matched mask over gt."""
    tp = np.zeros(len(ps_boxes), dtype=bool)
    used = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-ps_score):
        cand = np.where((gt_cls == ps_cls[i]) & ~used)[0]
        if not len(cand):
            continue
        d = np.linalg.norm(gt_boxes[cand, :2] - ps_boxes[i, :2], axis=1)
        j = np.argmin(d)
        if d[j] <= MATCH_DIST:
            tp[i] = True
            used[cand[j]] = True
    return tp, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', required=True)
    ap.add_argument('--ps_label', required=True)
    ap.add_argument('--frames', type=int, default=1000)
    ap.add_argument('--source_frames', type=int, default=1000)
    ap.add_argument('--out', default=None, help='pickle the raw records here')
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    classes = cfg.CLASS_NAMES
    nc = len(classes)

    # ---------------- target: pseudo-labelled and real, same frames, no augmentation -------------
    tar_ps, _, _ = build_dataloader(cfg.DATA_CONFIG_TAR, classes, 1, False, workers=0, logger=logger,
                                    training=True, model_ontology=cfg.get('ONTOLOGY', None))
    real_cfg = copy.deepcopy(cfg.DATA_CONFIG_TAR)
    real_cfg.USE_PSEUDO_LABEL = False
    tar_gt, _, _ = build_dataloader(real_cfg, classes, 1, False, workers=0, logger=logger,
                                    training=True, model_ontology=cfg.get('ONTOLOGY', None))
    disable_augmentation(tar_ps)
    disable_augmentation(tar_gt)

    ps = pickle.load(open(args.ps_label, 'rb'))
    self_training_utils.PSEUDO_LABELS.update(ps)
    logger.info('loaded %d pseudo-labelled frames from %s' % (len(ps), args.ps_label))

    src_dc = next(iter(cfg.DATA_CONFIGS.values()))
    nb, md = src_dc.get('HIST_DIST_BINS', 50), src_dc.get('HIST_DIST_MAX_DIST', 75.0)
    ch = {k: Channel(nb, md, nc) for k in ['pseudo', 'gt', 'tp', 'fp', 'gt_missed']}
    scores = {'tp': [], 'fp': []}
    raw_ps_per_frame = []

    idxs = strided(len(tar_ps), args.frames)
    for k, idx in enumerate(idxs):
        dp, dg = tar_ps[idx], tar_gt[idx]
        pts = dp['points']
        assert len(pts) == len(dg['points']), 'MAX_SWEEPS 1 and no aug: clouds must be identical'
        # pseudo boxes as the calibration saw them (range-masked, ignored ones dropped)
        pb = np.asarray(dp['gt_boxes'])
        pb = pb[pb[:, 7] > 0] if len(pb) else pb.reshape(0, 8)
        # scores: recover from the stored labels by exact position (no aug, so identical)
        raw = self_training_utils.PSEUDO_LABELS[dp['frame_id']]['gt_boxes']
        raw_ps_per_frame.append((raw[:, 7] > 0).sum())
        sc = np.zeros(len(pb))
        for i, b in enumerate(pb):
            j = np.argmin(np.abs(raw[:, :3] - b[:3]).sum(axis=1))
            sc[i] = raw[j, 8]
        gb = np.asarray(dg['gt_boxes']).reshape(-1, 8)

        ch['pseudo'].add(pts, pb[:, :7], pb[:, 7])
        ch['gt'].add(pts, gb[:, :7], gb[:, 7])
        tp, used = match(pb, pb[:, 7], sc, gb, gb[:, 7])
        ch['tp'].add(pts, pb[tp, :7], pb[tp, 7])
        ch['fp'].add(pts, pb[~tp, :7], pb[~tp, 7])
        ch['gt_missed'].add(pts, gb[~used, :7], gb[~used, 7])
        scores['tp'] += list(zip(pb[tp, 7], sc[tp]))
        scores['fp'] += list(zip(pb[~tp, 7], sc[~tp]))
        if (k + 1) % 100 == 0:
            logger.info('target %d/%d' % (k + 1, len(idxs)))

    # ---------------- sources, exactly as the calibration measured them (augmentation on) ---------
    src_ch = {}
    for name, dc in cfg.DATA_CONFIGS.items():
        sset, _, _ = build_dataloader(dc, classes, 1, False, workers=0, logger=logger,
                                      training=True, model_ontology=cfg.get('ONTOLOGY', None))
        c = Channel(nb, md, nc)
        for k, idx in enumerate(strided(len(sset), args.source_frames)):
            d = sset[idx]
            b = np.asarray(d['gt_boxes']).reshape(-1, 8)
            c.add(d['points'], b[:, :7], b[:, 7])
        src_ch[name] = c
        logger.info('source %s done' % name)

    # ---------------- report --------------------------------------------------------------------
    P = print
    P('\n=== boxes per frame (occupied, >=1 point) and mean points per box, by box-centre ring ===')
    for c in [-1] + list(range(1, nc + 1)):
        cname = 'ALL CLASSES' if c < 0 else classes[c - 1]
        P('\n-- %s --' % cname)
        P('%-7s | %-15s | %-15s | %-15s | %-15s | %-6s %-6s | %s' % (
            'ring', 'GT  n/fr  pts', 'PS  n/fr  pts', 'TP  n/fr  pts', 'FP  n/fr  pts',
            'prec', 'recall', '  '.join('%s pts' % s for s in src_ch)))
        tabs = {k: v.table() for k, v in ch.items()}
        stabs = {k: v.table() for k, v in src_ch.items()}
        for ri, rn in enumerate(RING_NAMES):
            g, p, t, f = (tabs[k][(ri, c)] for k in ['gt', 'pseudo', 'tp', 'fp'])
            prec = t[0] / p[0] if p[0] else np.nan
            rec = t[0] / g[0] if g[0] else np.nan
            P('%-7s | %5.2f %8.1f | %5.2f %8.1f | %5.2f %8.1f | %5.2f %8.1f | %5.2f  %5.2f  | %s' % (
                rn, g[0], g[1], p[0], p[1], t[0], t[1], f[0], f[1], prec, rec,
                '  '.join('%8.1f' % stabs[s][(ri, c)][1] for s in src_ch)))

    P('\n=== points per box tgt/src ===')
    P('%-12s | %-26s | %-26s' % ('source', 'calibration definition', 'box-attributed, per ring'))
    for s, sc_ in src_ch.items():
        P('%-12s | PS %.3f  GT %.3f  TP %.3f | PS %.3f  GT %.3f  TP %.3f' % (
            s, calib_ratio(ch['pseudo'], sc_), calib_ratio(ch['gt'], sc_), calib_ratio(ch['tp'], sc_),
            attributed_ratio(ch['pseudo'], sc_), attributed_ratio(ch['gt'], sc_),
            attributed_ratio(ch['tp'], sc_)))

    P('\n=== calibration per-bin pts/box (fg points by POINT radius / boxes by CENTRE radius) ===')
    edges = ch['gt'].edges
    P('%-11s %10s %10s %10s ' % ('bin', 'PS', 'GT', 'TP') + ' '.join('%10s' % s for s in src_ch))
    for b in range(nb):
        row = [ch[k].per_bin()[b] for k in ['pseudo', 'gt', 'tp']] + \
              [v.per_bin()[b] for v in src_ch.values()]
        P('%4.1f-%4.1f  ' % (edges[b], edges[b + 1]) + ' '.join('%10.1f' % x for x in row))

    P('\n=== pseudo-label scores (median / p90) ===')
    for k in ['tp', 'fp']:
        a = np.array(scores[k]).reshape(-1, 2)
        for c in range(1, nc + 1):
            s = a[a[:, 0] == c, 1]
            if len(s):
                P('%s %-10s n=%6d  median %.3f  p90 %.3f' % (k.upper(), classes[c - 1], len(s),
                                                           np.median(s), np.percentile(s, 90)))
    P('\nstored positive pseudo-labels/frame (before range mask): %.1f' % np.mean(raw_ps_per_frame))

    if args.out:
        pickle.dump({'target': {k: v.records for k, v in ch.items()},
                     'target_bins': {k: (v.fg, v.nbox, v.frames) for k, v in ch.items()},
                     'source': {k: v.records for k, v in src_ch.items()},
                     'source_bins': {k: (v.fg, v.nbox, v.frames) for k, v in src_ch.items()},
                     'scores': scores}, open(args.out, 'wb'))


if __name__ == '__main__':
    main()
