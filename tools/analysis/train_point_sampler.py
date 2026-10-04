"""Train the learned point sampler for one source-target pair (experiments_md/20261003_04 sections 3-4).

What it learns: the per-bin density rule is measured once (link_point_calibration, target train split,
source with augmentation OFF) and stored only as the sampler's INITIALISATION (so the L0 ablation is
exact); then a small MLP over per-point local-geometry features (range, height, elevation,
nearest-neighbour distance, local counts, azimuth) is trained so that, per frame, the EXPECTED
(a) elevation-angle histogram per 10 m ring and (b) point count per (azimuth sector, range bin) of
the sampled source match the target's. (a) is the descriptor the rule leaves untouched (20261003_04
section 11); (b) replaces the rule's azimuth-pooled count matching and the HIST_DIST_FOV_DEGREE key
with something learned: a target that only covers a cone shows up as zero counts outside it, and the
sampler learns to drop the source there. A keep probability cannot add points, so where the source is
below the target the count term saturates (the accumulate-first design). Everything is on cached
features; minutes on CPU.

    python train_point_sampler.py <cfg> --out /home/koyama/data/samplers/<pair>.npz   (OUTSIDE the repo: jobs run from a snapshot without output/) [--frames 200] [--steps 400] [--count_weight 1.0]

The weights go into a config as
    DATA_PROCESSOR: - NAME: sample_points_learned
                      WEIGHTS: <weights.npz>
in place of sample_points_hist_based. Prints the gate descriptor (elevation JS per ring) before and
after training, with a Bernoulli-sampled check, so the number the sampler was trained on is visible.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader, link_point_calibration  # noqa: E402
from pcdet.datasets.point_calibration import _augmentation_off, calibration_target_config  # noqa: E402
from pcdet.datasets.processor.point_sampler import (LearnedPointSampler, point_features, cell_index,  # noqa: E402
                                                    N_SECTORS, RANGE_EDGES)

RINGS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
ELEV_EDGES = np.arange(-30, 10.01, 0.25)
# N_SECTORS / RANGE_EDGES (24 sectors of 15 deg x 30 range bins of 2.5 m) live in point_sampler.py: the sampler's
# gate uses the same cells at inference.
COVER_RING = (2, 20)  # range bins 5-50 m: where a sector's coverage is judged
COVER_FRAC = 0.05     # a sector is covered if its 5-50 m count is >= 5% of the best-covered sector's


def strided(dataset, n_frames):
    n = len(dataset); step = max(1, n // n_frames); used = 0
    for idx in range(0, n, step):
        if used >= n_frames:
            return
        pts = dataset[idx]['points'][:, :3]
        if len(pts):
            used += 1
            yield pts


def elev_bins(feats):
    return np.clip(np.digitize(feats[:, 2], ELEV_EDGES) - 1, 0, len(ELEV_EDGES) - 2)


def ring_of(feats):
    r = feats[:, 0]
    return np.clip(np.floor(r / 10).astype(np.int64), 0, len(RINGS))  # index len(RINGS) = beyond 50 m


def cell_of(feats):
    """(azimuth sector, range bin) cell index per point, from the cos/sin azimuth features."""
    return cell_index(feats[:, 0], np.arctan2(feats[:, 7], feats[:, 6]))


def target_cone(Tc):
    """The target's azimuth coverage, read off its own per-cell counts: (fov_degree, heading_degree) of the
    largest contiguous arc of covered sectors, or None when every sector is covered (a 360 deg sensor).
    This is what HIST_DIST_FOV_DEGREE / _HEADING declared by hand for PandarGT; here it is measured."""
    n_rb = len(RANGE_EDGES) - 1
    per_sec = Tc.reshape(N_SECTORS, n_rb)[:, COVER_RING[0]:COVER_RING[1]].sum(1)
    cov = per_sec >= COVER_FRAC * per_sec.max()
    if cov.all():
        return None
    best, start = (0, 0), None
    for k in range(2 * N_SECTORS):  # circular: walk twice
        if cov[k % N_SECTORS]:
            start = k if start is None else start
            if k - start + 1 > best[0] and k - start + 1 <= N_SECTORS:
                best = (k - start + 1, start)
        else:
            start = None
    n, s0 = best
    width = 360.0 / N_SECTORS
    heading = -180.0 + (s0 + n / 2.0) * width
    heading = (heading + 180.0) % 360.0 - 180.0
    return n * width, heading


def js_torch(p, q, eps=1e-8):
    p = p / (p.sum() + eps); q = q / (q.sum() + eps); m = 0.5 * (p + q)
    def kl(a, b):
        return (a * (torch.log2(a + eps) - torch.log2(b + eps))).sum()
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cfg'); ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=200); ap.add_argument('--hist_frames', type=int, default=300)
    ap.add_argument('--steps', type=int, default=400); ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--count_weight', type=float, default=1.0, help='weight of the (azimuth, range) count term against the elevation term')
    ap.add_argument('--source', default=None, help='DATA_CONFIGS key when the config has several sources (default: first)')
    ap.add_argument('--count_cells', default='observed', choices=['observed', 'all'],
                    help="'observed' (default since 2026-10-04): the count term ignores cells the target never returns a "
                         "point in; 'all' reproduces the samplers trained before (S2 s2_n0*_kitti, S3 s3_spin4_flash)")
    ap.add_argument('--cache', default=None, help='feature cache to reuse (default: <out>.cache.npz)')
    ap.add_argument('--rule_fov', default='auto', choices=['auto', 'none'],
                    help="'auto' (default since 2026-10-04): if the target covers only an azimuth arc (read off its own "
                         "cell counts), measure the rule inside that arc; 360 deg targets are unchanged")
    ap.add_argument('--gate', default='observed', choices=['observed', 'none'],
                    help="'observed' (default since 2026-10-04): the learned correction acts only in cells the target "
                         "returned points in, the rule alone elsewhere")
    args = ap.parse_args()
    os.chdir(TOOLS)
    import logging
    logger = logging.getLogger('sampler'); logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)
    cfg = EasyDict(); cfg_from_yaml_file(args.cfg, cfg)
    data_configs = cfg.get('DATA_CONFIGS') or {'DATA_CONFIG': cfg.DATA_CONFIG}
    key = args.source or next(iter(data_configs)); dc = data_configs[key]
    calib_cfg, split = calibration_target_config(cfg.DATA_CONFIG_TAR)
    target, _, _ = build_dataloader(dataset_cfg=calib_cfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                    workers=0, logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    source, _, _ = build_dataloader(dataset_cfg=dc, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                    workers=0, logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
    tz = float(calib_cfg.get('SHIFT_COOR', [0, 0, 0])[2]); sz = float(dc.get('SHIFT_COOR', [0, 0, 0])[2])
    print(f'source {key} ({type(source).__name__}, augmentation off), target {type(target).__name__} {split} split', flush=True)

    cache = Path(args.cache) if args.cache else Path(args.out).with_suffix('.cache.npz')
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        rate, max_dist = z['rate'], float(z['max_dist']); F, R, E, C, T, Tc = z['F'], z['R'], z['E'], z['C'], z['T'], z['Tc']
        feats, rings, ebins = list(z['feats']), list(z['rings']), list(z['ebins']); n_src_frames = len(feats); n_tgt = int(z['n_tgt'])
        n_cells = N_SECTORS * (len(RANGE_EDGES) - 1)
        print(f'loaded cached features from {cache} ({len(F)} points, {n_src_frames} source frames, {n_tgt} target frames)', flush=True)
    else:
        rate, max_dist, F, R, E, C, T, Tc, feats, rings, ebins, n_src_frames, n_tgt = _measure(source, target, dc, sz, tz, args, logger)
        n_cells = N_SECTORS * (len(RANGE_EDGES) - 1)
        np.savez(cache, rate=rate, max_dist=max_dist, F=F, R=R, E=E, C=C, T=T, Tc=Tc, n_tgt=n_tgt,
                 feats=np.array(feats, dtype=object), rings=np.array(rings, dtype=object), ebins=np.array(ebins, dtype=object))
        print(f'cached features to {cache}', flush=True)

    if args.rule_fov == 'auto':
        cone = target_cone(Tc)
        if cone is None:
            print('target covers every azimuth sector: the rule is measured over 360 deg (unchanged)', flush=True)
        else:
            # The pooled rule compares the source's 360 deg against a target that covers only this arc, i.e.
            # totals, not density (the inverted S3 measurement of 20260928_02). Measure it inside the arc.
            print(f'target covers a {cone[0]:.0f} deg arc at heading {cone[1]:.0f} deg (from its own cell counts): '
                  f'the rule is re-measured inside it', flush=True)
            with _augmentation_off(source):
                link_point_calibration(source, target, num_frames=args.hist_frames, num_bins=dc.get('HIST_DIST_BINS', 50),
                                       max_dist=dc.get('HIST_DIST_MAX_DIST', 75.0), logger=logger,
                                       fov_degree=cone[0], fov_heading=cone[1])
                proc = source.data_processor
                rate = proc.per_bin_sample_rate().astype(np.float32); max_dist = float(proc.hist_max_dist)
                proc.hist_dist_src = None
    sampler = LearnedPointSampler.from_rule(rate, max_dist, sz, F.mean(0), F.std(0) + 1e-6)
    if args.gate == 'observed':
        sampler.obs_mask = Tc > 0  # the learned correction acts only where the target returned points
    _train_and_report(sampler, F, R, E, C, T, Tc, n_src_frames, n_cells, feats, rings, ebins, args)


def _measure(source, target, dc, sz, tz, args, logger):
    # 1. the rule, measured as train.py would (gives the sampler its initialisation)
    with _augmentation_off(source):
        link_point_calibration(source, target, num_frames=args.hist_frames, num_bins=dc.get('HIST_DIST_BINS', 50),
                               max_dist=dc.get('HIST_DIST_MAX_DIST', 75.0), logger=logger,
                               fov_degree=dc.get('HIST_DIST_FOV_DEGREE', None), fov_heading=dc.get('HIST_DIST_FOV_HEADING', 0.0))
        proc = source.data_processor
        rate = proc.per_bin_sample_rate().astype(np.float32); max_dist = float(proc.hist_max_dist)
        proc.hist_dist_src = None  # the rule off: the trainer sees the raw accumulated cloud
        # 2. cache source features
        feats, rings, ebins, cells = [], [], [], []
        for i, pts in enumerate(strided(source, args.frames)):
            f = point_features(pts, sz); feats.append(f); rings.append(ring_of(f)); ebins.append(elev_bins(f)); cells.append(cell_of(f))
            if (i + 1) % 50 == 0:
                print(f'  source features: {i + 1} frames', flush=True)
    F = np.concatenate(feats); R = np.concatenate(rings); E = np.concatenate(ebins); C = np.concatenate(cells)
    n_src_frames = len(feats)
    n_cells = N_SECTORS * (len(RANGE_EDGES) - 1)
    # 3. target: elevation histograms per ring (pooled) and mean count per (sector, range) cell per frame
    T = np.zeros((len(RINGS), len(ELEV_EDGES) - 1)); Tc = np.zeros(n_cells); n_tgt = 0
    for pts in strided(target, args.frames):
        f = point_features(pts, tz); rg = ring_of(f); eb = elev_bins(f); n_tgt += 1
        Tc += np.bincount(cell_of(f), minlength=n_cells)
        for k in range(len(RINGS)):
            m = rg == k
            T[k] += np.bincount(eb[m], minlength=T.shape[1])
    Tc /= max(n_tgt, 1)
    print(f'measured {len(F)} source points from {n_src_frames} frames; target histograms from {n_tgt} frames', flush=True)
    return rate, max_dist, F, R, E, C, T, Tc, feats, rings, ebins, n_src_frames, n_tgt


def _train_and_report(sampler, F, R, E, C, T, Tc, n_src_frames, n_cells, feats, rings, ebins, args):
    rate, max_dist = sampler.rate, sampler.max_dist
    # 4. train the MLP on cached features
    x = torch.tensor((F - sampler.feat_mean) / sampler.feat_std)
    base = torch.tensor(sampler.rule_logit(_points_from_feats(F)))  # the rule's logit per cached point
    w1 = torch.tensor(sampler.w1, requires_grad=True); b1 = torch.tensor(sampler.b1, requires_grad=True)
    w2 = torch.tensor(sampler.w2, requires_grad=True); b2 = torch.tensor(sampler.b2, requires_grad=True)
    w3 = torch.tensor(sampler.w3, requires_grad=True); b3 = torch.tensor(sampler.b3, requires_grad=True)
    Tt = torch.tensor(T, dtype=torch.float32); Tct = torch.tensor(Tc, dtype=torch.float32)
    ring_t = torch.tensor(R); ebin_t = torch.tensor(E); cell_t = torch.tensor(C)
    # Source points in cells the target returned points in. The elevation term compares only these: the
    # target's histograms are, by construction, measured there, and the correction is gated to them - so on
    # a cone-shaped target (PandarGT) the out-of-cone 5/6 of the source would otherwise dominate a histogram
    # the sampler is not allowed to change (S3, 2026-10-04: no learning at all, 0.89 -> 0.90).
    seen_t = torch.ones(len(F), dtype=torch.bool) if sampler.obs_mask is None else torch.tensor(sampler.obs_mask[C])
    p0 = torch.sigmoid(base)
    opt = torch.optim.Adam([w1, b1, w2, b2, w3, b3], lr=args.lr)

    gate = torch.ones(len(F)) if sampler.obs_mask is None else torch.tensor(sampler.obs_mask[C], dtype=torch.float32)

    def forward():
        h = torch.relu(x @ w1 + b1); h = torch.relu(h @ w2 + b2)
        return torch.sigmoid(base + gate * (h @ w3 + b3)[:, 0])

    def losses(p):
        js_total = 0.0; per_ring = []
        for k in range(len(RINGS)):
            m = (ring_t == k) & seen_t  # only where the target observes: its histograms come from there
            hist = torch.zeros(T.shape[1]).index_add_(0, ebin_t[m], p[m])
            d = js_torch(hist, Tt[k]); per_ring.append(float(d)); js_total = js_total + d
        # expected count per (sector, range) cell PER FRAME against the target's mean; relative error,
        # so a sparse far cell and a dense near cell weigh alike. Where the source cannot reach the
        # target (count below it at p = 1) the term saturates, which is the accumulate-first design.
        count = torch.zeros(n_cells).index_add_(0, cell_t, p) / n_src_frames
        # log-count error: scale-free, so a sparse far cell and a dense near cell weigh alike (a relative
        # error drove EVERY keep probability to 0 on S3, 2026-10-03).
        # Cells the target NEVER returns a point in (count exactly 0 over the measured frames) are left
        # out by default (--count_cells observed): they say where the target sensor cannot see, not how
        # dense it is. Pulling the source to zero there cut everything outside PandarGT's cone, which
        # emptied 80% of the source's TRAINING boxes - and the zero-point GT filter runs before the data
        # processor, so those empty boxes stayed as labels (L5, job 27261, 20261003_04 section 16). On S1
        # / S2 the excluded cells are the 0-2.5 m bin in a few sectors (the ego body), nothing else.
        err = torch.log1p(count) - torch.log1p(Tct)
        if args.count_cells == 'observed':
            err = err[Tct > 0]
        return js_total, (err ** 2).mean(), per_ring

    with torch.no_grad():
        j0, c0, ring0 = losses(p0)
        _, c_raw, _ = losses(torch.ones_like(p0))
    print('before training (= the rule): elevation JS per ring ' + ' '.join(f'{v:.4f}' for v in ring0)
          + f'  (sum {float(j0):.4f}); (sector, range) count error {float(c0):.4f} (raw source {float(c_raw):.4f})', flush=True)
    for step in range(args.steps):
        opt.zero_grad()
        p = forward(); j, c, _ = losses(p)
        loss = j + args.count_weight * c
        loss.backward(); opt.step()
        if (step + 1) % 50 == 0:
            print(f'  step {step + 1}: elev JS sum {float(j):.4f}  count error {float(c):.4f}  mean keep {float(p.mean()):.3f} (rule {float(p0.mean()):.3f})', flush=True)
    with torch.no_grad():
        p = forward(); j1, c1, ring1 = losses(p)
    print('after training:  elevation JS per ring ' + ' '.join(f'{v:.4f}' for v in ring1)
          + f'  (sum {float(j1):.4f}); count error {float(c1):.4f}; mean keep {float(p.mean()):.3f}', flush=True)

    sampler.w1, sampler.b1, sampler.w2, sampler.b2, sampler.w3, sampler.b3 = [t.detach().numpy().astype(np.float32) for t in (w1, b1, w2, b2, w3, b3)]
    sampler.save(args.out)
    # 5. a Bernoulli-sampled check through the real inference path, on the cached frames
    rng = np.random.RandomState(0)
    H = np.zeros_like(T); n = 0
    for f_frame, rg, eb in zip(feats, rings, ebins):
        pts = _points_from_feats(f_frame)
        keep = rng.rand(len(pts)) < sampler.keep_probability(pts, f_frame)  # features cached, xyz only for the rule's range
        if sampler.obs_mask is not None:
            keep &= sampler.obs_mask[cell_of(f_frame)]
        for k in range(len(RINGS)):
            m = (rg == k) & keep
            H[k] += np.bincount(eb[m], minlength=H.shape[1])
        n += 1
    ring_s = [float(js_torch(torch.tensor(H[k], dtype=torch.float32), Tt[k])) for k in range(len(RINGS))]
    print('sampled check:   elevation JS per ring ' + ' '.join(f'{v:.4f}' for v in ring_s), flush=True)
    print(f'saved {args.out}', flush=True)


def _points_from_feats(F):
    """(x, y, z) consistent with the cached features (range and azimuth restored); only the range is read."""
    pts = np.zeros((len(F), 3), np.float32)
    pts[:, 0] = F[:, 0] * F[:, 6]; pts[:, 1] = F[:, 0] * F[:, 7]; pts[:, 2] = F[:, 1]
    return pts


if __name__ == '__main__':
    main()
