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

    python train_point_sampler.py <cfg> --out <weights.npz> [--frames 200] [--steps 400] [--count_weight 1.0]

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
from pcdet.datasets.processor.point_sampler import LearnedPointSampler, point_features  # noqa: E402

RINGS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
ELEV_EDGES = np.arange(-30, 10.01, 0.25)
N_SECTORS = 24  # 15 deg azimuth sectors
RANGE_EDGES = np.arange(0, 75.01, 2.5)  # 30 range bins for the (sector, range) count target


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
    az = np.arctan2(feats[:, 7], feats[:, 6])
    sec = np.clip(np.floor((az + np.pi) / (2 * np.pi) * N_SECTORS).astype(np.int64), 0, N_SECTORS - 1)
    rb = np.clip(np.digitize(feats[:, 0], RANGE_EDGES) - 1, 0, len(RANGE_EDGES) - 2)
    return sec * (len(RANGE_EDGES) - 1) + rb


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
    print(f'cached {len(F)} source points from {n_src_frames} frames; target histograms from {n_tgt} frames', flush=True)

    sampler = LearnedPointSampler.from_rule(rate, max_dist, sz, F.mean(0), F.std(0) + 1e-6)
    # 4. train the MLP on cached features
    x = torch.tensor((F - sampler.feat_mean) / sampler.feat_std)
    base = torch.tensor(sampler.rule_logit(_points_from_feats(F)))  # the rule's logit per cached point
    w1 = torch.tensor(sampler.w1, requires_grad=True); b1 = torch.tensor(sampler.b1, requires_grad=True)
    w2 = torch.tensor(sampler.w2, requires_grad=True); b2 = torch.tensor(sampler.b2, requires_grad=True)
    w3 = torch.tensor(sampler.w3, requires_grad=True); b3 = torch.tensor(sampler.b3, requires_grad=True)
    Tt = torch.tensor(T, dtype=torch.float32); Tct = torch.tensor(Tc, dtype=torch.float32)
    ring_t = torch.tensor(R); ebin_t = torch.tensor(E); cell_t = torch.tensor(C)
    p0 = torch.sigmoid(base)
    opt = torch.optim.Adam([w1, b1, w2, b2, w3, b3], lr=args.lr)

    def forward():
        h = torch.relu(x @ w1 + b1); h = torch.relu(h @ w2 + b2)
        return torch.sigmoid(base + (h @ w3 + b3)[:, 0])

    def losses(p):
        js_total = 0.0; per_ring = []
        for k in range(len(RINGS)):
            m = ring_t == k
            hist = torch.zeros(T.shape[1]).index_add_(0, ebin_t[m], p[m])
            d = js_torch(hist, Tt[k]); per_ring.append(float(d)); js_total = js_total + d
        # expected count per (sector, range) cell PER FRAME against the target's mean; relative error,
        # so a sparse far cell and a dense near cell weigh alike. Where the source cannot reach the
        # target (count below it at p = 1) the term saturates, which is the accumulate-first design.
        count = torch.zeros(n_cells).index_add_(0, cell_t, p) / n_src_frames
        rel = (count - Tct) / (Tct + 1.0)
        return js_total, (rel ** 2).mean(), per_ring

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
