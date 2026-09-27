"""Evidence figures for the two label-free estimators of the real object count (20260927_04 section 2b).

Fig A (persistence):  needs <work>/persistence.pkl from persistence.py
   left  - fraction of kept Car boxes that persist in both adjacent keyframes, by score bin (label-free),
           with P(persist | matched) and P(persist | unmatched) as validation lines;
   right - kept boxes/frame against the cut, the persistence-unmixed real-count estimate
           N_est = pi(0.1) * kept(0.1) (label-free) and the GT count (valid.); count balance = where the
           kept curve crosses N_est.
Fig B (joint mixture): needs <work>/nusc_k500_records.pkl from selection_bias_sweep.py
   left  - Car boxes at 10-20 m in (logit score, log points) with the 2-component GMM fitted on
           score >= 0.1, matched boxes drawn darker (valid.);
   right - per-ring real count from the component weight (label-free) against the GT count (valid.),
           and the score cut each implies.

    python estimator_evidence_figures.py <work dir> <out dir>
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
import numpy as np  # noqa: E402

W, OUT = sys.argv[1], sys.argv[2]
# optional: a records file for Fig B other than nuScenes (e.g. the KITTI sweep), with a name/label
REC = sys.argv[3] if len(sys.argv) > 3 else os.path.join(W, 'nusc_k500_records.pkl')
LABEL = sys.argv[4] if len(sys.argv) > 4 else 'nuScenes'
SUFFIX = '' if len(sys.argv) <= 3 else '_' + LABEL.lower()
CUTS = np.array([0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80])
BINS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.17, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0]
C_LF, C_VAL, C_GT = '#1f77b4', '#7f7f7f', '#d62728'


def logit(s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def cross(x, y, target):
    for i in range(1, len(x)):
        a, b = y[i - 1], y[i]
        if (a - target) * (b - target) <= 0 and a != b:
            return x[i - 1] + (target - a) * (x[i] - x[i - 1]) / (b - a)
    return np.nan


# ------------------------------------------------------------------ Fig A: persistence
pf = os.path.join(W, 'persistence.pkl')
if os.path.exists(pf) and len(sys.argv) <= 3:
    PE = pickle.load(open(pf, 'rb'))          # cls, score, range, persistent, tp, frame ; GT markers: score=-1, tp=n_gt
    gtm = PE[PE[:, 1] < 0]
    frames = len(np.unique(gtm[:, 5]))
    car = PE[(PE[:, 0] == 1) & (PE[:, 1] >= 0) & (PE[:, 2] < 50)]
    n_gt = gtm[gtm[:, 0] == 1][:, 4]           # per anchor frame, all ranges < 70 m in persistence.py
    # the GT count marker in persistence.py is RMAX = 70 m; restrict kept boxes the same way for the right panel
    car70 = PE[(PE[:, 0] == 1) & (PE[:, 1] >= 0)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # left: persistence by score bin
    xs, ys, ns = [], [], []
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = (car[:, 1] >= a) & (car[:, 1] < b)
        if m.sum() >= 50:
            xs.append(np.sqrt(a * b)); ys.append(car[m, 3].mean()); ns.append(m.sum())
    ax[0].plot(xs, ys, 'o-', color=C_LF, label='Car, all kept boxes (label-free)')
    tp = car[:, 4] > 0
    ax[0].axhline(car[tp, 3].mean(), color=C_VAL, ls='--', label='P(persist | matched to GT)  valid.')
    ax[0].axhline(car[~tp, 3].mean(), color=C_VAL, ls=':', label='P(persist | unmatched)  valid.')
    ax[0].axvline(0.1, color='k', lw=0.8, ls=':')
    ax[0].set_xscale('log'); ax[0].set_ylim(0, 1); ax[0].set_xlabel('teacher score (bin centre)')
    ax[0].set_ylabel('fraction persisting in both adjacent keyframes')
    ax[0].set_title('Flat across the plateau (< 0.1), rises with score above it', fontsize=9)
    ax[0].legend(fontsize=8, loc='lower right'); ax[0].grid(alpha=0.3, which='both')
    # right: unmixing
    p_real = car70[car70[:, 1] >= 0.6, 3].mean()
    p_fp = car70[(car70[:, 1] >= 0.10) & (car70[:, 1] < 0.12), 3].mean()
    kept = np.array([(car70[:, 1] >= t).sum() / frames for t in CUTS])
    pers = np.array([car70[car70[:, 1] >= t, 3].mean() for t in CUTS])
    pi = np.clip((pers - p_fp) / (p_real - p_fp), 0, 1)
    n_est = pi[0] * kept[0]
    gt_per_frame = n_gt.mean()
    t_star = cross(CUTS, kept, n_est)
    t_gt = cross(CUTS, kept, gt_per_frame)
    ax[1].plot(CUTS, kept, 'o-', color=C_LF, label='kept Car boxes / frame (label-free)')
    ax[1].plot(CUTS, pi * kept, 's--', color='#2ca02c', ms=4, label='persistence-unmixed real share x kept (label-free)')
    ax[1].axhline(n_est, color='#2ca02c', lw=1, label='N_est = pi(0.1) x kept(0.1) = %.2f / frame' % n_est)
    ax[1].axhline(gt_per_frame, color=C_GT, ls='--', lw=1, label='real Car boxes / frame = %.2f  valid.' % gt_per_frame)
    if np.isfinite(t_star):
        ax[1].axvline(t_star, color='#2ca02c', lw=1, ls=':', label='count balance (label-free) t = %.2f' % t_star)
    if np.isfinite(t_gt):
        ax[1].axvline(t_gt, color=C_GT, lw=1, ls=':', label='count balance against GT t = %.2f  valid.' % t_gt)
    ax[1].set_xscale('log'); ax[1].set_xlabel('score cut t'); ax[1].set_ylabel('boxes per frame (< 70 m)')
    ax[1].set_title('Real count from persistence (p_real = %.2f at score >= 0.6, p_fp = %.2f at 0.10-0.12)' % (p_real, p_fp), fontsize=9)
    ax[1].legend(fontsize=7.5); ax[1].grid(alpha=0.3, which='both')
    fig.suptitle('Estimator 1 - temporal persistence, nuScenes Car, %d anchor frames (teacher iwg6l5v1 ep30, K = 500)' % frames, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'pseudo_label_estimator_fig1_persistence.png'), dpi=150)
    print('fig1: p_real %.3f p_fp %.3f N_est %.2f GT %.2f t_star %.3f t_gt %.3f' % (p_real, p_fp, n_est, gt_per_frame, t_star, t_gt))
else:
    print('no persistence.pkl - skipping fig 1')

# ------------------------------------------------------------------ Fig B: joint mixture
rf = REC
if os.path.exists(rf):
    from sklearn.mixture import GaussianMixture
    R = pickle.load(open(rf, 'rb'))
    ps, gt, frames = R['ps'], R['gt'], R['frames']   # ps: ring cls score npts tp r platform frame
    car = ps[(ps[:, 1] == 1) & (ps[:, 3] >= 1) & (ps[:, 2] >= 0.1)]
    gcar = gt[(gt[:, 1] == 1) & (gt[:, 2] >= 1)]
    rings = [(0, '0-10'), (1, '10-20'), (2, '20-30'), (3, '30-40'), (4, '40-50')]
    fits = {}
    for ri, _ in rings:
        b = car[car[:, 0] == ri]
        z = np.column_stack([logit(b[:, 2]), np.log(b[:, 3])])
        g = GaussianMixture(2, covariance_type='full', random_state=0, n_init=8).fit(z)
        hi = int(np.argmax(g.means_[:, 1] + 0.01 * g.means_[:, 0]))
        fits[ri] = (b, z, g, hi)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    b, z, g, hi = fits[1]
    tp = b[:, 4] > 0
    ax[0].scatter(z[~tp, 0], z[~tp, 1], s=3, alpha=0.25, color='#9ecae1', label='unmatched box  valid.')
    ax[0].scatter(z[tp, 0], z[tp, 1], s=3, alpha=0.5, color='#08519c', label='matched to a real car  valid.')
    for k, col, lab in [(hi, '#d62728', 'fitted "real" component'), (1 - hi, '#ff7f0e', 'fitted "clutter" component')]:
        vals, vecs = np.linalg.eigh(g.covariances_[k])
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        for nsig in (1, 2):
            ax[0].add_patch(Ellipse(g.means_[k], 2 * nsig * np.sqrt(vals[1]), 2 * nsig * np.sqrt(vals[0]), angle=ang,
                                    fill=False, color=col, lw=1.5 if nsig == 1 else 0.8, label=lab if nsig == 1 else None))
    sx = [0.1, 0.2, 0.3, 0.5, 0.8]
    ax[0].set_xticks(logit(np.array(sx))); ax[0].set_xticklabels([str(v) for v in sx])
    py = [2, 5, 10, 30, 100, 300]
    ax[0].set_yticks(np.log(py)); ax[0].set_yticklabels([str(v) for v in py])
    ax[0].set_xlabel('teacher score (logit axis)'); ax[0].set_ylabel('points in box (log axis)')
    ax[0].set_title('Car, 10-20 m, score >= 0.1: "real" component weight %.2f\n-> %.2f boxes/frame (label-free) vs %.2f real (valid.)'
                    % (g.weights_[hi], g.weights_[hi] * len(b) / frames, (gcar[:, 0] == 1).sum() / frames), fontsize=9)
    ax[0].legend(fontsize=8, loc='lower right'); ax[0].grid(alpha=0.3)
    # right: per-ring counts and implied cuts
    xs = np.arange(len(rings)); est, real, tcut, tgt = [], [], [], []
    for ri, name in rings:
        b, z, g, hi = fits[ri]
        n_est = g.weights_[hi] * len(b) / frames
        n_real = (gcar[:, 0] == ri).sum() / frames
        est.append(n_est); real.append(n_real)
        srt = np.sort(b[:, 2])[::-1]
        k_est, k_gt = int(round(n_est * frames)), int(round(n_real * frames))
        tcut.append(srt[min(k_est, len(srt)) - 1] if k_est > 0 else np.nan)
        tgt.append(srt[min(k_gt, len(srt)) - 1] if k_gt > 0 else np.nan)
    ax[1].bar(xs - 0.18, est, 0.36, color='#2ca02c', label='real count from component weight (label-free)')
    ax[1].bar(xs + 0.18, real, 0.36, color=C_GT, alpha=0.7, label='real Car boxes / frame  valid.')
    for i in range(len(rings)):
        ax[1].text(xs[i] - 0.18, est[i] + 0.05, 't=%.2f' % tcut[i], ha='center', fontsize=7.5, color='#2ca02c')
        ax[1].text(xs[i] + 0.18, real[i] + 0.05, 't=%.2f' % tgt[i], ha='center', fontsize=7.5, color=C_GT)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([n + ' m' for _, n in rings]); ax[1].set_ylabel('Car boxes per frame')
    ax[1].set_title('Per-ring real count from the component weight, and the score cut it implies', fontsize=9)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis='y')
    fig.suptitle('Estimator 2 - joint (score, points-in-box) mixture per range ring, %s Car, %d frames (%s)' % (LABEL, frames, os.path.basename(os.path.dirname(R['ps_label']))), fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'pseudo_label_estimator_fig2_joint_mixture%s.png' % SUFFIX), dpi=150)
    print('fig2: est/frame', np.round(est, 2), 'real/frame', np.round(real, 2), 'cut est', np.round(tcut, 3), 'cut gt', np.round(tgt, 3))
else:
    print('no nusc_k500_records.pkl - skipping fig 2')
