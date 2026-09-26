"""Figures for experiments_md/20260926_06 (pseudo-label score threshold).

    python figures.py <dir with score_pts.pkl, persistence.pkl, kitti_oos.pkl, audit_full.pkl> <out_dir>

Palette and style follow tools/analysis/domain_gap_figures.py so the report's figures read as one
set. Class hue is fixed across every panel: Car blue, Pedestrian orange, Cyclist red. Ground truth
(validation only) is always ink, dashed.
"""
import pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import bins

S, OUT = sys.argv[1], sys.argv[2]
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9'
BLUE, ORANGE, RED, DARK = '#2a78d6', '#eb6834', '#d03b3b', '#17457c'
HUE = {'Car': BLUE, 'Pedestrian': ORANGE, 'Cyclist': RED}
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'axes.edgecolor': GRID,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 9, 'axes.titlesize': 10,
                     'axes.titleweight': 'bold', 'legend.frameon': False})

SP = pickle.load(open(S + '/score_pts.pkl', 'rb'))          # class, score, npts, range, tp, frame
PE = pickle.load(open(S + '/persistence.pkl', 'rb'))        # class, score, range, persistent, tp, frame
KO = pickle.load(open(S + '/kitti_oos.pkl', 'rb'))          # rows: class, score, npts, range, tp, frame ; gt
AU = pickle.load(open(S + '/audit_full.pkl', 'rb'))
GT = np.array(AU['target']['gt'], dtype=float).reshape(-1, 4); GT = GT[(GT[:, 2] >= 1) & (GT[:, 3] < 70)]
NF = 1000


def logit(s):
    return np.log(s / (1 - s))


def gmm_threshold(s):
    z = logit(s).reshape(-1, 1)
    g = GaussianMixture(bins.NCOMP, random_state=0, n_init=10).fit(z); hi = int(np.argmax(g.means_.ravel()))
    grid = np.linspace(bins.LO, 0.95, 851); post = g.predict_proba(logit(grid).reshape(-1, 1))[:, hi]
    return g, hi, grid[np.argmax(post >= .5)]


# ---------------------------------------------------------------- Fig 1: score histograms + GMM
sets = [('nuScenes Car', SP[SP[:, 0] == 1]), ('nuScenes Pedestrian', SP[SP[:, 0] == 2]),
        ('nuScenes Cyclist', SP[SP[:, 0] == 3]), ('KITTI Car', KO['rows'][KO['rows'][:, 0] == 3]),
        ('KITTI Pedestrian', KO['rows'][KO['rows'][:, 0] == 4])]
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), constrained_layout=True)
for ax, (name, b) in zip(axes, sets):
    s, tp = b[:, 1], b[:, 4] > 0; z = logit(s); cls = name.split()[-1]
    g, hi, t = gmm_threshold(s)
    edges = np.linspace(logit(bins.LO), logit(0.97), 60)
    ax.hist(z, bins=edges, color=HUE[cls], alpha=0.85, density=True, label='all pseudo-labels')
    ax.hist(z[tp], bins=edges, histtype='step', color=INK, density=False,
            weights=np.full(tp.sum(), 1 / (len(z) * (edges[1] - edges[0]))), lw=1.2, ls='--',
            label='matched to GT (validation)')
    xs = np.linspace(edges[0], edges[-1], 300).reshape(-1, 1)
    for k in range(bins.NCOMP):
        pdf = g.weights_[k] * np.exp(-0.5 * ((xs[:, 0] - g.means_[k, 0]) ** 2) / g.covariances_[k, 0, 0]) / np.sqrt(2 * np.pi * g.covariances_[k, 0, 0])
        ax.plot(xs[:, 0], pdf, color=DARK if k == hi else INK2, lw=1.6, label='GMM confident mode' if k == hi else ('GMM other mode' if k == 0 else None))
    ax.axvline(logit(t), color=INK, lw=1.2)
    prec = tp[s >= t].mean()
    ax.set_title('%s\nt = %.2f, precision %.2f (valid.)' % (name, t, prec))
    ticks = ([0.001, 0.01] if bins.FULL else []) + [0.1, 0.2, 0.3, 0.5, 0.8]; ax.set_xticks(logit(np.array(ticks))); ax.set_xticklabels([str(x) for x in ticks])
    ax.set_xlabel('teacher score (logit axis)'); ax.set_yticks([])
axes[0].set_ylabel('density'); axes[0].legend(loc='upper right', fontsize=7)
fig.suptitle('Fig. 1  Full-range teacher scores: one mode at the background plateau (~0.05) and a monotone tail; the true detections are a low shoulder, not a mode (KITTI Car is the exception)' if bins.FULL else 'Fig. 1  Pseudo-label scores are bimodal on the logit axis in every class, including one that is 98% wrong', x=0.01, ha='left', fontsize=10)
fig.savefig(OUT + '/pseudo_label_' + ('full_' if bins.FULL else '') + 'fig1_score_gmm.png', dpi=160); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: points vs score by ring
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), constrained_layout=True)
panels = [(1, 'Car', 10, 20), (1, 'Car', 20, 30), (1, 'Car', 30, 40), (2, 'Pedestrian', 10, 20)]
edges = np.concatenate([bins.LOW_SB, np.arange(0.10, 0.62, 0.02)])
for ax, (c, cn, lo, hi) in zip(axes, panels):
    b = SP[(SP[:, 0] == c) & (SP[:, 3] >= lo) & (SP[:, 3] < hi)]; s, n, tp = b[:, 1], b[:, 2], b[:, 4] > 0
    g = GT[(GT[:, 1] == c) & (GT[:, 3] >= lo) & (GT[:, 3] < hi), 2]
    xs, med, q1, q3, medtp = [], [], [], [], []
    for a, e in zip(edges[:-1], edges[1:]):
        m = (s >= a) & (s < e)
        if m.sum() < 15: continue
        xs.append((a + e) / 2); med.append(np.median(n[m])); q1.append(np.percentile(n[m], 25)); q3.append(np.percentile(n[m], 75))
        medtp.append(np.median(n[m & tp]) if (m & tp).sum() >= 10 else np.nan)
    ax.fill_between(xs, q1, q3, color=HUE[cn], alpha=0.18, lw=0, label='IQR, all pseudo-labels')
    ax.plot(xs, med, color=HUE[cn], lw=2, label='median, all pseudo-labels')
    ax.plot(xs, medtp, color=INK, lw=1.2, ls=':', label='median, matched only (valid.)')
    ax.axhline(np.median(g), color=INK, lw=1.2, ls='--', label='GT median (valid.)')
    ax.set_yscale('log'); ax.set_xlim(bins.LO, 0.6); ax.set_xscale('log' if bins.FULL else 'linear'); ax.set_title('%s, %d–%d m  (n = %d)' % (cn, lo, hi, len(b)))
    ax.set_xlabel('teacher score')
axes[0].set_ylabel('points inside box'); axes[0].legend(fontsize=7, loc='upper left')
fig.suptitle('Fig. 2  Full range: points-in-box is flat from the plateau up to ~0.2 at every ring, then rises for Car at 10–30 m; the sub-0.1 boxes are the same population as 0.10–0.12' if bins.FULL else 'Fig. 2  Points-in-box rises with score for Car at 10–30 m, is flat beyond 30 m, and flat for Pedestrian', x=0.01, ha='left', fontsize=10)
fig.savefig(OUT + '/pseudo_label_' + ('full_' if bins.FULL else '') + 'fig2_points_vs_score.png', dpi=160); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: what a threshold keeps (Car)
b = SP[(SP[:, 0] == 1) & (SP[:, 3] < 70)]; s, n, r, tp = b[:, 1], b[:, 2], b[:, 3], b[:, 4] > 0
g = GT[GT[:, 1] == 1]; ngt = len(g)
T = bins.GRID[bins.GRID <= 0.5]
R = [0, 10, 20, 30, 40, 50, 70]
kept = np.array([(s >= t).sum() for t in T]) / NF
prec = np.array([tp[s >= t].mean() for t in T]); rec = np.array([(tp & (s >= t)).sum() / ngt for t in T])
f1 = 2 * prec * rec / (prec + rec)
def tv(x):
    hx = np.histogram(x, R)[0] / len(x); hy = np.histogram(g[:, 3], R)[0] / len(g); return .5 * np.abs(hx - hy).sum()
def w1(x, y):
    q = np.linspace(.005, .995, 199); return np.abs(np.quantile(x, q) - np.quantile(y, q)).mean()
tvs = np.array([tv(r[s >= t]) for t in T]); w1s = np.array([w1(np.log(n[s >= t]), np.log(g[:, 2])) for t in T])
bias = {}
for lo, hi in [(10, 20), (20, 30), (30, 40)]:
    gm = g[(g[:, 3] >= lo) & (g[:, 3] < hi), 2].mean()
    bias[(lo, hi)] = np.array([n[(s >= t) & (r >= lo) & (r < hi)].mean() / gm for t in T])
_, _, t_gmm = gmm_threshold(s)
t_cb = T[np.argmin(np.abs(kept - ngt / NF))]

fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)
ax = axes[0]
ax.plot(T, prec, color=BLUE, lw=2, label='precision'); ax.plot(T, rec, color=DARK, lw=2, label='recall'); ax.plot(T, f1, color=INK2, lw=1.4, ls=':', label='F1')
ax.set_ylim(0, 1); ax.set_title('Training-label quality of the kept set (validation)'); ax.set_ylabel('fraction')
ax = axes[1]
for (lo, hi), col in zip(bias, [BLUE, DARK, INK2]):
    ax.plot(T, bias[(lo, hi)], color=col, lw=2, label='%d–%d m' % (lo, hi))
ax.axhline(1, color=INK, lw=1.2, ls='--', label='unbiased'); ax.set_ylim(0.5, 2.6)
ax.set_title('Density estimate: kept-set mean pts/box ÷ GT mean (validation)'); ax.set_ylabel('ratio')
ax = axes[2]
ax.plot(T, tvs, color=BLUE, lw=2, label='TV distance, range distribution'); ax.plot(T, w1s / 3, color=DARK, lw=2, label='W1 distance, log points (÷3)')
ax.plot(T, kept / (ngt / NF), color=INK2, lw=1.4, ls=':', label='kept ÷ real boxes per frame')
ax.set_ylim(0, 1.2); ax.set_title('Kept-set distance to GT and box-count balance'); ax.set_ylabel('distance / ratio')
for ax in axes:
    ax.axvline(t_gmm, color=INK, lw=1.2); ax.text(t_gmm + 0.005, ax.get_ylim()[1] * 0.96, 'GMM t=%.2f\n(label-free)' % t_gmm, fontsize=7, va='top')
    ax.set_xlabel('SCORE_THRESH  (keeps every box ≥ t)'); ax.set_xlim(bins.LO, 0.5); ax.set_xscale('log' if bins.FULL else 'linear'); ax.legend(fontsize=7)
fig.suptitle('Fig. 3  nuScenes Car, full range: below 0.1 nothing changes; count balance and the distance minimum stay at 0.18–0.20, the 3-component GMM lands at 0.24' if bins.FULL else 'Fig. 3  nuScenes Car: the label-free GMM threshold sits at box-count balance; precision keeps rising past it while the density estimate goes over 1', x=0.01, ha='left', fontsize=10)
fig.savefig(OUT + '/pseudo_label_' + ('full_' if bins.FULL else '') + 'fig3_threshold_tradeoff.png', dpi=160); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: persistence and range mix
fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), constrained_layout=True)
ax = axes[0]
edges = np.concatenate([bins.LOW_SB, np.arange(0.10, 0.50, 0.02), [0.6, 0.8, 1.01]])
for c, cn in [(1, 'Car'), (2, 'Pedestrian'), (3, 'Cyclist')]:
    pe = PE[(PE[:, 0] == c) & (PE[:, 1] >= 0)]; s_, p_, tp_ = pe[:, 1], pe[:, 3] > 0, pe[:, 4] > 0
    xs, ys = [], []
    for a, e in zip(edges[:-1], edges[1:]):
        m = (s_ >= a) & (s_ < e)
        if m.sum() >= 30: xs.append((a + e) / 2); ys.append(p_[m].mean())
    ax.plot(xs, ys, color=HUE[cn], lw=2, marker='o', ms=3, label=cn)
    if c == 1:
        ax.axhline(p_[tp_].mean(), color=INK, ls='--', lw=1, label='Car P(persist | matched) (valid.)')
        ax.axhline(p_[~tp_].mean(), color=INK, ls=':', lw=1, label='Car P(persist | unmatched) (valid.)')
ax.set_xscale('log'); tk = ([0.001, 0.01] if bins.FULL else []) + [0.1, 0.2, 0.3, 0.5, 1.0]; ax.set_xticks(tk); ax.set_xticklabels([str(x) for x in tk])
ax.set_ylim(0, 1); ax.set_xlabel('teacher score'); ax.set_ylabel('fraction persisting in adjacent keyframes'); ax.set_title('Temporal persistence vs score (label-free)'); ax.legend(fontsize=7)
ax = axes[1]
SB = bins.SB
for c, cn in [(1, 'Car'), (2, 'Pedestrian'), (3, 'Cyclist')]:
    bb = SP[(SP[:, 0] == c) & (SP[:, 3] < 70)]
    xs = [(a + min(e, 1)) / 2 for a, e in zip(SB[:-1], SB[1:])]
    ys = [np.median(bb[(bb[:, 1] >= a) & (bb[:, 1] < e), 3]) for a, e in zip(SB[:-1], SB[1:])]
    ax.plot(xs, ys, color=HUE[cn], lw=2, marker='o', ms=3, label=cn)
    ax.axhline(np.median(GT[GT[:, 1] == c, 3]), color=HUE[cn], lw=1, ls='--')
ax.plot([], [], color=INK, ls='--', lw=1, label='GT median range, same hue (valid.)')
ax.set_xlabel('teacher score (bin centre)'); ax.set_ylabel('median box range (m)'); ax.set_title('Where each score bin sits in range'); ax.legend(fontsize=7)
ax = axes[2]
Rr = [0, 10, 20, 30, 40, 50, 70]; b = SP[(SP[:, 0] == 1) & (SP[:, 3] < 70)]; g = GT[GT[:, 1] == 1]
TH = ([0.0001, 0.01] if bins.FULL else []) + [0.10, 0.18, 0.30, 0.50]; cats = ['GT'] + ['≥%g' % t for t in TH]
shares = [np.histogram(g[:, 3], Rr)[0] / len(g)] + [np.histogram(b[b[:, 1] >= t, 3], Rr)[0] / (b[:, 1] >= t).sum() for t in TH]
ramp = ['#dce9f9', '#9cc3ee', '#4a8fdd', '#2a78d6', '#17457c', '#0d2a4d']
bottom = np.zeros(len(cats))
for k, (lo, hi) in enumerate(zip(Rr[:-1], Rr[1:])):
    vals = [sh[k] for sh in shares]
    ax.bar(cats, vals, bottom=bottom, color=ramp[k], edgecolor=SURFACE, linewidth=1, label='%d–%d m' % (lo, hi)); bottom += vals
ax.set_ylim(0, 1); ax.set_ylabel('share of Car boxes'); ax.set_title('Range mix of the kept Car set vs GT'); ax.legend(fontsize=7, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.12), handlelength=1)
fig.suptitle('Fig. 4  Full range: plateau boxes (<0.1) persist MORE than low-score detections - static clutter re-fires every frame - so persistence is not monotone across the plateau' if bins.FULL else 'Fig. 4  Persistence separates score bands inside the same class; each score band lives at a different range, so a threshold also picks a range mix', x=0.01, ha='left', fontsize=10)
fig.savefig(OUT + '/pseudo_label_' + ('full_' if bins.FULL else '') + 'fig4_persistence_range.png', dpi=160); plt.close(fig)
print('wrote 4 figures to', OUT)
