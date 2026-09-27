"""Figure: points-per-kept-box / points-per-GT-box against the score cut, per range ring, on the
labelled SOURCE (Lyft val) and the TARGET (nuScenes, GT read for the answer only). Markers: the
count-balance cut per ring. One panel per domain; Car only.

    python analysis/pseudo_label_threshold/selection_bias_figure.py <src_records> <tgt_records> <out.png>
"""
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from selection_bias_report import Pop, RING_NAMES, THRS  # noqa: E402

src, tgt, out = sys.argv[1:4]
recs = [('SOURCE: Lyft val (labels legal)', pickle.load(open(src, 'rb'))),
        ('TARGET: nuScenes train (GT read for diagnosis only)', pickle.load(open(tgt, 'rb')))]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
for ax, (title, rec) in zip(axes, recs):
    for ring, c in zip(range(5), colors):
        pop = Pop(rec, 1, ring)
        cur = pop.curve()
        ax.plot(THRS, cur[:, 2], '-', color=c, label='%s m (GT %.1f/fr, %.0f pts)' % (RING_NAMES[ring], pop.gt_n, pop.gt_mean))
        tc = pop.t_count()
        if np.isfinite(tc):
            ax.plot([tc], [pop.at(tc)[2]], 'o', color=c, ms=7)
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.axvline(0.1, color='grey', lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_xlim(0.04, 0.9); ax.set_ylim(0.3, 2.6)
    ax.set_xlabel('teacher score cut t (kept: score >= t)')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(alpha=0.3, which='both')
axes[0].set_ylabel('mean pts per KEPT box / mean pts per GT box (Car)')
fig.suptitle('Selection bias of the teacher (iwg6l5v1 ep30): dots = count-balance cut; dotted = plateau edge 0.1', fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=150)
print('wrote', out)
