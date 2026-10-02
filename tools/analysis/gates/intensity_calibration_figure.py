"""Figures for the intensity-calibration gate (experiments_md 20261002_02): how nuScenes intensities
compare with KITTI's, per range ring, before and after a per-ring quantile map - one map for all
points ("global") or one per channel (car points / background points, "channel-aware").

Maps are fitted on even frames and shown on odd frames, so the channel-aware curves are not trivially
exact. GT boxes on both sides, diagnosis only. Writes two PNGs into the directory given as argv[1].

    python analysis/gates/intensity_calibration_figure.py /home/koyama/code/experiments_md
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intensity_gate as g

out = sys.argv[1] if len(sys.argv) > 1 else '.'
S, T = g.nuscenes(), g.kitti()
rings = list(zip(g.EDGES[:-1], g.EDGES[1:]))
CH = {'Car points': 2, 'Background points': 3}
SERIES = [('KITTI (target)', 'k', '-'), ('nuScenes raw / 255', 'tab:gray', ':'),
          ('nuScenes, global map', 'tab:blue', '--'), ('nuScenes, channel-aware map', 'tab:red', '-')]

def series_values(lo, hi, col):
    s = S[(S[:, 0] >= lo) & (S[:, 0] < hi)]; t = T[(T[:, 0] >= lo) & (T[:, 0] < hi)]
    A = lambda x: x[x[:, 5] == 0, 1]; B = lambda x: x[x[:, 5] == 1, 1]
    sc, tc = s[s[:, col] == 1], t[t[:, col] == 1]
    glob = g.qmap(A(s), A(t)); chan = g.qmap(A(sc), A(tc))
    return [B(tc), g.dequantize(B(sc)) / 255.0, glob(B(sc)), chan(B(sc))]

vals = {name: [series_values(lo, hi, col) for lo, hi in rings] for name, col in CH.items()}
mid = np.array([(lo + hi) / 2 for lo, hi in rings])

# 1. range profile: median and inter-quartile band per ring
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (name, per_ring) in zip(axes, vals.items()):
    for k, (label, colour, ls) in enumerate(SERIES):
        q = np.array([np.percentile(v[k], [25, 50, 75]) for v in per_ring])
        ax.plot(mid, q[:, 1], ls, color=colour, marker='o', ms=4, label=label)
        ax.fill_between(mid, q[:, 0], q[:, 2], color=colour, alpha=0.12)
    ax.set_title(name); ax.set_xlabel('range ring centre (m)'); ax.grid(alpha=0.3)
axes[0].set_ylabel('intensity, KITTI units (median, band = p25-p75)')
axes[1].legend(fontsize=8, loc='upper right')
fig.suptitle('Intensity by range, nuScenes -> KITTI: maps fitted on even frames, shown on odd frames (GT boxes, diagnosis only)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(out, 'intensity_calibration_profile.png'), dpi=150); plt.close(fig)

# 2. per-ring histograms
bins = np.linspace(0, 0.6, 41)
fig, axes = plt.subplots(2, len(rings), figsize=(3.1 * len(rings), 6.2), sharex=True)
for r, (name, per_ring) in enumerate(vals.items()):
    for c, ((lo, hi), v) in enumerate(zip(rings, per_ring)):
        ax = axes[r, c]
        for k, (label, colour, ls) in enumerate(SERIES):
            ax.hist(np.clip(v[k], 0, 0.6 - 1e-6), bins=bins, density=True, histtype='step', color=colour,
                    linestyle=ls, linewidth=1.4 if k in (0, 3) else 1.0, label=label)
        ax.set_yscale('log'); ax.set_title('%s, %d-%d m' % (name.split()[0], lo, hi), fontsize=9)
        if r == 1: ax.set_xlabel('intensity (KITTI units, clipped at 0.6)')
axes[0, 0].legend(fontsize=7)
fig.suptitle('Intensity histograms per range ring (density, log scale)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(out, 'intensity_calibration_histograms.png'), dpi=150); plt.close(fig)
print('wrote', out)
