"""How deep must a source accumulate for its objects to match the target AT EACH RANGE?

The depth criteria in 20260922_07 are not parallel. The GLOBAL one demands the source's radial
histogram meet the target's in every 5-70 m bin. The OBJECT one pools every box into a single
median, so a source can be declared at parity while its far objects are an order of magnitude
short - which is exactly what 20260922_08's curves show for nuScenes at N=15: parity by 15 m, a
factor of ten down by 60 m.

This applies the global criterion's own standard to objects: the smallest N at which median points
per Car box reaches the target's IN THAT RANGE BIN. It answers whether the far-field object gap can
be closed by accumulation at all.

Counting is additive - a box's points are the sum of each frame's contribution to it - so one
backward pass per anchor yields every depth.
"""
import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from domain_gap_figures import SURFACE, INK, INK2, BLUE, ORANGE, RED, DARK, style, grid
from domain_gap_analysis import build_platforms, mask_range, points_in_boxes, quat_to_rot, DATA

R10 = np.arange(0, 70, 10)                  # 0-10 ... 60-70
GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150]
TARGET = 'KITTI'


def nuscenes_frames(P):
    from accumulation_deep_nuscenes import frames_back, INFOS
    tok = {inf['token']: i for i, inf in enumerate(INFOS)}
    return lambda plat, info, nmax: frames_back(tok[info['token']], nmax)


def lyft_frames(plat, info, nmax):
    """Stored sweeps only - Lyft infos carry 9, so this caps at 10 and says so by stopping."""
    root = DATA / 'lyft/trainval'

    def raw(path):
        a = np.fromfile(str(root / path), dtype=np.float32)
        return a[:len(a) - (len(a) % 5)].reshape([-1, 5])[:, :3]
    yield raw(info['lidar_path'])
    for s in info['sweeps'][:nmax - 1]:
        p = raw(s['lidar_path'])
        tm = s['transform_matrix']
        yield ((tm[:3, :3] @ p.T).T + tm[:3, 3]) if tm is not None else p


def pandaset_frames_factory(device):
    infos = pickle.load(open(DATA / 'pandaset/pandaset_infos_train.pkl', 'rb'))
    by_seq = collections.defaultdict(dict)
    for i in infos:
        by_seq[i['sequence']][i['frame_idx']] = i
    fix = lambda q: str(q).replace('/root/ST3D/data/pandaset', str(DATA / 'pandaset'))
    cache = {}

    def frames(plat, info, nmax):
        import pandas as pd
        seq, a = info['sequence'], info['frame_idx']
        if seq not in cache:
            cache[seq] = json.load(open(DATA / f'pandaset/dataset/{seq}/lidar/poses.json'))
        q = cache[seq][a]
        R = quat_to_rot(*[q['heading'][k] for k in 'wxyz'])
        t = np.array([q['position'][k] for k in 'xyz'])
        for k in range(a, max(-1, a - nmax), -1):
            if k not in by_seq[seq]:
                return
            df = pd.read_pickle(fix(by_seq[seq][k]['lidar_path']))
            df = df[df.d == device]
            ego = (R.T @ (df[['x', 'y', 'z']].to_numpy() - t).T).T
            yield np.column_stack([ego[:, 1], -ego[:, 0], ego[:, 2]])
    return frames


def pad(p):
    return np.column_stack([p, np.zeros(len(p))]) if p.shape[1] == 3 else p


def curve(plat, frames_fn, anchors, nmax):
    """median points per Car box per 10 m bin, at every depth in GRID."""
    got = {n: {i: [] for i in range(len(R10))} for n in GRID}
    for info in plat.sample(anchors):
        fr = plat.frame(info)
        m = fr.names == plat.car_class
        if not m.sum():
            continue
        b = fr.boxes[m]
        r = np.linalg.norm(b[:, :2], axis=1)
        keep = (r >= 5.0) & (r < 70.0)
        b, r = b[keep], r[keep]
        if not len(b):
            continue
        counts = np.zeros(len(b))
        n = 0
        for p in frames_fn(plat, info, nmax):
            counts = counts + points_in_boxes(mask_range(pad(p)), b)
            n += 1
            if n in got:
                for c, d in zip(counts, r):
                    got[n][int(d // 10)].append(c)
        # depths beyond what this anchor could reach are simply absent for it
    med = {n: np.array([np.median(got[n][i]) if got[n][i] else np.nan
                        for i in range(len(R10))]) for n in GRID}
    # How many boxes each bin's median rests on. A far bin backed by a handful of boxes gives a
    # median that moves several-fold between samples, which is what made the 25-anchor run
    # non-monotonic - so the count belongs beside every depth quoted from this.
    med['n_boxes'] = np.array([len(got[GRID[0]][i]) for i in range(len(R10))])
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchors', type=int, default=25)
    ap.add_argument('--out', type=Path, default=Path('.'))
    args = ap.parse_args()
    P = build_platforms()
    P['PandaSet Pandar64'].infos = [i for i in
                                    pickle.load(open(DATA / 'pandaset/pandaset_infos_train.pkl', 'rb'))
                                    if i['frame_idx'] % 4 == 0]

    tgt_c = curve(P[TARGET], lambda plat, info, nmax: iter([plat.frame(info).points]),
                  args.anchors, 1)
    tgt = tgt_c[1]
    print('KITTI target, median pts/Car box by range (n = boxes behind each median):')
    print('   ' + '  '.join('%d-%d:%.0f(n=%d)' % (R10[i], R10[i] + 10, tgt[i], tgt_c['n_boxes'][i])
                            for i in range(len(R10))))

    sources = [('nuScenes n015 Singapore', nuscenes_frames(P), 150),
               ('Lyft 40-beam', lyft_frames, 10),
               ('PandaSet Pandar64', pandaset_frames_factory(0), 75)]
    out = {}
    for name, fn, nmax in sources:
        c = curve(P[name], fn, args.anchors, nmax)
        need = []
        for i in range(len(R10)):
            hit = next((n for n in GRID if n <= nmax and np.isfinite(c[n][i])
                        and c[n][i] >= tgt[i]), None)
            need.append(hit)
        out[name] = (c, need, nmax)
        print('\n%s (reachable depth %d)' % (name, nmax))
        print('   range   :  ' + '  '.join('%5s' % ('%d-%d' % (R10[i], R10[i] + 10))
                                           for i in range(len(R10))))
        print('   N needed:  ' + '  '.join('%5s' % (need[i] if need[i] else '>%d' % nmax)
                                           for i in range(len(R10))))
        print('   boxes    :  ' + '  '.join('%5d' % c['n_boxes'][i] for i in range(len(R10))))

    fig, axes = grid(1, 1, (7.6, 4.8))
    ax = axes[0, 0]
    style(ax)
    x = R10 + 5
    for (name, _, nmax), col in zip(sources, [BLUE, ORANGE, DARK]):
        need = out[name][1]
        y = [n if n else np.nan for n in need]
        ax.plot(x, y, color=col, lw=2, marker='o', ms=5, label='%s' % name)
        for xi, n in zip(x, need):
            if n is None:
                ax.annotate('>%d' % nmax, (xi, nmax), color=col, fontsize=7.5,
                            ha='center', va='bottom')
    ax.set_yscale('log')
    ax.set_xlabel('distance to box (m)', color=INK2, fontsize=9.5)
    ax.set_ylabel('frames needed for object parity in that bin', color=INK2, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle('One accumulation depth cannot serve every range',
                 fontsize=13, color=INK, y=.975)
    ax.set_title('frames needed for median points per Car box to reach KITTI, in each 10 m ring',
                 fontsize=9.5, color=INK2, pad=8)
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(args.out / 'object_parity_depth_by_range.png', dpi=150, facecolor=SURFACE)
    print('\nwrote', args.out / 'object_parity_depth_by_range.png')


if __name__ == '__main__':
    main()
