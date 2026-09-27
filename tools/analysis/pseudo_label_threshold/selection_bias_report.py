"""Selection-bias report over `selection_bias_sweep.py` records.

Per class and range ring: at which score cut t does mean points per KEPT teacher box equal mean
points per GT box (t_density, ratio = 1), and at which does the kept count equal the GT count
(t_count, count balance)? Then the transfer: t_density measured on the SOURCE (Lyft val, labels
legal) applied to the TARGET (nuScenes) records - is the target density estimate unbiased there?
Target GT is used to READ the answer (valid.), never to choose t.

    python analysis/pseudo_label_threshold/selection_bias_report.py \
        --source <lyft_val_records.pkl> --target <nuscenes_records.pkl> [--flat 0.21 0.19 0.18]
"""
import argparse
import pickle

import numpy as np

RING_NAMES = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+']
THRS = np.array([0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.30,
                 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
# ps: ring cls score npts tp r platform frame ; gt: ring cls npts matched r platform frame


class Pop:
    """One (dataset, class, ring, platform) population."""

    def __init__(self, rec, cls, ring, plat=None):
        ps, gt = rec['ps'], rec['gt']
        if plat:
            fr = set(ps[ps[:, 6] == plat][:, 7]) | set(gt[gt[:, 5] == plat][:, 6])
            self.frames = len(fr)
            ps, gt = ps[ps[:, 6] == plat], gt[gt[:, 5] == plat]
        else:
            self.frames = rec['frames']
        self.g = gt[(gt[:, 0] == ring) & (gt[:, 1] == cls) & (gt[:, 2] >= 1)]
        self.p = ps[(ps[:, 0] == ring) & (ps[:, 1] == cls) & (ps[:, 3] >= 1)]
        self.gt_n = len(self.g) / max(self.frames, 1)
        self.gt_mean = self.g[:, 2].mean() if len(self.g) else np.nan

    def at(self, t):
        """(kept/frame, mean pts, ratio to GT, precision, recall) at cut t."""
        k = self.p[self.p[:, 2] >= t]
        n = len(k) / max(self.frames, 1)
        m = k[:, 3].mean() if len(k) else np.nan
        prec = k[:, 4].mean() if len(k) else np.nan
        rec = k[:, 4].sum() / len(self.g) if len(self.g) else np.nan
        return n, m, m / self.gt_mean, prec, rec

    def curve(self):
        return np.array([self.at(t) for t in THRS])

    def t_density(self):
        """Lowest t at which ratio crosses 1 (linear interpolation on the grid)."""
        y = self.curve()[:, 2]
        return _cross(THRS, y, 1.0)

    def t_count(self):
        y = self.curve()[:, 0]
        return _cross(THRS, y, self.gt_n)


def _cross(x, y, target):
    for i in range(1, len(x)):
        a, b = y[i - 1], y[i]
        if np.isfinite(a) and np.isfinite(b) and (a - target) * (b - target) <= 0 and a != b:
            return x[i - 1] + (target - a) * (x[i] - x[i - 1]) / (b - a)
    return np.nan


def fmt(v, f='%.2f'):
    return f % v if np.isfinite(v) else '  -  '


def per_dataset(rec, name, classes, rings, plat=None):
    print('\n=== %s%s: %d frames ===' % (name, ' platform %d' % plat if plat else '', Pop(rec, 1, 1, plat).frames))
    for ci, cname in enumerate(classes, 1):
        print('\n-- %s --   ratio = mean pts per KEPT box / mean pts per GT box (occupied boxes, centre ring)' % cname)
        print('%-6s | %-12s | %-8s %-8s | %s' % ('ring', 'GT n/fr  pts', 't_dens', 't_count',
                                                 '  '.join('t=%.2f: n ratio prec' % t for t in (0.10, 0.18, 0.21, 0.30))))
        for ring in rings:
            pop = Pop(rec, ci, ring, plat)
            cells = ['%5.2f %5.2f %4.2f' % (pop.at(t)[0], pop.at(t)[2], pop.at(t)[3]) for t in (0.10, 0.18, 0.21, 0.30)]
            print('%-6s | %5.2f %6.1f | %-8s %-8s | %s' % (RING_NAMES[ring], pop.gt_n, pop.gt_mean,
                                                            fmt(pop.t_density()), fmt(pop.t_count()), '  '.join(cells)))


def transfer(src, tgt, classes, rings, flat, src_plats=(None, 40, 64)):
    print('\n=== TRANSFER: t_density measured on the SOURCE, applied to the TARGET (target GT read for the answer only) ===')
    for ci, cname in enumerate(classes, 1):
        print('\n-- %s --' % cname)
        hdr = '%-6s | %-14s |' % ('ring', 'TGT GT n/fr pts')
        for p in src_plats:
            hdr += ' src%s: t -> tgt n ratio prec |' % ('' if p is None else str(p))
        hdr += ' tgt own t_dens (valid.) | flat %.2f: n ratio prec' % flat[ci - 1]
        print(hdr)
        for ring in rings:
            tp_ = Pop(tgt, ci, ring)
            row = '%-6s | %5.2f %8.1f |' % (RING_NAMES[ring], tp_.gt_n, tp_.gt_mean)
            for p in src_plats:
                ts = Pop(src, ci, ring, p).t_density()
                if np.isfinite(ts):
                    n, _, r, pr, _ = tp_.at(ts)
                    row += ' %.2f -> %5.2f %4.2f %4.2f |' % (ts, n, r, pr)
                else:
                    row += '   -  ->   -     -    -   |'
            n, _, r, pr, _ = tp_.at(flat[ci - 1])
            row += ' %-23s | %5.2f %4.2f %4.2f' % (fmt(tp_.t_density()), n, r, pr)
            print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=None)
    ap.add_argument('--target', default=None)
    ap.add_argument('--flat', type=float, nargs='+', default=[0.21, 0.19, 0.18])
    ap.add_argument('--rings', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    src = pickle.load(open(args.source, 'rb')) if args.source else None
    tgt = pickle.load(open(args.target, 'rb')) if args.target else None
    classes = (src or tgt)['classes'][:2]          # Car, Pedestrian; Cyclist is ~all FP everywhere
    if tgt is not None:
        per_dataset(tgt, 'TARGET ' + tgt['ps_label'].split('/')[-2], classes, args.rings)
    if src is not None:
        per_dataset(src, 'SOURCE ' + src['ps_label'].split('/')[-2], classes, args.rings)
        plats = sorted(set(int(v) for v in src['ps'][:, 6]) - {0})   # Lyft platforms only; 0 = no platform
        for p in plats:
            per_dataset(src, 'SOURCE ' + src['ps_label'].split('/')[-2], classes, args.rings, plat=p)
    if src is not None and tgt is not None:
        transfer(src, tgt, classes, args.rings, args.flat, src_plats=[None] + plats)


if __name__ == '__main__':
    main()
