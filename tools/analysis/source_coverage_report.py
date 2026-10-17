"""Tables and readings of source_coverage.py (experiments_md 20261018_01 §3). Reads the npz files in --dir:
nus_base.npz (P1, stride 4), nus_aug.npz (P1 augmented, stride 16), nus_ops_*.npz (Part B shards, stride 8),
waymo_train.npz (P2, stride 4), waymo_val.npz (P3). Writes tables.md and numbers.json into --dir.
ANALYSIS: P2 and P3 are Waymo labels; nothing here selects a method or a value."""
import glob
import json
from pathlib import Path

import numpy as np

from source_coverage import (BOX_COLS, RINGS, RING_NAMES, NVOX_EDGES, NVOX_NAMES, NLAY_EDGES, NLAY_NAMES, LONG_M,
                             VOXEL_CAP, R_NEW, support, reachable_total, covered, cell_of, uncovered_share, a2_reading,
                             reach, bin_of)

EPOCHS_SRC, EPOCHS_TGT = 20, 7
A2_RINGS = (4, 5, 6)                       # 40-50, 50-60, 60-75 m
SECTORS = ('ahead', 'sides', 'behind')
PLATS = ('Boston', 'Singapore')
G1_OPS = ('base', 'B1a', 'B2a', 'B1b', 'B2b')


def _rows(a):
    a = np.asarray(a, np.float64)
    return a.reshape(-1, len(BOX_COLS)) if a.size else np.zeros((0, len(BOX_COLS)))


def _load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _q(v, f='{:.0f}'):
    v = np.asarray(v, np.float64)
    if not len(v):
        return '-'
    return ' / '.join(f.format(x) for x in np.percentile(v, [25, 50, 75]))


def _pct(x):
    return '-' if x is None or not np.isfinite(x) else f'{100 * x:.1f}%'


def _f(x, f='{:.2f}'):
    return '-' if x is None or not np.isfinite(x) else f.format(x)


class Out:
    def __init__(self):
        self.lines = []

    def __call__(self, s=''):
        self.lines.append(s); print(s)

    def table(self, head, rows):
        self('| ' + ' | '.join(head) + ' |'); self('|' + '---|' * len(head))
        for r in rows:
            self('| ' + ' | '.join(str(x) for x in r) + ' |')
        self()


def main_report(a):
    d = Path(a.dir); out = Out(); num = {}
    P1 = _load(d / 'nus_base.npz'); P2 = _load(d / 'waymo_train.npz'); V = _load(d / 'waymo_val.npz')
    AUG = _load(d / 'nus_aug.npz') if (d / 'nus_aug.npz').exists() else None
    shards = [_load(p) for p in sorted(glob.glob(str(d / 'nus_ops_*.npz')))]
    s1, s2 = float(P1['stride']), float(P2['stride'])
    C1, O1, C2, CV = _rows(P1['car']), _rows(P1.get('other', np.zeros(0))), _rows(P2['car']), _rows(V['car'])
    out(f'# source_coverage report ({d})'); out()
    out(f'P1 frames {int(P1["n_tasks"])} (stride {int(s1)}), P2 frames {int(P2["n_tasks"])} (stride {int(s2)} of '
        f'{int(P2["n_sampled"])}), P3 frames {int(V["n_tasks"])}, P3 GT {len(CV)}; ops shards {len(shards)}'); out()

    # ---------------------------------------------------------------- supports and coverage
    s_src = support(C1, s1); s_tgt = support(C2, s2); s_oth = support(O1, s1)
    S_src, S_tgt = reachable_total(s_src), reachable_total(s_tgt)
    cov = {'share': covered(s_src, s_tgt, 'share'), 'count': covered(s_src, s_tgt, 'count'),
           'run': covered(s_src, s_tgt, 'run', scale_src=EPOCHS_SRC, scale_tgt=EPOCHS_TGT)}
    if AUG is not None:
        s_aug = support(_rows(AUG['car']), float(AUG['stride']))
        cov['augmented'] = covered(s_aug, s_tgt, 'share')
    s_co = s_src + s_oth
    cov_co = covered(s_co, s_tgt, 'share')
    fr1, fr2 = np.asarray(P1['frame']).reshape(-1, 6), np.asarray(P2['frame']).reshape(-1, 6)
    out('## Totals (instances with n_vox >= 1 at r < 75 m)'); out()
    out.table(['population', 'per epoch', 'per run', 'frames', 'frames over 150k voxels', 'points per frame (median)'],
              [['P1 nuScenes `car` (27471)', f'{S_src:,.0f}', f'{S_src * EPOCHS_SRC:,.0f}', len(fr1),
                _pct(np.mean(fr1[:, 3] > VOXEL_CAP)), f'{np.median(fr1[:, 2]):,.0f}'],
               ['P1 nuScenes other vehicles (background)', f'{reachable_total(s_oth):,.0f}',
                f'{reachable_total(s_oth) * EPOCHS_SRC:,.0f}', '', '', ''],
               ['P2 Waymo train Vehicle (26398)', f'{S_tgt:,.0f}', f'{S_tgt * EPOCHS_TGT:,.0f}', len(fr2),
                _pct(np.mean(fr2[:, 3] > VOXEL_CAP)), f'{np.median(fr2[:, 2]):,.0f}']])
    out(f'Ratio of per-epoch totals S_src / S_tgt = {S_src / S_tgt:.3f}; per run {S_src * EPOCHS_SRC / (S_tgt * EPOCHS_TGT):.3f}.')
    out()
    num['totals'] = dict(S_src=S_src, S_tgt=S_tgt, S_other=reachable_total(s_oth), ratio_epoch=S_src / S_tgt,
                         ratio_run=S_src * EPOCHS_SRC / (S_tgt * EPOCHS_TGT),
                         p1_frames_over_cap=float(np.mean(fr1[:, 3] > VOXEL_CAP)),
                         p2_frames_over_cap=float(np.mean(fr2[:, 3] > VOXEL_CAP)))

    # ---------------------------------------------------------------- P3 groups
    o5, u5, iou = V['oracle_found5'].astype(bool), V['ours27596_found5'].astype(bool), V['ours27596_best_iou']
    grp = {'oracle-only (misses)': o5 & ~u5, 'no-overlap misses': o5 & ~u5 & (iou <= 0), 'found by both': o5 & u5,
           'missed by both': ~o5 & ~u5, 'ours-only': ~o5 & u5}
    cv_cells = cell_of(CV[:, 0], CV[:, 2], CV[:, 3])
    short, reach_ok = CV[:, 5] < LONG_M, CV[:, 2] >= 1
    pop = short & reach_ok
    rv = np.searchsorted(RINGS, CV[:, 0], side='right') - 1
    out('## Group sizes (Waymo val Vehicle GT, IoU 0.5)'); out()
    out.table(['group', 'all', '< 6 m', '< 6 m and n_vox >= 1', 'n_vox = 0 (< 6 m)', '>= 6 m', 'r >= 75 m'],
              [[g, f'{m.sum():,}', f'{(m & short).sum():,}', f'{(m & pop).sum():,}', f'{(m & short & ~reach_ok).sum():,}',
                f'{(m & ~short).sum():,}', f'{(m & (CV[:, 0] >= 75)).sum():,}'] for g, m in grp.items()])
    num['groups'] = {g: dict(all=int(m.sum()), pop=int((m & pop).sum()), zero_vox=int((m & short & ~reach_ok).sum()),
                             long=int((m & ~short).sum())) for g, m in grp.items()}

    # ---------------------------------------------------------------- medians per ring
    out('## Medians per ring: n / median n_vox / median n_layers / median ppv'); out()
    pops = [('P1 `car`', C1), ('P1 other vehicles', O1), ('P2 Waymo train', C2)] + \
           [(g, CV[m]) for g, m in grp.items() if g != 'ours-only']
    head = ['population'] + list(RING_NAMES)
    rows = []
    for name, R in pops:
        rg = np.searchsorted(RINGS, R[:, 0], side='right') - 1
        cells = []
        for k in range(len(RING_NAMES)):
            q = R[(rg == k) & (R[:, 0] < 75)]
            cells.append('-' if not len(q) else
                         f'{len(q):,} / {np.median(q[:, 2]):.0f} / {np.median(q[:, 3]):.0f} / {np.median(q[:, 4]):.2f}')
        rows.append([name] + cells)
    out.table(head, rows)
    out('(P1 and P2 counts are frames processed, not per epoch; all lengths; n_vox = 0 included.)'); out()

    # ---------------------------------------------------------------- A2
    def a2(cv, members, found=grp['found by both'], extra=pop):
        per = []
        for k in A2_RINGS:
            ring = rv == k
            per.append((uncovered_share(cv_cells[members & extra & ring], cv),
                        uncovered_share(cv_cells[found & extra & ring], cv)))
        return per

    out('## A2: uncovered share per ring (misses / found by both), length < 6 m, n_vox >= 1'); out()
    rows, num['A2'] = [], {}
    variants = [('DECIDING: misses, rho_share', cov['share'], grp['oracle-only (misses)'], pop),
                ('no-overlap misses', cov['share'], grp['no-overlap misses'], pop),
                ('missed by both', cov['share'], grp['missed by both'], pop),
                ('misses, all lengths', cov['share'], grp['oracle-only (misses)'], reach_ok),
                ('sensitivity: rho_count per epoch', cov['count'], grp['oracle-only (misses)'], pop),
                ('sensitivity: per-run count', cov['run'], grp['oracle-only (misses)'], pop)]
    if 'augmented' in cov:
        variants.append(('augmented P1 support', cov['augmented'], grp['oracle-only (misses)'], pop))
    for name, cv, m, ex in variants:
        per = a2(cv, m, extra=ex)
        reading, fa, fab = a2_reading(per)
        rows.append([name] + [f'{_pct(x)} / {_pct(y)} ({"a" if fa[i] else "-"}{"b" if fab[i] else "-"})'
                              for i, (x, y) in enumerate(per)] + [reading])
        num['A2'][name] = dict(per_ring=per, a=fa, ab=fab, reading=reading)
    out.table(['variant', '40-50 m', '50-60 m', '60-75 m', 'reading'], rows)
    out('Each cell: uncovered share of the group / of found-by-both; (a) and (b) as in §3.2.'); out()

    # cells holding the misses
    out('## The cells holding the misses (rho_share coverage), 50-60 and 60-75 m: top 8 by misses'); out()
    for k in (5, 6):
        m = grp['oracle-only (misses)'] & pop & (rv == k)
        f = grp['found by both'] & pop & (rv == k)
        cm, cf = np.bincount(cv_cells[m], minlength=len(s_src)), np.bincount(cv_cells[f], minlength=len(s_src))
        top = np.argsort(-cm)[:8]
        rows = []
        for c in top:
            if cm[c] == 0:
                continue
            vb, lb = (c // len(NLAY_EDGES)) % len(NVOX_EDGES), c % len(NLAY_EDGES)
            a_, b_ = s_src[c] / max(S_src, 1e-12), s_tgt[c] / max(S_tgt, 1e-12)
            rows.append([NVOX_NAMES[vb], NLAY_NAMES[lb], f'{cm[c]:,}', f'{cf[c]:,}', f'{s_src[c]:,.0f}', f'{s_tgt[c]:,.0f}',
                         _f(a_ / b_ if b_ > 0 else np.inf), 'yes' if cov['share'][c] else 'NO'])
        out(f'{RING_NAMES[k]} m:'); out()
        out.table(['n_vox', 'n_layers', 'misses', 'found by both', 's_src / epoch', 's_tgt / epoch', 'rho_share', 'covered'], rows)

    # ---------------------------------------------------------------- A3
    out('## A3: long vehicles (>= 6 m)'); out()
    rows, num['A3'] = [], {}
    lm = grp['oracle-only (misses)'] & ~short
    lf = grp['found by both'] & ~short
    for k in range(len(RING_NAMES)):
        m = lm & (rv == k) & reach_ok
        f = lf & (rv == k) & reach_ok
        if not (lm & (rv == k)).sum():
            continue
        q = CV[m]
        uc, uco = uncovered_share(cv_cells[m], cov['share']), uncovered_share(cv_cells[m], cov_co)
        rows.append([RING_NAMES[k], f'{(lm & (rv == k)).sum():,}', f'{m.sum():,}',
                     f'{np.median(q[:, 2]):.0f} / {np.median(q[:, 3]):.0f}' if len(q) else '-',
                     _pct(uc), _pct(uco), f'{f.sum():,}', _pct(uncovered_share(cv_cells[f], cov['share']))])
        num['A3'][RING_NAMES[k]] = dict(misses=int((lm & (rv == k)).sum()), uncovered_car=uc, uncovered_car_other=uco)
    out.table(['ring', 'long misses', 'with n_vox >= 1', 'median n_vox / n_layers', 'uncovered under `car`',
               'under `car` + other vehicles', 'long found by both', 'their uncovered share (`car`)'], rows)
    lo1 = C1[(C1[:, 5] >= LONG_M) & (C1[:, 0] >= 40) & (C1[:, 0] < 75)]
    lo2 = O1[(O1[:, 5] >= LONG_M) & (O1[:, 0] >= 40) & (O1[:, 0] < 75)]
    out(f'P1 instances >= 6 m at 40-75 m (frames processed): `car` {len(lo1):,}, other vehicles {len(lo2):,}; '
        f'P2 {int(((C2[:, 5] >= LONG_M) & (C2[:, 0] >= 40) & (C2[:, 0] < 75)).sum()):,}.'); out()

    # ---------------------------------------------------------------- Part B
    if not shards:
        Path(d / 'tables.md').write_text('\n'.join(out.lines)); json.dump(num, open(d / 'numbers.json', 'w'), indent=1, default=float)
        return
    so = float(shards[0]['stride'])
    cat = lambda k: np.concatenate([np.asarray(s[k]).reshape(len(np.asarray(s[k])), -1) for s in shards if k in s and np.asarray(s[k]).size])
    ops = {'base (15 / 10)': _rows(cat('car')), 'B1a (30 / 20)': _rows(cat('car_B1a')), 'B2a (30 / 20 + z-buffer)': _rows(cat('car_B2a')),
           'B1b (60 / 60)': _rows(cat('car_B1b')), 'B2b (60 / 60 + z-buffer)': _rows(cat('car_B2b'))}
    B3 = np.asarray(cat('b3'), np.float64) if any('b3' in s for s in shards) else np.zeros((0, 16))
    B3rows = _rows(cat('car_B3')) if any('car_B3' in s for s in shards) else np.zeros((0, len(BOX_COLS)))
    B3in = _rows(cat('car_B3in')) if any('car_B3in' in s for s in shards) else np.zeros((0, len(BOX_COLS)))
    frames_ops = {n: np.asarray(cat(k)).reshape(-1, 6) for n, k in (('base (15 / 10)', 'frame'), ('B1a (30 / 20)', 'frame_B1a'),
                  ('B2a (30 / 20 + z-buffer)', 'frame_B2a'), ('B1b (60 / 60)', 'frame_B1b'), ('B2b (60 / 60 + z-buffer)', 'frame_B2b'))}
    out(f'# Part B (stride {int(so)}, {len(frames_ops["base (15 / 10)"])} frames)'); out()
    out.table(['source', 'points per frame (median)', 'voxels per frame (median)', 'frames over 150k voxels', 'load s (median)'],
              [[k, f'{np.median(v[:, 2]):,.0f}', f'{np.median(v[:, 3]):,.0f}', _pct(np.mean(v[:, 3] > VOXEL_CAP)),
                _f(np.median(v[:, 5]))] for k, v in frames_ops.items()])
    num['ops_cost'] = {k: dict(points=float(np.median(v[:, 2])), voxels=float(np.median(v[:, 3])),
                               over_cap=float(np.mean(v[:, 3] > VOXEL_CAP))) for k, v in frames_ops.items()}

    out('## B1 / B2 analysis: A2 shares with the operator\'s `car` support (rho_share), misses / found by both'); out()
    rows, num['B_A2'] = [], {}
    for name, R in ops.items():
        so_ = support(R, so); cv_ = covered(so_, s_tgt, 'share')
        per = a2(cv_, grp['oracle-only (misses)'])
        reading, fa, fab = a2_reading(per)
        covers = [bool(np.isfinite(x) and x <= 0.5) for x, _ in per]
        rows.append([name] + [f'{_pct(x)} / {_pct(y)}' for x, y in per] +
                    [f'{sum(covers)} of 3 -> {"CAN produce them" if sum(covers) >= 2 else "cannot"}', reading])
        num['B_A2'][name] = dict(per_ring=per, covers=covers, reading=reading)
    out.table(['source', '40-50 m', '50-60 m', '60-75 m', 'covers the misses (<= 50% uncovered)', 'A2 reading'], rows)

    out('## Reach: share of the misses (< 6 m, n_vox >= 1) in cells holding >= 1% of the source\'s instances in the ring'); out()
    miss_rows = CV[grp['oracle-only (misses)'] & pop]
    found_rows = CV[grp['found by both'] & pop]
    srcs = [('P1 `car` (stride 4)', C1), ('P2 Waymo train', C2)] + list(ops.items()) + [('B3 (relocated + rendered)', B3rows), ('B3, in-box variant', B3in)]
    rows, num['reach'] = [], {}
    for name, R in srcs:
        R = R[R[:, 2] >= 1]
        rr = [reach(miss_rows, R, lo, hi) for lo, hi in ((40, 50), (50, 60), (60, 75))]
        rf = [reach(found_rows, R, lo, hi) for lo, hi in ((50, 60), (60, 75))]
        verdict = ''
        if name.startswith('B3 ('):
            verdict = 'CAN produce them' if all(np.isfinite(x) and x >= 0.5 for x in rr[1:]) else 'cannot'
        rows.append([name] + [_pct(x) for x in rr] + [_pct(x) for x in rf] + [verdict])
        num['reach'][name] = dict(misses=rr, found=rf)
    out.table(['source', 'misses 40-50', 'misses 50-60', 'misses 60-75', 'found 50-60', 'found 60-75', 'B3 reading (>= 50% in both)'], rows)

    out('## Distributions at 50-60 and 60-75 m: p25 / p50 / p75 of n_vox, n_layers, ppv'); out()
    rows = []
    for name, R in [('misses', miss_rows), ('found by both', found_rows), ('P2 Waymo train', C2), ('P1 `car`', C1)] + \
            list(ops.items()) + [('B3', B3rows), ('B3, in-box variant', B3in)]:
        for lo, hi in ((50, 60), (60, 75)):
            q = R[(R[:, 0] >= lo) & (R[:, 0] < hi) & (R[:, 2] >= 1)]
            rows.append([name, f'{lo}-{hi}', f'{len(q):,}', _q(q[:, 2]), _q(q[:, 3]), _q(q[:, 4], '{:.2f}')])
    out.table(['population', 'ring', 'n', 'n_vox', 'n_layers', 'ppv'], rows)

    # ---------------------------------------------------------------- gate
    g1 = np.asarray(cat('g1'), np.float64) if any('g1' in s for s in shards) else np.zeros((0, 21))
    out('## Label-free gate, B1 / B2: G1 median RC and G2 median PPP per cell (base-cloud sparse far cars, moving ego)'); out()
    mv = g1[:, 4] > 0
    rows, passes, g2ok = [], {k: 0 for k in G1_OPS}, {k: True for k in G1_OPS}
    for p in range(2):
        for lo, hi in ((20, 40), (40, 75)):
            for s in range(3):
                q = g1[mv & (g1[:, 0] == p) & (g1[:, 2] >= lo) & (g1[:, 2] < hi) & (g1[:, 3] == s)]
                cells = []
                for k, op in enumerate(G1_OPS):
                    rc, ppp = q[:, 6 + 3 * k], q[:, 7 + 3 * k]
                    mrc = np.nanmedian(rc) if np.isfinite(rc).any() else np.nan
                    mpp = np.nanmedian(ppp) if np.isfinite(ppp).any() else np.nan
                    passes[op] += bool(np.isfinite(mrc) and mrc >= 0.8)
                    g2ok[op] &= bool(np.isfinite(mpp) and 1.0 <= mpp <= 1.1)
                    cells.append(f'{_f(mrc)} / {_f(mpp)}')
                rows.append([PLATS[p], f'{lo}-{hi}', SECTORS[s], f'{len(q)}' + (' (< 15)' if len(q) < 15 else '')] + cells)
    out.table(['platform', 'ring', 'sector', 'cars'] + [f'{k}: RC / PPP' for k in G1_OPS], rows)
    out('G1 cells with median RC >= 0.8 (of 12): ' + ', '.join(f'{k} {passes[k]}' for k in G1_OPS) +
        '; G2 (PPP 1.0-1.1 in every cell): ' + ', '.join(f'{k} {"PASS" if g2ok[k] else "FAIL"}' for k in G1_OPS) +
        '. G3: 0 by construction for B1 / B2 (every point is a source return in place).'); out()
    num['gate_B12'] = dict(G1_cells=passes, G2=g2ok)

    out('## Label-free gate, B3 (relocated donors, moving ego; 40-75 m ring only)'); out()
    if len(B3):
        mv3 = B3[:, 5] > 0
        rows, p3, g2b3 = [], 0, True
        for p in range(2):
            for s in range(3):
                q = B3[mv3 & (B3[:, 0] == p) & (B3[:, 4] == s)]
                mrc = np.nanmedian(q[:, 8]) if len(q) and np.isfinite(q[:, 8]).any() else np.nan
                mpp = np.nanmedian(q[:, 9]) if len(q) and np.isfinite(q[:, 9]).any() else np.nan
                p3 += bool(np.isfinite(mrc) and mrc >= 0.8); g2b3 &= bool(np.isfinite(mpp) and 1.0 <= mpp <= 1.1)
                small = q[(B3rows[mv3 & (B3[:, 0] == p) & (B3[:, 4] == s), 1] >= 10) &
                          (B3rows[mv3 & (B3[:, 0] == p) & (B3[:, 4] == s), 1] < 50)] if len(q) else q
                rows.append([PLATS[p], SECTORS[s], f'{len(q)} ({len(np.unique(q[:, :3], axis=0)) if len(q) else 0} donors)',
                             _f(mrc), _f(mpp), _f(np.nanmedian(small[:, 8]) if len(small) else np.nan) + f' ({len(small)})',
                             _f(np.nanmedian(q[:, 13]) if len(q) and np.isfinite(q[:, 13]).any() else np.nan)])
        out.table(['platform', 'sector', 'renders', 'median RC', 'median PPP', 'median RC, 10-49 rendered points',
                   'median RC, in-box variant'], rows)
        rows = []
        for p in range(2):
            rows.append([PLATS[p]] + [_f(np.nanmedian(B3[(B3[:, 0] == p) & mv3 & (B3[:, 6] == rn), 8]))
                                      if ((B3[:, 0] == p) & mv3 & (B3[:, 6] == rn)).any() else '-' for rn in R_NEW])
        out.table(['platform'] + [f"r' = {rn:.0f} m" for rn in R_NEW], rows)
        ph = []
        for rn in R_NEW:
            q = B3[B3[:, 6] == rn]
            ph.append(q[:, 11].sum() / max(q[:, 12].sum(), 1))
        out(f'G1 (B3): {p3} of 6 cells with median RC >= 0.8 -> {"PASS" if p3 >= 5 else "FAIL"}; G2: '
            f'{"PASS" if g2b3 else "FAIL"}; G3 phantom share per r\': ' +
            ', '.join(f'{rn:.0f} m {100 * x:.3f}%' for rn, x in zip(R_NEW, ph)) +
            f' -> {"PASS" if max(ph) < 0.01 else "FAIL"}'); out()
        out(f'B3 donors: {len(np.unique(B3[:, :3], axis=0)):,} ({int((B3[:, 0] == 0).sum() / 4):,} Boston, '
            f'{int((B3[:, 0] == 1).sum() / 4):,} Singapore renders / 4); donor points p25 / p50 / p75 {_q(B3[::4, 3])}; '
            f'rendered points per car p25 / p50 / p75 at r\' = 40 / 50 / 60 / 70: ' +
            '; '.join(_q(B3[B3[:, 6] == rn, 7]) for rn in R_NEW)); out()
        num['gate_B3'] = dict(G1_cells=p3, G2=g2b3, G3=ph)
    Path(d / 'tables.md').write_text('\n'.join(out.lines))
    json.dump(num, open(d / 'numbers.json', 'w'), indent=1, default=float)
    print(f'wrote {d / "tables.md"} and {d / "numbers.json"}')
