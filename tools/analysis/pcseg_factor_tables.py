"""Derive the S3a operator files of P1 from the label-free TRAIN statistics of pcseg_factor_stats.py
(experiments_md episodes/20261116_01 §8.9; pcseg/factor_ops.py). Labels are never read: the input holds point counts.

Writes into --out_dir:
  a1_density_rate_pandargt_to_pandar64.json   DENSITY (arm a1): keep rate per range bin for PandarGT (device 1),
        min(1, rho_Pandar64 / rho_PandarGT) with rho = in-cone points per frame per bin, by detection's
        DataProcessor.per_bin_sample_rate (its guard unchanged: a bin below 1% of either side's mean bin keeps rate 1);
  a2_intensity_pandargt_to_pandar64_rings.npz INTENSITY (arm a2): PandarGT intensities -> Pandar64's response;
  a4_intensity_pandar64_to_pandargt_rings.npz INTENSITY (arm a4): Pandar64 intensities -> PandarGT's response;
        both per range ring from the IN-CONE dequantised histograms, by detection's table construction
        (intensity_testtime_tables.py: quantiles at 201 levels, a ring with < 2000 points on either side takes the
        all-range map), in the npz format detection's map_intensity_to_reference reads, plus from_device / to_device;
  p1_factor_summary.json  the magnitudes, the a6 share, and the check of detection's spin2flash / flash2spin tables.

Every file carries provenance: this script's sha256, the statistics file's sha256 and its own provenance.
    python3 analysis/pcseg_factor_tables.py --stats /home/koyama/data/pcseg/factor_stats/p1_train_stats.npz \
        --out_dir /home/koyama/data/pcseg/factor_stats
"""
import argparse
import json
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.dirname(TOOLS))
from analysis.pcseg_factor_stats import provenance, sha256, SPEC_RANGE  # noqa: E402

MIN_HIST_BIN_FRACTION = 0.01          # detection's default guard (DataProcessor.per_bin_sample_rate)
MIN_POINTS = 2000                     # detection's ring fallback (intensity_testtime_tables.MIN_POINTS)
A6_THRESHOLD = 0.01                   # a6 is built only if >= 1% of PandarGT's in-cone TRAIN points lie beyond 200 m
DETECTION_TABLES = {'spin2flash': '/home/koyama/data/intensity_tables/spin2flash_rings.npz',
                    'flash2spin': '/home/koyama/data/intensity_tables/flash2spin_rings.npz'}
NAMES = {0: 'Pandar64', 1: 'PandarGT'}


def _intervals(edges, bins):
    """Contiguous runs of bin indices as [start_m, end_m] intervals."""
    out = []
    for b in bins:
        if out and out[-1][1] == float(edges[b]):
            out[-1][1] = float(edges[b + 1])
        else:
            out.append([float(edges[b]), float(edges[b + 1])])
    return out


def _median(h):
    from pcdet.datasets.intensity_calibration import quantiles
    return float(quantiles(h, np.array([0.5]))[0]) if np.sum(h) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', required=True)
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    from pcseg.factor_ops import density_rates_from_histograms, intensity_tables_from_histograms
    s = np.load(a.stats)
    stats_prov = json.loads(str(s['provenance']))
    frames = int(s['frames'])
    prov = provenance({'derived_by': os.path.abspath(__file__),
                       'derived_by_sha256': sha256(os.path.abspath(__file__)),
                       'stats_file': os.path.abspath(a.stats), 'stats_sha256': sha256(a.stats),
                       'stats_provenance': stats_prov['provenance'], 'frames_per_device': frames,
                       'sequences': stats_prov['sequences'], 'split': 'train', 'labels_read': 'none'})
    os.makedirs(a.out_dir, exist_ok=True)
    summary = {'provenance': prov}

    # --- a1: in-cone density per range bin, PandarGT thinned to Pandar64 ---
    max_dist, num_bins = float(s['density_max_dist']), int(s['density_num_bins'])
    rho = {d: s['hist_density_d%d' % d] / frames for d in (0, 1)}
    rate = density_rates_from_histograms(rho[1], rho[0], MIN_HIST_BIN_FRACTION)
    guarded = ~((rho[1] > MIN_HIST_BIN_FRACTION * rho[1].mean()) & (rho[0] > MIN_HIST_BIN_FRACTION * rho[0].mean()))
    edges = np.linspace(0, max_dist, num_bins + 1)
    kept = float((rate * rho[1]).sum() / rho[1].sum())
    a1 = {'kind': 'pcseg_density_rate', 'arm': 'a1', 'apply_to_device': 1, 'reference_device': 0,
          'max_dist': max_dist, 'num_bins': num_bins, 'bin_edges_m': edges.tolist(), 'rate': rate.tolist(),
          'rho_apply_per_frame': rho[1].tolist(), 'rho_reference_per_frame': rho[0].tolist(),
          'min_hist_bin_fraction': MIN_HIST_BIN_FRACTION, 'guarded_bins': np.flatnonzero(guarded).tolist(),
          'cone': s['cone'].tolist(),
          'note': 'rate[b] = min(1, rho_reference / rho_apply) by detection per_bin_sample_rate; rho = in-cone points '
                  'per frame per planar range bin (about the sensor xy origin) of the TRAIN split; applied by radius '
                  'to every PandarGT point, in the cone or not',
          'provenance': prov}
    p_a1 = os.path.join(a.out_dir, 'a1_density_rate_pandargt_to_pandar64.json')
    with open(p_a1, 'w') as f:
        json.dump(a1, f, indent=1)
    summary['a1'] = {
        'file': p_a1, 'sha256': sha256(p_a1),
        'in_cone_points_per_frame': {NAMES[d]: float(rho[d].sum()) for d in (0, 1)},
        'expected_kept_share_of_pandargt_in_cone': kept,
        'expected_pandargt_in_cone_per_frame_after': float((rate * rho[1]).sum()),
        'rate_by_range_m': {'%g-%g' % (lo, hi): float(np.average(rate[(edges[:-1] >= lo) & (edges[:-1] < hi)],
                                                                 weights=rho[1][(edges[:-1] >= lo) & (edges[:-1] < hi)]
                                                                 + 1e-12))
                            for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 50), (50, 75), (75, 100), (100, 150),
                                           (150, 200), (200, 300))},
        'guarded_bins_m': _intervals(edges, np.flatnonzero(guarded)),
        'pandargt_in_cone_share_in_guarded_bins': float(rho[1][guarded].sum() / rho[1].sum()),
        'pandar64_in_cone_share_in_guarded_bins': float(rho[0][guarded].sum() / rho[0].sum()),
        'bins_with_rate_1_unguarded': int(((rate >= 1) & ~guarded).sum())}

    # --- a2 / a4: per-ring intensity response, both directions, fitted in the cone ---
    ring_edges = np.asarray(s['ring_edges'], dtype=np.float64)
    step = float(s['intensity_step'])
    summary['intensity'] = {}
    for arm, frm, to in (('a2', 1, 0), ('a4', 0, 1)):
        hf, ht = s['hist_int_cone_d%d' % frm], s['hist_int_cone_d%d' % to]
        from_q, to_q, own = intensity_tables_from_histograms(hf, ht, MIN_POINTS)
        name = '%s_intensity_%s_to_%s_rings.npz' % (arm, NAMES[frm].lower(), NAMES[to].lower())
        path = os.path.join(a.out_dir, name)
        np.savez(path, edges=ring_edges, from_q=from_q, to_q=to_q, from_step=step, from_device=frm, to_device=to,
                 own=own, hist_from=hf, hist_to=ht, min_points=MIN_POINTS,
                 provenance=json.dumps(dict(prov, arm=arm, fitted_on='in-cone points (%s) of both sensors'
                                            % s['cone'].tolist())))
        summary['intensity'][arm] = {
            'file': path, 'sha256': sha256(path), 'from': NAMES[frm], 'to': NAMES[to],
            'rings': [{'ring_m': [float(ring_edges[k]), float(ring_edges[k + 1])], 'own_map': bool(own[k]),
                       'points_from': int(hf[k].sum()), 'points_to': int(ht[k].sum()),
                       'p50_from': _median(hf[k]), 'p50_to': _median(ht[k])} for k in range(len(ring_edges) - 1)]}

    # --- the cone matters? per-ring medians in the cone against all azimuths ---
    summary['cone_vs_all_azimuths_p50'] = {
        NAMES[d]: [{'ring_m': [float(ring_edges[k]), float(ring_edges[k + 1])],
                    'cone_p50': _median(s['hist_int_cone_d%d' % d][k]), 'all_p50': _median(s['hist_int_all_d%d' % d][k]),
                    'cone_points': int(s['hist_int_cone_d%d' % d][k].sum()),
                    'all_points': int(s['hist_int_all_d%d' % d][k].sum())}
                   for k in range(len(ring_edges) - 1)] for d in (0, 1)}

    # --- detection's tables: what they were fitted on ---
    det = {}
    for name, path in DETECTION_TABLES.items():
        if not os.path.exists(path):
            continue
        t = np.load(path)
        det[name] = {'file': path, 'sha256': sha256(path), 'edges': t['edges'].tolist(),
                     'hist_src_points_per_frame_300': (t['hist_src'].sum() / 300.0) if 'hist_src' in t.files else None,
                     'hist_tgt_points_per_frame_300': (t['hist_tgt'].sum() / 300.0) if 'hist_tgt' in t.files else None,
                     'hist_src_ring_p50': [_median(h) for h in t['hist_src']] if 'hist_src' in t.files else None,
                     'hist_tgt_ring_p50': [_median(h) for h in t['hist_tgt']] if 'hist_tgt' in t.files else None}
    within75 = ring_edges[1:] <= 75.0
    det['pcseg_points_per_frame_within_75m'] = {
        NAMES[d]: {'cone': float(s['hist_int_cone_d%d' % d][within75].sum() / frames),
                   'all_azimuths': float(s['hist_int_all_d%d' % d][within75].sum() / frames)} for d in (0, 1)}
    summary['detection_tables'] = det

    # --- a6: PandarGT's in-cone TRAIN points beyond Pandar64's 200 m spec range ---
    fr = s['frame_rows']
    m = fr[:, 0] == 1
    share_planar = float(fr[m, 3].sum() / fr[m, 2].sum())
    share_3d = float(fr[m, 4].sum() / fr[m, 2].sum())
    summary['a6'] = {'pandargt_in_cone_share_ge_200m_planar': share_planar,
                     'pandargt_in_cone_share_ge_200m_3d': share_3d,
                     'pandargt_in_cone_share_ge_300m_planar': float(fr[m, 5].sum() / fr[m, 2].sum()),
                     'pandar64_in_cone_share_ge_200m_planar': float(fr[~m, 3].sum() / fr[~m, 2].sum()),
                     'spec_range_m': {NAMES[d]: SPEC_RANGE[d] for d in (0, 1)},
                     'threshold': A6_THRESHOLD, 'build_arm': bool(max(share_planar, share_3d) >= A6_THRESHOLD)}
    p_sum = os.path.join(a.out_dir, 'p1_factor_summary.json')
    with open(p_sum, 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'provenance'}, indent=1))


if __name__ == '__main__':
    main()
