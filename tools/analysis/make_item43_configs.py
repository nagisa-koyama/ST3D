"""Write the configs of experiments_md 20261003_05 §5 item 43 (dense -> sparse training-time swaps on nuScenes by accumulation;
design fixed 2026-11-09 01:18 JST, before any number). ANALYSIS: nuScenes TRAIN labels train the rows, nothing feeds a method.
Every file is FULLY RESOLVED (no _BASE_CONFIG_): a child DATA_CONFIG_TAR would otherwise inherit the KITTI target keys of the
accumulation teacher's config. Written into cfgs/da-ieee-access-analysis/item43/; do not edit them by hand.

43a, own-input twins (eval-only): the nuScenes oracle config (centerpoint-sourceonly-nuscenes.yaml; its MODEL / CLASS_NAMES /
  ONTOLOGY equal those of every accumulated checkpoint scored here, analysis/item43_config_check.py) with ONE block changed,
  DATA_CONFIG_TAR: nuScenes val accumulated per platform to the depth the checkpoint trained at (MAX_SWEEPS_BY_VEHICLE, one
  dataset, one AP over the 6,019 frames), on the 200-sweep infos, with per-object motion compensation at evaluation
  (COMPENSATE_AT_EVAL, as trained; reads val GT boxes, analysis) and the point shuffle on, so the voxel cap and the 10 points
  per voxel act at random as in training. The unperturbed single-sweep val is the oracle config itself.
43b, L'_K (training): the S2 accumulation teacher 27468's config (centerpoint-accum-legaldepth-nuscenes2kitti.yaml: Boston
  n008 15 / Singapore n015 10 sweeps, compensation on) with, on each platform block, the count-matching arm: per 1.5-m range
  bin random thinning (sample_points_hist_based, measured on the fly over 1,000 frames) to the SAME platform's single-sweep
  in-grid count (HIST_DIST_REFERENCE_CFG = the oracle recipe restricted to that VEHICLE, HIST_DIST_IN_GRID), then the label
  recount (drop_empty_gt_boxes, as the item 29 count-matched rows). DATA_CONFIG_TAR = the oracle's (in-run eval on single-sweep
  nuScenes val). Plus two gate-only views (one platform block each) for analysis/arm_point_count_bins.py, never trained.

    python analysis/make_item43_configs.py [--check]     # from ST3D/tools; --check: exit 1 if a file on disk differs

Tests: tests/test_item43_configs.py.
"""
import argparse
import contextlib
import copy
import io
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
ORACLE = 'cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml'
TEACHER = 'cfgs/da-ieee-access/centerpoint-accum-legaldepth-nuscenes2kitti.yaml'      # 27468 / 27804
WAYMO_DEPTH = 'cfgs/da-ieee-access/centerpoint-accum-depthwaymo-nuscenes2waymo.yaml'  # 27185 (27183 resumed)
PROC_DEPTH = 'cfgs/da-ieee-access/centerpoint-accum-procdepth-nuscenes2waymo.yaml'    # 29354
D = 'cfgs/da-ieee-access-analysis/item43/'
REF = {'n008': D + 'centerpoint-i43-ref-single-n008.yaml', 'n015': D + 'centerpoint-i43-ref-single-n015.yaml'}
LPRIME = D + 'centerpoint-i43b-lprime-k.yaml'
GATE = {'n008': D + 'centerpoint-i43b-lprime-k-gate-n008.yaml', 'n015': D + 'centerpoint-i43b-lprime-k-gate-n015.yaml'}
BLOCK = {'n008': 'NUSCENES_N008', 'n015': 'NUSCENES_N015'}
# own-input twin per training config: (file, source config whose per-platform MAX_SWEEPS it copies)
OWN = {TEACHER: D + 'centerpoint-i43a-ownval-n15-10.yaml', WAYMO_DEPTH: D + 'centerpoint-i43a-ownval-n30-20.yaml',
       PROC_DEPTH: D + 'centerpoint-i43a-ownval-n22-17.yaml'}
ARM = {'HIST_DIST_ON_THE_FLY': True, 'HIST_DIST_FRAMES': 1000, 'HIST_DIST_BINS': 50, 'HIST_DIST_MAX_DIST': 75.0,
       'HIST_DIST_IN_GRID': True}
# MIN_HIST_BIN_FRACTION 0.0 (amendment A7 of experiments_md 20261109_01, 2026-11-09 07:33 JST): the default guard (0.01) leaves a bin
# uncorrected when the REFERENCE holds under 1% of its mean bin. The single-sweep reference holds ~1 point per frame at 0-1.5 m
# (the keyframe's 1.5-m ego cut) where the accumulated source holds 2,204 / 1,264 (sweeps are cut at 1.0 m), so the guard kept
# them all and the first gate (job 32416) read n008 DIFFER (0-5 m x1.28). Count matching needs the rate in every bin;
# a bin with no reference point at all still keeps rate 1 (per_bin_sample_rate's tgt > 0 test).
THIN = [{'NAME': 'sample_points_hist_based', 'MIN_HIST_BIN_FRACTION': 0.0},
        {'NAME': 'drop_empty_gt_boxes', 'MIN_POINTS': 1}]


def load(path):
    """The resolved config at `path` (relative to ST3D/tools, where _BASE_CONFIG_ paths resolve), from any cwd."""
    import os
    from easydict import EasyDict
    sys.path.insert(0, str(TOOLS))
    import _init_path  # noqa: F401
    from pcdet.config import cfg_from_yaml_file
    cwd = os.getcwd()
    os.chdir(TOOLS)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return cfg_from_yaml_file(path, EasyDict())
    finally:
        os.chdir(cwd)


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items() if k not in ('_BASE_CONFIG_', 'ROOT_DIR', 'LOCAL_RANK')}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def depths(train_cfg):
    """{vehicle: MAX_SWEEPS} of a two-block accumulation config."""
    c = plain(load(train_cfg))
    return {v: int(c['DATA_CONFIGS'][b]['MAX_SWEEPS']) for v, b in BLOCK.items()}


def with_thinning(processor):
    """The block's processor with the arm right after shuffle_points (the C1 / item 29 order: the shuffle draws first, so at
    rate 1 the arm leaves points, order and voxels exactly as without it)."""
    names = [p['NAME'] for p in processor]
    assert 'sample_points_hist_based' not in names and names.index('shuffle_points') == 1, names
    return processor[:2] + copy.deepcopy(THIN) + processor[2:]


def ref_config(vehicle):
    c = plain(load(ORACLE))
    assert 'VEHICLE' not in c['DATA_CONFIG']
    c['DATA_CONFIG']['VEHICLE'] = vehicle
    return c


def lprime():
    c = plain(load(TEACHER))
    for v, b in BLOCK.items():
        blk = c['DATA_CONFIGS'][b]
        assert blk['VEHICLE'] == v and not any(k.startswith('HIST_DIST') for k in blk), (b, sorted(blk))
        blk.update(copy.deepcopy(ARM))
        blk['HIST_DIST_REFERENCE_CFG'] = REF[v]
        blk['DATA_PROCESSOR'] = with_thinning(blk['DATA_PROCESSOR'])
    c['DATA_CONFIG_TAR'] = plain(load(ORACLE))['DATA_CONFIG_TAR']
    return c


def gate(vehicle):
    c = lprime()
    c['DATA_CONFIGS'] = {BLOCK[vehicle]: c['DATA_CONFIGS'][BLOCK[vehicle]]}
    return c


def own_input(train_cfg):
    c = plain(load(ORACLE))
    t = c['DATA_CONFIG_TAR']
    dep = depths(train_cfg)
    src = plain(load(train_cfg))['DATA_CONFIGS'][BLOCK['n008']]
    t['MAX_SWEEPS'] = max(dep.values())
    t['MAX_SWEEPS_BY_VEHICLE'] = dep
    t['INFO_PATH'] = copy.deepcopy(src['INFO_PATH'])
    t['GT_BOXES_MOTION_COMPENSATION'] = True
    t['COMPENSATE_AT_EVAL'] = True
    shuffle = [p for p in t['DATA_PROCESSOR'] if p['NAME'] == 'shuffle_points']
    assert len(shuffle) == 1
    shuffle[0]['SHUFFLE_ENABLED'] = {'train': True, 'test': True}
    return c


HEAD = {
    'ref': ('# REFERENCE-ONLY config of item 43b (experiments_md 20261003_05 §5 item 43; generated by analysis/make_item43_configs.py):\n'
            '# the nuScenes oracle recipe (%s, single sweep) restricted to VEHICLE %s. Its SOURCE, in training mode, is what\n'
            '# L\'_K\'s %s block is thinned to per 1.5-m bin (HIST_DIST_REFERENCE_CFG). Never trained. ANALYSIS.\n'),
    'lprime': ('# ANALYSIS training row L\'_K of item 43b (experiments_md 20261003_05 §5 item 43, design fixed 2026-11-09 01:18 JST;\n'
               '# generated by analysis/make_item43_configs.py, do not edit). The S2 accumulation teacher 27468\'s config (%s:\n'
               '# Boston n008 15 / Singapore n015 10 sweeps, per-object motion compensation, MeanVFE, 20 epochs) with ONE arm on\n'
               '# each platform block: random thinning per 1.5-m range bin to that platform\'s SINGLE-SWEEP in-grid point count\n'
               '# (HIST_DIST_REFERENCE_CFG, HIST_DIST_IN_GRID, 1,000 calibration frames), then the label recount. It keeps the\n'
               '# accumulated structure (ego-motion-shifted lattice, smear) at the single-sweep count. In-run evaluation on\n'
               '# single-sweep nuScenes val (DATA_CONFIG_TAR = the oracle\'s). Contrasts: O_N - L\'_K (structure at equal count),\n'
               '# L\'_K - R_acc (27468 / 27804; count part of the density term). nuScenes TRAIN labels train it: ANALYSIS.\n'),
    'gate': ('# GATE-ONLY view of L\'_K (%s), its %s block alone, for analysis/arm_point_count_bins.py measure and\n'
             '# analysis/item43b_checks.py (generated by analysis/make_item43_configs.py). Never trained. ANALYSIS.\n'),
    'own': ('# EVAL-ONLY own-input twin of item 43a (experiments_md 20261003_05 §5 item 43; generated by analysis/make_item43_configs.py):\n'
            '# the nuScenes oracle config (%s) with DATA_CONFIG_TAR = nuScenes val accumulated per platform as %s\n'
            '# trains (n008 %d / n015 %d sweeps, MAX_SWEEPS_BY_VEHICLE, one AP over all 6,019 frames), 200-sweep infos,\n'
            '# compensation at evaluation (COMPENSATE_AT_EVAL; reads val boxes), shuffle on at test so the voxel cap and the\n'
            '# 10 points per voxel act at random as in training. Scores a model on its OWN input. ANALYSIS.\n'),
}


def outputs():
    out = {}
    for v in BLOCK:
        out[REF[v]] = (HEAD['ref'] % (ORACLE, v, BLOCK[v]), ref_config(v))
        out[GATE[v]] = (HEAD['gate'] % (LPRIME, BLOCK[v]), gate(v))
    out[LPRIME] = (HEAD['lprime'] % TEACHER, lprime())
    for train_cfg, path in OWN.items():
        dep = depths(train_cfg)
        out[path] = (HEAD['own'] % (ORACLE, train_cfg, dep['n008'], dep['n015']), own_input(train_cfg))
    return out


def render(header, body):
    import textwrap
    import yaml
    lines = []
    for line in header.rstrip('\n').split('\n'):       # a comment line over 118 columns is re-wrapped
        lines += textwrap.wrap(line[2:], 116, initial_indent='# ', subsequent_indent='# ', break_on_hyphens=False) \
            if len(line) > 118 else [line]
    return '\n'.join(lines) + '\n' + yaml.safe_dump(body, sort_keys=False, default_flow_style=None, width=118)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    bad = 0
    for path, (header, body) in outputs().items():
        text, f = render(header, body), TOOLS / path
        if args.check:
            same = f.exists() and f.read_text() == text
            bad += not same
            print('%s %s' % ('up to date:' if same else 'DIFFERS:', path))
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        print('wrote', path)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
