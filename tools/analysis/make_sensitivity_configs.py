"""Eval-only twins for experiments_md 20261010_01: test-time TARGET accumulation (A: nuScenes target, B: Waymo target)
and the target-oracle degradation table (density doses, nuScenes scan-pattern render). ANALYSIS unless 20261010_01
says a pre-declared rule makes an arm a method candidate.

Each twin is `_BASE_CONFIG_: <as-scored config>` plus changes in DATA_CONFIG_TAR only:
  - accumulation: MAX_SWEEPS N (nuScenes N > 10 also switches the test INFO_PATH to the 200-sweep val infos, which the
    loader needs for keyframe chaining), and the test voxel cap raised so accumulated points are never dropped (the
    cap never binds at N = 1, so it changes nothing for the as-scored config);
  - density dose: a `sample_points` step with RATIO {'test': r} inserted before voxelisation (uniform random keep);
  - pattern: RING_PATTERN with APPLY_AT_EVAL (27807's nuScenes HDL-32E rule applied to the evaluation cloud);
  - thin: the scan pattern cut at evaluation - lines or columns halved, or their count-matched random controls
    (EVAL_TOP_THIN on Waymo's TOP block, EVAL_RING_THIN on nuScenes' stored ring index).
Box-based motion compensation never runs at evaluation (should_compensate requires training), so accumulation reads no
label. `--check` resolves every twin and passes only if nothing outside DATA_CONFIG_TAR differs from its base.

    python analysis/make_sensitivity_configs.py [--check] [--voxel_cap N]
"""
import sys
from pathlib import Path

import yaml
from easydict import EasyDict

sys.path.insert(0, '.')
import _init_path  # noqa
from pcdet.config import cfg_from_yaml_file

OUT = Path('cfgs/da-ieee-access-analysis/sensitivity')
CAP = int(sys.argv[sys.argv.index('--voxel_cap') + 1]) if '--voxel_cap' in sys.argv else 600000
RING = {'TOP_CALIB': '/home/koyama/data/waymo_top_calib/top_calib.pkl', 'SPACING_DEG': 1.33, 'AZ_RES_DEG': 0.332,
        'TOP_ONLY': True, 'RECOUNT_GT_POINTS': True, 'APPLY_AT_EVAL': True}
# twin name suffix -> (kind, value)
TWINS = {
    # A: nuScenes target accumulation (Waymo-trained models + the nuScenes oracle)
    'centerpoint-sourceonly-waymo': [('acc', 10), ('acc', 20), ('acc', 30)],
    'centerpoint-gblobs-sourceonly-waymo': [('acc', 20)],
    'centerpoint-ringpattern-waymo2nuscenes': [('acc', 20)],
    'centerpoint-sourceonly-nuscenes': [('acc', 10), ('acc', 20), ('acc', 30), ('ratio', 0.75), ('ratio', 0.5),
                                        ('ratio', 0.25), ('thin', 'rows'), ('thin', 'random'), ('thin', 'cols'),
                                        ('thin', 'random_cols')],
    # B: Waymo target accumulation (nuScenes-trained models + the Waymo oracle)
    'centerpoint-accum-legaldepth-nuscenes2waymo': [('acc', 3), ('acc', 7)],
    'centerpoint-gblobs-sourceonly-nuscenes2waymo': [('acc', 7)],
    'centerpoint-sourceonly-waymo2waymo': [('acc', 3), ('acc', 7), ('ratio', 0.75), ('ratio', 0.5), ('ratio', 0.25),
                                           ('pattern', 'hdl32e'), ('thin', 'cols'), ('thin', 'random_cols')],
}


def plain(o):
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    return o


def resolved(path):
    cfg = EasyDict(); cfg_from_yaml_file(str(path), cfg); return plain(cfg)


def with_cap(processors, cap):
    out = []
    for p in processors:
        p = dict(p)
        if p['NAME'] == 'transform_points_to_voxels':
            p['MAX_NUMBER_OF_VOXELS'] = {**p['MAX_NUMBER_OF_VOXELS'], 'test': cap}
        out.append(p)
    return out


def with_ratio(processors, r):
    out = []
    for p in processors:
        if p['NAME'] == 'transform_points_to_voxels':
            out.append({'NAME': 'sample_points', 'NUM_POINTS': {'train': -1, 'test': -1}, 'RATIO': {'test': r}})
        out.append(dict(p))
    return out


def tar_change(rb, kind, val):
    tb = rb['DATA_CONFIG_TAR']
    if kind == 'acc':
        tar = {'MAX_SWEEPS': val, 'DATA_PROCESSOR': with_cap(tb['DATA_PROCESSOR'], CAP)}
        if tb['DATASET'] == 'NuScenesDataset' and val > 10:
            tar['INFO_PATH'] = {**tb['INFO_PATH'], 'test': ['nuscenes_infos_200sweeps_val.pkl']}
        return tar, f'acc{val}'
    if kind == 'ratio':
        return {'DATA_PROCESSOR': with_ratio(tb['DATA_PROCESSOR'], val)}, 'ratio%03d' % int(round(val * 100))
    if kind == 'thin':   # scan-pattern cut at evaluation: lines / columns halved, or a count-matched random control
        if tb['DATASET'] == 'WaymoDataset':
            return {'EVAL_TOP_THIN': {'TOP_CALIB': RING['TOP_CALIB'], 'MODE': val, 'STRIDE': 2}}, 'top' + val.replace('_', '')
        return {'EVAL_RING_THIN': {'MODE': val, 'STRIDE': 2}}, 'ring' + val.replace('_', '')
    if kind == 'pattern':
        assert tb['DATASET'] == 'WaymoDataset'
        return {'RING_PATTERN': RING}, 'pattern' + val
    raise ValueError(kind)


def main(check=False):
    OUT.mkdir(parents=True, exist_ok=True)
    for base, arms in TWINS.items():
        base_path = Path('cfgs/da-ieee-access') / f'{base}.yaml'
        rb = resolved(base_path)
        for kind, val in arms:
            tar, tag = tar_change(rb, kind, val)
            child = OUT / f'{base}-{tag}.yaml'
            if not check:
                header = (f'# EVAL-ONLY twin of {base_path} for experiments_md 20261010_01 ({tag}): target-side change only.\n'
                          f'# Generated by analysis/make_sensitivity_configs.py (test voxel cap {CAP} on accumulation twins). ANALYSIS.\n')
                child.write_text(header + yaml.safe_dump({'_BASE_CONFIG_': str(base_path), 'DATA_CONFIG_TAR': tar},
                                                         sort_keys=False, default_flow_style=None))
            rc = resolved(child)
            diffs = [k for k in set(rb) | set(rc) if k not in ('DATA_CONFIG_TAR', '_BASE_CONFIG_') and rb.get(k) != rc.get(k)]
            tb, tc = rb['DATA_CONFIG_TAR'], rc['DATA_CONFIG_TAR']
            changed = sorted(k for k in set(tb) | set(tc) if tb.get(k) != tc.get(k) and k != '_BASE_CONFIG_')
            ok = not diffs and set(changed) == set(tar)
            print(f'{child.name}: {"OK" if ok else "DIFF " + str(diffs) + str(changed)}  changed={changed}')


if __name__ == '__main__':
    main(check='--check' in sys.argv)
