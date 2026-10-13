"""Eval-only twins of the KITTI IA-SSD oracles for part B of experiments_md 20261011_07 (ANALYSIS): the arms of the KITTI
target-oracle degradation table (20261011_02, `make_kitti_sensitivity_configs.ARMS`), adapted to a fixed point budget.

Each twin is `_BASE_CONFIG_: <oracle config>` plus changes in the EVALUATED block only (DATA_CONFIG_TAR when the config has
one, else DATA_CONFIG, as test.py reads them). Two arms differ from the CenterPoint generator on purpose:
  - density dose: the uniform `sample_points` RATIO step goes right BEFORE the budget `sample_points` (16,384). The shared
    helper `with_ratio` inserts it before `transform_points_to_voxels`, which an IA-SSD config does not have, so on IA-SSD
    it would be a silent no-op;
  - vertical-FOV cut: SENSOR_ORIGIN = the block's own SHIFT_COOR (27600's published recipe has none, so [0, 0, 0]; the
    family oracle 27569 has [0, 0, 1.7]).
The thin arms (EVAL_RING_THIN: lines, columns, the 0.332-deg bin, their count-matched random controls, the HDL-32E render)
are the CenterPoint generator's, unchanged.
The Waymo base (W2, 20261011_07 §21) takes the arms of the CenterPoint Waymo oracle's table (20261010_01 §5.1 / §5.2, plus
the lines-halved pair of 27953) from `make_sensitivity_configs.tar_change` unchanged (EVAL_TOP_THIN, RING_PATTERN, the
elevation cut about the TOP sensor), except the density dose, which goes before its budget step `sample_points_by_voxels`.
The PandaSet base (P2, 20261011_07 §22) likewise takes the CenterPoint PandaSet oracle's arms (20261011_03) from
`make_pandaset_sensitivity_configs.tar_change` unchanged (EVAL_RING_THIN on the recovered Pandar64 channels: lines, columns,
the 0.332-deg bin, their random controls, channel-elevation cuts, the HDL-32E render), the density dose again before the
budget; the accumulation arms are left out, as for W2.
`--check` resolves every twin and passes only if nothing outside the evaluated block differs from its oracle and the block
changed in exactly the intended keys.

    python analysis/make_iassd_sensitivity_configs.py [--check]
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, '.')
from analysis.make_kitti_sensitivity_configs import ARMS  # noqa: E402  (imports _init_path)
from analysis.make_kitti_sensitivity_configs import tar_change as kitti_change  # noqa: E402
from analysis.make_sensitivity_configs import resolved  # noqa: E402
from analysis.make_sensitivity_configs import tar_change as waymo_change  # noqa: E402
from analysis.make_pandaset_sensitivity_configs import tar_change as pandaset_change  # noqa: E402

OUT = Path('cfgs/da-ieee-access-analysis/sensitivity-iassd')
BASES = ['cfgs/da-ieee-access-pointrcnn/iassd-published-kitti.yaml',          # 27600: published recipe, FOV-cropped
         'cfgs/da-ieee-access-pointrcnn/iassd-sourceonly-kitti2kitti.yaml',   # 27569: family protocol, 360 deg
         'cfgs/da-ieee-access-pointrcnn/iassd-sourceonly-kitti2kitti-fovcrop-noros-zshift.yaml']   # 28141: row K
WAYMO_BASES = ['cfgs/da-ieee-access-pointrcnn/iassd-sourceonly-waymo2waymo-vox65k.yaml']        # 28286: row W2
WAYMO_ARMS = [('ratio', 0.75), ('ratio', 0.5), ('ratio', 0.25), ('thin', 'rows'), ('thin', 'random'),
              ('thin', 'cols'), ('thin', 'random_cols'), ('thin', 'cols:4'), ('thin', 'random_cols:4'),
              ('thin', 'azbin:0.332'), ('thin', 'random_azbin:0.332'), ('pattern', 'hdl32e'),
              ('elev', 'max:0'), ('elev', 'max:-3'), ('elev', 'min:-14'), ('elev', 'min:-10')]
PANDASET_BASES = ['cfgs/da-ieee-access-pointrcnn/iassd-sourceonly-pandaset-vox65k.yaml']      # 28287: row P2
PANDASET_ARMS = [('ratio', 0.75), ('ratio', 0.5), ('ratio', 0.25), ('thin', 'rows'), ('thin', 'random'),
                 ('thin', 'cols'), ('thin', 'random_cols'), ('thin', 'cols:4'), ('thin', 'random_cols:4'),
                 ('thin', 'azbin:0.332'), ('thin', 'random_azbin:0.332'), ('thin', 'elmax:0'), ('thin', 'elmax:-3'),
                 ('thin', 'elmin:-14'), ('thin', 'elmin:-10'), ('thin', 'pattern:hdl32e')]
BUDGET_STEPS = ('sample_points', 'sample_points_by_voxels')
DATASET = {id(ARMS): 'KittiDataset', id(WAYMO_ARMS): 'WaymoDataset', id(PANDASET_ARMS): 'PandasetDataset'}


def with_ratio_before_budget(processors, r):
    out, done = [], False
    for p in processors:
        if p['NAME'] in BUDGET_STEPS and p['NUM_POINTS']['test'] != -1 and not done:
            out.append({'NAME': 'sample_points', 'NUM_POINTS': {'train': -1, 'test': -1}, 'RATIO': {'test': r}})
            done = True
        out.append(dict(p))
    assert done, 'no budget sample_points step to put the density dose in front of'
    return out


def block_change(blk, kind, val):
    if kind == 'ratio':
        return {'DATA_PROCESSOR': with_ratio_before_budget(blk['DATA_PROCESSOR'], val)}, 'ratio%03d' % int(round(val * 100))
    if blk['DATASET'] == 'WaymoDataset':
        assert not any(blk.get('SHIFT_COOR') or []), 'the TOP-sensor elevation origin assumes no SHIFT_COOR'
        return waymo_change({'DATA_CONFIG_TAR': blk}, kind, val)
    if blk['DATASET'] == 'PandasetDataset':
        return pandaset_change({'DATA_CONFIG_TAR': blk}, kind, val)
    if kind == 'elev':
        side, _, deg = val.partition(':')
        step = {'NAME': 'mask_points_by_spec', 'SENSOR_ORIGIN': list(blk.get('SHIFT_COOR') or [0.0, 0.0, 0.0]),
                ('ELEVATION_MAX_DEG' if side == 'max' else 'ELEVATION_MIN_DEG'): float(deg)}
        return {'DATA_PROCESSOR': [step] + blk['DATA_PROCESSOR']}, 'el' + side + deg.replace('-', 'm').replace('.', 'p')
    return kitti_change(blk, kind, val)


def main(check=False):
    OUT.mkdir(parents=True, exist_ok=True)
    ok_all = True
    for base, arms in [(b, ARMS) for b in BASES] + [(b, WAYMO_ARMS) for b in WAYMO_BASES] + \
            [(b, PANDASET_ARMS) for b in PANDASET_BASES]:
        base_path = Path(base)
        rb = resolved(base_path)
        key = 'DATA_CONFIG_TAR' if rb.get('DATA_CONFIG_TAR') else 'DATA_CONFIG'
        blk = rb[key]
        assert blk['DATASET'] == DATASET[id(arms)], blk['DATASET']
        for kind, val in arms:
            change, tag = block_change(blk, kind, val)
            child = OUT / f'{base_path.stem}-{tag}.yaml'
            if not check:
                header = (f'# EVAL-ONLY twin of {base_path} for experiments_md 20261011_07 part B ({tag}): {key} change only.\n'
                          f'# Generated by analysis/make_iassd_sensitivity_configs.py. ANALYSIS.\n')
                child.write_text(header + yaml.safe_dump({'_BASE_CONFIG_': str(base_path), key: change},
                                                         sort_keys=False, default_flow_style=None))
            rc = resolved(child)
            diffs = [k for k in set(rb) | set(rc) if k not in (key, '_BASE_CONFIG_') and rb.get(k) != rc.get(k)]
            cb = rc[key]
            changed = sorted(k for k in set(blk) | set(cb) if blk.get(k) != cb.get(k) and k != '_BASE_CONFIG_')
            ok = not diffs and set(changed) == set(change)
            ok_all &= ok
            print(f'{child.name}: {"OK" if ok else "DIFF " + str(diffs) + str(changed)}  changed={changed}')
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main(check='--check' in sys.argv)
