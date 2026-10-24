"""Write the rule PB IA-SSD configs (experiments_md 20261024_05; T05 §5 "Rule PB, FIXED 15:53 JST") from the census.

Each PB config is a CHILD of its parent row's config (`_BASE_CONFIG_`) that restates the DATA_PROCESSOR list of every data
block with only the budget step's NUM_POINTS changed (both mode keys = the block's PB value: rule PB item 2, one value per
block), plus, where the parent has none, `drop_empty_gt_boxes` (MIN_POINTS 1) right after the budget (item 4: every PB row
carries the recount; N and K gain it). A list cannot be appended to by inheritance, so the whole list is restated, copied
from the parent's RESOLVED block. Budgets are read from the census JSONs (`iassd_pb_budget_census.py`, TRAIN split only;
a `val` JSON is refused). Nothing else is written.

    python analysis/make_iassd_pb_configs.py --census logs/iassd_pb_20261024/census [--write]

Without --write it prints each row's blocks and budgets only.
"""
import argparse
import json
import os
import sys

import yaml

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = 'cfgs/da-ieee-access-pointrcnn/'
BUDGET_STEPS = ('sample_points', 'sample_points_by_voxels')
RECOUNT = {'NAME': 'drop_empty_gt_boxes', 'MIN_POINTS': 1}

# row: (PB config stem, _BASE_CONFIG_ stem, parent stem for the diff, {block: census stem}, blocks gaining the recount,
#       extra top-level keys for DATA_CONFIG_TAR, header)
ROWS = {
    'W2N_control': ('iassd-sourceonly-pb-waymo2nuscenes', 'iassd-sourceonly-waymo2nuscenes',
                    'iassd-sourceonly-waymo2nuscenes',
                    {'DATA_CONFIG': 'iassd-sourceonly-waymo2nuscenes',
                     'DATA_CONFIG_TAR': 'iassd-sourceonly-waymo2nuscenes'}, [], {},
                    'W->N control (27930 -> 27970, a1eylf2f / 8fbory3l)'),
    'W2N_P': ('iassd-ringpattern-pb-waymo2nuscenes', 'iassd-ringpattern-waymo2nuscenes', 'iassd-ringpattern-waymo2nuscenes',
              {'DATA_CONFIG': 'iassd-ringpattern-waymo2nuscenes', 'DATA_CONFIG_TAR': 'iassd-ringpattern-waymo2nuscenes'},
              [], {}, 'W->N P, scan-pattern arm (27926, ozcyyhl9)'),
    'W2N_C1': ('iassd-global-toponly-recount-pb-waymo2nuscenes', 'iassd-global-toponly-recount-waymo2nuscenes',
               'iassd-global-toponly-recount-waymo2nuscenes',
               {'DATA_CONFIG': 'iassd-global-toponly-recount-waymo2nuscenes',
                'DATA_CONFIG_TAR': 'iassd-global-toponly-recount-waymo2nuscenes'}, [], {},
               'W->N C1, density-matched arm (27922 -> 27968, p530ort7 / f2ilt9ex)'),
    'N': ('iassd-sourceonly-recount-pb-nuscenes2nuscenes', 'iassd-sourceonly-nuscenes2nuscenes',
          'iassd-sourceonly-nuscenes2nuscenes',
          # the training block is S2 A1's (one PB run serves both, coordinator 16:06): its value is A1's census
          {'DATA_CONFIG': 'iassd-sourceonly-recount-nuscenes2kitti',
           'DATA_CONFIG_TAR': 'iassd-sourceonly-nuscenes2nuscenes'}, ['DATA_CONFIG'], {},
          'nuScenes in-domain oracle N (28130, ggqy3c2b / goc1f5hn; A1 27501 / 27547 rescored); gains the recount'),
    'A1': ('iassd-sourceonly-recount-pb-nuscenes2kitti', 'iassd-sourceonly-recount-nuscenes2kitti',
           'iassd-sourceonly-recount-nuscenes2kitti',
           {'DATA_CONFIG': 'iassd-sourceonly-recount-nuscenes2kitti',
            'DATA_CONFIG_TAR': 'iassd-sourceonly-recount-nuscenes2kitti'}, [], {},
           'S2 A1 = A1R (30379, 6pi3566m), 360-degree in-run scoring block'),
    'A1_crop': ('iassd-sourceonly-recount-pb-nuscenes2kitti-fovtar', 'iassd-sourceonly-recount-pb-nuscenes2kitti',
                'iassd-sourceonly-recount-nuscenes2kitti-fovtar',
                {'DATA_CONFIG': 'iassd-sourceonly-recount-nuscenes2kitti',
                 'DATA_CONFIG_TAR': 'iassd-sourceonly-recount-nuscenes2kitti-fovtar'}, [],
                {'FOV_POINTS_ONLY': True, 'ALLOW_FOV_POINTS_ONLY': True},
                'S2 A1 crop scoring twin (27688 protocol): the target cropped to the camera FOV before the budget'),
    'A2': ('iassd-accum-legaldepth-recount-pb-nuscenes2kitti', 'iassd-accum-legaldepth-recount-nuscenes2kitti',
           'iassd-accum-legaldepth-recount-nuscenes2kitti',
           {'NUSCENES_N008': 'iassd-accum-legaldepth-recount-nuscenes2kitti',
            'NUSCENES_N015': 'iassd-accum-legaldepth-recount-nuscenes2kitti',
            'DATA_CONFIG_TAR': 'iassd-accum-legaldepth-recount-nuscenes2kitti'}, [], {},
           'S2 A2 = A2R (30377, 3a71sry2), Boston 15 / Singapore 10 sweeps, 360-degree in-run scoring block'),
    'A2_crop': ('iassd-accum-legaldepth-recount-pb-nuscenes2kitti-fovtar', 'iassd-accum-legaldepth-recount-pb-nuscenes2kitti',
                'iassd-accum-legaldepth-recount-nuscenes2kitti-fovtar',
                {'NUSCENES_N008': 'iassd-accum-legaldepth-recount-nuscenes2kitti',
                 'NUSCENES_N015': 'iassd-accum-legaldepth-recount-nuscenes2kitti',
                 'DATA_CONFIG_TAR': 'iassd-accum-legaldepth-recount-nuscenes2kitti-fovtar'}, [],
                {'FOV_POINTS_ONLY': True, 'ALLOW_FOV_POINTS_ONLY': True},
                'S2 A2 crop scoring twin (27688 protocol)'),
    'K': ('iassd-sourceonly-recount-pb-kitti2kitti-fovcrop-noros-zshift', 'iassd-sourceonly-kitti2kitti-fovcrop-noros-zshift',
          'iassd-sourceonly-kitti2kitti-fovcrop-noros-zshift',
          {'DATA_CONFIG': 'iassd-sourceonly-kitti2kitti-fovcrop-noros-zshift',
           'DATA_CONFIG_TAR': 'iassd-sourceonly-kitti2kitti-fovcrop-noros-zshift'}, ['DATA_CONFIG'], {},
          'cropped KITTI oracle K (28141, be3g5cbn); gains the recount (coordinator 16:06, T05 §5 experiments_md cbfbcf2)'),
    'K_budget_only': ('iassd-sourceonly-pb-kitti2kitti-fovcrop-noros-zshift',
                      'iassd-sourceonly-kitti2kitti-fovcrop-noros-zshift',
                      'iassd-sourceonly-kitti2kitti-fovcrop-noros-zshift',
                      {'DATA_CONFIG': 'iassd-sourceonly-kitti2kitti-fovcrop-noros-zshift',
                       'DATA_CONFIG_TAR': 'iassd-sourceonly-kitti2kitti-fovcrop-noros-zshift'}, [], {},
                      'K, budget only: a TESTED CANDIDATE, NEVER LAUNCHED (coordinator 16:06: K gets the recount)'),
    'I2_P': ('iassd-ringpattern-pb-lyft40-lyft2nuscenes', 'iassd-ringpattern-lyft40-lyft2nuscenes',
             'iassd-ringpattern-lyft40-lyft2nuscenes',
             {'LYFT_40BEAM': 'iassd-ringpattern-lyft40-lyft2nuscenes',
              'DATA_CONFIG_TAR': 'iassd-ringpattern-lyft40-lyft2nuscenes'}, [], {},
             'I2 P, Lyft 40-beam scan pattern (ST3D 49bae4e config; replaces I2 at 16,384, never launched)'),
    'I2_C1': ('iassd-global-recount-pb-lyft40-lyft2nuscenes', 'iassd-global-recount-lyft40-lyft2nuscenes',
              'iassd-global-recount-lyft40-lyft2nuscenes',
              {'LYFT_40BEAM': 'iassd-global-recount-lyft40-lyft2nuscenes',
               'DATA_CONFIG_TAR': 'iassd-global-recount-lyft40-lyft2nuscenes'}, [], {},
              'I2 C1-analog, Lyft 40-beam density twin (ST3D 49bae4e config)'),
}


def block_of(cfg, key):
    if key in ('DATA_CONFIG', 'DATA_CONFIG_TAR'):
        return cfg[key]
    return cfg['DATA_CONFIGS'][key]


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def pb_processor(parent_procs, budget, add_recount):
    out, seen = [], False
    for p in plain(parent_procs):
        if p['NAME'] in BUDGET_STEPS and not seen:
            seen = True
            p = dict(p, NUM_POINTS={'train': int(budget), 'test': int(budget)})
            out.append(p)
            if add_recount:
                assert p['NAME'] == 'sample_points'
                out.append(dict(RECOUNT))
            continue
        out.append(p)
    assert seen, 'no budget step'
    if add_recount:
        names = [p['NAME'] for p in out]
        assert names.count('drop_empty_gt_boxes') == 1, names
    return out


def census_value(census_dir, stem, block):
    path = os.path.join(census_dir, '%s__%s__train.json' % (stem, block))
    with open(path) as f:
        res = json.load(f)
    assert res['split'] == 'train' and res['pb_value'], path
    assert res['summary']['frames'] == min(1000, res['dataset_len']), path
    return res['summary']['pb_budget'], res


def render(row, census_dir, cfg_loader):
    stem, base, parent, blocks, recount, tar_extra, header = ROWS[row]
    pcfg = cfg_loader(C + parent + '.yaml')
    doc, budgets = {'_BASE_CONFIG_': C + base + '.yaml'}, {}
    for key, cstem in blocks.items():
        b, _ = census_value(census_dir, cstem, key)
        budgets[key] = b
        procs = pb_processor(block_of(pcfg, key).DATA_PROCESSOR, b, key in recount)
        node = dict(tar_extra) if key == 'DATA_CONFIG_TAR' else {}
        node['DATA_PROCESSOR'] = procs
        if key in ('DATA_CONFIG', 'DATA_CONFIG_TAR'):
            doc[key] = node
        else:
            doc.setdefault('DATA_CONFIGS', {})[key] = node
    lines = ['# IA-SSD, rule PB (experiments_md 20261024_05; T05 §5 "Rule PB, FIXED 15:53 JST", experiments_md 711912e /',
             '# b68c164 / cbfbcf2): %s.' % header,
             '# Parent: %s.yaml. This file differs from it ONLY in each data block\'s budget (NUM_POINTS, both mode' % parent,
             '# keys = the block\'s PB value)%s.' % (' and in drop_empty_gt_boxes right after the budget' if recount else ''),
             '# PB value = the smallest multiple of 4,096 >= the 99th percentile of the in-range point count entering the',
             '# budget step, on 1,000 strided frames of the block\'s TRAIN split (training blocks in training mode with',
             '# augmentation; the eval block in evaluation mode on the target TRAIN split; val never sets a value).',
             '# Legal: point counts of unlabelled TRAIN clouds only; no label, no AP read; one pre-declared rule, not tuned.',
             '# Budgets: ' + ', '.join('%s %d' % (k, v) for k, v in budgets.items()) + '.',
             '# Generated by tools/analysis/make_iassd_pb_configs.py from logs/iassd_pb_20261024/census (census job JSONs).']
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=120)
    return stem, '\n'.join(lines) + '\n' + body, budgets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--census', required=True)
    ap.add_argument('--rows', nargs='*', default=list(ROWS))
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    os.chdir(TOOLS)
    sys.path.insert(0, TOOLS)
    import _init_path  # noqa: F401
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    for row in a.rows:
        stem, text, budgets = render(row, a.census, lambda p: cfg_from_yaml_file(p, EasyDict()))
        print('%-14s %s.yaml  %s' % (row, stem, budgets))
        if a.write:
            path = C + stem + '.yaml'
            assert not os.path.exists(path) or open(path).read() == text or os.environ.get('PB_OVERWRITE') == '1', \
                'refusing to overwrite %s (PB_OVERWRITE=1 to replace an earlier draft of this file)' % path
            with open(path, 'w') as f:
                f.write(text)


if __name__ == '__main__':
    main()
