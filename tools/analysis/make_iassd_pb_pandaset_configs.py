"""Write the rule PB IA-SSD configs of row (d), PandaSet -> nuScenes P / C1-analog (experiments_md 20261024_06; T05 §5
rule PB item 11, fixed 10-24 18:42 JST, experiments_md 932dc88), from the census. The (a)-(c) generator
`make_iassd_pb_configs.py` is reused, not edited: its processor rewrite (`pb_processor`), census reader (`census_value`)
and voxel step (`voxel_step`) build these files exactly as they built the other twelve.

Each PB config is a CHILD (`_BASE_CONFIG_`) of its 16,384 twin (`iassd-ringpattern-pandaset2nuscenes`,
`iassd-global-fovcut-recount-pandaset2nuscenes`, ST3D ad2407e; item 11: "each PB config's own 16,384 twin") that restates
the DATA_PROCESSOR list of every data block with only the budget step's NUM_POINTS changed (both mode keys = the block's
PB value). The twins already carry the recount right after the budget, so no block gains one. Budgets: the source
blocks from this row's census JSONs (`logs/iassd_pb_20261024/pandaset/census`, TRAIN only), the nuScenes eval block from
the 20261024_05 census (`logs/iassd_pb_20261024/census`; its resolved block is identical to these twins' target block).
If the timing smoke sends a block to the cost fallback (rule PB item 5: > 3x its 16,384 twin), list it in VOXEL_FALLBACK
and rerun: the authors' voxel step (cap 150,000) replaces the plain budget, the budget being the P99 of its output.

    python analysis/make_iassd_pb_pandaset_configs.py [--write]
"""
import argparse
import os
import sys

import yaml

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TOOLS, 'analysis'))
import make_iassd_pb_configs as base  # noqa: E402

C = base.C
CENSUS_PS = 'logs/iassd_pb_20261024/pandaset/census'
CENSUS_AC = 'logs/iassd_pb_20261024/census'          # 20261024_05: the nuScenes eval block (TRAIN split, eval mode)
EVAL_STEM = 'iassd-sourceonly-waymo2nuscenes'           # the census stem of that block (resolved identical, §3 of the report)
VOXEL_FALLBACK = set()                                  # (row, block) pairs sent to the fallback by the timing smoke
VOXEL_CAP = base.VOXEL_CAP

# row: (PB config stem, its 16,384 twin (= _BASE_CONFIG_ and the diff reference), header)
ROWS = {
    'PS_P': ('iassd-ringpattern-pb-pandaset2nuscenes', 'iassd-ringpattern-pandaset2nuscenes',
             'PandaSet -> nuScenes P, scan-pattern arm (data block of CenterPoint 28034, lf9c36yn, frozen 34db697)'),
    'PS_C1': ('iassd-global-fovcut-recount-pb-pandaset2nuscenes', 'iassd-global-fovcut-recount-pandaset2nuscenes',
              'PandaSet -> nuScenes C1-analog, density twin (data block of CenterPoint 28041 / 28280, frozen 6ac42f8 / 5adaac6)'),
}


def render(row, cfg_loader):
    stem, twin, header = ROWS[row]
    tcfg = cfg_loader(C + twin + '.yaml')
    doc, budgets = {'_BASE_CONFIG_': C + twin + '.yaml'}, {}
    for key in ('DATA_CONFIG', 'DATA_CONFIG_TAR'):
        if key == 'DATA_CONFIG_TAR':
            b, res = base.census_value(CENSUS_AC, EVAL_STEM, key)
        else:
            b, res = base.census_value(CENSUS_PS, twin, key)
        voxel = (row, key) in VOXEL_FALLBACK
        if voxel:
            assert key != 'DATA_CONFIG_TAR', 'rule PB item 5 covers training blocks only (coordinator 17:05)'
            v = res['summary']['voxel_step']
            assert v['voxel'] == [0.1, 0.1, 0.15] and v['occupied_max'] <= VOXEL_CAP, v
            b = v['pb_budget_on_voxel_output']
        budgets[key] = b
        doc[key] = {'DATA_PROCESSOR': base.pb_processor(tcfg[key].DATA_PROCESSOR, b, False, voxel)}
    lines = ['# IA-SSD, rule PB, row (d) (experiments_md 20261024_06; T05 §5 rule PB item 11, fixed 10-24 18:42 JST,',
             '# experiments_md 932dc88; the rule: items 1-10, fixed 15:53, clarified 16:06 / 17:05): %s.' % header,
             '# Parent: %s.yaml, its own 16,384 twin (never launched). This file differs from it ONLY in each data' % twin,
             '# block\'s budget (NUM_POINTS, both mode keys = the block\'s PB value)%s.' % (
                 ' and, on the cost-fallback block below, the voxel step' if any((row, k) in VOXEL_FALLBACK
                                                                               for k in budgets) else ''),
             '# PB value = the smallest multiple of 4,096 >= the 99th percentile of the in-range point count entering the',
             '# budget step, on 1,000 strided frames of the block\'s TRAIN split (training block in training mode with',
             '# augmentation; the eval block in evaluation mode on the target TRAIN split, measured in 20261024_05 §2;',
             '# val never sets a value). The recount (drop_empty_gt_boxes) sits right after the budget, as in the twin.',
             '# Legal: point counts of unlabelled TRAIN clouds only; no label, no AP read; one pre-declared rule, not tuned.',
             '# Budgets: ' + ', '.join('%s %d' % (k, v) for k, v in budgets.items()) + '.']
    vox = [k for k in budgets if (row, k) in VOXEL_FALLBACK]
    if vox:
        lines += ['# COST FALLBACK (rule PB item 5): the PB step at the plain P99 budget timed > 3x its 16,384 twin on %s,' % ' / '.join(vox),
                  '# so the authors\' voxel step (shuffle_points, then sample_points_by_voxels, raw, 0.1 x 0.1 x 0.15 m, 5 per',
                  '# voxel) replaces the plain budget there, its cap lifted to %d as on S2 A2 (coordinator 17:05); the budget is' % VOXEL_CAP,
                  '# the P99 rule on its output. An OPERATOR on that block: it keeps one point per occupied voxel.']
    lines += ['# Generated by tools/analysis/make_iassd_pb_pandaset_configs.py from the census JSONs named above.']
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=120)
    return stem, '\n'.join(lines) + '\n' + body, budgets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', nargs='*', default=list(ROWS))
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    os.chdir(TOOLS)
    sys.path.insert(0, TOOLS)
    import _init_path  # noqa: F401
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    for row in a.rows:
        stem, text, budgets = render(row, lambda p: cfg_from_yaml_file(p, EasyDict()))
        print('%-6s %s.yaml  %s' % (row, stem, budgets))
        if a.write:
            path = C + stem + '.yaml'
            assert not os.path.exists(path) or open(path).read() == text or os.environ.get('PB_OVERWRITE') == '1', \
                'refusing to overwrite %s (PB_OVERWRITE=1 to replace an earlier draft of this file)' % path
            with open(path, 'w') as f:
                f.write(text)


if __name__ == '__main__':
    main()
