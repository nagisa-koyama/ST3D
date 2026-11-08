"""Write the eval-only config of experiments_md 20261003_05 §5 item 42a' (design fixed 2026-11-08 22:40 JST, before any number).

42a' scores the S2 count-matched control L_K (32289 / 32290, centerpoint-factorswap-s2-l-latcount.yaml) on KITTI val thinned the
way its training clouds were: L_K's own TRAIN_FACTOR_SWAP RANGE_KEEP (EDGES and RATE copied exactly) applied to the val scan by
EVAL_FACTOR_SWAP, with the same frame-seeded draw (kitti_factor_swap.apply_points), as item 33e scored R3 at its own density
(factorswap-s2/centerpoint-factorswap-s2-eval-r3-density.yaml). The twin's base is the oracle config, as 33e's: L_K's config is
that base plus DATA_CONFIG.TRAIN_FACTOR_SWAP, which does nothing in evaluation. ANALYSIS (KITTI val labels score only).

It also holds the ROW TABLE of item 42a (one source for the launcher scripts/_item42a_evals_20261108.sh and for
tests/test_item42_configs.py): every checkpoint scored after full AdaBN on its target's val, with the eval config that already
scores it there (KITTI rows: their own; Waymo / Pandar64 / Lyft-40: the item 26 / 34 / 37 eval twins) and the config it was
trained with.

    python analysis/make_item42_configs.py [--check]     # from ST3D/tools; --check: exit 1 if the file on disk differs
    python analysis/make_item42_configs.py --rows K      # "tag eval_cfg job epoch" per row of part K / W / P / L / A
    python analysis/make_item42_configs.py --rule_a_json # rows JSON of the Pandar64 AdaBN results (pandaset_eval_protocols.py)

Tests: tests/test_item42_configs.py.
"""
import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
REPO = TOOLS.parent
BASE = 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml'
L_K = TOOLS / 'cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-l-latcount.yaml'
OUT = TOOLS / 'cfgs/da-ieee-access-analysis/item42/centerpoint-factorswap-s2-eval-l-latcount.yaml'
TWIN_42AP = 'cfgs/da-ieee-access-analysis/item42/centerpoint-factorswap-s2-eval-l-latcount.yaml'

_A = 'cfgs/da-ieee-access-analysis'
_S2 = _A + '/factorswap-s2/centerpoint-factorswap-s2-'
_FP = _A + '/factorswap-pandaset/centerpoint-factorswap-pandaset-evalonly-'
_F40 = _A + '/factorswap-lyft40/centerpoint-factorswap-lyft40-'
_DA = 'cfgs/da-ieee-access/centerpoint-'
# part, row, seed, training job, final epoch, eval config (AdaBN statistics come from its DATA_CONFIG_TAR TRAIN split), training config
ROWS = [
    ('K', 'O', 666, 26843, 152, BASE, BASE),  # first: the smoke (T05 item 42a, ~80.33 expected within ~1 AP)
    ('K', 'O', 667, 31817, 152, BASE, BASE),
    ('K', 'P', 666, 32283, 152, _S2 + 'p-pattern.yaml', _S2 + 'p-pattern.yaml'),
    ('K', 'P', 667, 32284, 152, _S2 + 'p-pattern.yaml', _S2 + 'p-pattern.yaml'),
    ('K', 'L', 666, 32289, 152, _S2 + 'l-latcount.yaml', _S2 + 'l-latcount.yaml'),
    ('K', 'L', 667, 32290, 152, _S2 + 'l-latcount.yaml', _S2 + 'l-latcount.yaml'),
    ('K', 'C1', 666, 31414, 152, _S2 + 'r3-density.yaml', _S2 + 'r3-density.yaml'),  # C1_K = R3
    ('K', 'C1', 667, 31948, 152, _S2 + 'r3-density.yaml', _S2 + 'r3-density.yaml'),
    ('K', 'R1', 666, 31409, 152, _S2 + 'r1-lines.yaml', _S2 + 'r1-lines.yaml'),
    ('K', 'R1', 667, 31946, 152, _S2 + 'r1-lines.yaml', _S2 + 'r1-lines.yaml'),
    ('K', 'R2', 666, 31410, 152, _S2 + 'r2-random.yaml', _S2 + 'r2-random.yaml'),
    ('K', 'R2', 667, 31947, 152, _S2 + 'r2-random.yaml', _S2 + 'r2-random.yaml'),
    ('W', 'O', 666, 31624, 7, _DA + 'sourceonly-waymo2waymo.yaml', _DA + 'sourceonly-waymo2waymo.yaml'),  # O2
    ('W', 'O', 667, 31811, 7, _DA + 'sourceonly-waymo2waymo.yaml', _DA + 'sourceonly-waymo2waymo.yaml'),
    ('W', 'P', 666, 27807, 7, _DA + 'sourceonly-waymo2waymo.yaml', _DA + 'ringpattern-waymo2nuscenes.yaml'),
    ('W', 'P', 667, 27840, 7, _DA + 'sourceonly-waymo2waymo.yaml', _DA + 'ringpattern-waymo2nuscenes.yaml'),
    ('W', 'L', 666, 31823, 7, _DA + 'sourceonly-waymo2waymo.yaml',
     _DA + 'global-toponly-recount-latcount-waymo2nuscenes.yaml'),
    ('W', 'L', 667, 31824, 7, _DA + 'sourceonly-waymo2waymo.yaml',
     _DA + 'global-toponly-recount-latcount-waymo2nuscenes.yaml'),
    ('W', 'C1', 666, 27850, 7, _DA + 'sourceonly-waymo2waymo.yaml', _DA + 'global-toponly-recount-waymo2nuscenes.yaml'),
    ('W', 'C1', 667, 28278, 7, _DA + 'sourceonly-waymo2waymo.yaml', _DA + 'global-toponly-recount-waymo2nuscenes.yaml'),
    ('P', 'O', 666, 26389, 115, _DA + 'pandaset-spin2spin.yaml', _DA + 'pandaset-spin2flash.yaml'),
    ('P', 'O', 667, 32124, 115, _DA + 'pandaset-spin2spin.yaml', _DA + 'pandaset-spin2spin.yaml'),
    ('P', 'P', 666, 28034, 115, _FP + 'p-on-spin.yaml', _DA + 'ringpattern-pandaset2nuscenes.yaml'),
    ('P', 'P', 667, 28279, 115, _FP + 'p-on-spin.yaml', _DA + 'ringpattern-pandaset2nuscenes.yaml'),
    ('P', 'L', 666, 31539, 115, _FP + 'l-on-spin.yaml', _DA + 'global-fovcut-recount-latcount-pandaset2nuscenes.yaml'),
    ('P', 'L', 667, 31550, 115, _FP + 'l-on-spin.yaml', _DA + 'global-fovcut-recount-latcount-pandaset2nuscenes.yaml'),
    ('P', 'C1', 666, 28041, 115, _FP + 'c1-on-spin.yaml', _DA + 'global-fovcut-recount-pandaset2nuscenes.yaml'),
    ('P', 'C1', 667, 28280, 115, _FP + 'c1-on-spin.yaml', _DA + 'global-fovcut-recount-pandaset2nuscenes.yaml'),
    ('L', 'O', 666, 31978, 30, _F40 + 'oracle.yaml', _F40 + 'oracle.yaml'),
    ('L', 'O', 667, 32128, 30, _F40 + 'oracle.yaml', _F40 + 'oracle.yaml'),
    ('L', 'P', 666, 30312, 30, _F40 + 'evalonly-p-on-lyft40.yaml', _DA + 'ringpattern-lyft40-lyft2nuscenes.yaml'),
    ('L', 'P', 667, 30363, 30, _F40 + 'evalonly-p-on-lyft40.yaml', _DA + 'ringpattern-lyft40-lyft2nuscenes.yaml'),
    ('L', 'L', 666, 32273, 30, _F40 + 'evalonly-l-on-lyft40.yaml', _DA + 'global-recount-latcount-lyft40-lyft2nuscenes.yaml'),
    ('L', 'L', 667, 32274, 30, _F40 + 'evalonly-l-on-lyft40.yaml', _DA + 'global-recount-latcount-lyft40-lyft2nuscenes.yaml'),
    ('L', 'C1', 666, 30313, 30, _F40 + 'evalonly-c1-on-lyft40.yaml', _DA + 'global-recount-lyft40-lyft2nuscenes.yaml'),
    ('L', 'C1', 667, 30364, 30, _F40 + 'evalonly-c1-on-lyft40.yaml', _DA + 'global-recount-lyft40-lyft2nuscenes.yaml'),
    # 42a': L_K on KITTI val thinned by its own RANGE_KEEP, NO AdaBN (eval_checkpoint.sh)
    ('A', 'L', 666, 32289, 152, TWIN_42AP, _S2 + 'l-latcount.yaml'),
    ('A', 'L', 667, 32290, 152, TWIN_42AP, _S2 + 'l-latcount.yaml'),
]
EXTRA_TAG_42AP = '20261108_i42ap'


def tag(part, row, seed, job):
    return 'i42a%s_%s_%s_s%d_%d' % ('p' if part == 'A' else '', part, row, seed, job)


def result_pkl(part, row, seed, job, epoch, cfg):
    """Where the evaluation of a row writes result.pkl (adabn_eval.py for K / W / P / L, test.py for A)."""
    group, stem, t = '/'.join(cfg.split('/')[1:-1]), Path(cfg).stem, tag(part, row, seed, job)
    if part == 'A':
        return REPO / 'output' / group / stem / EXTRA_TAG_42AP / 'eval' / ('epoch_%d' % epoch) / 'val' / t / 'result.pkl'
    return REPO / 'output' / group / stem / t / 'eval' / ('epoch_%d' % epoch) / t / 'result.pkl'

HEADER = """\
# EVAL-ONLY twin of the KITTI oracle base for experiments_md 20261003_05 §5 item 42a' (design fixed 2026-11-08 22:40 JST,
# before any number). KITTI val cut by L_K's own RANGE_KEEP (centerpoint-factorswap-s2-l-latcount.yaml; EDGES and RATE copied
# exactly): L_K's training point count per 1.5-m bin (P_K's in-grid count). Same frame-seeded draw as training
# (EVAL_FACTOR_SWAP -> kitti_factor_swap.apply_points, evaluation only), as item 33e's R3 twin. Scores L_K 32289 / 32290 at
# epoch 152 WITHOUT AdaBN. Written by analysis/make_item42_configs.py; do not edit by hand. ANALYSIS.
"""


def flow_list(values):
    return '[' + ', '.join(repr(float(v)) if isinstance(v, float) else str(v) for v in values) + ']'


def render():
    import yaml
    arm = yaml.safe_load(open(L_K))['DATA_CONFIG']['TRAIN_FACTOR_SWAP']
    assert list(arm) == ['RANGE_KEEP'], arm
    rk = arm['RANGE_KEEP']
    assert set(rk) == {'EDGES', 'RATE'} and len(rk['EDGES']) == len(rk['RATE']) + 1, rk
    return (HEADER + '_BASE_CONFIG_: %s\n' % BASE + 'DATA_CONFIG_TAR:\n    EVAL_FACTOR_SWAP:\n        RANGE_KEEP:\n'
            + '            EDGES: %s\n' % flow_list(rk['EDGES']) + '            RATE: %s\n' % flow_list(rk['RATE']))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check', action='store_true')
    p.add_argument('--rows', choices=['K', 'W', 'P', 'L', 'A'])
    p.add_argument('--rule_a_json', action='store_true')
    args = p.parse_args()
    if args.rows:
        for part, row, seed, job, epoch, cfg, _ in ROWS:
            if part == args.rows:
                print(tag(part, row, seed, job), cfg, job, epoch)
        return
    if args.rule_a_json:  # [label, result.pkl, device 0 = spin, cone False] as analysis/pandaset_eval_protocols.py reads
        print(json.dumps([['%s %s s%d (job %d)' % (part, row, seed, job), str(result_pkl(part, row, seed, job, epoch, cfg)),
                           0, False] for part, row, seed, job, epoch, cfg, _ in ROWS if part == 'P'], indent=1))
        return
    text = render()
    if args.check:
        same = OUT.exists() and OUT.read_text() == text
        print('%s %s' % ('up to date:' if same else 'DIFFERS:', OUT))
        sys.exit(0 if same else 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
