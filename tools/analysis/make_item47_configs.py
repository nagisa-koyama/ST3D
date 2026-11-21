"""Eval-only configs and the row table of experiments_md 20261003_05 §5 item 47 (design fixed 2026-11-21 19:30 JST, before any
number): the structure term in the other direction.

Item 42a read the structure term L - P on the DENSE sensor's val. Item 47 reads it on S', the dense sensor's own clouds thinned
to nuScenes' lattice by P's own training cut, where P is the oracle and L (P's count per range bin, on the dense lattice) is the
swapped detector: structure_rev = P - L, with and without full AdaBN on S' TRAIN clouds (thinned the same way, eval-mode view).

  A (the reading): KITTI (part K), Waymo (W), Pandar64 (P); item 42a's checkpoints O, P, L, C1 x seeds 666 / 667.
    - KITTI: each arm's item 42a eval config + DATA_CONFIG_TAR.EVAL_FACTOR_SWAP.RING_THIN = P_K's TRAIN_FACTOR_SWAP.RING_THIN
      (item 39 B2's cut; frame-seeded phases through kitti_factor_swap.apply_points). Written here.
    - Waymo: every arm is scored with centerpoint-sourceonly-waymo2waymo.yaml in item 42a, so one existing twin serves all:
      sensitivity/centerpoint-sourceonly-waymo2waymo-patternhdl32e.yaml (P's RING_PATTERN + APPLY_AT_EVAL; 31713's config).
    - Pandar64: O's existing twin sensitivity-pandaset/centerpoint-pandaset-spin2spin-ringpatternhdl32e.yaml; P / L / C1: their
      item 42a eval twins + DATA_CONFIG_TAR.EVAL_RING_THIN = P's RING_PATTERN with MODE pattern (written here).
  B (beside): Waymo->nuScenes (part BW) and PandaSet->nuScenes (BP) P, L, C1 on nuScenes val with their training configs.

Every checkpoint is scored twice by analysis/adabn_eval.sh: stats `none` and `target`.

    python analysis/make_item47_configs.py [--check]     # from ST3D/tools; --check: exit 1 if a file on disk differs
    python analysis/make_item47_configs.py --rows K      # "tag eval_cfg job epoch stats" per row of part K / W / P / BW / BP
    python analysis/make_item47_configs.py --rule_a_json # rows JSON of the Pandar64 rows (pandaset_eval_protocols.py)

Tests: tests/test_item47_configs.py. ANALYSIS: target labels score only.
"""
import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS / 'analysis'))
import make_item42_configs as m42  # noqa: E402

OUT_DIR = 'cfgs/da-ieee-access-analysis/item47'
_DA = 'cfgs/da-ieee-access/centerpoint-'
P_K_TRAIN = m42._S2 + 'p-pattern.yaml'                                   # KITTI P: TRAIN_FACTOR_SWAP.RING_THIN
P_PANDASET_TRAIN = _DA + 'ringpattern-pandaset2nuscenes.yaml'             # Pandar64 P: DATA_CONFIG.RING_PATTERN
P_WAYMO_TRAIN = _DA + 'ringpattern-waymo2nuscenes.yaml'                   # Waymo P: DATA_CONFIG.RING_PATTERN
W_TWIN = 'cfgs/da-ieee-access-analysis/sensitivity/centerpoint-sourceonly-waymo2waymo-patternhdl32e.yaml'
P_O_TWIN = 'cfgs/da-ieee-access-analysis/sensitivity-pandaset/centerpoint-pandaset-spin2spin-ringpatternhdl32e.yaml'
STATS = ('none', 'target')


def twin_path(part, row):
    return '%s/centerpoint-item47-%s-%s-on-nuscenes-lattice.yaml' % (OUT_DIR, {'K': 'kitti', 'P': 'pandar64'}[part],
                                                                      row.lower())


def written_twins():
    """(part, row, base eval config) of every twin this script writes: KITTI O / P / L / C1, Pandar64 P / L / C1."""
    out = []
    for part, row, seed, _, _, cfg, _ in m42.ROWS:
        if seed != 666 or row not in ('O', 'P', 'L', 'C1'):
            continue
        if part == 'K' or (part == 'P' and row != 'O'):
            out.append((part, row, cfg))
    return out


def eval_cfg_of(part, row, m42_cfg):
    if part == 'W':
        return W_TWIN
    if part == 'P' and row == 'O':
        return P_O_TWIN
    return twin_path(part, row)


def _rows():
    rows = []
    for part, row, seed, job, epoch, cfg, train_cfg in m42.ROWS:
        if part not in ('K', 'W', 'P') or row not in ('O', 'P', 'L', 'C1'):
            continue
        rows.append([part, row, seed, job, epoch, eval_cfg_of(part, row, cfg), train_cfg])
    # B: the cross-dataset rows, scored with their own training configs on nuScenes val
    for part, row, seed, job, epoch, _, train_cfg in m42.ROWS:
        if part in ('W', 'P') and row in ('P', 'L', 'C1'):
            rows.append(['B' + part, row, seed, job, epoch, train_cfg, train_cfg])
    out = []
    for r in rows:
        for stats in STATS:
            out.append(tuple(r[:5]) + (stats,) + tuple(r[5:]))
    smoke = ('K', 'P', 666, 32283, 152, 'none')  # P_K on its own lattice, its own statistics: goes first
    out.sort(key=lambda r: r[:6] != smoke)
    # Waymo draw check (procedure check 4): P seed 666 `none` once more, in its own process, at the end of part W
    w = [r for r in out if r[:6] == ('W', 'P', 666, 27807, 7, 'none')][0]
    out.append(('W', 'P', 666, 27807, 7, 'none2') + w[6:])
    return out


# part, row, seed, training job, final epoch, stats (none / target / none2 = the Waymo redraw), eval config, training config
ROWS = _rows()


def stats_mode(stats):
    return 'none' if stats == 'none2' else stats


def tag(part, row, seed, job, stats):
    return 'i47%s_%s_%s_s%d_%d' % (part, row, stats, seed, job)


def result_pkl(part, row, seed, job, epoch, stats, cfg):
    """Where adabn_eval.py (--extra_tag tag --eval_tag tag) writes result.pkl for a row."""
    group, stem, t = '/'.join(cfg.split('/')[1:-1]), Path(cfg).stem, tag(part, row, seed, job, stats)
    return REPO / 'output' / group / stem / t / 'eval' / ('epoch_%d' % epoch) / t / 'result.pkl'


HEADER = """\
# EVAL-ONLY twin for experiments_md 20261003_05 §5 item 47 (design fixed 2026-11-21 19:30 JST, before any number): the
# {what} scored on S', the target's own clouds thinned to nuScenes' lattice by P's own training cut ({cut}), with and
# without full AdaBN on S' TRAIN clouds (adabn_eval.py --stats target reads this DATA_CONFIG_TAR's train split in the
# eval-mode view, so the statistics clouds are thinned the same way). Base: the row's item 42a eval config, unchanged.
# Written by analysis/make_item47_configs.py; do not edit by hand. ANALYSIS (target labels score only).
"""


def flow(v):
    if isinstance(v, list):
        return '[' + ', '.join(flow(x) for x in v) + ']'
    if isinstance(v, float):
        return repr(v)
    return str(v)


def kitti_cut():
    import yaml
    arm = yaml.safe_load(open(TOOLS / P_K_TRAIN))['DATA_CONFIG']['TRAIN_FACTOR_SWAP']
    assert list(arm) == ['RING_THIN'] and arm['RING_THIN']['MODE'] == 'pattern', arm
    return arm


def pandar64_cut():
    import yaml
    rp = yaml.safe_load(open(TOOLS / P_PANDASET_TRAIN))['DATA_CONFIG']['RING_PATTERN']
    assert set(rp) == {'LABEL_CACHE', 'SPACING_DEG', 'AZ_RES_DEG', 'TARGET_FOV_DEG'}, rp
    return {'LABEL_CACHE': rp['LABEL_CACHE'], 'MODE': 'pattern', 'SPACING_DEG': rp['SPACING_DEG'],
            'AZ_RES_DEG': rp['AZ_RES_DEG'], 'TARGET_FOV_DEG': rp['TARGET_FOV_DEG']}


def render(part, row, base):
    if part == 'K':
        arm = kitti_cut()['RING_THIN']
        body = ('DATA_CONFIG_TAR:\n    EVAL_FACTOR_SWAP:\n        RING_THIN: {MODE: %s, SPACING_DEG: %s, AZ_RES_DEG: %s}\n'
                % (arm['MODE'], flow(arm['SPACING_DEG']), flow(arm['AZ_RES_DEG'])))
        what, cut = 'KITTI %s row' % row, "P_K's TRAIN_FACTOR_SWAP RING_THIN pattern, copied exactly; frame-seeded phases"
    else:
        cut_cfg = pandar64_cut()
        body = 'DATA_CONFIG_TAR:\n    EVAL_RING_THIN:\n' + ''.join(
            '        %s: %s\n' % (k, flow(v)) for k, v in cut_cfg.items())
        what, cut = 'Pandar64 %s row' % row, "P's RING_PATTERN with MODE pattern, copied exactly; frame-seeded phases"
    return HEADER.format(what=what, cut=cut) + '_BASE_CONFIG_: %s\n' % base + body


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check', action='store_true')
    p.add_argument('--rows', choices=['K', 'W', 'P', 'BW', 'BP'])
    p.add_argument('--rule_a_json', action='store_true')
    args = p.parse_args()
    if args.rows:
        for part, row, seed, job, epoch, stats, cfg, _ in ROWS:
            if part == args.rows:
                print(tag(part, row, seed, job, stats), cfg, job, epoch, stats_mode(stats))
        return
    if args.rule_a_json:  # [label, result.pkl, device 0 = spin, cone False] as analysis/pandaset_eval_protocols.py reads
        print(json.dumps([['%s %s %s s%d (job %d)' % (part, row, stats, seed, job),
                           str(result_pkl(part, row, seed, job, epoch, stats, cfg)), 0, False]
                          for part, row, seed, job, epoch, stats, cfg, _ in ROWS if part == 'P'], indent=1))
        return
    bad = 0
    for part, row, base in written_twins():
        path, text = TOOLS / twin_path(part, row), render(part, row, base)
        if args.check:
            same = path.exists() and path.read_text() == text
            print('%s %s' % ('up to date:' if same else 'DIFFERS:', path))
            bad += not same
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print('wrote', path)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
