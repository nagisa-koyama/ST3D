"""Item 63 (experiments_md 20261003_05 §5 item 63, design fixed 2026-11-27 13:02 JST, before any number): rule A per class and
x-y range bin on 59b's P0 floor pass (label-free), and the per-seed leaf configs. ANALYSIS.

    python3 analysis/item63_range_cuts.py cuts       # CPU job: /home/koyama/data/item63/range_cuts.json
    python3 analysis/item63_range_cuts.py configs    # cfgs/da-ieee-access-analysis/item63/centerpoint-item63-b-bn-rr-s<seed>.yaml

Bins: item 48's rings by box-centre x-y range, 0-10 / 10-20 / 20-30 / 30-40 / 40-75.2 m, inside rule A's own |x|, |y| <= 75.2
window. A bin with fewer than MIN_BOXES floor-pass boxes of a class takes the class's global rule-A cut (59b's), flagged.
Tests: tests/test_item63_range_cuts.py.
"""
import json
import pickle
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

OUT = Path('/home/koyama/data/item63')
P0 = Path('/home/koyama/data/item59')
CUTS_59B = P0 / 'cuts.json'
EDGES = (0.0, 10.0, 20.0, 30.0, 40.0)
CLASSES = ('Car', 'Pedestrian', 'Cyclist')
SEEDS = (666, 667)
MIN_BOXES = 1000
CFG_DIR = 'cfgs/da-ieee-access-analysis/item63'
LEAF_59B = 'cfgs/da-ieee-access-analysis/item59/centerpoint-item59-b-bn-s%d.yaml'


def bins_of(rho):
    import numpy as np
    return np.searchsorted(np.asarray(EDGES), rho, side='right') - 1


def range_rule_a(boxes, global_cuts):
    """{class: dict(cuts per bin, boxes per bin, fallback flags)} from floor-pass boxes (N, 9: ..., label, score)."""
    import numpy as np
    sys.path.insert(0, str(TOOLS / 'analysis' / 'pseudo_label_threshold'))
    from plateau_neg_thresh import RANGE, plateau_neg
    b = boxes[(np.abs(boxes[:, 0]) <= RANGE) & (np.abs(boxes[:, 1]) <= RANGE)]
    rb = bins_of(np.hypot(b[:, 0], b[:, 1]))
    out = {}
    for ci, name in enumerate(CLASSES, start=1):
        cuts, counts, fb = [], [], []
        for k in range(len(EDGES)):
            s = b[(np.abs(b[:, 7]) == ci) & (rb == k), 8]
            counts.append(int(len(s)))
            if len(s) < MIN_BOXES:
                cuts.append(float(global_cuts[name])); fb.append(True)
                continue
            neg, _, _, _ = plateau_neg(s)
            cuts.append(round(float(neg), 3)); fb.append(False)
        out[name] = dict(cuts=cuts, boxes=counts, fallback=fb)
    return out


def cmd_cuts():
    import numpy as np
    g = json.load(open(CUTS_59B))
    res = {}
    for seed in SEEDS:
        ps = pickle.load(open(P0 / ('p0_s%d' % seed) / 'ps_label_e0.pkl', 'rb'))
        boxes = np.concatenate([np.asarray(v['gt_boxes']).reshape(-1, 9) for v in ps.values()])
        gc = {c: g[str(seed)]['rule_a'][c]['neg'] for c in CLASSES}
        res[str(seed)] = dict(frames=len(ps), edges=list(EDGES), rule_a=range_rule_a(boxes, gc), global_cuts=gc)
        # descriptive: kept pseudo-labels per frame by bin at the per-range and at the global cut
        sys.path.insert(0, str(TOOLS / 'analysis' / 'pseudo_label_threshold'))
        from plateau_neg_thresh import RANGE
        b = boxes[(np.abs(boxes[:, 0]) <= RANGE) & (np.abs(boxes[:, 1]) <= RANGE)]
        rb = bins_of(np.hypot(b[:, 0], b[:, 1]))
        car = np.abs(b[:, 7]) == 1
        table = np.asarray(res[str(seed)]['rule_a']['Car']['cuts'])
        res[str(seed)]['car_per_frame'] = dict(
            range_cut=[float(((rb == k) & car & (b[:, 8] >= table[k])).sum() / len(ps)) for k in range(len(EDGES))],
            global_cut=[float(((rb == k) & car & (b[:, 8] >= gc['Car'])).sum() / len(ps)) for k in range(len(EDGES))])
        print(seed, json.dumps(res[str(seed)]['rule_a']), res[str(seed)]['car_per_frame'])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'range_cuts.json', 'w'), indent=1)


def leaf_text(seed, res):
    r = res[str(seed)]['rule_a']
    cuts = [r[c]['cuts'] for c in CLASSES]
    fb = {c: [EDGES[k] for k, f in enumerate(r[c]['fallback']) if f] for c in CLASSES}
    return ('# experiments_md 20261003_05 §5 item 63 (design fixed 2026-11-27 13:02 JST, before any number), seed %d. ANALYSIS:\n'
            '# 59b\'s b_bn (the base leaf, unchanged) with one rule-A cut per class and x-y range bin (item 48\'s rings) from\n'
            '# 59b\'s P0 floor pass (label-free), replacing NEG_THRESH / SCORE_THRESH (no ignore band). Written by\n'
            '# analysis/item63_range_cuts.py configs from /home/koyama/data/item63/range_cuts.json; do not edit by hand.\n'
            '# Bins that fell back to the global cut (fewer than %d boxes): %s\n'
            '_BASE_CONFIG_: %s\n'
            'SELF_TRAIN:\n'
            '    THRESH_BY_RANGE:\n'
            '        EDGES: %s\n'
            '        CUTS: %s\n') % (seed, MIN_BOXES, json.dumps(fb), LEAF_59B % seed, json.dumps(list(EDGES)), json.dumps(cuts))


def leaf_path(seed):
    return TOOLS / CFG_DIR / ('centerpoint-item63-b-bn-rr-s%d.yaml' % seed)


def cmd_configs():
    res = json.load(open(OUT / 'range_cuts.json'))
    for seed in SEEDS:
        p = leaf_path(seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(leaf_text(seed, res))
        print('wrote', p)


def main():
    a = sys.argv[1:]
    if a == ['cuts']:
        return cmd_cuts()
    if a == ['configs']:
        return cmd_configs()
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
