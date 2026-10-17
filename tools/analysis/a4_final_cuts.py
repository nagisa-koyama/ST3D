"""A4 stage 2 (experiments_md 20261017_09 §4): turn the printed outputs of the two pre-declared cut rules into the final cuts.

    python analysis/a4_final_cuts.py <rule B output.txt> <rule A output.txt> <out.json>

- Rule B (`source_precision_cut.py`) prints `cuts: {'Car': x, 'Pedestrian': y, 'Cyclist': z}` (1.01 = gated off).
- Rule A (`plateau_neg_thresh.py`) prints `NEG_THRESH: {'Car': x, 'Pedestrian': y, 'Cyclist': z}`.
- Rule A step 6 (20261004_01 §5e): a SCORE_THRESH below its NEG_THRESH is raised to it.
Writes {"rule_B": ..., "rule_A": ..., "SCORE_THRESH": [...], "NEG_THRESH": [...], "raised": [...]} and prints the two YAML lines
the stage-3 config takes. Exits 2 if either line is missing or a class is absent. No data is read; nothing is chosen.
"""
import ast
import json
import re
import sys

CLASSES = ('Car', 'Pedestrian', 'Cyclist')


def last_dict(path, tag):
    found = None
    for line in open(path, errors='replace').read().replace('\r', '\n').split('\n'):
        m = re.match(r'^\s*%s:\s*(\{.*\})\s*$' % re.escape(tag), line)
        if m:
            found = ast.literal_eval(m.group(1))
    if found is None or any(c not in found for c in CLASSES):
        print(f'a4_final_cuts: no complete "{tag}: {{...}}" line in {path}', file=sys.stderr)
        sys.exit(2)
    return {c: float(found[c]) for c in CLASSES}


def final_cuts(rule_b, rule_a):
    score = [max(rule_b[c], rule_a[c]) for c in CLASSES]
    neg = [rule_a[c] for c in CLASSES]
    raised = [c for c in CLASSES if rule_b[c] < rule_a[c]]
    return score, neg, raised


def main():
    b, a, out = sys.argv[1], sys.argv[2], sys.argv[3]
    rule_b, rule_a = last_dict(b, 'cuts'), last_dict(a, 'NEG_THRESH')
    score, neg, raised = final_cuts(rule_b, rule_a)
    res = {'rule_B': rule_b, 'rule_A': rule_a, 'SCORE_THRESH': score, 'NEG_THRESH': neg, 'raised': raised,
           'classes': list(CLASSES)}
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res))
    print('  SCORE_THRESH: [%s]' % ', '.join('%g' % v for v in score))
    print('  NEG_THRESH: [%s]' % ', '.join('%.3f' % v for v in neg))


if __name__ == '__main__':
    main()
