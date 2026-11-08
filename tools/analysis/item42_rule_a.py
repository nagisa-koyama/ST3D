"""Rule-A (and as-scored) rescoring of the item 42a Pandar64 rows at IoU 0.7 AND IoU 0.5 BEV, with and without AdaBN
(experiments_md 20261108_05; T05 §5 item 42a, plus the round-5 review's request for IoU 0.5 BEV beside IoU 0.7).

`analysis/pandaset_eval_protocols.py` (the rule-A tool every PandaSet number of the paper went through) returns Car BEV / 3D
moderate at IoU 0.7 only. This script calls ITS `score()` unchanged, with the official evaluator wrapped so that the result
string the evaluator prints is kept; the IoU 0.5 BEV moderate is read from that string's `AP_R40@0.70, 0.50, 0.50` block.
So the IoU 0.7 numbers are bit-identical to pandaset_eval_protocols.py's, and the IoU 0.5 ones come from the same call.

Rows: the 8 AdaBN results of item 42a (make_item42_configs.py --rule_a_json) and the 8 banked no-AdaBN results of the same
checkpoints (NO_ADABN_PKL below: item 34 E1 31983, item 37a 32267, O_P seed 667's in-run evaluation 32124), all spin val,
360 deg, device 0.

    python3 analysis/item42_rule_a.py --out /home/koyama/data/item42a/rule_a_iou.json   # inside a Slurm CPU job

Tests: tests/test_item42_rule_a.py.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

OUT = TOOLS.parent / 'output'
_FP = OUT / 'da-ieee-access-analysis' / 'factorswap-pandaset'
NO_ADABN_PKL = {  # (row, seed, job) -> result.pkl of the evaluation the banked rule-A number came from
    ('O', 666, 26389): OUT / 'da-ieee-access/centerpoint-pandaset-spin2spin/20261106_fs34/eval/epoch_115/val/fs34_OP/result.pkl',
    ('O', 667, 32124): Path('/home/koyama/data/wandb/run-20261107_131332-1v4pelr3/files/eval/eval_with_train/epoch_115/val/'
                            'result.pkl'),
    ('P', 666, 28034): _FP / 'centerpoint-factorswap-pandaset-evalonly-p-on-spin/20261106_fs34/eval/epoch_115/val/fs34_P_28034/result.pkl',
    ('P', 667, 28279): _FP / 'centerpoint-factorswap-pandaset-evalonly-p-on-spin/20261106_fs34/eval/epoch_115/val/fs34_P_28279/result.pkl',
    ('L', 666, 31539): _FP / 'centerpoint-factorswap-pandaset-evalonly-l-on-spin/20261108_fs37/eval/epoch_115/val/fs37_L_31539/result.pkl',
    ('L', 667, 31550): _FP / 'centerpoint-factorswap-pandaset-evalonly-l-on-spin/20261108_fs37/eval/epoch_115/val/fs37_L_31550/result.pkl',
    ('C1', 666, 28041): _FP / 'centerpoint-factorswap-pandaset-evalonly-c1-on-spin/20261106_fs34/eval/epoch_115/val/fs34_C1_28041/result.pkl',
    ('C1', 667, 28280): _FP / 'centerpoint-factorswap-pandaset-evalonly-c1-on-spin/20261106_fs34/eval/epoch_115/val/fs34_C1_28280/result.pkl',
}
PROTOCOLS = ('as_scored', 'rule_A')


def bev05_moderate(result_str, cls='Car'):
    """Car BEV AP_R40 moderate at IoU 0.5 from the evaluator's printed result string."""
    lines = result_str.splitlines()
    head = '%s AP_R40@0.70, 0.50, 0.50:' % cls
    for i, line in enumerate(lines):
        if line.strip() == head:
            for nxt in lines[i + 1:i + 6]:
                if nxt.startswith('bev  AP:'):
                    return float(nxt.split(':', 1)[1].split(',')[1])
    raise ValueError('no "%s" block with a bev line in the result string' % head)


class KeepResultString:
    """The official evaluator, unchanged, except that the result string of the last call is kept."""

    def __init__(self, official):
        self.official, self.last = official, None

    def get_official_eval_result(self, *args, **kwargs):
        result_str, ap = self.official.get_official_eval_result(*args, **kwargs)
        self.last = result_str
        return result_str, ap


def score_both(score, gt, dets, cone, protocol, official):
    """(BEV @0.7, 3D @0.7, BEV @0.5), Car, moderate (= every column on this target), via pandaset_eval_protocols.score."""
    keep = KeepResultString(official)
    bev07, d307 = score(gt, dets, cone, protocol, keep)
    return float(bev07), float(d307), bev05_moderate(keep.last)


def rows():
    import make_item42_configs as m42
    out = []
    for part, row, seed, job, epoch, cfg, _ in m42.ROWS:
        if part != 'P':
            continue
        out.append(dict(adabn=True, row=row, seed=seed, job=job, pkl=str(m42.result_pkl(part, row, seed, job, epoch, cfg))))
        out.append(dict(adabn=False, row=row, seed=seed, job=job, pkl=str(NO_ADABN_PKL[(row, seed, job)])))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--dry', action='store_true', help='list the rows and whether each result.pkl exists, then stop')
    args = p.parse_args()
    rs = rows()
    if args.dry:
        for r in rs:
            print(r['adabn'], r['row'], r['seed'], r['job'], os.path.exists(r['pkl']), r['pkl'])
        return
    import _init_path  # noqa: F401
    import pandaset_eval_protocols as pep
    from kitti_eval_cpu import load_official_eval
    os.chdir(TOOLS)
    official = load_official_eval()
    gt = pep.gt_with_counts(0)
    print('%-34s %-10s %8s %8s %8s' % ('row', 'protocol', 'BEV@0.7', '3D@0.7', 'BEV@0.5'))
    for r in rs:
        label = '%s %s s%d (job %d)' % ('AdaBN' if r['adabn'] else 'no AdaBN', r['row'], r['seed'], r['job'])
        if not os.path.exists(r['pkl']):
            print('%-34s missing %s' % (label, r['pkl']), flush=True)
            r['missing'] = True
            continue
        dets = pickle.load(open(r['pkl'], 'rb'))
        for proto in PROTOCOLS:
            r[proto] = score_both(pep.score, gt, dets, False, proto, official)
            print('%-34s %-10s %8.2f %8.2f %8.2f' % ((label, proto) + r[proto]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rs, open(args.out, 'w'), indent=1)
    print('wrote', args.out)
    if any(r.get('missing') for r in rs):
        sys.exit(1)


if __name__ == '__main__':
    main()
