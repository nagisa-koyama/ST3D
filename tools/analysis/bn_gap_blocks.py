"""Per-block BatchNorm gap profile from adabn_eval.py's bn_gap.json files: mean |log sd ratio| (and mean shift) over
each block's BN layers - conv_input (the first BN), conv1..conv4, conv_out, backbone_2d, dense_head. The gap is computed
before any mixing, so a mix0.5 file gives the same profile as a mix1 one (20261005_01 §8).

    python analysis/bn_gap_blocks.py <label>=<bn_gap.json> [...] [--shift]
"""
import collections
import json
import sys

ORDER = ['conv_input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv_out', 'backbone_2d', 'dense_head']


def block(name):
    return name.split('.')[1] if name.startswith('backbone_3d.') else name.split('.')[0]


def profile(path, key='log_sd_ratio'):
    agg = collections.defaultdict(list)
    for n, v in json.load(open(path))['per_layer'].items():
        agg[block(n)].append(abs(v[key]))
    return {b: sum(agg[b]) / len(agg[b]) for b in ORDER if agg[b]}


if __name__ == '__main__':
    key = 'mean_shift' if '--shift' in sys.argv else 'log_sd_ratio'
    print(f"{'|' + key + '|':24s}" + ''.join(f'{b:>12s}' for b in ORDER))
    for arg in [a for a in sys.argv[1:] if not a.startswith('--')]:
        label, path = arg.split('=', 1); pr = profile(path, key)
        print(f'{label:24s}' + ''.join(f'{pr[b]:12.3f}' if b in pr else f'{"-":>12s}' for b in ORDER))
