"""Compare a config resolved at the LIVE checkout with the resolved config a past run printed in its training log.

For runs whose W&B `config.yaml` was never written (a run killed before W&B flushed it, e.g. 27185 / p1a5erux, which
died on an NCCL timeout), the log is the record: train.py prints every resolved key through
`pcdet.config.log_config_to_file` as `cfg.A.B: value`. This reads the FIRST such block of the log (the training run's;
a later recovery evaluation prints a second one) and compares it key by key, after formatting the live config the same
way. Companion of resolved_cfg_vs_wandb.py (20261017_05); written for A4 (experiments_md 20261017_09 §2).

Usage, from ST3D/tools inside the container:
    python3 analysis/resolved_cfg_vs_log.py <cfg.yaml> <logs/error_<job>_*.txt>
Prints every differing key path (value here | value in the log). Values are compared after ast.literal_eval when they
parse (so 1 and 1.0 agree), else as strings. Run-time keys (ROOT_DIR, LOCAL_RANK, TAG, EXP_GROUP_PATH) are skipped.
"""
import ast
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402  (must precede pcdet: the live checkout, not the image's copy)
from easydict import EasyDict  # noqa: E402

from pcdet.config import cfg_from_yaml_file  # noqa: E402

SKIP = ('cfg.ROOT_DIR', 'cfg.LOCAL_RANK', 'cfg.TAG', 'cfg.EXP_GROUP_PATH')
LINE = re.compile(r'config\.py \d+\s+INFO\]\s+(cfg\.[^:]+): (.*)$')


def norm(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s.strip()


def from_log(path):
    out, started, gap = {}, False, 0
    with open(path, errors='replace') as f:
        for raw in f:
            for line in raw.split('\r'):
                m = LINE.search(line.rstrip('\n'))
                if m:
                    started, gap = True, 0
                    out[m.group(1)] = m.group(2)
                elif started and line.strip() and 'edict()' not in line:
                    gap += 1
                    if gap > 3:          # the block has ended
                        return out
    return out


def flatten(cfg, pre='cfg', out=None):
    out = {} if out is None else out
    for k, v in cfg.items():
        if isinstance(v, EasyDict):
            flatten(v, pre + '.' + k, out)
        else:
            out[pre + '.' + k] = '%s' % (v,)
    return out


def main():
    cfg_file, log = sys.argv[1], sys.argv[2]
    here = flatten(cfg_from_yaml_file(cfg_file, EasyDict()))
    there = from_log(log)
    print(f'{cfg_file} (live)  vs  {log} (first logged config block, {len(there)} keys)')
    n = 0
    for k in sorted(set(here) | set(there)):
        if k.startswith(SKIP):
            continue
        a, b = here.get(k, '<absent>'), there.get(k, '<absent>')
        if norm(a) != norm(b):
            n += 1
            print(f'  {k}: {a[:150]}  |  {b[:150]}')
    print(f'{n} differing keys')


if __name__ == '__main__':
    main()
