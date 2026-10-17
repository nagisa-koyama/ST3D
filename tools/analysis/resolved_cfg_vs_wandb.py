"""Compare a config resolved at the LIVE checkout with the resolved config a past run stored in W&B.

Why: a row meant to differ from a banked run in one key must be checked key by key against what that run
actually trained with (its W&B `config.yaml` is `vars(cfg)` right after resolution, train.py), not against the
YAML files as they read today (experiments_md 20261013_01 §1 did this by hand; 20261017_05 uses this tool).

Usage, from ST3D/tools inside the container (import _init_path first, or the image's stale pcdet is used):
    python3 analysis/resolved_cfg_vs_wandb.py <cfg.yaml> <run-dir/files/config.yaml> [<cfg.yaml> <config.yaml> ...]
Each pair prints every differing key path (value here | value in the run). Run-time keys that are not part of
the recipe (TAG, EXP_GROUP_PATH, LOCAL_RANK, ROOT_DIR, slurm, launch, _wandb) are skipped. Exit 0 always: the
reader decides which differences are intended.
"""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402  (must precede pcdet: the live checkout, not the image's copy)
import yaml  # noqa: E402
from easydict import EasyDict  # noqa: E402

from pcdet.config import cfg_from_yaml_file  # noqa: E402

SKIP = {'_wandb', 'slurm', 'launch', 'TAG', 'EXP_GROUP_PATH', 'LOCAL_RANK', 'ROOT_DIR'}


def plain(x):
    """EasyDict / tuples / W&B's string 'None' -> plain comparable python."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if x == 'None':
        return None
    return x


def diff(a, b, path=''):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f'{path}.{k}' if path else k
            if k not in a:
                out.append((p, '<absent>', b[k]))
            elif k not in b:
                out.append((p, a[k], '<absent>'))
            else:
                out += diff(a[k], b[k], p)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (u, v) in enumerate(zip(a, b)):
            out += diff(u, v, f'{path}[{i}]')
    elif a != b:
        out.append((path, a, b))
    return out


def short(v, n=160):
    s = repr(v)
    return s if len(s) <= n else s[:n] + ' ...'


def main(argv):
    assert len(argv) >= 2 and len(argv) % 2 == 0, __doc__
    for cfg_path, wb_path in zip(argv[0::2], argv[1::2]):
        here = EasyDict()
        cfg_from_yaml_file(cfg_path, here)
        here = {k: v for k, v in plain(here).items() if k not in SKIP}
        with open(wb_path) as f:
            wb = yaml.safe_load(f)
        run = {k: plain(v['value']) for k, v in wb.items() if k not in SKIP}
        commit = wb.get('_wandb', {}).get('value', {}).get('e', {})
        commit = next((e.get('git', {}).get('commit') for e in commit.values() if isinstance(e, dict)), None)
        d = diff(here, run)
        print(f'=== {cfg_path} (live checkout) vs {wb_path} (run commit {commit}): {len(d)} difference(s); '
              f'top-level keys compared: {sorted(set(here) | set(run))}')
        for p, u, v in d:
            print(f'  {p}: here {short(u)} | run {short(v)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
