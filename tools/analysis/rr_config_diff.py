"""Resolved-config diff for the relocate-and-render row (experiments_md 20261018_02 §2). CPU, seconds.

Prints the config the way train.py logs it (`log_config_to_file`) and compares it, line by line, with the config
printed in a past job's log. Allowed differences for the RR row: DATA_CONFIG_TAR (Waymo instead of 27471's KITTI; it
must equal 27596's), the relocate_render entry of each platform's AUG_CONFIG_LIST, and the config's own TAG.

    python analysis/rr_config_diff.py <cfg> <job log to compare> [--only PREFIX] [--skip PREFIX ...]
"""
import argparse
import logging
import re
import sys

sys.path.insert(0, '.')
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file, log_config_to_file  # noqa: E402

LINE = re.compile(r'(cfg(?:\.[A-Za-z0-9_]+)+)(?:: (.*)| = edict\(\))$')


def from_log(path):
    out = {}
    txt = open(path, 'rb').read().decode('utf-8', 'replace').replace('\r', '\n')
    for ln in txt.split('\n'):
        i = ln.find('cfg.')
        if i < 0:
            continue
        m = LINE.match(ln[i:].rstrip())
        if m:
            out[m.group(1)] = m.group(2) if m.group(2) is not None else 'edict()'
    return out


def from_cfg(path):
    cfg = EasyDict()
    cfg.ROOT_DIR = None; cfg.LOCAL_RANK = 0
    cfg_from_yaml_file(path, cfg)
    cfg.TAG = path.split('/')[-1].replace('.yaml', '')
    cfg.EXP_GROUP_PATH = '/'.join(path.split('/')[1:-1])
    lines = []

    class H(logging.Handler):
        def emit(self, rec):
            lines.append(rec.getMessage())
    lg = logging.getLogger('rr_cfg_diff'); lg.handlers = [H()]; lg.setLevel(logging.INFO); lg.propagate = False
    log_config_to_file(cfg, logger=lg)
    out = {}
    for ln in '\n'.join(lines).split('\n'):
        m = LINE.match(ln.strip())
        if m:
            out[m.group(1)] = m.group(2) if m.group(2) is not None else 'edict()'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg'); ap.add_argument('log')
    ap.add_argument('--only', default=None, help='compare keys under this prefix only')
    ap.add_argument('--skip', nargs='*', default=['cfg.ROOT_DIR', 'cfg.LOCAL_RANK'])
    a = ap.parse_args()
    new, old = from_cfg(a.cfg), from_log(a.log)
    keys = sorted(set(new) | set(old))
    if a.only:
        keys = [k for k in keys if k.startswith(a.only)]
    keys = [k for k in keys if not any(k.startswith(s) for s in a.skip)]
    nd = 0
    for k in keys:
        if new.get(k) != old.get(k):
            nd += 1
            print(f'DIFF {k}\n   log: {old.get(k, "<absent>")}\n   cfg: {new.get(k, "<absent>")}')
    print(f'{len(keys)} keys compared ({len(new)} in cfg, {len(old)} in log), {nd} differ')


if __name__ == '__main__':
    main()
