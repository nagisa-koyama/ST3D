"""Recursive-diff check of every config of experiments_md 20261003_05 §5 item 34 (factor swaps on PandaSet Pandar64, Lyft-64,
Lyft-40): each resolves to its base plus ONLY the intended change. ANALYSIS tooling; reads configs and job logs only.

  1. Oracles against their parents, both as the files resolve today and as the parent JOB logged its config at start
     (train.py's log_config_to_file lines, `cfg.<path>: <value>`):
       - centerpoint-pandaset-spin2spin.yaml (O_P's training recipe + spin val) vs 26389 (spin2flash): DATA_CONFIG, MODEL,
         OPTIMIZATION, CLASS_NAMES, ONTOLOGY identical;
       - O_L64 vs 25769 (lyft2lyft): only LIDAR_CONFIG 64 and SHIFT_COOR 2.04 on both blocks;
       - O_L40 vs 30313 (Lyft-40 C1-analog): only the sampler entry and the HIST_DIST_* keys gone, DATA_CONFIG_TAR Lyft-40.
  2. Training rows against their oracle: only <training block>.TRAIN_FACTOR_SWAP (D: its four keys) differ.
  3. Eval twins: (a)-(c) only DATA_CONFIG_TAR differs from the source row, and it equals the target oracle's (the Lyft
     controls: but for SHIFT_COOR); MODEL / CLASS_NAMES / ONTOLOGY also equal what the checkpoint's job logged;
     (d) only DATA_CONFIG_TAR.EVAL_OSENS2 differs from the oracle.
Prints one line per check and "ALL CONFIG CHECKS PASSED" or raises. Run inside the container from ST3D/tools:

    python analysis/factorswap_pl_config_check.py
"""
import ast
import contextlib
import glob
import io
import re
import sys

sys.path.insert(0, '.')
import _init_path  # noqa: F401,E402

A = 'cfgs/da-ieee-access-analysis/'
D = 'cfgs/da-ieee-access/'
FAILS = []
RELEVANT = ('DATA_CONFIG', 'DATA_CONFIGS', 'DATA_CONFIG_TAR', 'MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'ONTOLOGY')


def load(path):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    with contextlib.redirect_stdout(io.StringIO()):
        return cfg_from_yaml_file(path, EasyDict())


def flat(x, pre=''):
    """{dotted path: value} of a resolved config, `_BASE_CONFIG_` bookkeeping dropped."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            if k == '_BASE_CONFIG_':
                continue
            out.update(flat(v, pre + '.' + k if pre else k))
        return out
    out[pre] = [dict(e) if isinstance(e, dict) else e for e in x] if isinstance(x, list) else x
    return out


def logged(job):
    """{dotted path: value} of the config a job logged at start (top-level keys of interest only)."""
    path = (glob.glob('logs/error_%s_*.txt' % job) or [None])[0]
    assert path, 'no log for job %s' % job
    out = {}
    for line in open(path, errors='replace').read().replace('\r', '\n').split('\n'):
        m = re.search(r'INFO\]  cfg\.(\S+): (.*)$', line)
        if not m or not m.group(1).split('.')[0] in RELEVANT or m.group(1).endswith('_BASE_CONFIG_'):
            continue
        if m.group(1) in out:           # an eval job may log several configs (26971: 360 deg, then in the cone): the first
            continue
        try:
            v = ast.literal_eval(m.group(2))
        except (ValueError, SyntaxError):
            v = m.group(2)
        out[m.group(1)] = v
    assert out, 'job %s logged no config' % job
    return out


def norm(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, list):
        return [norm(e) for e in v]
    if isinstance(v, dict):
        return {k: norm(e) for k, e in v.items() if k != '_BASE_CONFIG_'}
    return v


def diff(a, b, keys=RELEVANT):
    """Paths whose values differ (or exist on one side only), restricted to the top-level `keys`."""
    a = {k: v for k, v in a.items() if k.split('.')[0] in keys}
    b = {k: v for k, v in b.items() if k.split('.')[0] in keys}
    return sorted(p for p in set(a) | set(b) if p not in a or p not in b or norm(a[p]) != norm(b[p]))


def check(ok, msg):
    print('[%s] %s' % ('ok' if ok else 'FAIL', msg))
    if not ok:
        FAILS.append(msg)


def expect(name, got, want):
    check(set(got) == set(want), '%s: differs in %s%s' % (name, got if len(got) < 12 else '%d paths' % len(got),
                                                          '' if set(got) == set(want) else ' (expected %s)' % sorted(want)))


def main():
    F = flat
    # ---- 1. oracles against their parents (today's file and the parent job's logged config)
    spin = F(load(D + 'centerpoint-pandaset-spin2spin.yaml'))
    s2f = F(load(D + 'centerpoint-pandaset-spin2flash.yaml'))
    keys = ('DATA_CONFIG', 'MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'ONTOLOGY')
    expect('spin2spin vs spin2flash (today) on the training recipe', diff(spin, s2f, keys), [])
    expect('spin2spin vs 26389 logged on the training recipe', diff(spin, logged(26389), keys), [])
    expect('spin2spin evaluation block vs 26971 logged (the eval job behind O_P 81.67)',
           diff(spin, logged(26971), ('DATA_CONFIG_TAR', 'MODEL')), [])
    tar_paths = [p for p in spin if p.startswith('DATA_CONFIG_TAR.')]
    print('     O_P evaluation block (spin2spin DATA_CONFIG_TAR): LIDAR_DEVICE %s, EVAL_FOV_FILTER %s, SHIFT_COOR %s, %d keys'
          % (spin['DATA_CONFIG_TAR.LIDAR_DEVICE'], spin['DATA_CONFIG_TAR.EVAL_FOV_FILTER'],
             spin['DATA_CONFIG_TAR.SHIFT_COOR'], len(tar_paths)))

    l64 = F(load(A + 'factorswap-lyft64/centerpoint-factorswap-lyft64-oracle.yaml'))
    l2l = F(load(D + 'centerpoint-sourceonly-lyft2lyft.yaml'))
    four = ['DATA_CONFIG.LIDAR_CONFIG', 'DATA_CONFIG.SHIFT_COOR', 'DATA_CONFIG_TAR.LIDAR_CONFIG', 'DATA_CONFIG_TAR.SHIFT_COOR']
    expect('O_L64 vs lyft2lyft (today)', diff(l64, l2l), four)
    expect('O_L64 vs 25769 logged', diff(l64, logged(25769)), four)
    check(l64['DATA_CONFIG.LIDAR_CONFIG'] == 64 and l64['DATA_CONFIG.SHIFT_COOR'] == [0.0, 0.0, 2.04]
          and l64['DATA_CONFIG_TAR.LIDAR_CONFIG'] == 64, 'O_L64: LIDAR_CONFIG 64, SHIFT_COOR 2.04 on both blocks')
    l64_direct = F(load('cfgs/da-ieee-access/da_lyft64_dataset.yaml'))
    expect('O_L64 DATA_CONFIG vs da_lyft64_dataset.yaml expanded',
           diff({'DATA_CONFIG.' + k: v for k, v in l64_direct.items()},
                {k: v for k, v in l64.items() if k.startswith('DATA_CONFIG.')}, ('DATA_CONFIG',)), [])

    l40 = F(load(A + 'factorswap-lyft40/centerpoint-factorswap-lyft40-oracle.yaml'))
    c1 = F(load(D + 'centerpoint-global-recount-lyft40-lyft2nuscenes.yaml'))
    hist = ['DATA_CONFIGS.LYFT_40BEAM.HIST_DIST_%s' % k for k in ('ON_THE_FLY', 'FRAMES', 'BINS', 'MAX_DIST')]
    for name, ref in (('today', c1), ('30313 logged', logged(30313))):
        got = diff(l40, ref, ('DATA_CONFIGS', 'MODEL', 'OPTIMIZATION', 'CLASS_NAMES', 'ONTOLOGY'))
        expect('O_L40 vs C1-analog (%s), training side' % name, got, hist + ['DATA_CONFIGS.LYFT_40BEAM.DATA_PROCESSOR'])
        a = [p['NAME'] for p in l40['DATA_CONFIGS.LYFT_40BEAM.DATA_PROCESSOR']]
        b = [p for p in ref['DATA_CONFIGS.LYFT_40BEAM.DATA_PROCESSOR'] if p['NAME'] != 'sample_points_hist_based']
        check(norm(l40['DATA_CONFIGS.LYFT_40BEAM.DATA_PROCESSOR']) == norm(b),
              'O_L40 processor = C1-analog (%s) minus sample_points_hist_based: %s' % (name, a))
    l40_tar = F(load('cfgs/da-ieee-access/da_lyft40_dataset.yaml'))
    expect('O_L40 DATA_CONFIG_TAR vs da_lyft40_dataset.yaml expanded',
           diff({'DATA_CONFIG_TAR.' + k: v for k, v in l40_tar.items()},
                {k: v for k, v in l40.items() if k.startswith('DATA_CONFIG_TAR.')}, ('DATA_CONFIG_TAR',)), [])

    # ---- 2. training rows against their oracle
    oracles = {'pandaset': (spin, 'DATA_CONFIG'), 'lyft64': (l64, 'DATA_CONFIG'), 'lyft40': (l40, 'DATA_CONFIGS.LYFT_40BEAM')}
    n_rows = 0
    for t, (base, block) in oracles.items():
        for path in sorted(glob.glob(A + 'factorswap-%s/centerpoint-factorswap-%s-r*.yaml' % (t, t))):
            row = F(load(path))
            got = diff(row, base)
            want = [p for p in row if p.startswith(block + '.TRAIN_FACTOR_SWAP.')]
            expect('%s vs its oracle' % path.split('/')[-1], got, want)
            check(len(want) > 0, '%s: the key is present' % path.split('/')[-1])
            n_rows += 1
    d = F(load(A + 'factorswap-lyft64/centerpoint-factorswap-lyft64-d-density.yaml'))
    expect('lyft64 d-density vs O_L64', diff(d, l64), ['DATA_CONFIG.DATA_PROCESSOR', 'DATA_CONFIG.POINT_CALIB_TAR',
                                                      'DATA_CONFIG.HIST_DIST_LYFT_PATH', 'DATA_CONFIG.HIST_DIST_NUSCENES_PATH'])
    c1_proc = [p['NAME'] for p in c1['DATA_CONFIGS.LYFT_40BEAM.DATA_PROCESSOR']]
    d_proc = [p['NAME'] for p in d['DATA_CONFIG.DATA_PROCESSOR']]
    check(d_proc == c1_proc and norm(d['DATA_CONFIG.DATA_PROCESSOR']) == norm(c1['DATA_CONFIGS.LYFT_40BEAM.DATA_PROCESSOR']),
          'lyft64 d-density processor == the Lyft-40 C1-analog\'s, entry for entry: %s' % d_proc)
    n_rows += 1
    print('     %d training rows checked' % n_rows)

    # ---- 3. eval twins
    srcs = {
        'pandaset-evalonly-p-on-spin': (D + 'centerpoint-ringpattern-pandaset2nuscenes.yaml', 'pandaset', [28034, 28279]),
        'pandaset-evalonly-c1-on-spin': (D + 'centerpoint-global-fovcut-recount-pandaset2nuscenes.yaml', 'pandaset',
                                         [28041, 28280]),
        'lyft40-evalonly-p-on-lyft40': (D + 'centerpoint-ringpattern-lyft40-lyft2nuscenes.yaml', 'lyft40', [30312, 30363]),
        'lyft40-evalonly-c1-on-lyft40': (D + 'centerpoint-global-recount-lyft40-lyft2nuscenes.yaml', 'lyft40',
                                         [30313, 30364]),
        'pandaset-evalonly-nuscenes-oracle-on-pandaset': (D + 'centerpoint-sourceonly-nuscenes.yaml', 'pandaset', [25743]),
        'lyft64-evalonly-nuscenes-oracle-on-lyft64': (D + 'centerpoint-sourceonly-nuscenes.yaml', 'lyft64', [25743]),
        'lyft40-evalonly-nuscenes-oracle-on-lyft40': (D + 'centerpoint-sourceonly-nuscenes.yaml', 'lyft40', [25743]),
    }
    tars = {'pandaset': spin, 'lyft64': l64, 'lyft40': l40}
    for stem, (src, t, jobs) in srcs.items():
        path = A + 'factorswap-%s/centerpoint-factorswap-%s.yaml' % (t, stem)
        tw = F(load(path))
        s = F(load(src))
        got = diff(tw, s)
        check(len(got) > 0 and all(p.startswith('DATA_CONFIG_TAR.') for p in got),
              '%s vs %s: only DATA_CONFIG_TAR differs (%d paths)' % (stem, src.split('/')[-1], len(got)))
        tg = diff(tw, tars[t], ('DATA_CONFIG_TAR',))
        allowed = ['DATA_CONFIG_TAR.SHIFT_COOR'] if ('nuscenes-oracle' in stem and t != 'pandaset') else []
        expect('%s DATA_CONFIG_TAR vs the %s oracle\'s' % (stem, t), tg, allowed)
        print('     %s: target SHIFT_COOR %s (oracle %s)' % (stem, tw['DATA_CONFIG_TAR.SHIFT_COOR'],
                                                        tars[t]['DATA_CONFIG_TAR.SHIFT_COOR']))
        for j in jobs:
            expect('%s MODEL / CLASS_NAMES / ONTOLOGY vs job %d logged' % (stem, j),
                   diff(tw, logged(j), ('MODEL', 'CLASS_NAMES', 'ONTOLOGY')), [])
        if t == 'pandaset':
            expect('%s MODEL vs the PandaSet oracle 26389 logged' % stem, diff(tw, logged(26389), ('MODEL',)), [])
    for t, (base, _) in oracles.items():
        path = A + 'factorswap-%s/centerpoint-factorswap-%s-evalbg050.yaml' % (t, t)
        tw = F(load(path))
        expect('%s vs its oracle' % path.split('/')[-1], diff(tw, base),
               ['DATA_CONFIG_TAR.EVAL_OSENS2.REGION_THIN.SIDE', 'DATA_CONFIG_TAR.EVAL_OSENS2.REGION_THIN.KEEP',
                'DATA_CONFIG_TAR.EVAL_OSENS2.BOX_MARGIN'])
    if FAILS:
        print('FAILED CONFIG CHECKS (%d):' % len(FAILS))
        for f in FAILS:
            print('  ' + f)
        raise SystemExit(1)
    print('ALL CONFIG CHECKS PASSED')


if __name__ == '__main__':
    main()
