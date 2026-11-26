"""Item 58 (experiments_md 20261003_05 §5 item 58, design fixed 2026-11-27 02:56 JST, experiments_md commit 0f9cb64, before any
number): an upper bound for normalization-level corrections of item 48's far residual. ANALYSIS (KITTI TRAIN labels train the
normalization layers of P_K; never a method row).

    python3 analysis/item58_bn_bound.py manifest OUT.json --jobs g666=J g667=J r666=J r667=J
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --real-kernel --workers 4
    python3 analysis/item58_bn_bound.py check OUT.json            # procedure check 1: non-normalization tensors bitwise equal
    python3 analysis/item58_bn_bound.py read AP.json [CHECK.json]

Arms: G (BatchNorm gamma / beta + statistics) and R (G + a per-range affine after every BN). References from item 57: P_K under
global AdaBN (P_RS1), L_K and O under global AdaBN. Readings, per arm and seed, over 20-30 and 30-40 m: s_X = L - P_X;
delta = mean(s_AdaBN - s_X), R_X = delta / mean(s_AdaBN). R_X >= 0.5 and delta >= 2 AP at both seeds: "a normalization-level
correction can close most of the far residual"; R_X <= 0.2 or delta < 1 AP at both seeds: "it cannot"; otherwise "partly".
Tests: tests/test_item58_bn_bound.py.
"""
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))

REF_MANIFEST = '/home/koyama/data/item57/item57b_manifest.json'
PK = {666: 32283, 667: 32284}
ARMS = {'g': 'P_BN_G', 'r': 'P_BN_R'}
FAR = (2, 3)
BN_BUFFERS = ('running_mean', 'running_var', 'num_batches_tracked')


def parse_jobs(args):
    out = {}
    for a in args:
        k, v = a.split('=')
        out[(k[0], int(k[1:]))] = int(v)
    return out


def manifest(jobs):
    import item38_kitti_rings as i38
    rows = []
    for (arm, seed), job in sorted(jobs.items()):
        r = i38.resolve('item58', ARMS[arm], seed, job, 'epoch_20')
        if r['status'] == 'ok' and not Path(r['result']).exists():
            r['status'] = 'result.pkl missing'
        rows.append(r)
    for r in json.load(open(REF_MANIFEST)):
        if r['row'] in ('P_RS1', 'L_ada', 'O_ada'):
            rows.append(dict(r, group='item58'))
    return rows


def norm_key(k):
    """A wrapped BatchNorm's tensor name (item 58 arm R: `<bn>.bn.<tensor>`) back to the original `<bn>.<tensor>`."""
    return re.sub(r'\.bn\.(weight|bias|running_mean|running_var|num_batches_tracked)$', r'.\1', k)


def compare(pre, post):
    """(non-normalization keys that differ, BN keys that changed, modulation max |a|, |b|, keys missing on either side)."""
    bn_prefixes = {k[:-len('.running_mean')] for k in pre if k.endswith('.running_mean')}
    post_n = {}
    mod = {'a': 0.0, 'b': 0.0}
    for k, v in post.items():
        m = re.search(r'\.mod\.(a|b)$', k)
        if m:
            mod[m.group(1)] = max(mod[m.group(1)], float(v.abs().max()))
            continue
        post_n[norm_key(k)] = v
    missing = sorted(set(pre) ^ set(post_n))
    differ, bn_changed = [], 0
    import torch
    for k, v in pre.items():
        if k not in post_n:
            continue
        same = torch.equal(v, post_n[k])
        if k.rsplit('.', 1)[0] in bn_prefixes:
            bn_changed += 0 if same else 1
        elif not same:
            differ.append(k)
    return dict(non_bn_differ=differ, bn_changed=bn_changed, mod_max=mod, missing=missing)


def ckpt_of(job, epoch):
    import subprocess
    return subprocess.check_output(['bash', str(TOOLS / 'analysis' / 'ckpt_of_job.sh'), str(job), str(epoch)], cwd=str(TOOLS),
                                   text=True).strip()


def cmd_check(manifest_path):
    import torch
    out = {}
    for r in json.load(open(manifest_path)):
        if r['row'] not in ARMS.values():
            continue
        pre = torch.load(ckpt_of(PK[r['seed']], 152), map_location='cpu')['model_state']
        post = torch.load(ckpt_of(r['job'], 20), map_location='cpu')['model_state']
        c = compare(pre, post)
        c['ok'] = not c['non_bn_differ'] and not c['missing']
        out['%s s%d' % (r['row'], r['seed'])] = c
        print('%s s%d: non-BN tensors differing %d, BN tensors changed %d, max |a| %.4f |b| %.4f, key mismatch %d -> %s' % (
            r['row'], r['seed'], len(c['non_bn_differ']), c['bn_changed'], c['mod_max']['a'], c['mod_max']['b'],
            len(c['missing']), 'ok' if c['ok'] else 'NOT READ'))
    return out


def rings(rows, name, seed):
    hit = [r for r in rows if r['row'] == name and r['seed'] == seed]
    return [g['bev'] for g in hit[0]['rings']] if len(hit) == 1 and hit[0].get('reproduced') else None


def mean(xs):
    return sum(xs) / len(xs)


def per_seed(rows, seed, name, check=None):
    if check is not None and not check.get('%s s%d' % (name, seed), {}).get('ok', False):
        return None
    L, P0, PX = rings(rows, 'L_ada', seed), rings(rows, 'P_RS1', seed), rings(rows, name, seed)
    if L is None or P0 is None or PX is None:
        return None
    s0 = [a - b for a, b in zip(L, P0)]
    sx = [a - b for a, b in zip(L, PX)]
    delta = mean([s0[i] - sx[i] for i in FAR])
    base = mean([s0[i] for i in FAR])
    return dict(s_adabn=s0, s_x=sx, delta=delta, R=delta / base if base else None, rings=PX)


def reading(seeds):
    if any(s is None for s in seeds):
        return 'not read'
    if all(s['R'] is not None and s['R'] >= 0.5 and s['delta'] >= 2.0 for s in seeds):
        return 'a normalization-level correction can close most of the far residual'
    if all((s['R'] is not None and s['R'] <= 0.2) or s['delta'] < 1.0 for s in seeds):
        return 'it cannot'
    return 'partly'


def cmd_read(path, check_path=None):
    rows = json.load(open(path))['rows']
    check = json.load(open(check_path)) if check_path else None
    for r in rows:
        print('%-7s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    for arm, name in ARMS.items():
        seeds = [per_seed(rows, s, name, check) for s in (666, 667)]
        print('\n== arm %s (%s): %s' % (arm.upper(), name, reading(seeds)))
        for s, d in zip((666, 667), seeds):
            print('  seed %d: %s' % (s, d))


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == 'manifest':
        rows = manifest(parse_jobs(sys.argv[4:]) if len(sys.argv) > 3 and sys.argv[3] == '--jobs' else {})
        json.dump(rows, open(sys.argv[2], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r['job'], r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if len(sys.argv) >= 3 and sys.argv[1] == 'check':
        out = cmd_check(sys.argv[2])
        dst = Path(sys.argv[2]).with_name(Path(sys.argv[2]).stem + '_check.json')
        json.dump(out, open(dst, 'w'), indent=1)
        print('wrote', dst)
        return
    if len(sys.argv) >= 3 and sys.argv[1] == 'read':
        cmd_read(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        return
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
