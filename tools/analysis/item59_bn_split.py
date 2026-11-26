"""Item 59 (experiments_md 20261003_05 §5 item 59, design fixed 2026-11-27 08:15 JST, experiments_md commit d61feae, before any
number). ANALYSIS on item 48's forward pair (P_K trained on S', scored on full KITTI val).

59a, where item 58's G correction sits (labelled analysis, rescore only):
    python3 analysis/item59_bn_split.py make [SEED...]           # hybrid checkpoints (G's gamma / beta per group) + bitwise check
    python3 analysis/item59_bn_split.py runs SEED                # "arm ckpt tag" for scripts/_item59a_bnsplit_20261127.sh
59b, label-free counterparts of G (pseudo-labels from P_K + AdaBN on KITTI testing clouds):
    python3 analysis/item59_bn_split.py base_config [--out F]    # (re)writes cfgs/.../item59/centerpoint-item59-b-base.yaml
    python3 analysis/item59_bn_split.py cuts                     # rule A on P0's floor passes -> /home/koyama/data/item59/cuts.json
    python3 analysis/item59_bn_split.py configs                  # the four per-seed leaf configs from cuts.json
Both:
    python3 analysis/item59_bn_split.py manifest OUT.json [--jobs bn666=J bn667=J all666=J all667=J]
    python3 analysis/item38_kitti_rings.py ap OUT.json AP.json --real-kernel --workers 4
    python3 analysis/item59_bn_split.py read AP.json

Readings: item 58's (analysis/item58_bn_bound.per_seed / reading) per arm and seed against P_RS1 and L_ada over 20-30 and 30-40 m.
Tests: tests/test_item59_bn_split.py, tests/test_item59_targets.py.
"""
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
# Before any pcdet import: run as a script, the image's stale baked-in pcdet would otherwise resolve _BASE_CONFIG_ with the
# old merge (base overwrites child); the first base_config written on 2026-11-27 lost G's z shift that way.
import _init_path  # noqa: F401,E402

REPO = TOOLS.parent
OUT = Path('/home/koyama/data/item59')
EVAL_CFG = 'cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml'   # P_K's (item 57's eval cfg)
PK = {666: 32283, 667: 32284}
PK_EPOCH = 152
G = {666: 37190, 667: 37191}
G_EPOCH = 20
GROUPS = {'aff_all': ('backbone_3d', 'backbone_2d', 'dense_head'),
          'aff_3d': ('backbone_3d',),
          'aff_2d': ('backbone_2d', 'dense_head')}
ROWS_A = {'rs1': 'P_RS1_59', 'aff_all': 'P_AFF_ALL', 'aff_3d': 'P_AFF_3D', 'aff_2d': 'P_AFF_2D'}
ROWS_B = {'bn': 'P_PL_BN', 'all': 'P_PL_ALL'}
B_DIR = 'cfgs/da-ieee-access-analysis/item59'
B_BASE = B_DIR + '/centerpoint-item59-b-base.yaml'
B_EPOCH = 10
CLASSES = ('Car', 'Pedestrian', 'Cyclist')
REF_57 = '/home/koyama/data/item57/item57b_manifest.json'
REF_58 = '/home/koyama/data/item58/item58_manifest.json'
TOL_RS1 = 0.05


# ---------------------------------------------------------------- 59a: hybrids
def bn_prefixes(state):
    return sorted(k[:-len('.running_mean')] for k in state if k.endswith('.running_mean'))


def hybrid(pk, g, groups):
    """P_K's state with G's BN weight / bias in the layers whose top-level module is in `groups`. Statistics stay P_K's
    (every arm re-estimates them by AdaBN)."""
    out = dict(pk)
    for p in bn_prefixes(pk):
        if p.split('.')[0] in groups:
            for t in ('weight', 'bias'):
                out[p + '.' + t] = g[p + '.' + t].clone()
    return out


def check_hybrid(h, pk, g, groups):
    """Bitwise: every non-BN tensor is P_K's; every BN weight / bias is G's inside `groups` and P_K's outside. Returns
    (number of layers taken from G, list of problems)."""
    import torch
    prob, n = [], 0
    pre = set(bn_prefixes(pk))
    if set(h) != set(pk):
        prob.append('key sets differ')
    for k, v in h.items():
        p, _, t = k.rpartition('.')
        if p in pre and t in ('weight', 'bias'):
            src = g if p.split('.')[0] in groups else pk
            if not torch.equal(v, src[k]):
                prob.append(k)
        elif not torch.equal(v, pk[k]):
            prob.append(k)
    for p in pre:
        if p.split('.')[0] in groups:
            n += 1
    return n, prob


def ckpt_of(job, epoch):
    import subprocess
    return subprocess.check_output(['bash', str(TOOLS / 'analysis' / 'ckpt_of_job.sh'), str(job), str(epoch)], cwd=str(TOOLS),
                                   text=True).strip()


def hybrid_path(arm, seed):
    # the stem carries only the epoch's digits: rs_adabn_eval names its output dir epoch_<digits of the stem>
    return OUT / 'ckpt' / ('%s_s%d' % (arm, seed)) / ('checkpoint_epoch_%d.pth' % PK_EPOCH)


def change_stats(pk, g):
    """Per group: mean |gamma_G - gamma_P| / |gamma_P| and mean |beta_G - beta_P| over channels."""
    import torch
    out = {}
    for grp in GROUPS['aff_all']:
        rg, db, n = [], [], 0
        for p in bn_prefixes(pk):
            if p.split('.')[0] != grp:
                continue
            n += 1
            gp, gg = pk[p + '.weight'].float(), g[p + '.weight'].float()
            rg.append(((gg - gp).abs() / gp.abs().clamp_min(1e-8)).flatten())
            db.append((g[p + '.bias'].float() - pk[p + '.bias'].float()).abs().flatten())
        out[grp] = dict(layers=n, rel_gamma=float(torch.cat(rg).mean()), abs_beta=float(torch.cat(db).mean()))
    return out


def cmd_make(seeds):
    import torch
    report = {}
    for seed in seeds:
        pk_ck = torch.load(ckpt_of(PK[seed], PK_EPOCH), map_location='cpu', weights_only=False)
        g_ck = torch.load(ckpt_of(G[seed], G_EPOCH), map_location='cpu', weights_only=False)
        pk, g = pk_ck['model_state'], g_ck['model_state']
        report['s%d' % seed] = dict(change=change_stats(pk, g))
        for arm, groups in GROUPS.items():
            h = hybrid(pk, g, groups)
            n, prob = check_hybrid(h, pk, g, groups)
            assert not prob, (arm, seed, prob[:5])
            dst = hybrid_path(arm, seed)
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state': h, 'epoch': PK_EPOCH, 'version': pk_ck.get('version')}, dst)
            report['s%d' % seed][arm] = dict(layers_from_G=n, path=str(dst), check='ok')
            print('s%d %-7s %2d BN layers take G\'s gamma / beta; bitwise check ok -> %s' % (seed, arm, n, dst))
        print('s%d change by group: %s' % (seed, report['s%d' % seed]['change']))
    for seed in seeds:
        json.dump(report['s%d' % seed], open(OUT / ('item59a_hybrids_s%d.json' % seed), 'w'), indent=1)


def tag(arm, seed):
    return 'i59_%s_s%d' % (arm, seed)


def runs(seed):
    out = [('rs1', ckpt_of(PK[seed], PK_EPOCH), tag('rs1', seed))]
    out += [(a, str(hybrid_path(a, seed)), tag(a, seed)) for a in GROUPS]
    return out


def result_dir(arm, seed):
    t = tag(arm, seed)
    return REPO / 'output' / 'da-ieee-access-analysis' / 'factorswap-s2' / Path(EVAL_CFG).stem / t / 'eval' / (
        'epoch_%d' % PK_EPOCH) / t


# ---------------------------------------------------------------- 59b: configs and cuts
def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def base_config_text():
    """The 59b base config: P_K's config with KITTI testing clouds as the pseudo-labelled target, G's augmentation, G's
    length in steps, a frozen P_K + AdaBN teacher (MODEL_TEACHER = P_K's MODEL, resolved), no thresholds (leaves add them)."""
    import os
    import yaml
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    cwd = os.getcwd()
    os.chdir(TOOLS)                  # _BASE_CONFIG_ paths are relative to tools/
    try:
        pk = cfg_from_yaml_file(EVAL_CFG, EasyDict())
        gcfg = cfg_from_yaml_file('cfgs/da-ieee-access-analysis/item58/centerpoint-item58-p-bnonly-g.yaml', EasyDict())
    finally:
        os.chdir(cwd)
    teacher = dict(CLASS_NAMES=list(pk.CLASS_NAMES), ONTOLOGY=pk.ONTOLOGY, **_plain(pk.MODEL))
    body = {
        'DATA_CONFIG_TAR': {
            'DATA_SPLIT': {'train': 'test', 'test': 'val'},
            'INFO_PATH': {'train': ['kitti_infos_test.pkl'], 'test': ['kitti_infos_val.pkl']},
            'USE_PSEUDO_LABEL': True,
            'DATA_AUGMENTOR': _plain(gcfg.DATA_CONFIG.DATA_AUGMENTOR),
        },
        'OPTIMIZATION': {'NUM_EPOCHS': B_EPOCH},
        'SELF_TRAIN': {
            'TEACHER_IS_STUDENT': False,
            'FROZEN_TEACHER_SINGLE_PASS': True,
            'EPOCH_FOLLOWS': 'target',
            'TEACHER_ADABN': {'FRAMES': 1000, 'MIX': 1.0, 'BATCH_SIZE': 4},
            'UPDATE_PSEUDO_LABEL': [0],
            'UPDATE_PSEUDO_LABEL_INTERVAL': 2,
            'INIT_PS': None,
            'SRC': {'USE_DATA': False, 'USE_GRAD': False, 'LOSS_WEIGHT': 1.0, 'BACKWARD_TOGETHER': True},
            'TAR': {'USE_DATA': True, 'LOSS_WEIGHT': 1.0, 'BACKWARD_TOGETHER': True},
            'PROG_AUG': {'ENABLED': False, 'UPDATE_AUG': [5, 10, 15], 'SCALE': 1.1},
            'MEMORY_ENSEMBLE': {'ENABLED': False, 'NAME': 'nms_ensemble', 'IOU_THRESH': 0.1,
                                'NMS_CONFIG': {'NMS_TYPE': 'nms_gpu', 'MULTI_CLASSES_NMS': False, 'NMS_PRE_MAXSIZE': 512,
                                               'NMS_POST_MAXSIZE': 100, 'NMS_THRESH': 0.1},
                                'MEMORY_VOTING': {'ENABLED': False, 'IGNORE_THRESH': 2, 'RM_THRESH': 3}},
            'MODEL_TEACHER': teacher,
        },
    }
    head = (
        "# experiments_md 20261003_05 §5 item 59b (design fixed 2026-11-27 08:15 JST, before any number). ANALYSIS on item 48's\n"
        "# forward pair: a label-free counterpart of item 58's G. P_K (pass it as --pretrained_model AND --pretrained_model_teacher)\n"
        "# is fine-tuned on KITTI testing clouds (7,518 frames, no labels exist, never seen by P_K) with pseudo-labels from a frozen\n"
        "# P_K + global AdaBN teacher (statistics from the same clouds), one pass. G's world augmentation, no PROG_AUG, no memory\n"
        "# ensemble, target only, 10 epochs x 1,253 iterations = 12,530 steps (G 12,380). Scored on KITTI val.\n"
        "# NOT RUNNABLE ALONE: SCORE_THRESH / NEG_THRESH come from rule A per seed (job P0) and live in the leaf configs.\n"
        "# GENERATED by analysis/item59_bn_split.py base_config (MODEL_TEACHER is P_K's resolved MODEL); do not edit by hand.\n"
        "_BASE_CONFIG_: %s\n\n" % EVAL_CFG)
    return head + yaml.safe_dump(body, sort_keys=False, default_flow_style=None, width=120)


def cmd_base_config(out=None):
    dst = Path(out) if out else TOOLS / B_BASE
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(base_config_text())
    print('wrote', dst)


def p0_dir(seed):
    return OUT / ('p0_s%d' % seed)


def rule_a(ps):
    """Rule A per class on a floor pass (ps_label_e0.pkl contents), exactly as plateau_neg_thresh.main does it."""
    import numpy as np
    sys.path.insert(0, str(TOOLS / 'analysis' / 'pseudo_label_threshold'))
    from plateau_neg_thresh import RANGE, plateau_neg
    b = np.concatenate([np.asarray(v['gt_boxes']).reshape(-1, 9) for v in ps.values()])
    b = b[(np.abs(b[:, 0]) <= RANGE) & (np.abs(b[:, 1]) <= RANGE)]
    out = {}
    for ci, name in enumerate(CLASSES, start=1):
        s = b[np.abs(b[:, 7]) == ci, 8]
        neg, mode, counts, m = plateau_neg(s)
        out[name] = dict(neg=round(neg, 3), mode=round(mode, 3), boxes=int(len(s)))
    return out


def cmd_cuts():
    import pickle
    res = {}
    for seed in (666, 667):
        ps = pickle.load(open(p0_dir(seed) / 'ps_label_e0.pkl', 'rb'))
        res[str(seed)] = dict(frames=len(ps), rule_a=rule_a(ps))
        print('s%d: %d frames; %s' % (seed, len(ps), res[str(seed)]['rule_a']))
    json.dump(res, open(OUT / 'cuts.json', 'w'), indent=1)
    print('wrote', OUT / 'cuts.json')


def leaf_text(arm, seed, cuts):
    v = [cuts[str(seed)]['rule_a'][c]['neg'] for c in CLASSES]
    lines = [
        "# experiments_md 20261003_05 §5 item 59b arm %s, seed %d (design fixed 2026-11-27 08:15 JST). ANALYSIS. GENERATED by" % (
            {'bn': 'b_bn (BatchNorm only)', 'all': 'b_all (every weight)'}[arm], seed),
        "# analysis/item59_bn_split.py configs from /home/koyama/data/item59/cuts.json (rule A on P0's floor pass of THIS seed's",
        "# teacher, P_K %d + AdaBN, over KITTI testing; label-free). SCORE_THRESH = NEG_THRESH: no ignore band (T05 item 59b)." % PK[seed],
        "_BASE_CONFIG_: %s" % B_BASE,
        "",
    ]
    if arm == 'bn':
        lines += ["OPTIMIZATION:", "    TRAIN_ONLY_BN: True", ""]
    lines += ["SELF_TRAIN:",
              "    SCORE_THRESH: [%s]" % ', '.join('%.3f' % x for x in v),
              "    NEG_THRESH: [%s]" % ', '.join('%.3f' % x for x in v), ""]
    return '\n'.join(lines)


def leaf_path(arm, seed):
    return '%s/centerpoint-item59-b-%s-s%d.yaml' % (B_DIR, arm, seed)


def cmd_configs():
    cuts = json.load(open(OUT / 'cuts.json'))
    for seed in (666, 667):
        for arm in ROWS_B:
            dst = TOOLS / leaf_path(arm, seed)
            dst.write_text(leaf_text(arm, seed, cuts))
            print('wrote', dst)


# ---------------------------------------------------------------- manifest and readings
def parse_jobs(args):
    out = {}
    for a in args:
        k, v = a.split('=')
        m = re.match(r'(bn|all)(\d+)$', k)
        out[(m.group(1), int(m.group(2)))] = int(v)
    return out


def manifest(jobs):
    from item42_collect import read_log
    rows = []
    for seed in (666, 667):
        for arm, name in ROWS_A.items():
            d = result_dir(arm, seed)
            got = read_log((d, 'log_rsadabn_*.txt', None))
            rows.append(dict(group='item59', row=name, seed=seed, job=None, result=str(d / 'result.pkl'),
                             logged_bev=got['bev07'] if got else None,
                             status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    if jobs:
        import item38_kitti_rings as i38
        for (arm, seed), job in sorted(jobs.items()):
            r = i38.resolve('item59', ROWS_B[arm], seed, job, 'epoch_%d' % B_EPOCH)
            if r['status'] == 'ok' and not Path(r['result']).exists():
                r['status'] = 'result.pkl missing'
            rows.append(r)
    for r in json.load(open(REF_57)):
        if r['row'] in ('P_RS1', 'L_ada', 'O_ada'):
            rows.append(dict(r, group='item59'))
    for r in json.load(open(REF_58)):
        if r['row'] == 'P_BN_G':
            rows.append(dict(r, group='item59'))
    return rows


def rs1_check(rows, seed):
    """Procedure check: this item's RS-1 rerun reproduces item 57's RS-1 pooled BEV within TOL_RS1."""
    a = [r for r in rows if r['row'] == ROWS_A['rs1'] and r['seed'] == seed]
    b = [r for r in rows if r['row'] == 'P_RS1' and r['seed'] == seed]
    if len(a) != 1 or len(b) != 1 or 'pooled' not in a[0] or 'pooled' not in b[0]:
        return False
    return abs(a[0]['pooled']['bev'] - b[0]['pooled']['bev']) <= TOL_RS1


def cmd_read(path):
    import item58_bn_bound as m58
    rows = json.load(open(path))['rows']
    for r in rows:
        print('%-10s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    ok = {s: rs1_check(rows, s) for s in (666, 667)}
    print('\nprocedure: RS-1 rerun within %.2f of item 57: %s' % (TOL_RS1, ok))
    g = {s: m58.per_seed(rows, s, 'P_BN_G') for s in (666, 667)}
    for name in list(ROWS_A.values())[1:] + list(ROWS_B.values()):
        if not any(r['row'] == name for r in rows):
            continue
        seeds = [m58.per_seed(rows, s, name) if (ok[s] or name in ROWS_B.values()) else None for s in (666, 667)]
        print('\n== %s: %s' % (name, m58.reading(seeds)))
        for s, d in zip((666, 667), seeds):
            if d is None:
                print('  seed %d: not read' % s)
                continue
            share = d['delta'] / g[s]['delta'] if g[s] and g[s]['delta'] else None
            print('  seed %d: s_x far %.2f / %.2f, delta %+.2f, R %.2f, share of G\'s delta %s; near %+.2f / %+.2f vs P_RS1' % (
                s, d['s_x'][2], d['s_x'][3], d['delta'], d['R'], '%.2f' % share if share is not None else '-',
                d['s_adabn'][0] - d['s_x'][0], d['s_adabn'][1] - d['s_x'][1]))


def main():
    a = sys.argv[1:]
    if a[:1] == ['make']:
        return cmd_make([int(x) for x in a[1:]] or [666, 667])
    if a[:1] == ['runs'] and len(a) == 2:
        for arm, ck, t in runs(int(a[1])):
            print(arm, ck, t)
        return
    if a[:1] == ['base_config']:
        return cmd_base_config(a[2] if len(a) == 3 and a[1] == '--out' else None)
    if a[:1] == ['cuts']:
        return cmd_cuts()
    if a[:1] == ['configs']:
        return cmd_configs()
    if a[:1] == ['manifest'] and len(a) >= 2:
        rows = manifest(parse_jobs(a[3:]) if len(a) > 2 and a[2] == '--jobs' else {})
        json.dump(rows, open(a[1], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r.get('job'), r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['read'] and len(a) == 2:
        return cmd_read(a[1])
    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
