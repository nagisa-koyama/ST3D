"""Build manifest.yaml from the result tables of the reports (experiments_md/20261003_02 §6). Host python.

    python3 analysis/viz/build_manifest.py

For every table row that names a job (a `job` column, else a job number in the first cell) and links a
W&B run: resolve the run directory, its entry point and the checkpoint behind the tabled number (the
`--ckpt` of an eval-only rescoring, else the training run's last epoch), and the target split its
density calibration measured (`test` before ST3D 20d1fef, `train` after; taken from the TRAINING
run's commit). One entry per job; the first table that names a job wins, except that an entry with a
W&B run replaces one without. Hand overrides live in OVERRIDES.
"""
import json
import re
import subprocess
from pathlib import Path

EXP = Path('/home/koyama/code/experiments_md')
ST3D = Path('/home/koyama/code/ST3D')
REPORTS = ['20261001_02_all_results_ranked_by_importance.md',
           '20261002_01_novelty_controls_thinning_is_generic_accumulation_is_mostly_default.md',
           '20261003_01_nuscenes_to_waymo_second_sparse_to_dense_pair.md',
           '20261001_03_pandaset_s3_s4_completion_plan.md']
WANDB_DIRS = [Path('/home/koyama/data/wandb'), ST3D / 'tools' / 'wandb']
JOB = re.compile(r'\b(2[5-7]\d{3})\b')
RUN = re.compile(r'wandb\.ai/nagisa/st3d/runs/([a-z0-9]{8})')
CALIB_FIX = '20d1fef'
# job -> fields that replace or add to the parsed ones
OVERRIDES = {
    '27171': dict(row='S2 · Pseudo-labelling from the target-free base 26814', run='mw1abaqq', ap='80.69 / 48.29'),
    # '27220': dict(row='S2 · Pseudo-labelling from the shrinking-prior base 26536', run='688kpq24'),  # still training
    '26970': dict(row='LiDAR Distillation, KITTI 64 → 32 beams (rerun, rings fixed)', run='6qen3mqi', ap='9.04 / 3.24',
                  report='20261001_02'),
    '26912': dict(row='Oracle, shrinking [0.75, 1.00]'),
}
SKIP = {'26391'}   # LiDAR Distillation with wrong rings: current code can only draw the fixed version (26970)


def run_dir(rid):
    for root in WANDB_DIRS:
        hits = sorted(root.glob('run-*-%s' % rid))
        if hits:
            return hits[-1]
    return None


def meta(d):
    f = d / 'files' / 'wandb-metadata.json'
    return json.load(open(f)) if f.exists() else {}


def cells(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def parse(report):
    out, section = [], ''
    lines = (EXP / report).read_text().split('\n')
    for i, line in enumerate(lines):
        if line.startswith('#'):
            section = line.lstrip('#').strip()
        if not (line.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|$', lines[i + 1].strip())):
            continue
        head = [c.lower() for c in cells(line)]
        jcol = head.index('job') if 'job' in head else None
        apcol = next((k for k, h in enumerate(head) if 'bev' in h), None)
        rowcol = next((k for k, h in enumerate(head) if h in ('row', 'source', 'checkpoint (training)', 'checkpoint')), None)
        k = i + 2
        while k < len(lines) and lines[k].startswith('|'):
            cs = cells(lines[k])
            jobs = JOB.findall(cs[jcol] if jcol is not None and jcol < len(cs) else cs[0])
            runs = RUN.findall(lines[k])
            if not jobs and runs and jcol is None:
                jobs = runs[:1]                      # a table keyed by W&B run only (Table E): the run id is the key
            if jobs:
                pairs = list(zip(jobs, runs)) if len(jobs) == len(runs) else [(jobs[0], runs[0] if runs else None)]
                ap = re.sub(r'\*', '', cs[apcol]) if apcol is not None and apcol < len(cs) else ''
                parts = [x.strip() for x in re.split(r' · | / (?=\d+\.\d+ / )', ap)] if len(pairs) > 1 else [ap]
                for n, (j, r) in enumerate(pairs):
                    out.append(dict(job=j, run=r, report=report[:11], section=section[:40],
                                    row=re.sub(r'[*`]', '', cs[rowcol]) if rowcol is not None and rowcol < len(cs) else '',
                                    ap=parts[n] if len(parts) == len(pairs) else ap))
            k += 1
    return out


def resolve(e):
    d = run_dir(e['run'])
    if d is None:
        e['error'] = 'no run directory for %s' % e['run']
        return e
    m = meta(d)
    args, code = m.get('args', []), m.get('codePath', '')
    e['entry'] = Path(code).name
    if '--ckpt' in args and e['entry'] == 'test.py':      # train.py's --ckpt means RESUME, not the scored model
        ck = args[args.index('--ckpt') + 1]
        mm = re.search(r'run-[\d_]+-([a-z0-9]{8})/files/ckpt/checkpoint_epoch_(\d+)\.pth', ck)
        if not mm:                                    # a checkpoint copied outside any W&B run (Table E)
            e['ckpt'] = ck.replace('/storage/', '/home/koyama/data/')
            return e
        e['ckpt'] = '%s:%s' % mm.groups()
        train = mm.group(1)
    else:
        eps = sorted(int(p.stem.split('_')[-1]) for p in (d / 'files' / 'ckpt').glob('checkpoint_epoch_*.pth'))
        if not eps:
            e['error'] = 'no checkpoint in %s' % d
            return e
        e['ckpt'] = '%s:%d' % (e['run'], eps[-1])
        train = e['run']
    td = run_dir(train)
    commit = meta(td).get('git', {}).get('commit') if td else None
    if commit:
        after = subprocess.run(['git', '-C', str(ST3D), 'merge-base', '--is-ancestor', CALIB_FIX, commit],
                               capture_output=True).returncode == 0
        e['calib_split'] = 'train' if after else 'test'
        e['commit'] = commit[:7]
        # Before ST3D 2e5ae50 a cone-restricted calibration measured the SOURCE with its training
        # augmentation on (world rotation brings side/rear returns into the cone); after, off.
        e['cone_aug_on'] = subprocess.run(['git', '-C', str(ST3D), 'merge-base', '--is-ancestor', '2e5ae50', commit],
                                          capture_output=True).returncode != 0
    return e


def main():
    entries = {}
    for rep in REPORTS:
        for e in parse(rep):
            if e['job'] not in entries or (entries[e['job']]['run'] is None and e['run']):
                entries[e['job']] = e
    for j, o in OVERRIDES.items():
        entries.setdefault(j, dict(job=j, report='', section='', row='', ap='', run=None)).update(o)
    rows = []
    for j in sorted(entries):
        e = entries[j]
        if j in SKIP:
            print('skip %s: %s' % (j, 'not reproducible with current code (see SKIP)'))
            continue
        if not e.get('run'):
            print('skip %s: no W&B run in any table (%s)' % (j, e['row'][:50]))
            continue
        e = resolve(e)
        rows.append(e)
        print('%s %-9s %-14s %-6s %-16s %s' % (j, e['run'], e.get('entry', '?'), e.get('calib_split', '?'),
                                             e.get('ckpt', e.get('error', '')), e['row'][:60]))
    with open(Path(__file__).resolve().parent / 'manifest.yaml', 'w') as f:
        f.write('# Generated by build_manifest.py from the report tables; edit OVERRIDES there, not here.\n')
        for e in rows:
            f.write('- %s\n' % json.dumps({k: v for k, v in e.items() if v not in (None, '')}, ensure_ascii=False))
    print(len(rows), 'entries')


if __name__ == '__main__':
    main()
