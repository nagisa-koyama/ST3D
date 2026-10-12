"""Build the row manifest for the IoU 0.7 / 0.5 rescoring of the paper's cross-domain rows (experiments_md 20261013_02).
Text only (reads job logs): for each job, the LAST evaluation it logged - result directory, config, W&B run, and the Car
(and Pedestrian) AP_R40 BEV / 3D it printed - so the rescoring at 0.7 can be checked against the job's own number.

    python3 analysis/multiiou_manifest.py rows.tsv > manifest.json
rows.tsv: setting <TAB> label <TAB> job <TAB> dataset (nuscenes|kitti|waymo|pandaset) <TAB> ledger "BEV/3D" <TAB> extra
"""
import glob
import json
import re
import sys

LOGS = '/home/koyama/code/ST3D/tools/logs/'


def find_eval(job, pick=None):
    """The job's evaluation whose result directory contains `pick` (the last one if `pick` is None): result directory,
    W&B run (the run directory in the result path for an in-run evaluation, else the evaluation's own run), config, and
    the Car / Pedestrian AP_R40 BEV / 3D it printed."""
    found = None
    for f in sorted(glob.glob(f'{LOGS}error_{job}_*.txt') + glob.glob(f'{LOGS}output_{job}_*.txt')):
        txt = open(f, errors='replace').read()
        ends = [m.end() for m in re.finditer(r'Result is save to ', txt)]
        prev = 0
        for pos in ends:
            rdir = txt[pos:txt.index('\n', pos)].strip().replace('/storage/', '/home/koyama/data/')
            block = txt[prev:pos]
            prev = pos
            if pick is not None and pick not in rdir:
                continue
            car = re.findall(r'Car AP_R40@0\.70, 0\.70, 0\.70:\s*\nbbox AP:[^\n]*\nbev  AP:([\d.]+), ([\d.]+), ([\d.]+)\n3d   AP:([\d.]+), ([\d.]+), ([\d.]+)', block)
            ped = re.findall(r'Pedestrian AP_R40@0\.50, 0\.50, 0\.50:\s*\nbbox AP:[^\n]*\nbev  AP:([\d.]+), ([\d.]+), ([\d.]+)\n3d   AP:([\d.]+), ([\d.]+), ([\d.]+)', block)
            inrun = re.search(r'run-\d{8}_\d{6}-([a-z0-9]{8})/files/eval', rdir)
            own = re.findall(r'Run data is saved locally in \S*run-\d{8}_\d{6}-([a-z0-9]{8})', txt[:pos])
            upto = txt[:pos]   # the config logged BEFORE this evaluation (a multi-evaluation job logs several)
            cfgs = re.findall(r'--cfg_file[ =]([^\s\'",]+)', upto) or re.findall(r"cfg_file\s+(cfgs/\S+\.yaml)", upto)
            found = {'log': f, 'result_dir': rdir, 'wandb': inrun.group(1) if inrun else (own[-1] if own else None),
                     'cfg': cfgs[-1] if cfgs else None,
                     'car_bev_3d_mod': [float(car[-1][1]), float(car[-1][4])] if car else None,
                     'ped_bev_3d_mod': [float(ped[-1][1]), float(ped[-1][4])] if ped else None}
    return found


def main(path):
    out = []
    for line in open(path):
        if not line.strip() or line.startswith('#'):
            continue
        setting, label, job, ds, ledger, *extra = [x.strip() for x in line.rstrip('\n').split('\t')]
        x = extra[0] if extra else ''
        pk = re.search(r'pick=(\S+?)(;|$)', x)
        e = find_eval(job, pk.group(1) if pk else None) or {}
        m = re.search(r'result=(\S+)', x)
        if m:   # a result named in the row (PandaSet rows: from pandaset_eval_protocols.ROWS)
            e['result_dir'] = m.group(1).rsplit('/', 1)[0]
        if ds == 'pandaset':   # the logged value is as scored; the ledger value is rule A
            e['car_bev_3d_mod'] = e['ped_bev_3d_mod'] = None
        out.append({'setting': setting, 'label': label, 'job': job, 'dataset': ds, 'ledger': ledger,
                    'extra': extra[0] if extra else '', **e})
    json.dump(out, sys.stdout, indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
