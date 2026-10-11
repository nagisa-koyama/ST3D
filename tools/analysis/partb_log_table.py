"""Read a multi-evaluation eval-only job log (consecutive `analysis/eval_checkpoint.sh` runs in one Slurm job) into a table:
eval tag, W&B run id, Car BEV / 3D AP_R40 at IoU 0.7 and Pedestrian BEV / 3D at 0.5, moderate column. Used for part B of
experiments_md 20261011_07. Host-side, no dependencies beyond the standard library.

    python3 analysis/partb_log_table.py logs/output_<job>_<name>.txt logs/error_<job>_<name>.txt
"""
import re
import sys

text = ''
for f in sys.argv[1:]:
    text += open(f, errors='replace').read().replace('\r', '\n') + '\n'
# one block per evaluation: the output dir names the eval tag; AP blocks follow in the error log
tags = re.findall(r'eval/epoch_\d+/val/([A-Za-z0-9_.]+)', text)
seen = []
for t in tags:
    if t not in seen:
        seen.append(t)
runs = []
for m in re.finditer(r'Run data is saved locally in \S*run-\d{8}_\d{6}-([a-z0-9]{8})', text):
    if m.group(1) not in runs:
        runs.append(m.group(1))


def blocks(cls, iou):
    pat = re.compile(re.escape(f'{cls} AP_R40@{iou}') + r'[^\n]*\n[^\n]*\nbev\s+AP:([\d.]+), ([\d.]+), ([\d.]+)\n3d\s+AP:([\d.]+), ([\d.]+), ([\d.]+)')
    return [(float(m.group(2)), float(m.group(5))) for m in pat.finditer(text)]


car, ped = blocks('Car', '0.70, 0.70, 0.70'), blocks('Pedestrian', '0.50, 0.50, 0.50')
print(f'# {len(seen)} eval tags, {len(runs)} W&B runs, {len(car)} Car blocks, {len(ped)} Pedestrian blocks')
for k, tag in enumerate(seen):
    c = car[k] if k < len(car) else (float('nan'),) * 2
    p = ped[k] if k < len(ped) else (float('nan'),) * 2
    r = runs[k] if k < len(runs) else '?'
    print(f'{tag}\t{r}\t{c[0]:.2f}\t{c[1]:.2f}\t{p[0]:.2f}\t{p[1]:.2f}')
