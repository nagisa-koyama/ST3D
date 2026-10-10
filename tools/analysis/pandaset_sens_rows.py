"""Rows for `pandaset_eval_protocols.py --rows_json` from the PandaSet oracle-degradation evaluations
(experiments_md 20261011_03): every result.pkl written under extra_tag 20261011_pandaset_sens.

    python analysis/pandaset_sens_rows.py <spin|flash> <out.json>

spin rows are scored on the Pandar64 (device 0) over 360 deg; flash rows on the PandarGT (device 1) in its 60-deg cone,
each with rule A's point counts taken from the UNMANIPULATED cloud (pandaset_eval_protocols.gt_with_counts).
"""
import glob
import json
import os
import sys

which, out = sys.argv[1], sys.argv[2]
cfg = {'spin': ('centerpoint-pandaset-spin2spin', 'epoch_115', 0, False),
       'flash': ('centerpoint-pandaset-flash2flash', 'epoch_15', 1, True)}[which]
name, epoch, device, cone = cfg
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
pats = [f'{root}/da-ieee-access/{name}/20261011_pandaset_sens/eval/{epoch}/val/*/result.pkl',
        f'{root}/da-ieee-access-analysis/sensitivity-pandaset/{name}-*/20261011_pandaset_sens/eval/{epoch}/val/*/result.pkl']
rows = []
for pat in pats:
    for p in sorted(glob.glob(pat)):
        tag = p.split('/')[-2]                      # psSens_<which>_<arm>
        rows.append([tag, os.path.realpath(p), device, cone])
json.dump(rows, open(out, 'w'), indent=1)
print(f'{len(rows)} rows -> {out}')
for r in rows:
    print('  ', r[0])
