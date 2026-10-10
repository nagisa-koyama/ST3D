"""Collect logs/box_error/*.jsonl into the 20261011_01 tables (pure python, reads JSON only)."""
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / 'logs/box_error'
SRC = [('Waymo', 'waymo_oracle', 'bev', '3d'), ('nuScenes', 'nuscenes_oracle', 'bev', '3d'),
       ('PandaSet spin (rule A)', 'pandaset_spin_oracle', 'bev_rule_A', '3d_rule_A'),
       ('KITTI (moderate)', 'kitti_oracle', 'bev_moderate', '3d_moderate')]
rows = {}
for name, f, kb, k3 in SRC:
    for l in open(D / f'{f}.jsonl'):
        r = json.loads(l); rows.setdefault(r['arm'], {})[name] = (r[kb], r[k3])
order = ['base', 'roundtrip'] + [a for a in rows if a not in ('base', 'roundtrip')]
names = [s[0] for s in SRC]
print('| arm | ' + ' | '.join(names) + ' |'); print('|---|' + '---|' * len(names))
for a in order:
    cells = []
    for n in names:
        v = rows[a].get(n)
        b = rows['base'].get(n)
        if v is None: cells.append('-'); continue
        if a == 'base': cells.append(f'{v[0]:.2f} / {v[1]:.2f}')
        else: cells.append(f'{v[0]:.2f} / {v[1]:.2f} ({v[0]-b[0]:+.1f} / {v[1]-b[1]:+.1f})')
    print(f'| {a} | ' + ' | '.join(cells) + ' |')
