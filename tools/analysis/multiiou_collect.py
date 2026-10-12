"""Collect the IoU 0.7 / 0.5 rescoring (experiments_md 20261013_02) into one markdown table per setting, plus the rows
whose IoU 0.7 value does not reproduce the ledger within 0.1. Text only.

    python3 analysis/multiiou_collect.py <manifest.json> <out_dir>
"""
import json
import sys
from pathlib import Path

SETTINGS = {
    'A': ('Table A - X -> nuScenes source-only', 'nuScenes val (all objects)'),
    'B': ('Table B - S1 Lyft -> nuScenes', 'nuScenes val (all objects)'),
    'C': ('Table C - S2 nuScenes -> KITTI', 'KITTI val (moderate, camera FOV)'),
    'D': ('Table D - S3 / S4 PandaSet, rule A', None),
    'W': ('nuScenes -> Waymo', 'Waymo val (all objects)'),
    'P': ('Waymo -> nuScenes, scan-pattern rows', 'nuScenes val (all objects)'),
    'Q': ('PandaSet -> nuScenes, scan-pattern rows', 'nuScenes val (all objects)'),
}


def model_domain(r):
    s, lab = r['setting'], r['label']
    if 'oracle' in lab:
        return {'A': 'nuScenes', 'C': 'KITTI', 'W': 'Waymo'}.get(s, 'PandarGT' if 'flash -> flash' in lab else 'Pandar64')
    return {'A': lab.split(' -> ')[0], 'B': 'Lyft', 'C': 'nuScenes', 'W': 'nuScenes', 'P': 'Waymo', 'Q': 'PandaSet Pandar64'}.get(
        s, 'Pandar64 (spin)' if lab.startswith('S3') else 'PandarGT (flash)')


def cloud_domain(r):
    if r['setting'] != 'D':
        return SETTINGS[r['setting']][1]
    x = r['extra']
    return ('PandarGT val, rule A, 60-deg cone' if 'device=1' in x else
            'Pandar64 val, rule A, ' + ('60-deg cone' if 'cone=1' in x else '360 deg'))


def load(out_dir, k, r, cls):
    p = Path(out_dir) / f"r{k:02d}_{r['setting']}_{r['job']}_{cls}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows[-1] if rows else None


def main(manifest, out_dir):
    m = json.load(open(manifest))
    bad = []
    for s, (title, _) in SETTINGS.items():
        rows = [(k, r) for k, r in enumerate(m) if r['setting'] == s]
        if not rows:
            continue
        ped = any('ped' in r['extra'].split(';') for _, r in rows)
        print(f'\n**{title}**\n')
        head = '| point cloud (evaluation) | model (training domain) | row | job | Car BEV / 3D @0.7 | Car BEV / 3D @0.5 |'
        if ped:
            head += ' Ped BEV / 3D @0.5 | Ped BEV / 3D @0.25 |'
        head += ' ledger @0.7 | W&B run |'
        print(head)
        print('|' + '---|' * (head.count('|') - 1))
        for k, r in rows:
            c = load(out_dir, k, r, 'Car')
            led = [float(x) for x in r['ledger'].split('/')]

            def pair(d, t):
                return '-' if d is None else f"{d[f'bev_moderate@{t}']:.2f} / {d[f'3d_moderate@{t}']:.2f}"
            line = f"| {cloud_domain(r)} | {model_domain(r)} | {r['label']} | {r['job']} | {pair(c, 0.7)} | {pair(c, 0.5)} |"
            if ped:
                p = load(out_dir, k, r, 'Pedestrian') if 'ped' in r['extra'].split(';') else None
                line += f" {pair(p, 0.5)} | {pair(p, 0.25)} |"
            wb = r.get('wandb') or '?'
            note = ' (checkpoint\'s training run)' if r['setting'] == 'D' and 'oracle' in r['label'] else ''
            line += f" {r['ledger']} | [{wb}](https://wandb.ai/nagisa/st3d/runs/{wb}){note} |"
            print(line)
            if c is None:
                bad.append((r, 'not scored'))
            elif abs(c['bev_moderate@0.7'] - led[0]) > 0.1 or abs(c['3d_moderate@0.7'] - led[1]) > 0.1:
                bad.append((r, f"@0.7 {c['bev_moderate@0.7']:.2f} / {c['3d_moderate@0.7']:.2f} vs ledger {r['ledger']}"))
    print('\n**Rows that do not reproduce the ledger at IoU 0.7 within 0.1:**\n')
    for r, why in bad:
        print(f"- {r['setting']} {r['job']} {r['label']}: {why}")
    if not bad:
        print('- none')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
