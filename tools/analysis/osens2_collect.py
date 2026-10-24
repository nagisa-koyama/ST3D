"""Collect OSENS-2 (experiments_md 20261025_03; ANALYSIS, target labels at evaluation only).

Per evaluation (eval tag osens2_<oracle>_<arm>, extra tag 20261025_osens2):
  - Car AP_R40 BEV / 3D at IoU 0.7 and 0.5 (easy / moderate / hard) from its own eval log (the official KITTI evaluator
    as test.py ran it: all objects on Waymo / nuScenes, real difficulties under the camera-FOV filter on KITTI);
  - its W&B run, found by the eval tag in tools/wandb/run-*/files/wandb-metadata.json;
  - beside AP, from its result.pkl: background false positives per frame at score >= 0.3 and Car recall, by
    20261017_13's matching and taxonomy (waymo_ranking_loss_fp_taxonomy.frame_detections, unchanged): Car detections in
    descending score, each takes the unmatched Car GT of highest BEV IoU above 0.7 (greedy); an FP is 'background' if
    it touches no Car GT (BEV IoU < 0.1) and no other-class GT (BEV IoU < 0.1, centre not inside). GT = every Car box
    of the infos (Waymo Vehicle, nuScenes car, KITTI Car), all ranges; other-class = every other annotated box
    (Waymo 'unknown' and KITTI 'DontCare' excluded). Recall at IoU 0.7 (0.5 beside) counting only detections with
    score >= 0.3, and the maximum recall over all detections. Background FPs are not split by points (no cloud read).
Readings (Car BEV R40 IoU 0.7; KITTI moderate): dAP = arm - as scored; 'sensitive' if dAP <= -2 (the design's rule).

    python analysis/osens2_collect.py --out /home/koyama/data/osens2/results_<date> [--workers 8] [--no_fp]
"""
import argparse
import glob
import json
import pickle
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS.parent, TOOLS, TOOLS / 'analysis'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EXTRA = '20261025_osens2'
ORACLE_DIR = {'waymo': 'centerpoint-sourceonly-waymo2waymo', 'nuscenes': 'centerpoint-sourceonly-nuscenes',
              'kitti': 'centerpoint-sourceonly-kitti2kitti-noros-zshift'}
INFOS = {'waymo': TOOLS.parent / 'data/waymo/waymo_infos_val.pkl',
         'nuscenes': TOOLS.parent / 'data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl',
         'kitti': TOOLS.parent / 'data/kitti/kitti_infos_val.pkl'}
CAR = {'waymo': 'Vehicle', 'nuscenes': 'car', 'kitti': 'Car'}
EXCLUDE = {'unknown', 'DontCare'}
SCORE = 0.3


def ap_block(text, iou):
    m = re.findall(r'Car AP_R40@0\.70, %s, %s:\s*\nbbox AP:[^\n]*\nbev  AP:([\d.]+), ([\d.]+), ([\d.]+)\n'
                   r'3d   AP:([\d.]+), ([\d.]+), ([\d.]+)' % (iou, iou), text)
    return [float(x) for x in m[-1]] if m else None


def eval_dirs():
    """{eval tag: eval dir} for every OSENS-2 evaluation on disk."""
    out = {}
    for d in glob.glob(str(TOOLS.parent / 'output' / '**' / EXTRA / 'eval' / 'epoch_*' / 'val' / 'osens2_*'), recursive=True):
        out[Path(d).name] = Path(d)
    return out


def wandb_runs():
    """{eval tag: [W&B run ids]} from the eval runs' metadata (test.py runs from tools/, so wandb/ is there)."""
    out = {}
    for f in glob.glob(str(TOOLS / 'wandb' / 'run-*' / 'files' / 'wandb-metadata.json')):
        try:
            args = json.load(open(f)).get('args', [])
        except Exception:
            continue
        if '--extra_tag' in args and args[args.index('--extra_tag') + 1] == EXTRA and '--eval_tag' in args:
            tag = args[args.index('--eval_tag') + 1]
            out.setdefault(tag, []).append(Path(f).parents[1].name.split('-')[-1])
    return out


def load_gt(oracle):
    infos = pickle.load(open(INFOS[oracle], 'rb'))
    gt = {}
    for inf in infos:
        if oracle == 'waymo':
            fid = inf['frame_id'] if 'frame_id' in inf else \
                inf['point_cloud']['lidar_sequence'] + '_%03d' % inf['point_cloud']['sample_idx']
            names = np.asarray(inf['annos']['name'])
            boxes = np.asarray(inf['annos']['gt_boxes_lidar'], np.float32).reshape(len(names), -1)[:, :7] \
                if len(names) else np.zeros((0, 7), np.float32)
        elif oracle == 'nuscenes':
            fid = Path(inf['lidar_path']).stem
            names = np.asarray(inf['gt_names'])
            boxes = np.asarray(inf['gt_boxes'], np.float32).reshape(len(names), -1)[:, :7] if len(names) \
                else np.zeros((0, 7), np.float32)
        else:
            fid = inf['point_cloud']['lidar_idx']
            a = inf['annos']
            names = np.asarray(a['name'])
            boxes = np.asarray(a['gt_boxes_lidar'], np.float32).reshape(-1, 7)
            assert (names[:len(boxes)] != 'DontCare').all() and (names[len(boxes):] == 'DontCare').all()
            names = names[:len(boxes)]
        car = names == CAR[oracle]
        other = ~car & ~np.isin(names, list(EXCLUDE))
        gt[str(fid)] = (boxes[car], boxes[other])
    return gt


GT = None


def fp_recall(args):
    oracle, tag, pkl = args
    from waymo_ranking_loss_fp_taxonomy import frame_detections, BG, TP
    res = pickle.load(open(pkl, 'rb'))
    n_gt = n_bg = 0
    hit3_7 = hit3_5 = hit_all7 = 0
    for r in res:
        gv, go = GT[str(r['frame_id'])]
        m = np.asarray(r['name']) == 'Car'
        b = np.asarray(r['boxes_lidar'], np.float32).reshape(len(m), -1)[:, :7] if len(m) else np.zeros((0, 7), np.float32)
        b, s = b[m], np.asarray(r['score'], np.float64)[m]
        o = frame_detections(b, s, gv, go, points=None)
        n_gt += len(gv)
        n_bg += int(((o['codes'] == BG) & (s >= SCORE)).sum())
        if len(gv):
            hit_all7 += int((o['gt_dt7'] >= 0).sum())
            hit3_7 += int(((o['gt_dt7'] >= 0) & (s[np.maximum(o['gt_dt7'], 0)] >= SCORE if len(s) else False)).sum())
            hit3_5 += int(((o['gt_dt5'] >= 0) & (s[np.maximum(o['gt_dt5'], 0)] >= SCORE if len(s) else False)).sum())
    nf = len(res)
    return tag, dict(frames=nf, car_gt=n_gt, bg_fp_per_frame=n_bg / nf, recall07_at03=hit3_7 / max(n_gt, 1),
                     recall05_at03=hit3_5 / max(n_gt, 1), max_recall07=hit_all7 / max(n_gt, 1))


def _init(gt):
    global GT
    GT = gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no_fp', action='store_true')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dirs, runs = eval_dirs(), wandb_runs()
    table = {}
    for tag, d in sorted(dirs.items()):
        logs = sorted(d.glob('log_eval_*.txt'))
        text = logs[-1].read_text() if logs else ''
        oracle = tag.split('_')[1]
        arm = tag[len('osens2_%s_' % oracle):]
        table[tag] = dict(oracle=oracle, arm=arm, ap07=ap_block(text, '0.70'), ap05=ap_block(text, '0.50'),
                          wandb=runs.get(tag, []), result=str(d / 'result.pkl') if (d / 'result.pkl').exists() else None)
    if not args.no_fp:
        for oracle in ORACLE_DIR:
            todo = [(oracle, t, v['result']) for t, v in table.items() if v['oracle'] == oracle and v['result']
                    and v['ap07'] is not None]
            if not todo:
                continue
            gt = load_gt(oracle)
            with Pool(min(args.workers, len(todo)), initializer=_init, initargs=(gt,)) as pool:
                for tag, fr in pool.imap_unordered(fp_recall, todo):
                    table[tag]['fp'] = fr
                    print(tag, fr, flush=True)
    (out / 'osens2_results.json').write_text(json.dumps(table, indent=1))
    # markdown, per oracle
    lines = []
    for oracle in ORACLE_DIR:
        rows = {v['arm']: v for v in table.values() if v['oracle'] == oracle and v['ap07']}
        if not rows:
            continue
        base = rows.get('asscored_a', {}).get('ap07')
        lines += ['', f'### {oracle}', '',
                  '| arm | Car BEV e / m / h | Car 3D e / m / h | dBEV (m) | d3D (m) | BEV / 3D IoU 0.5 (m) | bg FP / frame @0.3 | '
                  'recall 0.7 @0.3 | max recall 0.7 | W&B run |', '|' + '---|' * 10]
        for arm, v in rows.items():
            a, a5, fp = v['ap07'], v['ap05'], v.get('fp', {})
            d = '' if base is None else '%+.2f' % (a[1] - base[1])
            d3 = '' if base is None else '%+.2f' % (a[4] - base[4])
            wb = ', '.join(f'[{w}](https://wandb.ai/nagisa/st3d/runs/{w})' for w in v['wandb']) or '?'
            lines.append(f'| {arm} | {a[0]:.2f} / {a[1]:.2f} / {a[2]:.2f} | {a[3]:.2f} / {a[4]:.2f} / {a[5]:.2f} | {d} | {d3} | '
                         f'{a5[1]:.2f} / {a5[4]:.2f} | {fp.get("bg_fp_per_frame", float("nan")):.3f} | '
                         f'{fp.get("recall07_at03", float("nan")):.4f} | {fp.get("max_recall07", float("nan")):.4f} | {wb} |')
    (out / 'osens2_results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
