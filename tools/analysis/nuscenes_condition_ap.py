"""ENV E2: rain and night inside each nuScenes city (labelled analysis, descriptive; experiments_md 20261024_02).

Design FIXED in experiments_md T05 §5, entry "User 10-24 ~02:40", item 4 (commit 3cddbb7); applied here unchanged:
frames of nuScenes val tagged by their scene description (substrings `rain`, `night`, case-insensitive) and city
(`log.location`: boston-seaport = Boston, singapore-* = Singapore). Cells: clear day (neither tag), rain only, night
only, both. Models: B->B on Boston frames, S->S on Singapore frames, the nuScenes in-domain oracle 25743 on each
city's frames. Per model and city: AP(condition) - AP(clear day) for rain, night and both, with frames and GT cars
per cell and a 90% interval from 1,000 scene-level bootstrap resamples drawn within each cell. Reading: "lower under
the condition" only if the difference is <= -2 and the interval excludes 0; else level at this resolution. Cells
under 300 frames are reported, not read.

AP is the KITTI metric of the evaluation (Car BEV AP_R40 at IoU 0.7, "all objects"), recomputed on the frame subset
from the saved predictions through `kitti_frame_decomposition` (exactly the official evaluator on those frames).

    singularity exec --bind /home/koyama/data/:/storage <sif> python3 analysis/nuscenes_condition_ap.py \\
        --model 'B->B=<cfg>=<result.pkl>=<logged>=Boston' ... --out <json>

Run from ST3D/tools. CPU only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kitti_frame_decomposition as kfd  # noqa: E402  (imports _init_path first)

META = Path('../data/nuscenes/v1.0-trainval/v1.0-trainval')
CELLS = ('clear', 'rain', 'night', 'both')
MIN_FRAMES = 300
N_BOOT = 1000
GATE_TOL = 0.05


def frame_conditions(tokens):
    """sample token -> (scene token, city, cell, description), from scene.json / log.json / sample.json."""
    logs = {l['token']: l for l in json.load(open(META / 'log.json'))}
    scenes = {s['token']: s for s in json.load(open(META / 'scene.json'))}
    sample_scene = {s['token']: s['scene_token'] for s in json.load(open(META / 'sample.json'))}
    out = []
    for t in tokens:
        sc = scenes[sample_scene[t]]
        loc = logs[sc['log_token']]['location']
        assert loc == 'boston-seaport' or loc.startswith('singapore-'), loc
        city = 'Boston' if loc == 'boston-seaport' else 'Singapore'
        d = sc['description'].lower()
        cell = {(False, False): 'clear', (True, False): 'rain', (False, True): 'night', (True, True): 'both'}[
            ('rain' in d, 'night' in d)]
        out.append((sc['token'], city, cell, sc['description']))
    return out


def scene_groups(frame_idx, scene_of):
    groups = {}
    for f in frame_idx:
        groups.setdefault(scene_of[f], []).append(f)
    return [np.asarray(v, np.int64) for _, v in sorted(groups.items())]


def bootstrap_diff(fr, clear_groups, cond_groups, rng, n_boot=N_BOOT):
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        c = rng.integers(0, len(clear_groups), len(clear_groups))
        d = rng.integers(0, len(cond_groups), len(cond_groups))
        fc = np.concatenate([clear_groups[i] for i in c])
        fd = np.concatenate([cond_groups[i] for i in d])
        diffs[b] = fr.ap(fd) - fr.ap(fc)
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', action='append', required=True,
                    help="'<label>=<cfg>=<result.pkl>=<logged AP over its eval set>=<city>[,<city>]'")
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261024)
    a = ap.parse_args()
    official = kfd.load_official_eval()
    result = dict(models={}, n_boot=N_BOOT, seed=a.seed, min_frames=MIN_FRAMES)
    for spec in a.model:
        label, cfg, pkl, logged, cities = spec.split('=')
        t0 = time.time()
        cap = kfd.capture_eval_annos(cfg, pkl, official)
        diff = cap['car_bev_r40'] - float(logged)
        ok = abs(diff) <= GATE_TOL
        print(f'== {label}: frames {len(cap["gt"])}, recomputed {cap["car_bev_r40"]:.4f} vs logged {float(logged):.4f} '
              f'(diff {diff:+.4f}, {"PASS" if ok else "FAIL"}); {pkl}', flush=True)
        if not ok:
            print(f'{label}: the recomputation does not reproduce the logged AP; no cell is computed.', flush=True)
            sys.exit(3)
        fr = kfd.CarBevFrames(cap['gt'], cap['dt'], official, verify=True)
        assert abs(fr.ap() - cap['car_bev_r40']) < 1e-6
        cond = frame_conditions([g['token'] for g in cap['gt']])
        scene_of = [c[0] for c in cond]
        entry = dict(recomputed=cap['car_bev_r40'], logged=float(logged), pkl=pkl, cfg=cfg, cities={})
        for city in cities.split(','):
            rows = {}
            idx = {cell: np.array([f for f, c in enumerate(cond) if c[1] == city and c[2] == cell], np.int64)
                   for cell in CELLS}
            n_city = sum(len(v) for v in idx.values())
            assert n_city > 0, (label, city)
            for cell in CELLS:
                f = idx[cell]
                rows[cell] = dict(frames=int(len(f)), scenes=len(set(scene_of[i] for i in f)),
                                  gt_cars=int(fr.n_valid[f].sum()) if len(f) else 0,
                                  ap=fr.ap(f) if len(f) else None)
            city_ap = fr.ap(np.concatenate([idx[c] for c in CELLS]))
            clear_groups = scene_groups(idx['clear'], scene_of)
            for k, cell in enumerate(('rain', 'night', 'both')):
                r = rows[cell]
                if r['frames'] == 0:
                    r['reading'] = 'no frames'
                    continue
                rng = np.random.default_rng([a.seed, sum(map(ord, label + city)), k])
                diffs = bootstrap_diff(fr, clear_groups, scene_groups(idx[cell], scene_of), rng)
                d = r['ap'] - rows['clear']['ap']
                lo, hi = (float(x) for x in np.percentile(diffs, [5, 95]))
                r.update(diff=d, ci90=[lo, hi], boot_mean=float(diffs.mean()))
                if r['frames'] < MIN_FRAMES:
                    r['reading'] = f'reported, not read (< {MIN_FRAMES} frames)'
                elif d <= -2 and (hi < 0 or lo > 0):
                    r['reading'] = 'lower under the condition'
                else:
                    r['reading'] = 'level at this resolution'
            entry['cities'][city] = dict(frames=n_city, ap_city=city_ap, cells=rows)
            print(f'  {label} on {city} ({n_city} frames, AP over the city {city_ap:.2f}):', flush=True)
            for cell in CELLS:
                r = rows[cell]
                if r['frames'] == 0:
                    print(f'    {cell:6s} no frames', flush=True)
                    continue
                extra = '' if cell == 'clear' else (f'  diff {r["diff"]:+.2f}  90% [{r["ci90"][0]:+.2f}, '
                                                    f'{r["ci90"][1]:+.2f}]  -> {r["reading"]}')
                print(f'    {cell:6s} frames {r["frames"]:5d}  scenes {r["scenes"]:3d}  GT cars {r["gt_cars"]:6d}  '
                      f'AP {r["ap"]:.2f}{extra}', flush=True)
        result['models'][label] = entry
        print(f'  ({time.time() - t0:.0f} s)', flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(result, f, indent=1, default=float)
    print(f'wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
