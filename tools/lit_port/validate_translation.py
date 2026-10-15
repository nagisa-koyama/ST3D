"""Completeness check of the full LiT translation (experiments_md 20261015_02 §7), before the row may train. Fails loudly (exit 1)
on ANY problem; there is no fallback. Plain numpy.
    python3 -I lit_port/validate_translation.py <root> <plan.json> <out.json>
Checks: every plan sequence has done/<seq>.gpu and .sim and no failed/ marker; its sim directory holds EXACTLY the sampled
sample_idx files (the frames WaymoDataset will read at SAMPLED_INTERVAL 2), each loadable by the training hook's own
`load_translated` (shape, finite, frame_index); the totals match the plan (79,041 declared). Also the per-frame point count
distribution (min / quantiles) and the number of empty frames."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pcdet.datasets.waymo.waymo_lit import load_translated  # noqa: E402


def main(root, plan, out):
    root = Path(root)
    plan = json.load(open(plan))
    problems, counts, empty, n = [], [], 0, 0
    failed = sorted(p.name for p in (root / 'failed').glob('*'))
    if failed:
        problems.append(f'failed markers: {failed[:5]} ({len(failed)})')
    for s in plan['sequences']:
        nm = s['name']
        for st in ('gpu', 'sim'):
            if not (root / 'done' / f'{nm}.{st}').exists():
                problems.append(f'{nm}: no done.{st}')
        d = root / 'sim_nuscenes' / nm
        have = sorted(int(p.stem) for p in d.glob('*.npz')) if d.is_dir() else []
        if have != s['sampled']:
            problems.append(f'{nm}: {len(have)} files, plan {len(s["sampled"])}')
            continue
        for k in s['sampled']:
            try:
                p = load_translated(root / 'sim_nuscenes', nm, k)
            except Exception as e:  # noqa: BLE001
                problems.append(f'{nm} {k}: {e}')
                continue
            counts.append(len(p)); empty += len(p) == 0; n += 1
    c = np.array(counts) if counts else np.zeros(1)
    res = dict(sequences=len(plan['sequences']), frames_loaded=n, plan_total_sampled=plan['total_sampled'],
               empty_frames=int(empty), points_min=int(c.min()), points_quantiles={q: float(np.quantile(c, q / 100)) for q in (1, 25, 50, 75, 99)},
               problems=problems[:50], n_problems=len(problems))
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res, indent=1))
    if problems or n != plan['total_sampled']:
        print('VALIDATION FAILED')
        sys.exit(1)
    print('VALIDATION OK')


if __name__ == '__main__':
    main(*sys.argv[1:4])
