"""Equivalence of ray-drop-during-cast and ray-drop-as-filter (experiments_md 20261017_01 §3): recast N sampled frames of one
sequence with the worker's own `sim_sequence(drop=...)` and compare with the stored no-drop frames filtered by
`ray_drop.keep_mask`. Also checks the drop origin (poses L2V translation) equals the Waymo TOP extrinsic used by the check.
Runs INSIDE the LiT container:  lit_port/lit_exec.sh python -I lit_port/check_ray_drop_equiv.py <root> <plan> <ray_drop.json> <calib.pkl> [nseq=2] [nframes=6]"""
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import lit_worker as W  # noqa: E402
import ray_drop  # noqa: E402


def main(root, plan, rdj, calib, nseq=2, nfr=6):
    plan = json.load(open(plan)); drop = json.load(open(rdj)); calib = pickle.load(open(calib, 'rb'))
    a = types.SimpleNamespace(root=root, lit='/home/koyama/code/lit', scene_tmp=str(Path(root) / 'scene_tmp'), engine='fast')
    paths = W.lit_setup(a, [s['name'] for s in plan['sequences']])
    bad = 0
    for s in plan['sequences'][7:7 + nseq]:
        n = s['name']
        frames = s['sampled'][::max(1, len(s['sampled']) // nfr)][:nfr]
        new = W.sim_sequence(a, paths, s, 'fast', frames=frames, drop=drop)
        z = np.load(Path(root) / 'poses' / f'{n}.npz')
        e = np.asarray(calib[n]['extrinsic'], np.float64).reshape(4, 4)[:3, 3]
        print(n, 'origin L2V vs extrinsic max diff', float(np.abs(z['L2V'][:3, 3] - e).max()))
        assert np.abs(z['L2V'][:3, 3] - e).max() < 1e-3
        for k in frames:
            old = np.load(Path(root) / 'sim_nuscenes' / n / ('%04d.npz' % k))['local_points']
            exp = old[ray_drop.keep_mask(old, z['L2V'][:3, 3], drop['p'], n, k)]
            same = new[k].local_points.shape == exp.shape and np.array_equal(new[k].local_points, exp)
            print(f'  frame {k}: stored {len(old)} -> filtered {len(exp)}; recast-with-drop {len(new[k].local_points)}; identical {same}')
            bad += not same
    print('EQUIVALENCE', 'FAILED' if bad else 'OK')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main(*sys.argv[1:5], *(int(x) for x in sys.argv[5:7]))
