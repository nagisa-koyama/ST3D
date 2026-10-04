"""How much of the target's per-ring elevation mass lies where the raw (cached) source has NO points - a keep
probability cannot reach it - and the elevation JS restricted to the shared support, rule vs trained sampler.
Diagnoses a learned sampler that does not learn (experiments_md/20261003_04 section 16: unaccumulated Lyft
40-beam has 19-52% of nuScenes' elevation mass outside its support).

    python analysis/sampler_support_check.py <name> [<name> ...]   # reads /home/koyama/data/samplers/<name>{.npz,.cache.npz}
"""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import _init_path  # noqa
from pcdet.datasets.processor.point_sampler import LearnedPointSampler
from analysis.train_point_sampler import _points_from_feats

def js(p, q):
    p = p / p.sum(); q = q / q.sum(); m = (p + q) / 2
    f = lambda a: np.sum(a[a > 0] * np.log2(a[a > 0] / m[a > 0]))
    return 0.5 * f(p) + 0.5 * f(q)

for name in sys.argv[1:]:
    z = np.load(f'/home/koyama/data/samplers/{name}.cache.npz', allow_pickle=True)
    F, R, E, T = z['F'], z['R'], z['E'], z['T']
    s = LearnedPointSampler.load(f'/home/koyama/data/samplers/{name}.npz')
    pts = _points_from_feats(F)
    p_rule = 1 / (1 + np.exp(-s.rule_logit(pts))); p_tr = s.keep_probability(pts, F)
    print(f'== {name}')
    print('ring | target mass outside source support | JS full rule/trained | JS on shared support rule/trained')
    for k in range(T.shape[0]):
        m = R == k
        raw = np.bincount(E[m], minlength=T.shape[1]).astype(float)
        sup = raw > 0.001 * raw.sum()
        out = T[k][~sup].sum() / T[k].sum()
        hr = np.bincount(E[m], weights=p_rule[m], minlength=T.shape[1]); ht = np.bincount(E[m], weights=p_tr[m], minlength=T.shape[1])
        print(f'{k*10:2d}-{k*10+10:2d} m |  {out:5.2f} | {js(hr, T[k]):.3f} / {js(ht, T[k]):.3f} | {js(hr[sup], T[k][sup]):.3f} / {js(ht[sup], T[k][sup]):.3f}')
