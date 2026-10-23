"""ENV E1, two-seed readings (experiments_md 20261024_02 §9; design fixed in T05 §5, sub-entry "User 10-24 (received between
03:22 and 03:25)", experiments_md 8d96563). Reads the E1 output JSON of seed 666 and of seed 667
(scene_error_decomposition.py, each seed paired within itself) and applies, as fixed:
- the E1 rule to the MEAN E_bkg, E_loc, E_miss over the two seeds (S->B - B->B), unchanged (`e1_reading`);
- the control on the means (B->S - S->S): every mean E_x < 2 expected, else the decomposition is flagged unreliable;
- the scene terms T_B = (B->B) - (S->B) and T_S = (S->S) - (B->S) per seed and their mean: present if the mean >= 2, else
  absent. Per seed from the LOGGED Car BEV AP_R40 @0.7 (as 20261022_06 §6.2 read them); recomputed values beside.
Reported: per-seed values and the spread (half the difference between the seeds). No third seed.

    python3 analysis/env_e1_two_seed.py --seed666 <json> --seed667 <json> --out <json>      (from ST3D/tools, container)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene_error_decomposition import CATS, e1_reading  # noqa: E402

READ, CONTROL = 'S->B vs B->B', 'B->S vs S->S'


def scene_terms(r, key):
    g = r['gate']
    return {'T_B': g['B->B'][key] - g['S->B'][key], 'T_S': g['S->S'][key] - g['B->S'][key]}


def mean_spread(a, b):
    return {k: dict(seed666=a[k], seed667=b[k], mean=(a[k] + b[k]) / 2, spread=abs(a[k] - b[k]) / 2) for k in a}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed666', required=True)
    ap.add_argument('--seed667', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    r6, r7 = json.load(open(a.seed666)), json.load(open(a.seed667))
    out = {}
    e = mean_spread(r6['pairs'][READ]['E'], r7['pairs'][READ]['E'])
    c = mean_spread(r6['pairs'][CONTROL]['E'], r7['pairs'][CONTROL]['E'])
    e_mean = {x: e[x]['mean'] for x in CATS}
    c_mean = {x: c[x]['mean'] for x in CATS}
    out['E_read'] = dict(values=e, reading_on_means=e1_reading(e_mean))
    flag = [x for x in CATS if c_mean[x] >= 2]
    out['E_control'] = dict(values=c, verdict='every mean E_x < 2, as expected' if not flag else
                            f'mean E >= 2 for {", ".join(flag)}: decomposition flagged unreliable')
    for key in ('logged', 'recomputed'):
        t = mean_spread(scene_terms(r6, key), scene_terms(r7, key))
        for k in t:
            t[k]['reading'] = 'present' if t[k]['mean'] >= 2 else 'absent at this resolution'
        out[f'scene_terms_{key}'] = t
    out['delta_ap'] = {m: mean_spread(r6['models'][m]['delta'], r7['models'][m]['delta']) for m in r6['models']}
    out['ap'] = {m: mean_spread({'AP': r6['models'][m]['ap']['AP']}, {'AP': r7['models'][m]['ap']['AP']})['AP']
                 for m in r6['models']}

    print('== E1 two seeds (S->B - B->B), per seed / mean / spread ==')
    for x in CATS:
        v = e[x]
        print(f'  E_{x:4s} 666 {v["seed666"]:+.2f}  667 {v["seed667"]:+.2f}  mean {v["mean"]:+.2f}  spread {v["spread"]:.2f}')
    print(f'  reading on the means: {out["E_read"]["reading_on_means"]}')
    print('== control (B->S - S->S) ==')
    for x in CATS:
        v = c[x]
        print(f'  E_{x:4s} 666 {v["seed666"]:+.2f}  667 {v["seed667"]:+.2f}  mean {v["mean"]:+.2f}  spread {v["spread"]:.2f}')
    print(f'  {out["E_control"]["verdict"]}')
    for key in ('logged', 'recomputed'):
        print(f'== scene terms ({key} AP) ==')
        for k, v in out[f'scene_terms_{key}'].items():
            print(f'  {k} 666 {v["seed666"]:+.2f}  667 {v["seed667"]:+.2f}  mean {v["mean"]:+.2f}  spread {v["spread"]:.2f}'
                  f'  -> {v["reading"]}')
    print('== AP and dAP per model, per seed ==')
    for m in r6['models']:
        d = out['delta_ap'][m]
        print(f'  {m:5s} AP {out["ap"][m]["seed666"]:.2f} / {out["ap"][m]["seed667"]:.2f}  ' + '  '.join(
            f'dAP_{x} {d[x]["seed666"]:+.2f} / {d[x]["seed667"]:+.2f}' for x in CATS))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
