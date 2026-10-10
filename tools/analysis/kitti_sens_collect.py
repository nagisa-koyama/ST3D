"""Collect the KITTI target-oracle degradation results (experiments_md 20261011_02) from job logs: per arm, Car BEV / 3D
AP_R40 (easy / moderate / hard) from its own evaluation log (found by its eval tag), and its W&B run id from the job's
error log (one W&B run per arm, in arm order). Prints a markdown table with D for each count-matched pair. Text only.

    python3 analysis/kitti_sens_collect.py <jobid>
"""
import glob
import re
import sys

ARMS = ('asscored ratio075 ratio050 ratio025 ringrows2 ringrandom2 ringrows3 ringrandom3 ringrows4 ringrandom4 '
        'ringcols2 ringrandomcols2 ringcols4 ringrandomcols4 ringazbin0p332 ringrandomazbin0p332 ringpatternhdl32e '
        'elmax0 elmaxm3 elminm14 elminm10 elminm17p6').split()
PAIRS = [('ringrows2', 'ringrandom2'), ('ringrows3', 'ringrandom3'), ('ringrows4', 'ringrandom4'),
         ('ringcols2', 'ringrandomcols2'), ('ringcols4', 'ringrandomcols4'), ('ringazbin0p332', 'ringrandomazbin0p332')]


def car_r40(text):
    m = re.findall(r'Car AP_R40@0\.70, 0\.70, 0\.70:\s*\nbbox AP:[^\n]*\nbev  AP:([\d.]+), ([\d.]+), ([\d.]+)\n3d   AP:([\d.]+), ([\d.]+), ([\d.]+)', text)
    return [float(x) for x in m[-1]] if m else None


def main(job):
    err = open(glob.glob(f'logs/error_{job}_*.txt')[0]).read()
    runs = re.findall(r'Run data is saved locally in \S*run-\d+_\d+-([a-z0-9]{8})', err)
    res = {}
    for a in ARMS:
        logs = sorted(glob.glob(f'../output/**/kittisens_{a}/log_eval_*.txt', recursive=True))
        if logs:
            r = car_r40(open(logs[-1]).read())
            if r:
                res[a] = r
    done = [a for a in ARMS if a in res]
    assert len(runs) >= len(done), (len(runs), len(done))
    wb = dict(zip(ARMS, runs))
    base = res.get('asscored')
    print('| point cloud (manipulation) | model | Car BEV e / m / h | Car 3D e / m / h | BEV mod change | W&B run |')
    print('|---|---|---|---|---|---|')
    for a in done:
        r = res[a]
        ch = '' if base is None or a == 'asscored' else '%+.2f' % (r[1] - base[1])
        print(f'| {a} | KITTI oracle 26843 | {r[0]:.2f} / {r[1]:.2f} / {r[2]:.2f} | {r[3]:.2f} / {r[4]:.2f} / {r[5]:.2f} | {ch} | '
              f'[{wb.get(a, "?")}](https://wandb.ai/nagisa/st3d/runs/{wb.get(a, "?")}) |')
    print()
    for s, c in PAIRS:
        if s in res and c in res:
            print(f'D {s} - {c}: BEV mod {res[s][1] - res[c][1]:+.2f}, 3D mod {res[s][4] - res[c][4]:+.2f}')


if __name__ == '__main__':
    main(sys.argv[1])
