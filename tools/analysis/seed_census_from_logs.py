"""Per-epoch Car AP_R40 (IoU 0.7) of every evaluated epoch in a job's log, for the seed census of experiments_md 20261108_03
§2 A (T05 §5 item 39). Reads logs only; no dataset, no GPU.

For each job id it finds logs/error_<id>_*.txt (else output_<id>_*.txt), converts tqdm's carriage returns, and pairs each
"EPOCH n of <dataset> EVALUATION end" line with the Car AP_R40@0.70, 0.70, 0.70 block printed before it. BEV and 3D are
taken from the MODERATE column (the second value; on a non-KITTI target the three columns are identical). Also reported:
the code sha the job froze, its W&B run id (first "Run data is saved locally" line) and its logged seed.

    python analysis/seed_census_from_logs.py --out census.json 25743 31818 ...
"""
import argparse
import glob
import json
import re
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
CAR = 'Car AP_R40@0.70, 0.70, 0.70:'
END = re.compile(r'EPOCH (\d+) of (\w+) EVALUATION end')
RUN = re.compile(r'Run data is saved locally in \S*/run-[0-9_]+-([a-z0-9]+)')
SEED = re.compile(r'INFO\]\s+seed\s+(\d+)')
SHA = re.compile(r'=== code frozen at (\w+)')


def moderate(line):
    vals = line.split(':', 1)[1].split(',')
    return float(vals[1]) if len(vals) > 1 else float(vals[0])


def parse(lines):
    """Returns (epochs, run, seed) from one log's lines. epochs: list of {epoch, dataset, bev, 3d}."""
    out, run, seed, pending = [], None, None, None
    it = iter(lines)
    for line in it:
        if run is None and (m := RUN.search(line)):
            run = m.group(1)
        if seed is None and (m := SEED.search(line)):
            seed = int(m.group(1))
        if line.strip() == CAR:
            block = [next(it, '') for _ in range(3)]  # bbox, bev, 3d
            bev = [b for b in block if b.startswith('bev')]
            d3 = [b for b in block if b.startswith('3d')]
            if bev and d3:
                pending = (moderate(bev[0]), moderate(d3[0]))
        elif (m := END.search(line)):
            if pending is not None:
                out.append({'epoch': int(m.group(1)), 'dataset': m.group(2), 'bev': pending[0], '3d': pending[1]})
            pending = None
    return out, run, seed


def job_logs(job):
    for kind in ('error', 'output'):
        hits = sorted(glob.glob(str(TOOLS / 'logs' / ('%s_%s_*.txt' % (kind, job)))))
        if hits:
            return hits
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('jobs', nargs='+')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    res = {}
    for job in args.jobs:
        logs = job_logs(job)
        rec = {'logs': logs, 'epochs': [], 'run': None, 'seed': None, 'sha': None}
        for path in logs:
            text = Path(path).read_text(errors='replace').replace('\r', '\n').splitlines()
            ep, run, seed = parse(text)
            rec['epochs'] += ep
            rec['run'] = rec['run'] or run
            rec['seed'] = rec['seed'] if rec['seed'] is not None else seed
        out_log = sorted(glob.glob(str(TOOLS / 'logs' / ('output_%s_*.txt' % job))))
        if out_log:
            for line in Path(out_log[0]).read_text(errors='replace').splitlines()[:5]:
                if (m := SHA.search(line)):
                    rec['sha'] = m.group(1)
        res[job] = rec
        last = rec['epochs'][-1] if rec['epochs'] else None
        print(job, rec['run'], rec['seed'], rec['sha'], len(rec['epochs']), last)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
