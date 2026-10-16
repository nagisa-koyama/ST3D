"""Release the work-queue claims of dead Slurm jobs (experiments_md 20261015_02 §7). Plain python3, run on the node before a
worker job starts:  python3 lit_port/lit_reclaim.py <root> [--clear-failed]
A claim `claims/<stage>/<seq>` with no `done/<seq>.<stage>` marker whose owner job (`<role>.<jobid>.<tag>`) is not in the
queue is removed, so the next worker redoes the task. A claim of a running job is never touched."""
import shutil
import subprocess
import sys
from pathlib import Path


def alive(jobid):
    out = subprocess.run(['squeue', '-h', '-j', jobid, '-o', '%T'], capture_output=True, text=True).stdout.split()
    return any(s in ('RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING') for s in out)


def main():
    root = Path(sys.argv[1])
    clear_failed = '--clear-failed' in sys.argv
    cache, n = {}, 0
    for stage in ('gpu', 'sim', 'simrd'):
        for c in (root / 'claims' / stage).glob('*') if (root / 'claims' / stage).is_dir() else []:
            if (root / 'done' / f'{c.name}.{stage}').exists():
                continue
            owner = (c / 'owner').read_text() if (c / 'owner').exists() else 'x.0.x'
            jid = owner.split('.')[1]
            if jid not in cache:
                cache[jid] = jid != 'nojob' and alive(jid)
            if not cache[jid]:
                shutil.rmtree(c)
                n += 1
    if clear_failed:
        for f in (root / 'failed').glob('*'):
            f.unlink()
    print('released', n, 'stale claims')


if __name__ == '__main__':
    main()
