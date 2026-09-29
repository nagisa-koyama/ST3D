#!/usr/bin/env python3
"""Where can a job actually start right now, and how many CPUs should it ask for?

Runs on the HOST (needs scontrol/squeue, no torch). The rule it encodes, learned 2026-09-29: on
this cluster the binding resource is usually CPUs, not GPUs. Thirteen GPUs sat idle on node03 /
node11 / node12 / node61 while five 2-GPU jobs pended, because those nodes had 0-16 free CPUs and
each job asked for 20 (the per-GPU share of the smallest node, node61). Shrinking two of them to
12 and 10 CPUs started them within seconds. So: size CPUs to what the LOADER needs (a floor), then
ask for the most a currently free slot can give up to the share cap - not the cap itself.

    free_slots.py                             table of usable nodes: free CPUs / free GPUs
    free_slots.py --gpus 2 [--min 5 --max 10] recommend --cpus-per-task for a 2-GPU job; exit 0
                                              if it fits somewhere now, 1 if it will have to wait
    free_slots.py --adjust <sbatch flags...>  rewrite --cpus-per-task in the given flags to what
                                              fits now (never below --min per GPU); prints flags
    free_slots.py --fit-pending [--apply]     for MY pending jobs: would a smaller CPU request
                                              fit a free slot? --apply does the three scontrol
                                              updates (NumCPUs, MinCPUsNode, CPUsPerTask)

Partitions considered: a6000_ada, a6000, rtx8000 (a100 is broken at the node level, pro6000 is
Blackwell and cannot run the container). A node in DOWN/DRAIN is skipped. The account cap is
12 GPUs at once (QOS limit12); the table says how many of them are in use.

Floors, per GPU: 4 for a plain source-only row (loader ahead of the GPU at NUM_WORKERS 4:
<0.2% data-wait on KITTI/Lyft/nuScenes), 5 for self-training rows (three loaders plus the
calibration and pseudo-label passes, which are CPU work), 8 for accumulating PandaSet rows
(loader-bound, 20260924_01). --min is the caller's statement of that floor.
"""
import argparse
import re
import subprocess
import sys

PARTITIONS = ('a6000_ada', 'a6000', 'rtx8000')
GPU_CAP = 12


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def parse_nodes(text):
    """scontrol show node output -> list of dicts. Pure, so it can be tested on a fixture."""
    nodes = []
    for block in re.split(r'\n\s*\n', text.strip()):
        kv = dict(re.findall(r'(\w+)=([^\s]+)', block))
        if 'NodeName' not in kv:
            continue
        parts = kv.get('Partitions', '').split(',')
        if not any(p in PARTITIONS for p in parts):
            continue
        state = kv.get('State', '')
        cpu_tot, cpu_alloc = int(kv.get('CPUTot', 0)), int(kv.get('CPUAlloc', 0))
        gpu_tot = int((re.search(r'gpu:(\d+)', kv.get('Gres', '')) or [0, 0])[1])
        alloc = kv.get('AllocTRES', '')
        gpu_alloc = int((re.search(r'gres/gpu=(\d+)', alloc) or [0, 0])[1])
        usable = not any(s in state for s in ('DOWN', 'DRAIN', 'NOT_RESPONDING'))
        nodes.append(dict(name=kv['NodeName'], partitions=parts, state=state, usable=usable,
                          cpu_free=cpu_tot - cpu_alloc, cpu_tot=cpu_tot,
                          gpu_free=gpu_tot - gpu_alloc, gpu_tot=gpu_tot))
    return nodes


def live_nodes():
    return parse_nodes(sh('scontrol show node'))


def my_gpus_in_use():
    out = sh('squeue -u $USER -h -t R -o "%b"')
    return sum(int((re.search(r'gpu:(\d+)', l) or [0, 0])[1]) for l in out.splitlines())


def best_fit(nodes, gpus, cpu_min, cpu_max):
    """The node giving the most CPUs per GPU (<= cpu_max) among those with >= gpus free GPUs and
    >= gpus*cpu_min free CPUs; None if nothing fits now."""
    best = None
    for n in nodes:
        if not n['usable'] or n['gpu_free'] < gpus:
            continue
        per_gpu = min(cpu_max, n['cpu_free'] // gpus)
        if per_gpu < cpu_min:
            continue
        # more CPUs per GPU first; on a tie the node with more CPUs left, so the next job fits too
        if best is None or (per_gpu, n['cpu_free']) > (best[1], best[0]['cpu_free']):
            best = (n, per_gpu)
    return best


def table(nodes):
    lines = ['%-7s %-10s %5s %5s %5s %5s' % ('node', 'partition', 'cpuF', 'cpuT', 'gpuF', 'gpuT')]
    for n in nodes:
        flag = '' if n['usable'] else '  (' + n['state'] + ')'
        lines.append('%-7s %-10s %5d %5d %5d %5d%s' % (
            n['name'], n['partitions'][0], n['cpu_free'], n['cpu_tot'], n['gpu_free'], n['gpu_tot'], flag))
    return '\n'.join(lines)


def adjust_flags(flags, nodes, cpu_min, cpu_max):
    """Rewrite --cpus-per-task in sbatch flags to what fits now. Returns (flags, note)."""
    gpus = 1
    for f in flags:
        m = re.match(r'--gres=gpu:(\d+)', f)
        if m:
            gpus = int(m.group(1))
    fit = best_fit(nodes, gpus, cpu_min, cpu_max)
    if fit is None:
        return flags, 'no node has %d free GPU(s) with >= %d free CPUs each right now; keeping the request' % (gpus, cpu_min)
    node, per_gpu = fit
    want = per_gpu * gpus
    out, changed = [], False
    for f in flags:
        m = re.match(r'--cpus-per-task=(\d+)', f)
        if m and int(m.group(1)) > want:
            out.append('--cpus-per-task=%d' % want); changed = True
        else:
            out.append(f)
    note = ('%s has %d free GPU(s) and %d free CPUs: %s' % (
        node['name'], node['gpu_free'], node['cpu_free'],
        ('asking %d CPUs (%d per GPU) so the job can start there' % (want, per_gpu)) if changed
        else 'the request already fits'))
    return out, note


def pending_jobs():
    out = sh('squeue -u $USER -h -t PD -o "%i|%j|%b|%C"')
    jobs = []
    for l in out.splitlines():
        jid, name, gres, cpus = l.split('|')
        jobs.append(dict(id=jid, name=name, gpus=int((re.search(r'gpu:(\d+)', gres) or [0, 1])[1]), cpus=int(cpus)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=int)
    ap.add_argument('--min', type=int, default=5, help='floor of CPUs per GPU (the loader need)')
    ap.add_argument('--max', type=int, default=10, help='cap of CPUs per GPU (node61 share)')
    ap.add_argument('--adjust', nargs=argparse.REMAINDER)
    ap.add_argument('--fit-pending', action='store_true')
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    nodes = live_nodes()
    if a.adjust is not None:
        flags, note = adjust_flags(a.adjust, nodes, a.min, a.max)
        print('free_slots: ' + note, file=sys.stderr)
        print(' '.join(flags))
        return 0
    print(table(nodes), file=sys.stderr)
    print('my GPUs in use: %d of %d (QOS limit12)' % (my_gpus_in_use(), GPU_CAP), file=sys.stderr)
    if a.fit_pending:
        for j in pending_jobs():
            fit = best_fit(nodes, j['gpus'], a.min, a.max)
            if fit is None:
                print('%s %-12s %d GPU %2d CPUs: nothing free' % (j['id'], j['name'], j['gpus'], j['cpus']), file=sys.stderr)
                continue
            node, per_gpu = fit
            want = per_gpu * j['gpus']
            if want >= j['cpus']:
                print('%s %-12s %d GPU %2d CPUs: fits %s as is' % (j['id'], j['name'], j['gpus'], j['cpus'], node['name']), file=sys.stderr)
                continue
            print('%s %-12s %d GPU %2d CPUs: %s would take it at %d CPUs%s' % (
                j['id'], j['name'], j['gpus'], j['cpus'], node['name'], want, ' - applying' if a.apply else ''), file=sys.stderr)
            if a.apply:
                for field in ('NumCPUs', 'MinCPUsNode', 'CPUsPerTask'):
                    subprocess.run(['scontrol', 'update', 'JobId=%s' % j['id'], '%s=%d' % (field, want)], check=False)
        return 0
    if a.gpus:
        fit = best_fit(nodes, a.gpus, a.min, a.max)
        if fit is None:
            print('no fit now for %d GPU(s) at >= %d CPUs each' % (a.gpus, a.min), file=sys.stderr)
            return 1
        node, per_gpu = fit
        print('--cpus-per-task=%d' % (per_gpu * a.gpus))
        print('fits %s now (%d free GPUs, %d free CPUs)' % (node['name'], node['gpu_free'], node['cpu_free']), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
