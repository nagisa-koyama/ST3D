#!/usr/bin/env python3
"""Size a Slurm request from an estimated single-GPU duration.

Called by scripts/submit.sh when EST_H is set, with submit.sh's own arguments. Prints sbatch
flags on stdout and the reasoning on stderr. Run it inside the container (it needs easydict, and
it loads the config with the repo's own resolver so `_BASE_CONFIG_` chains are followed exactly
as train.py follows them).

  EST_H   expected wall-clock hours on ONE GPU, evaluation included. Take it from the closest
          comparable run's real elapsed time (`sacct -j <job> -o Elapsed,AllocTRES`); most of
          those landed on rtx8000/node61, the slowest partition in the list, so a limit sized
          from them holds wherever the job lands.

Rules (experiments_md/20260927_04):
  GPUs    2 if EST_H > DDP_ABOVE_H (24), else 1. A 2-GPU job needs both GPUs free on ONE node,
          which is rarer: the only >10 h queue waits since 2026-09-18 were 2-GPU jobs. Below a
          day of work the ~2x speedup (20260922_06 section 2e) is smaller than that wait. Only
          run_sourceonly_2gpu.sh can use 2 GPUs; any other script is sized to 1.
  CPUs    per GPU = training loaders x workers + 2, clamped to [4, CPUS_PER_GPU_MAX=10]. 10 is
          node61's CPUs-per-GPU share (40 / 4) and the smallest in a6000_ada,a6000,rtx8000, so a
          job asking <= 10 per GPU never strands a GPU on the node it lands on. Before this, every
          launch asked 20, and node61 showed load ~9 on the 36 CPUs those jobs held.
  time    EST_H / speedup x SAFETY (1.5) + 1 h, rounded up to the hour. Over 7 days only rtx8000
          (28-day MaxTime) can run it, so the partition is narrowed and a warning printed.

Any flag given explicitly to submit.sh comes after these on the sbatch command line and wins.
Overrides: NGPU, CPUS_PER_GPU (exact, skips the formula), DDP_ABOVE_H, CPUS_PER_GPU_MAX, SAFETY.
"""
import contextlib
import math
import os
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
REPO = TOOLS.parent
DDP_SCRIPT = 'run_sourceonly_2gpu.sh'
SHORTHAND = {'kitti', 'lyft', 'nuscenes', 'pandaset', 'waymo'}
PARTITION_MAXTIME_H = {'a6000_ada': 168, 'a6000': 168, 'rtx8000': 672}


def say(msg):
    print(f'size_job: {msg}', file=sys.stderr)


def tokens(argv):
    # --wrap="bash analysis/eval_checkpoint.sh <cfg> ..." carries its config inside one argument.
    for arg in argv:
        if arg.startswith('--wrap='):
            yield from arg[len('--wrap='):].split()
        else:
            yield from arg.split()


def find_config(toks):
    for i, tok in enumerate(toks):
        if tok.endswith('.yaml'):
            return tok
    for i, tok in enumerate(toks):
        if tok.endswith(DDP_SCRIPT) and i + 1 < len(toks) and toks[i + 1] in SHORTHAND:
            return f'cfgs/da-ieee-access/centerpoint-sourceonly-{toks[i + 1]}.yaml'
    return None


def load_config(path):
    # The container also carries a STALE editable pcdet at /code/ST3D; resolve the live checkout.
    sys.path.insert(0, str(REPO))
    import pcdet
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    if not Path(pcdet.__file__).resolve().is_relative_to(REPO):
        raise SystemExit(f'size_job: pcdet resolved to {pcdet.__file__}, not {REPO}')
    with contextlib.redirect_stdout(sys.stderr):  # cfg_from_yaml_file prints "... is loaded"
        return cfg_from_yaml_file(path, EasyDict())


def env_float(name, default):
    return float(os.environ.get(name, default))


def main(argv):
    est_h = float(os.environ['EST_H'])
    if est_h <= 0:
        raise SystemExit('size_job: EST_H must be positive hours on one GPU')
    toks = list(tokens(argv))
    ddp_capable = any(t.endswith(DDP_SCRIPT) for t in toks)
    entry = os.environ.get('ENTRY', 'train.py')

    # --- GPUs ---------------------------------------------------------------------------------
    ddp_above = env_float('DDP_ABOVE_H', 24)
    # An explicit --gres wins on the sbatch line anyway; follow it here too, so CPUs and time are
    # sized for the GPU count the job will actually get.
    explicit = [m.group(1) for t in (' '.join(toks).replace('--gres gpu:', '--gres=gpu:')).split()
                for m in [re.fullmatch(r'--gres=gpu:(\d+)', t)] if m]
    if 'NGPU' in os.environ:
        ngpu = int(os.environ['NGPU'])
        why = 'NGPU override'
    elif explicit:
        ngpu = int(explicit[-1])
        why = f'explicit --gres=gpu:{ngpu}'
    else:
        ngpu = 2 if est_h > ddp_above else 1
        why = f'EST_H {est_h:g} h {">" if ngpu == 2 else "<="} {ddp_above:g} h'
    if ngpu > 1 and not ddp_capable:
        if explicit or 'NGPU' in os.environ:
            say(f'WARNING: {ngpu} GPUs requested explicitly, but only {DDP_SCRIPT} runs DDP')
        else:
            ngpu, why = 1, f'{why}, but only {DDP_SCRIPT} runs DDP'
    say(f'GPUs {ngpu} ({why})')

    # --- CPUs ---------------------------------------------------------------------------------
    cfg_path = find_config(toks)
    workers, loaders = 4, 1
    if cfg_path is None:
        say('no config among the arguments - assuming 1 loader x 4 workers')
    else:
        os.chdir(TOOLS)  # configs and their _BASE_CONFIG_ paths are relative to tools/
        if not Path(cfg_path).is_file():
            raise SystemExit(f'size_job: config {cfg_path} does not exist (relative to {TOOLS})')
        cfg = load_config(cfg_path)
        workers = cfg.get('OPTIMIZATION', {}).get('NUM_WORKERS', 4)
        if '--workers' in toks:
            workers = int(toks[toks.index('--workers') + 1])
        if not ddp_capable:
            loaders = 1  # eval / pseudo-label generation / a smoke script: one loader at a time
        elif entry.endswith('adaptive_train.py'):
            loaders = 2  # source + target, both iterated every step
        else:
            loaders = len(cfg.DATA_CONFIGS) if cfg.get('DATA_CONFIGS', None) else 1
            loaders += 1 if cfg.get('SELF_TRAIN', None) else 0
    if 'CPUS_PER_GPU' in os.environ:
        per_gpu = int(os.environ['CPUS_PER_GPU'])
        say(f'CPUs {per_gpu}/GPU (CPUS_PER_GPU override)')
    else:
        cap = int(env_float('CPUS_PER_GPU_MAX', 10))
        want = loaders * workers + 2
        per_gpu = max(4, min(cap, want))
        say(f'CPUs {per_gpu}/GPU = min({cap}, {loaders} loader(s) x {workers} workers + 2)'
            + (f'  [capped from {want}]' if want > cap else '')
            + (f'  [{cfg_path}, entry {entry}]' if cfg_path else ''))

    # --- time ---------------------------------------------------------------------------------
    safety = env_float('SAFETY', 1.5)
    speedup = 2.0 if ngpu == 2 and ddp_capable else 1.0  # measured 2.03-2.06x, 20260922_06
    hours = max(1, math.ceil(est_h / speedup * safety + 1))
    say(f'time {hours} h = {est_h:g} h / {speedup:g} x {safety:g} + 1 h')
    flags = [f'--gres=gpu:{ngpu}', f'--cpus-per-task={ngpu * per_gpu}',
             f'--time={hours // 24}-{hours % 24:02d}:00:00']
    if hours > max(PARTITION_MAXTIME_H.values()):
        raise SystemExit(f'size_job: {hours} h exceeds every partition limit')
    if hours > PARTITION_MAXTIME_H['a6000_ada']:
        say(f'WARNING: {hours} h exceeds the 7-day limit of a6000_ada/a6000 - rtx8000 only')
        flags.append('--partition=rtx8000')
    print(' '.join(flags))


if __name__ == '__main__':
    main(sys.argv[1:])
