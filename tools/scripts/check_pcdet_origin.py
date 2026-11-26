"""Which `pcdet` does a tools/ script import when it is run the way jobs run it (cwd tools/, `python3 <script>`)?

The container image carries a STALE baked-in copy of pcdet in site-packages. A script run directly gets its own directory as
sys.path[0], not tools/ or the repo root, so unless it imports `_init_path` (or puts the repo root on sys.path) before its
first pcdet import, it silently uses that copy. The copy predates fixes such as the `_BASE_CONFIG_` merge (`e107939`): on
2026-11-27 item 59's tool resolved a config with it and lost G's z-shift augmentation (experiments_md 20261003_05 §5 item 59).

    python3 scripts/check_pcdet_origin.py FILE [FILE ...]      # from tools/, inside the container

For each file the module is executed as a script would be, but under a name other than '__main__' (so its main block does not
run), then the probe reports where pcdet is (if the module imported it) or would be (find_spec, for an import made later):
"repo", "stale", or "error". A module whose pcdet imports sit in functions and which adds the repo root only inside main()
reads "stale" here although its main() may be fine; such files are judged by hand (tests/test_pcdet_import_order.py).
Never imports pcdet itself. Tests: tests/test_pcdet_import_order.py.
"""
import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
REPO = TOOLS.parent

PROBE = r'''
import importlib.util, json, os, runpy, sys
script = os.path.abspath(sys.argv[1])
sys.argv = [script]
sys.path[0] = os.path.dirname(script)          # what `python3 <script>` puts first
err = None
try:
    runpy.run_path(script, run_name='__pcdet_probe__')
except SystemExit:
    pass
except BaseException as e:                       # report, never hide
    err = '%s: %s' % (type(e).__name__, str(e)[:200])
mod = sys.modules.get('pcdet')
if mod is not None:
    origin, loaded = getattr(mod, '__file__', None), True
else:
    spec = importlib.util.find_spec('pcdet')
    origin, loaded = (spec.origin if spec else None), False
print('PCDET_PROBE ' + json.dumps(dict(origin=origin, loaded=loaded, error=err)))
'''


def probe(path, timeout=180):
    """{'file', 'origin', 'loaded', 'error', 'verdict'} for one file (path relative to tools/ or absolute)."""
    p = subprocess.run([sys.executable, '-c', PROBE, str(path)], cwd=str(TOOLS), capture_output=True, text=True,
                       timeout=timeout)
    line = [x for x in p.stdout.splitlines() if x.startswith('PCDET_PROBE ')]
    if not line:
        return dict(file=str(path), origin=None, loaded=False, error='no probe output: ' + p.stderr[-300:], verdict='error')
    d = json.loads(line[-1][len('PCDET_PROBE '):])
    repo_pcdet = str(REPO / 'pcdet')
    d['verdict'] = 'repo' if d['origin'] and str(Path(d['origin']).resolve()).startswith(repo_pcdet) else 'stale'
    if d['error'] and d['verdict'] != 'repo':
        d['verdict'] = 'error'
    d['file'] = str(path)
    return d


def main():
    out = [probe(f) for f in sys.argv[1:]]
    for d in out:
        print('%-6s %-55s %s %s%s' % (d['verdict'], d['file'], 'loaded' if d['loaded'] else 'deferred', d['origin'],
                                     ('  [' + d['error'] + ']') if d['error'] else ''))
    sys.exit(0 if all(d['verdict'] == 'repo' for d in out) else 1)


if __name__ == '__main__':
    main()
