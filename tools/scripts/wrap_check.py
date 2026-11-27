"""Refuse an sbatch `--wrap` that cannot run under /bin/sh, BEFORE it queues (called by scripts/submit.sh).

sbatch turns `--wrap="..."` into a `#!/bin/sh` script, and /bin/sh on the nodes is dash, which has no `pipefail`. On
2026-11-27 item 59a's scoring job 37205 (`--wrap="set -eo pipefail; singularity exec ... | tee ..."`) died at its first
line (`set: Illegal option -o pipefail`) after waiting for its predecessors, and its 59b twin 37213 was pending with the same
line (experiments_md 20261003_05 §5 item 59). Put pipes and bash options inside `bash -c '...'`, where bash runs them.

    python3 scripts/wrap_check.py [sbatch args ...]      # exit 0 = fine or no --wrap, 2 = refused (reason on stderr)

Two checks on the wrap text: (1) `pipefail` outside any quoted string (inside quotes it belongs to an inner `bash -c`);
(2) `dash -n` (syntax only, nothing runs), when dash is installed. Stdlib only, so it runs on the host python3.
Tests: tests/test_submit_wrap_check.py.
"""
import re
import shutil
import subprocess
import sys


def wrap_texts(argv):
    """Every --wrap value in an sbatch argument list (`--wrap=X` or `--wrap X`), in order."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith('--wrap='):
            out.append(a[len('--wrap='):])
        elif a == '--wrap' and i + 1 < len(argv):
            out.append(argv[i + 1])
            i += 1
        i += 1
    return out


def top_level(text):
    """The text with every quoted string and backslash-escaped character blanked (what /bin/sh itself interprets)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '\\':
            out.append('  ')
            i += 2
        elif c == "'":
            j = text.find("'", i + 1)
            j = n if j < 0 else j
            out.append(' ' * (j + 1 - i))
            i = j + 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == '\\' else 1
            out.append(' ' * (min(j, n - 1) + 1 - i))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def problems(text, dash=None):
    """Reasons this wrap text would fail under /bin/sh (dash); empty if none found."""
    bad = []
    if re.search(r'\bpipefail\b', top_level(text)):
        bad.append('`pipefail` at the top level of --wrap: sbatch runs it with /bin/sh (dash), which has no pipefail, so '
                   'the job dies at its first line. Move the pipe and `set -o pipefail` inside `bash -c \'...\'`.')
    dash = shutil.which('dash') if dash is None else dash
    if dash:
        p = subprocess.run([dash, '-n', '-c', text], capture_output=True, text=True)
        if p.returncode != 0:
            bad.append('/bin/sh (dash) syntax error in --wrap: ' + (p.stderr.strip() or 'exit %d' % p.returncode))
    return bad


def main(argv):
    bad = [b for t in wrap_texts(argv) for b in problems(t)]
    for b in bad:
        print('submit.sh: refused: ' + b, file=sys.stderr)
    return 2 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
