"""Add (or refresh) a `viz` column in the result tables of a report (experiments_md/20261003_02 §6).

A row's figures are found by key: the job numbers in its `job` column; failing that, the job numbers
in its first cell; failing that, the W&B run ids it links (Table E is keyed by run). Each key with a
figure at <experiments_md>/viz/<key>.jpg gets a link, plus its <key>_pseudo.jpg if there is one -
unless the row cites W&B runs and none is the run the figure was made from (same checkpoint, another
target).
A table gains the column only if at least one row has a figure, so plan and status tables stay
untouched. Re-runnable: an existing `viz` column is rewritten in place, never duplicated.

    python3 analysis/viz/insert_viz_links.py <report.md> [...]
"""
import re
import sys
from pathlib import Path

ROOT = Path('/home/koyama/code/experiments_md')
JOB = re.compile(r'\b(2[5-7]\d{3})\b')
RUN = re.compile(r'wandb\.ai/nagisa/st3d/runs/([a-z0-9]{8})')


def cells(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def fmt(cs):
    return '| ' + ' | '.join(cs) + ' |'


def manifest_runs():
    """key -> the W&B run its figure was made from (manifest.yaml is one JSON object per line)."""
    import json
    out = {}
    for line in (Path(__file__).resolve().parent / 'manifest.yaml').read_text().split('\n'):
        if line.startswith('- {'):
            e = json.loads(line[2:])
            out[str(e['job'])] = e.get('run')
    return out


MANIFEST = manifest_runs()


def links(line, cs, jcol):
    keys = JOB.findall(cs[jcol]) if jcol is not None and jcol < len(cs) else JOB.findall(cs[0])
    if not keys and jcol is None:
        keys = RUN.findall(line)
    runs = set(RUN.findall(line))
    out = []
    for k in dict.fromkeys(keys):
        # a row that cites W&B runs must cite the one the figure was made from: the same checkpoint
        # scored on another target (S4's spin-val rescoring of 26389) is not that figure
        if runs and k in MANIFEST and MANIFEST[k] not in runs and k not in runs:
            continue
        if (ROOT / 'viz' / ('%s.jpg' % k)).exists():
            out.append('[%s](viz/%s.jpg)' % (k, k))
        if (ROOT / 'viz' / ('%s_pseudo.jpg' % k)).exists():
            out.append('[pseudo](viz/%s_pseudo.jpg)' % k)
    return ' · '.join(out)


def process(path):
    lines = path.read_text().split('\n')
    out, i, changed = [], 0, 0
    while i < len(lines):
        line = lines[i]
        is_header = line.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|$', lines[i + 1].strip())
        if not is_header:
            out.append(line); i += 1; continue
        k = i + 2
        while k < len(lines) and lines[k].startswith('|'):
            k += 1
        head, body = cells(line), lines[i + 2:k]
        low = [c.lower() for c in head]
        jcol = low.index('job') if 'job' in low else None
        has = 'viz' in head
        vals = [links(b, cells(b), jcol) for b in body]
        if not has and not any(vals):
            out.extend(lines[i:k]); i = k; continue
        vcol = head.index('viz') if has else len(head)
        if not has:
            head.append('viz')
        sep = cells(lines[i + 1])
        if not has:
            sep.append('---')
        out += [fmt(head), fmt(sep)]
        for b, v in zip(body, vals):
            cs = cells(b)
            v = v or '—'
            if has and vcol < len(cs):
                cs[vcol] = v
            else:
                cs.append(v)
            out.append(fmt(cs)); changed += 1
        i = k
    path.write_text('\n'.join(out))
    return changed


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(p, process(Path(p)), 'table rows')
