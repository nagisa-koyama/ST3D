"""Add (or refresh) a `viz` column in every result table of a report (experiments_md/20261003_02 §6).

A table qualifies when its header has a `job` column. Each job number in that column that has a
figure at <experiments_md>/viz/<job>.jpg gets a link; rows without one get an em dash. Re-runnable:
an existing `viz` column is rewritten in place, never duplicated.

    python3 analysis/viz/insert_viz_links.py <report.md> [...]
"""
import re
import sys
from pathlib import Path

ROOT = Path('/home/koyama/code/experiments_md')


def cells(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def fmt(cs):
    return '| ' + ' | '.join(cs) + ' |'


def process(path):
    lines = path.read_text().split('\n')
    out, i, changed = [], 0, 0
    while i < len(lines):
        line = lines[i]
        is_header = line.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|$', lines[i + 1].strip())
        if not is_header or 'job' not in [c.lower() for c in cells(line)]:
            out.append(line); i += 1; continue
        head = cells(line)
        jcol = [c.lower() for c in head].index('job')
        has = 'viz' in head
        vcol = head.index('viz') if has else len(head)
        if not has:
            head.append('viz')
        out.append(fmt(head))
        sep = cells(lines[i + 1])
        if not has:
            sep.append('---')
        out.append(fmt(sep))
        i += 2
        while i < len(lines) and lines[i].startswith('|'):
            cs = cells(lines[i])
            jobs = re.findall(r'\b(2\d{4})\b', cs[jcol]) if jcol < len(cs) else []
            links = ['[%s](viz/%s.jpg)' % (j, j) for j in jobs if (ROOT / 'viz' / ('%s.jpg' % j)).exists()]
            v = ' · '.join(links) if links else '—'
            if has and vcol < len(cs):
                cs[vcol] = v
            else:
                cs.append(v)
            out.append(fmt(cs)); changed += 1; i += 1
    path.write_text('\n'.join(out))
    return changed


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(p, process(Path(p)), 'table rows')
