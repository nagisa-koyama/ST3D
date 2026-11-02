#!/usr/bin/env python3
"""Read out the FACTOR-SWAP S2 rows (experiments_md T05 §5 item 24) from their job logs.

Design (pre-declared, 2026-11-02): the S2 oracle 26843 (KITTI -> KITTI) is retrained with ONE
factor set to nuScenes' value per row, R1-R10. The term of a factor is the AP the oracle loses
when that factor is swapped: term = O - Rk, except that R1 (lines) is read against R2 (random
thinning, the control): term(R1) = R2 - R1, and R10 (fg+bg) against R3 (density): term(R10) =
R3 - R10. The core terms (R1, R3, R4, R5, R6, R7) are summed and compared with the oracle-control
gap 80.33 - 41.57 = 38.8; R2 is the control, R8-R10 are listed but not summed.

The final in-run evaluation is read from `tools/logs/{error,output}_<job>_*.txt` (the metrics come
through `logging.info`, usually in the error log; tqdm writes `\\r`, so the file is split on both
`\\r` and `\\n`). The block parsed is the one printed by
`pcdet/datasets/kitti/kitti_object_eval_python/eval.py:get_official_eval_result`:

    Car AP_R40@0.70, 0.70, 0.70:        <- (bbox, bev, 3d) IoU; ret_dict keys Car_bev/moderate_R40 ...
    bbox AP:91.8136, 86.5900, 86.7175   <- easy, moderate, hard
    bev  AP:88.7970, 80.3301, 80.1218
    3d   AP:82.5437, 70.5866, 69.3276
    Car AP_R40@0.70, 0.50, 0.50:        <- the IoU-0.5 block (not in ret_dict; text only)
    ...

Pure Python, no torch. Runs on the master in seconds:

    python3 tools/analysis/factorswap_s2_readout.py --missing-ok
"""
import argparse
import glob
import os
import re
import sys
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(REPO, 'tools', 'logs')

# row -> (job, swapped factor)
DEFAULT_JOBS = OrderedDict([
    ('O', (26843, 'oracle (KITTI -> KITTI, 26843)')),
    ('R1', (31409, 'lines (32-beam ring pattern)')),
    ('R2', (31410, 'random thinning (control for R1)')),
    ('R3', (31414, 'density')),
    ('R4', (31411, 'azimuth binning')),
    ('R5', (31415, 'pitch')),
    ('R6', (31416, 'label size')),
    ('R7', (31417, 'label range')),
    ('R8', (31412, 'foreground 0.50')),
    ('R9', (31413, 'background 0.50')),
    ('R10', (31420, 'fg + bg (read against R3)')),
])
ROW_ORDER = ['O', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']
CORE_ROWS = ['R1', 'R3', 'R4', 'R5', 'R6', 'R7']
# term = minuend - row;  minuend 'O' unless listed here
TERM_MINUEND = {'R1': 'R2', 'R10': 'R3'}
ORACLE_FALLBACK = {'Car_bev/moderate_R40': 80.33, 'Car_3d/moderate_R40': 70.59}
CONTROL_BEV = 41.57          # 26842 recipe control, Car BEV moderate R40 (T06)
GAP = 80.33 - CONTROL_BEV    # 38.76, quoted as 38.8
VERDICT_TOL = 5.0
READ_THRESH = 3.0

RUN_ID_RE = re.compile(r'run-(\d{8}_\d{6})-([0-9a-z]+)')
EPOCH_RE = re.compile(r'Performance of EPOCH (\d+)')
HEADER_RE = re.compile(r'^(?:\[.*?\]\s+)?(\w+) AP_R40@([\d.]+), ([\d.]+), ([\d.]+):\s*$')
AP_LINE_RE = re.compile(r'^(bbox|bev|3d|aos)\s+AP:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\s*$')
LEVELS = ('easy', 'moderate', 'hard')


def split_log_lines(text):
    """tqdm progress bars end lines with \\r: treat \\r like \\n (tr '\\r' '\\n')."""
    return text.replace('\r', '\n').split('\n')


def parse_eval_blocks(lines):
    """Yield (epoch, metrics) for every evaluation block; metrics keyed like ret_dict.

    Keys: '<cls>_<bev|3d|image>/<easy|moderate|hard>_R40' for the first (IoU 0.7 for Car) block,
    and the same with '@0.5' appended for the second block whose bev/3d IoU is 0.50.
    """
    epoch = None
    cur = None
    blocks = []
    header = None
    for raw in lines:
        line = raw.strip()
        m = EPOCH_RE.search(line)
        if m:
            epoch = int(m.group(1))
            cur = {}
            blocks.append((epoch, cur))
            header = None
            continue
        if cur is None:
            continue
        m = HEADER_RE.match(line)
        if m:
            cls, _bbox_iou, bev_iou, _3d_iou = m.group(1), m.group(2), m.group(3), m.group(4)
            header = (cls, float(bev_iou))
            continue
        m = AP_LINE_RE.match(line)
        if m and header is not None:
            kind = {'bbox': 'image', 'bev': 'bev', '3d': '3d', 'aos': 'aos'}[m.group(1)]
            cls, iou = header
            suffix = '' if iou >= 0.7 or cls != 'Car' else '@%.1f' % iou
            if cls != 'Car' and iou < 0.5:
                suffix = '@%.2f' % iou
            for lvl, val in zip(LEVELS, m.group(2, 3, 4)):
                cur['%s_%s/%s_R40%s' % (cls, kind, lvl, suffix)] = float(val)
            continue
        if line.startswith('Pedestrian AP@') or line.startswith('Cyclist AP@') or \
                line.startswith('Car AP@'):
            header = None   # the 11-point block: skip its AP lines
    return blocks


def final_eval(blocks):
    """The last COMPLETE Car block (both IoU-0.7 and IoU-0.5 R40 present), or None."""
    for epoch, m in reversed(blocks):
        if 'Car_bev/moderate_R40' in m and 'Car_bev/moderate_R40@0.5' in m:
            return epoch, m
    return None


def find_logs(job, log_dir=LOG_DIR):
    return sorted(glob.glob(os.path.join(log_dir, 'error_%d_*.txt' % job))) + \
        sorted(glob.glob(os.path.join(log_dir, 'output_%d_*.txt' % job)))


def read_job(job, log_dir=LOG_DIR):
    """Return dict(job, run_id, epoch, metrics, progress, logs). metrics is None when pending."""
    out = {'job': job, 'run_id': None, 'epoch': None, 'metrics': None, 'progress': None, 'logs': []}
    for path in find_logs(job, log_dir):
        out['logs'].append(path)
        with open(path, 'r', errors='replace') as f:
            text = f.read()
        lines = split_log_lines(text)
        if out['run_id'] is None:
            m = RUN_ID_RE.search(text)
            if m:
                out['run_id'] = m.group(2)
        fe = final_eval(parse_eval_blocks(lines))
        if fe is not None and (out['epoch'] is None or fe[0] > out['epoch']):
            out['epoch'], out['metrics'] = fe
        for line in reversed(lines):
            pm = re.search(r'epochs:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)', line)
            if pm:
                out['progress'] = '%s/%s epochs' % (pm.group(2), pm.group(3))
                break
    return out


def fmt(v, nd=2):
    return '-' if v is None else '%.*f' % (nd, v)


def build_table(results, rows=ROW_ORDER, jobs=DEFAULT_JOBS):
    """Return (table_rows, terms). terms: row -> float or None."""
    bev = {}
    for r in rows:
        m = results[r]['metrics']
        bev[r] = m['Car_bev/moderate_R40'] if m else None
    if bev.get('O') is None:
        bev['O'] = ORACLE_FALLBACK['Car_bev/moderate_R40']
    terms = {}
    table = []
    for r in rows:
        res = results[r]
        m = res['metrics']
        if r == 'O':
            term = None
            term_s = 'gap vs control %.2f: %.1f' % (CONTROL_BEV, GAP)
        else:
            minuend = TERM_MINUEND.get(r, 'O')
            if bev[r] is not None and bev[minuend] is not None:
                term = bev[minuend] - bev[r]
                reading = 'read' if term >= READ_THRESH else 'not detectable at one seed'
                term_s = '%s − %s = %+.2f (%s)' % (minuend, r, term, reading)
            else:
                term = None
                term_s = '%s − %s: pending' % (minuend, r)
        terms[r] = term
        if m is None and r == 'O':
            m = {'Car_bev/moderate_R40': ORACLE_FALLBACK['Car_bev/moderate_R40'],
                 'Car_3d/moderate_R40': ORACLE_FALLBACK['Car_3d/moderate_R40']}
            status = 'hardcoded (log has no final eval)'
        elif m is None:
            status = 'pending' + (' (%s)' % res['progress'] if res['progress'] else '')
        else:
            status = 'epoch %d' % res['epoch']
        table.append({
            'row': r, 'factor': jobs[r][1], 'job': res['job'], 'run_id': res['run_id'] or '-',
            'status': status,
            'bev07': m.get('Car_bev/moderate_R40') if m else None,
            '3d07': m.get('Car_3d/moderate_R40') if m else None,
            'bev05': m.get('Car_bev/moderate_R40@0.5') if m else None,
            '3d05': m.get('Car_3d/moderate_R40@0.5') if m else None,
            'easy': m.get('Car_bev/easy_R40') if m else None,
            'hard': m.get('Car_bev/hard_R40') if m else None,
            'term': term, 'term_s': term_s,
        })
    return table, terms


def summarize(terms, core=CORE_ROWS, gap=GAP, tol=VERDICT_TOL):
    """Return (sum_or_None, missing_rows, verdict_string)."""
    missing = [r for r in core if terms.get(r) is None]
    present = [terms[r] for r in core if terms.get(r) is not None]
    if missing:
        partial = sum(present) if present else None
        return partial, missing, 'verdict pending (%s not final)' % ', '.join(missing)
    s = sum(present)
    if abs(s - gap) <= tol:
        verdict = 'consistent at one seed'
    else:
        verdict = 'interaction + unmeasured'
    return s, [], '|%.2f − %.1f| = %.2f %s %.0f → %s' % (
        s, gap, abs(s - gap), '≤' if abs(s - gap) <= tol else '>', tol, verdict)


def render(table, terms, missing_ok):
    out = []
    out.append('| row | factor | job | W&B | Car BEV / 3D mod. R40 @0.7 | @0.5 | BEV easy / hard | status | term (BEV @0.7) |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for t in table:
        out.append('| %s | %s | %d | %s | %s / %s | %s / %s | %s / %s | %s | %s |' % (
            t['row'], t['factor'], t['job'], t['run_id'],
            fmt(t['bev07']), fmt(t['3d07']), fmt(t['bev05']), fmt(t['3d05']),
            fmt(t['easy']), fmt(t['hard']), t['status'], t['term_s']))
    out.append('')
    s, missing, verdict = summarize(terms)
    core_s = ' + '.join('%s %s' % (r, fmt(terms.get(r))) for r in CORE_ROWS)
    out.append('Core terms (R1, R3, R4, R5, R6, R7; R2 control, R8-R10 not summed): %s' % core_s)
    if missing:
        out.append('Sum so far: %s (missing %s); gap O − control = %.1f; %s' % (
            fmt(s), ', '.join(missing), GAP, verdict))
    else:
        out.append('Sum = %.2f; gap O − control = %.1f; %s' % (s, GAP, verdict))
    out.append('Rule: |sum − 38.8| ≤ 5 → consistent at one seed, else interaction + unmeasured; '
               'a term ≥ 3 is read, smaller is not detectable at one seed.')
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--missing-ok', action='store_true',
                    help='rows without a final evaluation are printed as pending, not errors')
    ap.add_argument('--log-dir', default=LOG_DIR)
    ap.add_argument('--job', action='append', default=[], metavar='ROW=JOB',
                    help='override a job id, e.g. --job R3=31999')
    args = ap.parse_args(argv)

    jobs = OrderedDict((k, v) for k, v in DEFAULT_JOBS.items())
    for spec in args.job:
        row, job = spec.split('=')
        jobs[row] = (int(job), jobs[row][1] if row in jobs else spec)

    results = {r: read_job(jobs[r][0], args.log_dir) for r in ROW_ORDER}
    pending = [r for r in ROW_ORDER if results[r]['metrics'] is None and r != 'O']
    if pending and not args.missing_ok:
        sys.stderr.write('no final evaluation yet for %s (pass --missing-ok)\n' % ', '.join(pending))
        return 1
    table, terms = build_table(results, jobs=jobs)
    print(render(table, terms, args.missing_ok))
    return 0


if __name__ == '__main__':
    sys.exit(main())
