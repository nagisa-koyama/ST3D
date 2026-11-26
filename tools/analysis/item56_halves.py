"""Item 56 (experiments_md 20261003_05 §5 item 56, design fixed 2026-11-27 00:02 JST, experiments_md commit fcedf64, before any
number): the projection-plane grid in two parts. Is O's far loss on S' the azimuth spacing? ANALYSIS (KITTI labels score only).

    python3 analysis/item56_halves.py eval_rows                                   # "tag cfg job epoch arm" for the 8 evaluations
    python3 analysis/item56_halves.py manifest OUT_DIR                            # the 16 scored rows (4 clouds x O / P x 2 seeds)
    python3 analysis/item56_halves.py count OUT_DIR [--workers 8]                 # CPU: S' lines of every detection; cloud checks
    python3 analysis/item56_halves.py score OUT_DIR [--real-kernel] [--workers 4]  # GPU node for --real-kernel
    python3 analysis/item56_halves.py read OUT_DIR/item56_ap.json

Clouds: full KITTI val (the rows' in-run values, item 48 forward `none`), S' (item 47 `none`), S'_rows and S'_az (kitti_rings
`pattern_rows` / `pattern_az`, this item). Cells, lines and the detection convention are item 54's (item54_lines): lines on a box
are counted on S' for every cloud, so a cell holds the same GT in all four. Readings (T05 item 56): 1, restoring azimuth
rho = (O_rows - O_S') / (P_S' - O_S') per ring; 2, item 54's reading 3 on q_rows = O_rows / P_S' in the ring x 1-2-line cells;
3, c_rows = O_full - O_rows against c_az = O_full - O_az. Tests: tests/test_item56_halves.py.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis'))
import item54_lines as m54  # noqa: E402

REPO = TOOLS.parent
CFG_DIR = 'cfgs/da-ieee-access-analysis/item56'
FWD, REV = '/home/koyama/data/item47/item48_manifest_fwd.json', '/home/koyama/data/item47/item48_manifest_rev.json'
CKPTS = [('O', 666, 26843), ('O', 667, 31817), ('P', 666, 32283), ('P', 667, 32284)]
EPOCH = 152
ARMS = ('rows', 'az')
CLOUDS = ('full', 'sprime', 'rows', 'az')
FAR = (3, 4)                                   # item 48's rings 30-40, 40-70.4 m
MIN_GAP, RHO_HI, RHO_LO, AXIS = 5.0, 0.5, 0.2, 1.5


def cfg_of(row, arm):
    return '%s/centerpoint-item56-kitti-%s-on-nuscenes-%s.yaml' % (CFG_DIR, row.lower(), arm)


def tag(row, arm, seed, job):
    return 'i56K_%s_%s_none_s%d_%d' % (arm, row, seed, job)


def result_pkl(row, arm, seed, job):
    cfg, t = cfg_of(row, arm), tag(row, arm, seed, job)
    return REPO / 'output' / 'da-ieee-access-analysis' / 'item56' / Path(cfg).stem / t / 'eval' / ('epoch_%d' % EPOCH) / t / 'result.pkl'


def eval_rows():
    return [(tag(r, a, s, j), cfg_of(r, a), j, EPOCH, a) for a in ARMS for r, s, j in CKPTS]


def manifest():
    """The 16 scored rows; `row` is unique per (detector, cloud) so item54_lines keys stay distinct."""
    from item42_collect import read_log
    out = []
    for path, cloud in ((FWD, 'full'), (REV, 'sprime')):
        for r in json.load(open(path)):
            if r['stats'] == 'none' and r['row'] in ('O', 'P'):
                out.append(dict(det=r['row'], cloud=cloud, row='%s@%s' % (r['row'], cloud), stats='none', seed=r['seed'],
                                job=r['job'], result=r['result'], logged_bev=r['logged_bev'], status=r['status']))
    for arm in ARMS:
        for row, seed, job in CKPTS:
            pkl = result_pkl(row, arm, seed, job)
            got = read_log((pkl.parent, 'log_adabn_*.txt', None))
            out.append(dict(det=row, cloud=arm, row='%s@%s' % (row, arm), stats='none', seed=seed, job=job, result=str(pkl),
                            logged_bev=got['bev07'] if got else None, status='ok' if pkl.exists() and got else 'missing'))
    assert len(out) == 16, len(out)
    return out


def _cloud_check(task):
    """Per val frame: S' = S'_rows AND S'_az through the loader path, and the kept point counts."""
    from pcdet.datasets.kitti import kitti_factor_swap as fs
    from pcdet.datasets.kitti import kitti_rings as kr
    import kitti_eval_cpu
    fid = task
    pts = np.fromfile(kitti_eval_cpu.VELODYNE / ('%s.bin' % fid), dtype=np.float32).reshape(-1, 4)
    cfg = dict(_C['swap']['RING_THIN'])
    k = {m: kr.eval_ring_thin_mask(pts, dict(cfg, MODE=m), fs.frame_rng(fid, 'RING_THIN'))
         for m in ('pattern', 'pattern_rows', 'pattern_az')}
    assert np.array_equal(k['pattern'], k['pattern_rows'] & k['pattern_az']), fid
    return {'fid': fid, 'n': len(pts), 'sprime': int(k['pattern'].sum()), 'rows': int(k['pattern_rows'].sum()),
            'az': int(k['pattern_az'].sum())}


_C = {}


def _check_init(swap):
    import _init_path  # noqa: F401
    _C['swap'] = swap


def cmd_count(args):
    import _init_path  # noqa: F401
    import kitti_eval_cpu
    os.chdir(TOOLS)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    swap, train_swap = m54.load_swaps()
    rows = manifest()
    bad = [r for r in rows if r['status'] != 'ok']
    assert not bad, bad
    dets = {m54.row_key(r): {d['frame_id']: d['boxes_lidar'] for d in pickle.load(open(r['result'], 'rb'))} for r in rows}
    infos = pickle.load(open(kitti_eval_cpu.INFOS, 'rb'))
    tasks = [(i['point_cloud']['lidar_idx'], i['annos'], {k: v[i['point_cloud']['lidar_idx']] for k, v in dets.items()})
             for i in infos]
    with mp.get_context('spawn').Pool(args.workers, initializer=m54._count_init, initargs=(swap, train_swap)) as pool:
        val = pool.map(m54._count_val, tasks, chunksize=8)
    with mp.get_context('spawn').Pool(args.workers, initializer=_check_init, initargs=(swap,)) as pool:
        chk = pool.map(_cloud_check, [t[0] for t in tasks], chunksize=16)
    pickle.dump({'rows': rows, 'swap': swap, 'min_height': m54.MIN_HEIGHT, 'val': val}, open(out_dir / 'item56_counts.pkl', 'wb'))
    n = sum(c['n'] for c in chk)
    share = {k: sum(c[k] for c in chk) / n for k in ('sprime', 'rows', 'az')}
    json.dump({'frames': len(chk), 'kept_share': share}, open(out_dir / 'item56_clouds.json', 'w'), indent=1)
    print('frames %d: S\' = S\'_rows AND S\'_az on every frame; kept share %s' % (len(chk), share))


def cmd_score(args):
    out_dir = Path(args.out_dir)
    counts = out_dir / 'item56_counts.pkl'
    rows = pickle.load(open(counts, 'rb'))['rows']
    with mp.get_context('spawn').Pool(args.workers, initializer=m54._score_init, initargs=(args.real_kernel, str(counts))) as pool:
        res = pool.map(m54._score, rows, chunksize=1)
    json.dump({'kernel': 'cuda' if args.real_kernel else 'cpu', 'cells': [c[0] for c in m54.cells()], 'rows': res},
              open(out_dir / 'item56_ap.json', 'w'), indent=1, default=str)
    print('wrote', out_dir / 'item56_ap.json')


# ---------------------------------------------------------------- readings

def values(rows):
    """{(det, cloud, seed): {cell: bev}} over reproduced rows, and {cell: n_gt}."""
    v, n = {}, {}
    for r in rows:
        if not r.get('reproduced'):
            continue
        v[(r['det'], r['cloud'], r['seed'])] = {c['cell']: c['bev'] for c in r['cells']}
        n.update({c['cell']: c['n_gt'] for c in r['cells']})
    return v, n


RING_CELLS = ['ring %g-%g' % g for g in m54.RINGS]


def reading1(v):
    """rho per ring and seed; far = mean over the read far rings; None if a needed row is missing."""
    seeds = (666, 667)
    need = [(d, c, s) for d, c in (('O', 'rows'), ('O', 'sprime'), ('P', 'sprime')) for s in seeds]
    if any(k not in v for k in need):
        return 'not read', None
    gap = {s: [v[('P', 'sprime', s)][c] - v[('O', 'sprime', s)][c] for c in RING_CELLS] for s in seeds}
    read = [i for i in range(len(RING_CELLS)) if all(gap[s][i] >= MIN_GAP for s in seeds)]
    rho = {s: [((v[('O', 'rows', s)][RING_CELLS[i]] - v[('O', 'sprime', s)][RING_CELLS[i]]) / gap[s][i]) if i in read else None
               for i in range(len(RING_CELLS))] for s in seeds}
    far_i = [i for i in FAR if i in read]
    if not far_i:
        return 'not read', rho
    far = {s: float(np.mean([rho[s][i] for i in far_i])) for s in seeds}
    if all(far[s] >= RHO_HI for s in seeds):
        out = 'azimuth carries most of the far loss'
    elif all(far[s] <= RHO_LO for s in seeds):
        out = 'azimuth does not carry it'
    else:
        out = 'partly'
    return out, dict(rho=rho, far=far, rings_read=[RING_CELLS[i] for i in far_i])


def reading2(v, n):
    """Item 54's reading 3 on q_rows = O_rows / P_S' in the ring x 1-2-line cells."""
    t = {}
    for cell, ngt in n.items():
        for s in (666, 667):
            if ('O', 'rows', s) in v and ('P', 'sprime', s) in v:
                t.setdefault(cell, {})[s] = {'O': v[('O', 'rows', s)][cell], 'P': v[('P', 'sprime', s)][cell], 'n_gt': ngt}
    return m54.reading3(t)['few']


def reading3(v):
    seeds = (666, 667)
    need = [('O', c, s) for c in ('full', 'rows', 'az', 'sprime') for s in seeds]
    if any(k not in v for k in need):
        return 'not read', None
    cost = {(c, s): [v[('O', 'full', s)][r] - v[('O', c, s)][r] for r in RING_CELLS] for c in ('rows', 'az', 'sprime') for s in seeds}
    far = {(c, s): float(np.mean([cost[(c, s)][i] for i in FAR])) for c in ('rows', 'az', 'sprime') for s in seeds}
    if all(far[('az', s)] >= AXIS * far[('rows', s)] for s in seeds):
        out = 'azimuth larger'
    elif all(far[('rows', s)] >= AXIS * far[('az', s)] for s in seeds):
        out = 'vertical larger'
    else:
        out = 'comparable'
    inter = {s: [cost[('sprime', s)][i] - cost[('rows', s)][i] - cost[('az', s)][i] for i in range(len(RING_CELLS))] for s in seeds}
    return out, dict(cost={'%s s%d' % k: c for k, c in cost.items()}, far={'%s s%d' % k: f for k, f in far.items()},
                     interaction=inter)


def cmd_read(path):
    d = json.load(open(path))
    rows = d['rows']
    for r in rows:
        if not r.get('reproduced'):
            print('NOT REPRODUCED (not read): %s s%d pooled %.4f logged %s' % (r['row'], r['seed'], r['cells'][0]['bev'], r['logged_bev']))
    v, n = values(rows)
    show = ['pooled'] + RING_CELLS + ['%s few' % c for c in RING_CELLS] + ['%s many' % c for c in RING_CELLS] + \
        ['lines %s' % m54._lname(b) for b in m54.LINE_BINS]
    print('%-22s %s n_gt' % ('cell', ' '.join('%7s' % ('%s%s%d' % (dt, c[:2], s % 10)) for dt in 'OP' for c in CLOUDS for s in (666, 667))))
    for cell in show:
        print('%-22s %s %d' % (cell, ' '.join(
            '%7s' % ('-' if (dt, c, s) not in v or v[(dt, c, s)].get(cell) is None else '%.2f' % v[(dt, c, s)][cell])
            for dt in 'OP' for c in CLOUDS for s in (666, 667)), n.get(cell, 0)))
    r1, d1 = reading1(v)
    print('\nreading 1 (restoring azimuth): %s\n  %s' % (r1, d1))
    r2 = reading2(v, n)
    print('reading 2 (range at fixed 1-2 lines, O_rows / P_S\'): %s\n  %s' % r2)
    r3, d3 = reading3(v)
    print('reading 3 (which axis costs O more, far): %s\n  %s' % (r3, d3))
    return r1, r2[0], r3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('eval_rows')
    mf = sub.add_parser('manifest'); mf.add_argument('out_dir')
    c = sub.add_parser('count'); c.add_argument('out_dir'); c.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('score'); s.add_argument('out_dir'); s.add_argument('--workers', type=int, default=4)
    s.add_argument('--real-kernel', action='store_true')
    r = sub.add_parser('read'); r.add_argument('path')
    args = p.parse_args()
    if args.cmd == 'eval_rows':
        for row in eval_rows():
            print(' '.join(str(x) for x in row))
    elif args.cmd == 'manifest':
        rows = manifest()
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(Path(args.out_dir) / 'item56_manifest.json', 'w'), indent=1)
        for x in rows:
            print(x['row'], x['seed'], x['job'], x['status'], x['logged_bev'])
    elif args.cmd == 'count':
        cmd_count(args)
    elif args.cmd == 'score':
        cmd_score(args)
    else:
        cmd_read(args.path)


if __name__ == '__main__':
    main()
