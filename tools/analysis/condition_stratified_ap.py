"""P2-A: condition-stratified reading of saved predictions on Waymo and nuScenes val (ANALYSIS).

experiments_md 20261028_02. Design FIXED in experiments_md 20261003_05 (T05) §5, entry "PHASE 2, design FIXED 12:04 JST
by the coordinator, before any AP", bullet P2-A (experiments_md af7680c); applied here unchanged. Target labels are
read to diagnose; nothing here selects or feeds a method value (memory/repo/uda_legality_rule.md, analysis exception).

Cells (tags from COND-TAB, experiments_md 20261028_01, job 30914):
- Waymo val at the eval sampling (SAMPLED_INTERVAL test 5, 7,998 frames), segment tags from
  /home/koyama/data/cond_tab_20261028/waymo_segments.csv: Day / Dawn/Dusk / Night with weather sunny, pooled over
  location, and the per-location Day | sunny cells (sf, phx, other). The 40 rain frames are in no cell.
- nuScenes val (all 6,019 keyframes), the E2 word rule (`nuscenes_condition_ap.frame_conditions`): Boston clear,
  Boston rain, Singapore clear, Singapore night. The 120 Singapore rain + night frames are in no cell.
Metric: Car BEV AP_R40 at IoU 0.7 ("all objects"; the target's paper metric), recomputed on a frame subset from the
saved result.pkl through `kitti_frame_decomposition` (the official evaluator per frame, CPU IoU kernel), with Car 3D
AP_R40 at IoU 0.7 beside (the same decomposition at metric 2). Every cell's AP is also asserted equal to the official
`eval_class` run on that cell's annos alone.

Quantities: per row and cell AP (BEV, 3D), frames, GT cars, cars per frame. Residual R = AP(oracle) - AP(row) per cell
(two-seed rows: the mean of the seeds' APs). Condition dependence D = R(condition) - R(base) inside one place:
Waymo Night - Day and Dawn/Dusk - Day (pooled), Boston rain - Boston clear, Singapore night - Singapore clear.
90% interval of D from scene / segment-level bootstrap resamples drawn within each cell (the same draws for every
row, so R and D are paired). Reading, mechanical: |D| >= 2 "the gap depends on the condition"; within +-2 "not
detectable"; a contrast with a cell under 300 val frames is reported, not read. Per-cell oracle AP is descriptive.

Two steps, so a row can be added later (the P2-B condition oracles, PL-GT 30904):

    # 1. capture one row (gate: the recomputed full-val Car BEV must equal the logged value within 0.05, else exit 3)
    python analysis/condition_stratified_ap.py capture --target waymo --name 30904 \\
        --from_log logs/error_30905_<x>.txt [--epoch 20] [--logged <bev> --logged_3d <3d>] [--cfg <cfg> --pkl <pkl>]
    # 2. read a target: oracle, row groups (two seeds = comma-separated names), optional pairs (AP(A) - AP(B) per cell)
    python analysis/condition_stratified_ap.py read --target waymo --oracle 27179 \\
        --row 'PL 27704/28563=27704,28563' --row 'PL-GT 30904=30904' [--pair 'PL-GT 30904:PL 27704/28563'] \\
        --n_boot 1000 --workers 8 --out <dir>/waymo

The bootstrap draws depend only on --seed and the cells' segment / scene lists, so re-running `read` with an extra row
reproduces every earlier number exactly. Run from ST3D/tools inside the container. CPU only.
"""
import argparse
import copy
import csv
import json
import pickle
import re
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kitti_frame_decomposition as kfd  # noqa: E402  (imports _init_path first)

COND_DIR = Path('/home/koyama/data/cond_tab_20261028')
WAYMO_SEGMENTS = COND_DIR / 'waymo_segments.csv'
NUSC_SCENES = COND_DIR / 'nuscenes_scenes.csv'
COND_TAB = COND_DIR / 'cond_tab.json'
DEFAULT_OUT = Path('/home/koyama/data/cond_p2a_20261028')
GATE_TOL = 0.05          # recomputed vs logged full-val Car BEV
MIN_FRAMES = 300         # cells under this are reported, not read (E2's rule)
BAR = 2.0                # |D| >= 2: the gap depends on the condition
METRIC_BEV, METRIC_3D = 1, 2

CELLS = {
    'waymo': ['Day', 'Dawn/Dusk', 'Night', 'sf Day', 'phx Day', 'other Day'],
    'nuscenes': ['Boston clear', 'Boston rain', 'Singapore clear', 'Singapore night'],
}
CONTRASTS = {   # (condition, base): D = R(condition) - R(base)
    'waymo': [('Night', 'Day'), ('Dawn/Dusk', 'Day')],
    'nuscenes': [('Boston rain', 'Boston clear'), ('Singapore night', 'Singapore clear')],
}
DATASET = {'waymo': 'WaymoDataset', 'nuscenes': 'NuScenesDataset'}


# ------------------------------------------------------------------------------------------------------------- logs
def host_path(p):
    """/storage/... (the container's bind of /home/koyama/data) -> a path that exists here."""
    p = str(p)
    if not Path(p).exists() and p.startswith('/storage/'):
        alt = '/home/koyama/data/' + p[len('/storage/'):]
        if Path(alt).exists():
            return alt
    return p


def parse_log(path, epoch=None):
    """cfg_file, and per evaluation (epoch, logged Car BEV / 3D AP_R40 @0.70, result dir) from a test.py / train.py log."""
    txt = open(path, errors='replace').read().replace('\r', '\n')
    lines = txt.split('\n')
    cfg, evals, cur, pending = None, [], None, None
    for i, line in enumerate(lines):
        if cfg is None and 'INFO' in line and re.search(r'\scfg_file\s', line):
            cfg = line.split('cfg_file')[-1].strip()
        m = re.search(r'EPOCH (\S+) of \S+ EVALUATION start', line)
        if m:
            cur, pending = m.group(1), None
        if line.startswith('Car AP_R40@0.70, 0.70, 0.70') and pending is None and cur is not None:
            vals = {}
            for k in range(i + 1, min(i + 5, len(lines))):
                for key in ('bev', '3d'):
                    if lines[k].startswith(key + ' ') and 'AP:' in lines[k]:
                        vals[key] = float(lines[k].split('AP:')[1].split(',')[0])
            assert set(vals) == {'bev', '3d'}, (path, i, vals)
            pending = dict(epoch=cur, bev=vals['bev'], d3=vals['3d'])
        if 'Result is save to' in line and pending is not None:
            pending['result_dir'] = line.split('Result is save to')[-1].strip()
            evals.append(pending)
            pending = None
    assert cfg is not None, f'no cfg_file line in {path}'
    assert evals, f'no completed Car AP_R40 evaluation in {path}'
    if epoch is None:
        ev = evals[-1]
    else:
        hits = [e for e in evals if e['epoch'] == str(epoch)]
        assert len(hits) >= 1, f'epoch {epoch} not evaluated in {path} (have {[e["epoch"] for e in evals]})'
        ev = hits[-1]
    return dict(cfg=cfg, **ev)


# ------------------------------------------------------------------------------------------------------ the capture
def capture_waymo(cfg_file, result_pkl, official):
    """kfd.capture_eval_annos for WaymoDataset: the dataset's own KITTI-metric evaluation on a saved result.pkl, its
    annos captured as they enter get_official_eval_result (class map, fabricated 2D boxes, every stored GT box)."""
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils

    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    eval_cfg = kfd.eval_data_config(cfg)
    assert eval_cfg.DATASET == 'WaymoDataset', eval_cfg.DATASET
    assert not eval_cfg.get('GT_FILTER', None), 'a GT filter would break the info <-> KITTI-anno box alignment'
    assert cfg.MODEL.POST_PROCESSING.EVAL_METRIC == 'kitti'
    class_names = list(cfg.CLASS_NAMES)
    assert ':' not in class_names[0], 'multi-dataset class names: eval_utils remaps them; not handled here'
    ontology = cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None)
    logger = common_utils.create_logger()
    dataset, _, _ = build_dataloader(dataset_cfg=eval_cfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False,
                                     workers=0, logger=logger, training=False, model_ontology=ontology)
    with open(result_pkl, 'rb') as f:
        det_annos = pickle.load(f)
    assert len(det_annos) == len(dataset.infos), (len(det_annos), len(dataset.infos))
    for d, info in zip(det_annos, dataset.infos):
        assert str(d['frame_id']) == str(info['frame_id']), 'result.pkl is not in the eval dataset order'

    captured = {}

    def capture(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
        captured.update(gt=gt_annos, dt=dt_annos, classes=list(current_classes))
        return '', {}

    shim = types.ModuleType(kfd._EVAL_MODULE)
    shim.get_official_eval_result = capture
    with kfd._module_swapped(shim):
        dataset.evaluation(copy.deepcopy(det_annos), class_names, eval_metric='kitti')
    assert 'Car' in captured['classes'], captured['classes']
    for g, info in zip(captured['gt'], dataset.infos):
        assert len(g['name']) == len(info['annos']['name'])
    ap = kfd.official_car_bev_ap(captured['gt'], captured['dt'], official)
    return dict(car_bev_r40=ap, gt=captured['gt'], dt=captured['dt'], infos=dataset.infos, class_names=class_names,
                eval_cfg=eval_cfg)


@contextmanager
def _metric(m):
    """kfd.CarBevFrames reads the module global METRIC_BEV at call time; metric 2 gives the same bookkeeping in 3D
    (compute_statistics_jit's only metric branch is the DontCare step of metric 0; verify=True asserts every state)."""
    saved = kfd.METRIC_BEV
    kfd.METRIC_BEV = m
    try:
        yield
    finally:
        kfd.METRIC_BEV = saved


def official_car_ap(gt_annos, dt_annos, official, metric):
    """Car AP_R40 at IoU 0.7, difficulty 0, official eval_class, one frame per part (as kfd.official_car_bev_ap)."""
    if not len(gt_annos):
        return float('nan')
    ret = official.eval_class(gt_annos, dt_annos, [kfd.CAR], [kfd.DIFFICULTY], metric,
                              np.full((1, 3, 1), kfd.MIN_OVERLAP), num_parts=len(gt_annos))
    return float(official.get_mAP_R40(ret['precision'])[0, 0, 0])


STATE_KEYS = ('n_frames', 'tp_offsets', 'tp_flat', 'n_valid', 'n_missed', 'ev_offsets', 'ev', 'total_valid_gt')


def frames_state(fr):
    return {k: getattr(fr, k) for k in STATE_KEYS}


def frames_from_state(state, official):
    fr = object.__new__(kfd.CarBevFrames)
    fr.__dict__.update(state)
    fr.official = official
    return fr


def waymo_tags():
    rows = [r for r in csv.DictReader(open(WAYMO_SEGMENTS)) if r['split'] == 'val']
    return {r['segment']: (r['location'].replace('location_', ''), r['time_of_day'], r['weather']) for r in rows}


def waymo_cells(loc, tod, weather):
    if weather != 'sunny':
        return []
    cells = [tod]
    if tod == 'Day':
        cells.append(f'{loc} Day')
    return cells


def frame_cells(target, cap):
    """Per frame: id, bootstrap group (segment / scene token), cell list."""
    if target == 'waymo':
        tags = waymo_tags()
        ids = [str(i['frame_id']) for i in cap['infos']]
        groups = [i['point_cloud']['lidar_sequence'] for i in cap['infos']]
        missing = sorted(set(groups) - set(tags))
        assert not missing, f'{len(missing)} val segments have no COND-TAB tag, e.g. {missing[:3]}'
        cells = [waymo_cells(*tags[g]) for g in groups]
    else:
        import nuscenes_condition_ap as nca  # E2's rule, unchanged
        ids = [str(g['token']) for g in cap['gt']]
        cond = nca.frame_conditions(ids)
        groups = [c[0] for c in cond]
        csv_cell = {r['scene_token']: (r['city'], r['cell']) for r in csv.DictReader(open(NUSC_SCENES))
                    if r['split'] == 'val'}
        for c in cond:   # COND-TAB's scene table agrees with E2's rule scene by scene
            assert csv_cell[c[0]] == (c[1], c[2]), (c, csv_cell.get(c[0]))
        cells = [[f'{c[1]} {c[2]}'] if f'{c[1]} {c[2]}' in CELLS['nuscenes'] else [] for c in cond]
    return ids, groups, cells


def cmd_capture(a):
    t0 = time.time()
    official = kfd.load_official_eval()
    info = dict(name=a.name, target=a.target)
    if a.from_log:
        lg = parse_log(a.from_log, a.epoch)
        info.update(log=str(a.from_log), log_epoch=lg['epoch'], log_result_dir=lg['result_dir'])
        cfg = a.cfg or lg['cfg']
        pkl = a.pkl or str(Path(host_path(lg['result_dir'])) / 'result.pkl')
        for key, given in (('bev', a.logged), ('d3', a.logged_3d)):
            if given is not None:
                assert abs(given - lg[key]) < 1e-4, f'--logged{"" if key == "bev" else "_3d"} {given} != log {lg[key]}'
        logged, logged_3d = lg['bev'], lg['d3']
    else:
        assert a.cfg and a.pkl and a.logged is not None, 'without --from_log give --cfg, --pkl and --logged'
        cfg, pkl, logged, logged_3d = a.cfg, a.pkl, a.logged, a.logged_3d
    pkl = host_path(pkl)
    assert Path(pkl).exists(), pkl
    try:   # git may be missing inside the container; the job script logs the commit on the host as well
        commit = subprocess.run(['git', '-C', str(kfd.TOOLS.parent), 'log', '-1', '--format=%h'], capture_output=True,
                                text=True).stdout.strip() or 'unknown'
    except OSError:
        commit = 'unknown'
    info.update(cfg=cfg, pkl=pkl, logged_bev=logged, logged_3d=logged_3d, code_commit=commit,
                pkl_mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(Path(pkl).stat().st_mtime)))
    print(f'== capture {a.name} ({a.target}): cfg {cfg}; pkl {pkl} (mtime {info["pkl_mtime"]}); '
          f'logged BEV {logged} / 3D {logged_3d}', flush=True)

    if a.target == 'nuscenes':
        cap = kfd.capture_eval_annos(cfg, pkl, official)        # E2's capture, unchanged
    else:
        cap = capture_waymo(cfg, pkl, official)
    assert cap['eval_cfg'].DATASET == DATASET[a.target], (cap['eval_cfg'].DATASET, a.target)
    diff = cap['car_bev_r40'] - float(logged)
    ok = abs(diff) <= GATE_TOL
    print(f'   frames {len(cap["gt"])}, recomputed BEV {cap["car_bev_r40"]:.4f} vs logged {float(logged):.4f} '
          f'(diff {diff:+.4f}, {"PASS" if ok else "FAIL"})', flush=True)
    if not ok:
        print(f'{a.name}: the recomputation does not reproduce the logged full-val Car BEV; nothing is stored.',
              flush=True)
        sys.exit(3)

    fr = kfd.CarBevFrames(cap['gt'], cap['dt'], official, verify=True)
    assert abs(fr.ap() - cap['car_bev_r40']) < 1e-6, (fr.ap(), cap['car_bev_r40'])
    with _metric(METRIC_3D):
        fr3 = kfd.CarBevFrames(cap['gt'], cap['dt'], official, verify=True)
    ap3_official = official_car_ap(cap['gt'], cap['dt'], official, METRIC_3D)
    assert abs(fr3.ap() - ap3_official) < 1e-6, (fr3.ap(), ap3_official)
    assert np.array_equal(fr.n_valid, fr3.n_valid)
    d3 = '' if logged_3d is None else f' vs logged {float(logged_3d):.4f} (diff {ap3_official - float(logged_3d):+.4f})'
    print(f'   3D recomputed {ap3_official:.4f}{d3}; frame decomposition = official on all frames, BEV and 3D; '
          f'{fr.n_state_checks} + {fr3.n_state_checks} threshold states verified', flush=True)

    ids, groups, cells = frame_cells(a.target, cap)
    cell_check = {}
    for cell in CELLS[a.target]:   # every cell's decomposed AP = the official evaluator on that cell's annos alone
        f = np.array([i for i, c in enumerate(cells) if cell in c], np.int64)
        gsub, dsub = [cap['gt'][i] for i in f], [cap['dt'][i] for i in f]
        ob, o3 = official_car_ap(gsub, dsub, official, METRIC_BEV), official_car_ap(gsub, dsub, official, METRIC_3D)
        db, d3c = fr.ap(f), fr3.ap(f)
        assert abs(ob - db) < 1e-6 and abs(o3 - d3c) < 1e-6, (cell, ob, db, o3, d3c)
        cell_check[cell] = dict(frames=int(len(f)), bev=db, d3=d3c)
        print(f'   {cell:16s} frames {len(f):5d}  GT cars {int(fr.n_valid[f].sum()):7d}  BEV {db:6.2f}  3D {d3c:6.2f}'
              f'  (= official on the cell)', flush=True)
    info.update(recomputed_bev=cap['car_bev_r40'], recomputed_3d=ap3_official, gate_diff=diff, frames=len(ids),
                frame_ids=ids, groups=groups, cells=cells, bev=frames_state(fr), d3=frames_state(fr3),
                cell_check=cell_check, seconds=time.time() - t0)
    out = Path(a.rows_dir) / a.target / f'{a.name}.pkl'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(info, f, protocol=4)
    tmp.replace(out)
    print(f'   wrote {out} ({time.time() - t0:.0f} s)', flush=True)


# ---------------------------------------------------------------------------------------------------------- the read
def expected_cells(target):
    """COND-TAB's val counts per cell (frames at the eval sampling, segments / scenes, raw GT Car)."""
    d = json.load(open(COND_TAB))
    out = {}
    if target == 'waymo':
        for cell in CELLS['waymo']:
            if cell in ('Day', 'Dawn/Dusk', 'Night'):
                e = d['waymo']['margins']['val'][f'{cell}|sunny']
            else:
                e = d['waymo']['tables']['val'][f'location_{cell.split()[0]}|Day|sunny']
            out[cell] = dict(frames=e['frames_i5'], groups=e['segments'], gt_cars=e['car_raw_i5'])
    else:
        for cell in CELLS['nuscenes']:
            e = d['nuscenes']['tables']['val'][cell.replace(' ', '|')]
            out[cell] = dict(frames=e['frames'], groups=e['scenes'], gt_cars=e['car_raw'])
    return out


_W = {}


def _boot_init(rows_dir, target, names, perms, cell_groups, draws):
    _W.update(official=kfd.load_official_eval(), rows_dir=rows_dir, target=target, perms=perms,
              cell_groups=cell_groups, draws=draws)


def _boot_one(name):
    with open(Path(_W['rows_dir']) / _W['target'] / f'{name}.pkl', 'rb') as f:
        st = pickle.load(f)['bev']
    fr = frames_from_state(st, _W['official'])
    perm = _W['perms'][name]
    out = {}
    for cell, groups in _W['cell_groups'].items():
        draws = _W['draws'][cell]
        vals = np.empty(len(draws))
        for b, idx in enumerate(draws):
            vals[b] = fr.ap(perm[np.concatenate([groups[i] for i in idx])])
        out[cell] = vals
    return name, out


def pct(v, q):
    return [float(x) for x in np.percentile(v, q)]


def reading(d, frames_c, frames_b):
    if min(frames_c, frames_b) < MIN_FRAMES:
        return f'reported, not read (a cell under {MIN_FRAMES} frames)'
    return 'the gap depends on the condition' if abs(d) >= BAR else 'not detectable'


def cmd_read(a):
    t0 = time.time()
    target = a.target
    official = kfd.load_official_eval()
    groups_spec = []                       # (label, [names])
    for spec in a.row:
        label, names = spec.split('=', 1)
        groups_spec.append((label.strip(), [n.strip() for n in names.split(',') if n.strip()]))
    all_names = [a.oracle] + [n for _, ns in groups_spec for n in ns]
    assert len(set(all_names)) == len(all_names), 'a name appears twice'
    rows = {}
    for n in all_names:
        p = Path(a.rows_dir) / target / f'{n}.pkl'
        assert p.exists(), f'no capture for {n} ({p}): run `capture` first (or it failed its gate)'
        with open(p, 'rb') as f:
            rows[n] = pickle.load(f)
        assert rows[n]['target'] == target
    canon = rows[a.oracle]
    ids, cells = canon['frame_ids'], canon['cells']
    pos = {fid: i for i, fid in enumerate(ids)}
    perms = {}
    for n, r in rows.items():   # align every row to the oracle's frame order by frame id
        assert len(r['frame_ids']) == len(ids) and set(r['frame_ids']) == set(ids), f'{n}: a different frame set'
        rpos = {fid: i for i, fid in enumerate(r['frame_ids'])}
        perm = np.array([rpos[fid] for fid in ids], np.int64)
        assert all(r['cells'][perm[i]] == cells[i] for i in range(len(ids))), f'{n}: cell tags differ'
        perms[n] = perm
    n_valid = rows[a.oracle]['bev']['n_valid']
    for n, r in rows.items():
        assert np.array_equal(r['bev']['n_valid'][perms[n]], n_valid), f'{n}: GT cars per frame differ'

    # cells, their bootstrap groups, and the COND-TAB check
    cell_frames, cell_groups = {}, {}
    exp = expected_cells(target)
    checks = {}
    for cell in CELLS[target]:
        f = np.array([i for i, c in enumerate(cells) if cell in c], np.int64)
        g = {}
        for i in f:
            g.setdefault(canon['groups'][i], []).append(i)
        cell_frames[cell] = f
        cell_groups[cell] = [np.asarray(v, np.int64) for _, v in sorted(g.items())]
        got = dict(frames=int(len(f)), groups=len(g), gt_cars=int(n_valid[f].sum()))
        checks[cell] = dict(expected=exp[cell], got=got, ok=got == exp[cell])
        assert got == exp[cell], f'{cell}: {got} != COND-TAB {exp[cell]}'
    print(f'== read {target}: {len(ids)} frames; every cell matches COND-TAB in frames, '
          f'{"segments" if target == "waymo" else "scenes"} and GT cars', flush=True)

    # point estimates
    def ap_of(name, metric, frames=None):
        fr = frames_from_state(rows[name]['bev' if metric == METRIC_BEV else 'd3'], official)
        return fr.ap(None if frames is None else perms[name][frames])
    per_row = {}
    for n in all_names:
        r = rows[n]
        per_row[n] = dict(cfg=r['cfg'], pkl=r['pkl'], log=r.get('log'), log_epoch=r.get('log_epoch'),
                          logged_bev=r['logged_bev'], logged_3d=r['logged_3d'], recomputed_bev=r['recomputed_bev'],
                          recomputed_3d=r['recomputed_3d'], gate_diff=r['gate_diff'], code_commit=r['code_commit'],
                          full_bev=ap_of(n, METRIC_BEV), full_3d=ap_of(n, METRIC_3D),
                          cells={c: dict(bev=ap_of(n, METRIC_BEV, f), d3=ap_of(n, METRIC_3D, f))
                                 for c, f in cell_frames.items()})
        assert abs(per_row[n]['full_bev'] - r['recomputed_bev']) < 1e-6
    cell_info = {c: dict(frames=int(len(f)), groups=len(cell_groups[c]), gt_cars=int(n_valid[f].sum()),
                         cars_per_frame=float(n_valid[f].sum() / len(f))) for c, f in cell_frames.items()}

    # bootstrap: the same draws for every row (paired), drawn per cell from --seed only
    rng = np.random.default_rng(a.seed)
    draws = {c: rng.integers(0, len(cell_groups[c]), size=(a.n_boot, len(cell_groups[c]))) for c in CELLS[target]}
    t1 = time.time()
    with get_context('spawn').Pool(min(a.workers, len(all_names)), initializer=_boot_init,
                                   initargs=(a.rows_dir, target, all_names, perms, cell_groups, draws)) as pool:
        boot = dict(pool.map(_boot_one, all_names, chunksize=1))
    print(f'   bootstrap: {a.n_boot} resamples x {len(all_names)} rows x {len(CELLS[target])} cells '
          f'({time.time() - t1:.0f} s)', flush=True)

    def group_vals(names, metric_key, cell):
        return float(np.mean([per_row[n]['cells'][cell][metric_key] for n in names]))

    def group_boot(names, cell):
        return np.mean([boot[n][cell] for n in names], axis=0)

    oracle_g = [a.oracle]
    groups_out = {}
    for label, names in groups_spec:
        g = dict(names=names, full_bev=float(np.mean([per_row[n]['full_bev'] for n in names])),
                 full_3d=float(np.mean([per_row[n]['full_3d'] for n in names])),
                 full_logged_bev=float(np.mean([per_row[n]['logged_bev'] for n in names])), cells={}, contrasts={},
                 per_seed_contrasts={})
        for cell in CELLS[target]:
            bev, d3 = group_vals(names, 'bev', cell), group_vals(names, 'd3', cell)
            o_bev, o_d3 = group_vals(oracle_g, 'bev', cell), group_vals(oracle_g, 'd3', cell)
            rb = group_boot(oracle_g, cell) - group_boot(names, cell)
            g['cells'][cell] = dict(bev=bev, d3=d3, R=o_bev - bev, R_3d=o_d3 - d3, R_ci90=pct(rb, [5, 95]),
                                    per_seed_bev={n: per_row[n]['cells'][cell]['bev'] for n in names})
        for cond, base in CONTRASTS[target]:
            d = g['cells'][cond]['R'] - g['cells'][base]['R']
            d3 = g['cells'][cond]['R_3d'] - g['cells'][base]['R_3d']
            db = ((group_boot(oracle_g, cond) - group_boot(names, cond))
                  - (group_boot(oracle_g, base) - group_boot(names, base)))
            g['contrasts'][f'{cond} - {base}'] = dict(
                D=d, D_3d=d3, ci90=pct(db, [5, 95]), boot_mean=float(db.mean()),
                reading=reading(d, cell_info[cond]['frames'], cell_info[base]['frames']))
            if len(names) > 1:
                for n in names:
                    dn = ((per_row[a.oracle]['cells'][cond]['bev'] - per_row[n]['cells'][cond]['bev'])
                          - (per_row[a.oracle]['cells'][base]['bev'] - per_row[n]['cells'][base]['bev']))
                    dbn = (boot[a.oracle][cond] - boot[n][cond]) - (boot[a.oracle][base] - boot[n][base])
                    g['per_seed_contrasts'].setdefault(n, {})[f'{cond} - {base}'] = dict(D=dn,
                                                                                        ci90=pct(dbn, [5, 95]))
        groups_out[label] = g

    pairs_out = {}
    lookup = dict(groups_spec + [('oracle', oracle_g)])
    for spec in a.pair or []:
        la, lb = (s.strip() for s in spec.split(':', 1))
        na, nb = lookup[la], lookup[lb]
        p = {}
        for cell in CELLS[target]:
            c = group_vals(na, 'bev', cell) - group_vals(nb, 'bev', cell)
            cb = group_boot(na, cell) - group_boot(nb, cell)
            p[cell] = dict(C=c, C_3d=group_vals(na, 'd3', cell) - group_vals(nb, 'd3', cell), ci90=pct(cb, [5, 95]),
                           frames=cell_info[cell]['frames'])
        pairs_out[f'{la} - {lb}'] = p

    res = dict(design='experiments_md T05 §5 P2-A (af7680c)', target=target, oracle=a.oracle, n_boot=a.n_boot,
               seed=a.seed, min_frames=MIN_FRAMES, bar=BAR, cells=cell_info, cond_tab_check=checks, rows=per_row,
               groups=groups_out, pairs=pairs_out, seconds=time.time() - t0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'p2a.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, default=float)
    md = markdown(res)
    (out / 'p2a.md').write_text(md, encoding='utf-8')
    print(md, flush=True)
    print(f'wrote {out}/p2a.json, p2a.md ({time.time() - t0:.0f} s)', flush=True)


def markdown(res):
    t, cells = res['target'], CELLS[res['target']]
    o = res['oracle']
    L = [f'## {t}: oracle {o}, {res["n_boot"]} bootstrap resamples (seed {res["seed"]})', '',
         '### Full val (gate: recomputed - logged within 0.05)', '',
         '| row | logged BEV / 3D | recomputed BEV / 3D | diff BEV |', '|---|---|---|---|']
    for n, r in res['rows'].items():
        l3 = '-' if r['logged_3d'] is None else f'{r["logged_3d"]:.4f}'
        L.append(f'| {n} | {r["logged_bev"]:.4f} / {l3} | {r["recomputed_bev"]:.4f} / {r["recomputed_3d"]:.4f} | '
                 f'{r["gate_diff"]:+.4f} |')
    L += ['', '### Cells', '', '| cell | frames | segments / scenes | GT cars | cars / frame |', '|---|---|---|---|---|']
    for c in cells:
        ci = res['cells'][c]
        L.append(f'| {c} | {ci["frames"]:,} | {ci["groups"]} | {ci["gt_cars"]:,} | {ci["cars_per_frame"]:.1f} |')
    L += ['', '### Car BEV AP_R40 @0.7 per cell (3D in brackets)', '',
          '| row | full val | ' + ' | '.join(cells) + ' |', '|---|---|' + '---|' * len(cells)]
    for n, r in res['rows'].items():
        L.append(f'| {n} | {r["full_bev"]:.2f} ({r["full_3d"]:.2f}) | ' +
                 ' | '.join(f'{r["cells"][c]["bev"]:.2f} ({r["cells"][c]["d3"]:.2f})' for c in cells) + ' |')
    L += ['', '### Residual R = oracle - row per cell (two-seed rows: mean of seeds); 3D in brackets', '',
          '| row | full val | ' + ' | '.join(cells) + ' |', '|---|---|' + '---|' * len(cells)]
    ofull, ofull3 = res['rows'][o]['full_bev'], res['rows'][o]['full_3d']
    for lab, g in res['groups'].items():
        L.append(f'| {lab} | {ofull - g["full_bev"]:.2f} ({ofull3 - g["full_3d"]:.2f}) | ' +
                 ' | '.join(f'{g["cells"][c]["R"]:.2f} ({g["cells"][c]["R_3d"]:.2f})' for c in cells) + ' |')
    L += ['', '### Condition dependence D = R(condition) - R(base), 90% bootstrap interval, reading', '',
          '| row | contrast | D BEV | 90% interval | D 3D | per seed (BEV) | reading |', '|---|---|---|---|---|---|---|']
    for lab, g in res['groups'].items():
        for k, c in g['contrasts'].items():
            ps = '; '.join(f'{n} {v[k]["D"]:+.2f}' for n, v in g['per_seed_contrasts'].items()) or '-'
            L.append(f'| {lab} | {k} | {c["D"]:+.2f} | [{c["ci90"][0]:+.2f}, {c["ci90"][1]:+.2f}] | '
                     f'{c["D_3d"]:+.2f} | {ps} | {c["reading"]} |')
    if res['pairs']:
        L += ['', '### Pairs: C = AP(A) - AP(B) per cell, 90% bootstrap interval (no reading here)', '',
              '| pair | ' + ' | '.join(cells) + ' |', '|---|' + '---|' * len(cells)]
        for k, p in res['pairs'].items():
            L.append(f'| {k} | ' + ' | '.join(f'{p[c]["C"]:+.2f} [{p[c]["ci90"][0]:+.2f}, {p[c]["ci90"][1]:+.2f}]'
                                             for c in cells) + ' |')
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('capture', help='recompute one row on the full val (gate) and store its per-frame records')
    c.add_argument('--target', required=True, choices=sorted(CELLS))
    c.add_argument('--name', required=True)
    c.add_argument('--from_log', help='test.py / train.py log: cfg, result dir and logged AP of the evaluation')
    c.add_argument('--epoch', help='the evaluated epoch to take from --from_log (default: the last)')
    c.add_argument('--cfg')
    c.add_argument('--pkl')
    c.add_argument('--logged', type=float, help='logged full-val Car BEV AP_R40 @0.7 (checked against the log)')
    c.add_argument('--logged_3d', type=float)
    c.add_argument('--rows_dir', default=str(DEFAULT_OUT / 'rows'))
    r = sub.add_parser('read', help='per-cell AP, R, D, bootstrap and readings for one target')
    r.add_argument('--target', required=True, choices=sorted(CELLS))
    r.add_argument('--oracle', required=True)
    r.add_argument('--row', action='append', required=True, help="'<label>=<name>[,<name of seed 2>]'")
    r.add_argument('--pair', action='append', help="'<label A>:<label B>' (labels of --row, or 'oracle')")
    r.add_argument('--n_boot', type=int, default=1000)
    r.add_argument('--seed', type=int, default=20261028)
    r.add_argument('--workers', type=int, default=8)
    r.add_argument('--rows_dir', default=str(DEFAULT_OUT / 'rows'))
    r.add_argument('--out', required=True)
    a = ap.parse_args()
    {'capture': cmd_capture, 'read': cmd_read}[a.cmd](a)


if __name__ == '__main__':
    main()
