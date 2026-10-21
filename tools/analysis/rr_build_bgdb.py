"""RR v8 background donor database (experiments_md 20261018_02 §13; T05 §5, experiments_md ae0cb00). CPU; SOURCE labels only
(to exclude objects).

Per keyframe of a platform's nuScenes TRAIN infos (sharded over scenes, as the car database `rr_build_db.py`): the loader's
accumulated, motion-compensated cloud (`get_lidar_with_sweeps(i, MAX_SWEEPS)` + SHIFT_COOR, as the car database), every
annotated box of ANY class, the local ground (relocate_render.GroundCells / local_bottom: the 20th percentile of the 1 m
cells' second-lowest z within 5 m, >= 3 cells, else the frame's 4-40 m plane). Candidate regions are upright boxes with the
dimensions of a car donor drawn at random from the car database, centred at r ~ U[5, 30] m with uniform azimuth and yaw, the
bottom on the local ground. A region is accepted if it intersects no annotated box grown by 0.5 m in BEV (nor an earlier
region of the same keyframe) and holds >= 200 points more than 0.3 m above the local ground; only those points are kept, in
the region's box frame with their intensity, 0.02 m centroid voxels (as the car donors). Up to 2 regions per keyframe, up to
100 candidates. The same pass counts the keyframes that hold >= 1 in-box point of an eligible car-database instance (the car
database's contributing keyframes), for the size condition.

    python analysis/rr_build_bgdb.py run --cfg <27471 cfg> --key NUSCENES_N008 --car_dir <v2 dir> --shard k --nshards N --out <dir>
    python analysis/rr_build_bgdb.py merge --key NUSCENES_N008 --nshards N --dir <dir> --car_dir <v2 dir>
Normals: `rr_build_db.py normals / normals_merge --dir <dir>` (the v5 procedure; the files are named as the car database's).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _init_path  # noqa: F401,E402

R_LO, R_HI = 5.0, 30.0
GROW = 0.5                      # BEV_MARGIN
CLEAR = 0.3                     # CLEARANCE
MIN_PTS = 200
MAX_TRIES, MAX_ACCEPT = 100, 2
GROUND = dict(cell=1.0, min_pts=3, radius=5.0, min_cells=3, pct=20.0)     # the GROUND_* rule
VOXEL = 0.02


def sample_regions(xyz, gt7, car_sizes, rng, plane, cells, grid=None, r_lo=R_LO, r_hi=R_HI, max_tries=MAX_TRIES,
                   max_accept=MAX_ACCEPT, min_pts=MIN_PTS, grow=GROW, clear=CLEAR):
    """Accepted background regions of one keyframe: list of (box (7,), indices of the kept points), and the tries made.
    xyz (n, 3) the loader cloud; gt7 (m, 7) every annotated box; car_sizes (k, 3) the car donors' (l, w, h)."""
    from pcdet.datasets.augmentor import relocate_render as rr
    grid = grid if grid is not None else rr._BevGrid(xyz)
    gt_c = rr.bev_corners(np.asarray(gt7, np.float64).reshape(-1, 7), grow) if len(gt7) else np.zeros((0, 4, 2))
    out, tries = [], 0
    taken = np.zeros((0, 4, 2))
    while tries < max_tries and len(out) < max_accept:
        tries += 1
        s = car_sizes[int(rng.integers(len(car_sizes)))]
        r = rng.uniform(r_lo, r_hi); phi = rng.uniform(0.0, 2 * np.pi); yaw = rng.uniform(0.0, 2 * np.pi)
        x, y = r * np.cos(phi), r * np.sin(phi)
        zb = rr.local_bottom(cells, x, y, plane, GROUND['radius'], GROUND['min_cells'], GROUND['pct'])[0]
        box = np.array([x, y, zb + s[2] / 2, s[0], s[1], s[2], yaw], np.float64)
        c4 = rr.bev_corners(box[None])[0]
        if len(gt_c) and rr.rects_overlap(c4, gt_c).any():
            continue
        if len(taken) and rr.rects_overlap(c4, taken).any():
            continue
        c = grid.query(rr.bev_corners(box[None], 0.02)[0])
        if not len(c):
            continue
        inb = c[rr.points_in_box(xyz[c], box)]
        keep = inb[xyz[inb, 2] > zb + clear]
        if len(keep) < min_pts:
            continue
        out.append((box, np.sort(keep)))
        taken = np.concatenate([taken, c4[None]])
    return out, tries


def run(a):
    import torch
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.datasets.augmentor import relocate_render as rr
    from pcdet.utils import common_utils
    torch.set_num_threads(1)
    cfg = EasyDict(); cfg_from_yaml_file(a.cfg, cfg)
    dcfg = cfg.DATA_CONFIGS[a.key]
    ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=common_utils.create_logger(), training=True, model_ontology=cfg.get('ONTOLOGY'))
    comp = ds._sweep_compensator
    assert comp is not None, 'the row compensates; the database must be built with the compensator'
    shift = np.asarray(dcfg.SHIFT_COOR, np.float64)
    fmin = int(dcfg.get('FILTER_MIN_POINTS_IN_GT', 0) or 0)
    max_sweeps = int(dcfg.MAX_SWEEPS)
    car = np.load(Path(a.car_dir) / f'db_{a.key}_index.npz')
    car_sizes = np.asarray(car['size'], np.float64); car_tids = set(str(t) for t in car['tids'])
    scenes = sorted({comp.tok2scene[info['token']] for info in ds.infos})
    mine = set(scenes[a.shard::a.nshards])
    idx = [i for i, info in enumerate(ds.infos) if comp.tok2scene[info['token']] in mine][:a.max_frames]
    print(f'[bg_db] {a.key} shard {a.shard}/{a.nshards}: {len(mine)} of {len(scenes)} scenes, {len(idx)} of {len(ds.infos)} '
          f'keyframes; MAX_SWEEPS {max_sweeps}, SHIFT_COOR {shift.tolist()}; car database {len(car_sizes)} donors', flush=True)
    pts_l, cnt, raw, size, alpha, rng_l, frame_l, tries_l = [], [], [], [], [], [], [], []
    car_frames = 0; no_plane = 0; acc_frames = 0
    t0 = time.time()
    for n, i in enumerate(idx):
        info = ds.infos[i]
        f = comp.frames[comp.tok2frame[info['token']]]
        names = np.asarray(info['gt_names']); gb = np.asarray(info['gt_boxes'])[:, :7].astype(np.float64).copy()
        gb[:, :3] += shift
        pts = ds.get_lidar_with_sweeps(i, max_sweeps=max_sweeps)
        pts[:, :3] += shift.astype(pts.dtype)
        xyz = np.asarray(pts[:, :3], np.float64)
        grid = rr._BevGrid(xyz)
        # the car database's contributing keyframes: >= 1 in-box point of an eligible car instance
        m = names == 'car'
        if fmin:
            m &= np.asarray(info['num_lidar_pts']) > fmin - 1
        hit = False
        for j in np.nonzero(m)[0]:
            tid = f['tids'][j]
            if tid is None or str(tid) not in car_tids:
                continue
            c = grid.query(rr.bev_corners(gb[j][None], 0.02)[0])
            if len(c) and rr.points_in_box(xyz[c], gb[j]).any():
                hit = True; break
        car_frames += int(hit)
        plane = rr.ground_plane(xyz, 40.0)
        if plane is None:
            no_plane += 1; continue
        cells = rr.GroundCells(xyz, GROUND['cell'], GROUND['min_pts'])
        rng = np.random.default_rng([a.seed, i])
        regs, tries = sample_regions(xyz, gb, car_sizes, rng, plane, cells, grid)
        acc_frames += int(len(regs) > 0)
        for box, keep in regs:
            va = rr.VoxelAccumulator(VOXEL, 4)
            va.add(np.c_[rr.to_box_frame(xyz[keep], box), pts[keep, 3]])
            p = va.points().astype(np.float32)
            pts_l.append(p); cnt.append(len(p)); raw.append(len(keep)); size.append(box[3:6].copy())
            alpha.append(rr.relative_view_angle(box)); rng_l.append(float(np.hypot(box[0], box[1]))); frame_l.append(i)
            tries_l.append(tries)
        if n % 200 == 0:
            print(f'[bg_db] {n + 1}/{len(idx)} keyframes, {len(cnt)} regions from {acc_frames} keyframes, car-contributing '
                  f'keyframes {car_frames}, {time.time() - t0:.0f} s', flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'bgshard_{a.key}_{a.shard:02d}.npz',
             points=np.concatenate(pts_l) if pts_l else np.zeros((0, 4), np.float32), count=np.array(cnt, np.int64),
             n_raw=np.array(raw, np.int64), size=np.array(size).reshape(-1, 3), obs_alpha=np.array(alpha),
             obs_range=np.array(rng_l), frame=np.array(frame_l, np.int64), tries=np.array(tries_l, np.int64),
             stats=np.array([len(idx), acc_frames, car_frames, no_plane]))
    print(f'[bg_db] done shard {a.shard}: {len(cnt)} regions from {acc_frames} of {len(idx)} keyframes (no plane {no_plane}); '
          f'car-contributing keyframes {car_frames}; {time.time() - t0:.0f} s', flush=True)


def merge(a):
    from pcdet.datasets.augmentor import relocate_render as rr
    D = Path(a.dir)
    sh = [np.load(D / f'bgshard_{a.key}_{k:02d}.npz') for k in range(a.nshards)]
    count = np.concatenate([s['count'] for s in sh]); size = np.concatenate([s['size'] for s in sh])
    P = np.concatenate([s['points'] for s in sh]).astype(np.float32)
    A = np.concatenate([s['obs_alpha'] for s in sh]); OR = np.concatenate([s['obs_range'] for s in sh])
    frame = np.concatenate([s['frame'] for s in sh]); n_raw = np.concatenate([s['n_raw'] for s in sh])
    st = np.sum([s['stats'] for s in sh], 0)
    n = len(count)
    pp = D / f'db_{a.key}_points.npy'; ip = D / f'db_{a.key}_index.npz'
    np.save(pp, P)
    np.savez(ip, offset=np.r_[0, np.cumsum(count)[:-1]].astype(np.int64), count=count, size=size,
             obs_offset=np.arange(n, dtype=np.int64), obs_count=np.ones(n, np.int64), obs_alpha=A, obs_range=OR,
             frame=frame, n_raw=n_raw)
    car = np.load(Path(a.car_dir) / f'db_{a.key}_index.npz')
    nfr = len(np.unique(frame))
    ok_n, ok_f = n >= len(car['count']), nfr >= int(st[2])
    print(f'[bg_db] {a.key}: shards {a.nshards}; keyframes processed {st[0]} (no plane {st[3]}); background donors {n} from '
          f'{nfr} keyframes; car database {len(car["count"])} donors, contributing keyframes {st[2]}; size condition: donors '
          f'{"PASS" if ok_n else "FAIL"}, keyframes {"PASS" if ok_f else "FAIL"}')
    if n:
        print(f'[bg_db] kept points per donor p10 / p50 / p90 {np.percentile(n_raw, 10):.0f} / {np.percentile(n_raw, 50):.0f} / '
              f'{np.percentile(n_raw, 90):.0f} (voxelised {np.percentile(count, 50):.0f} median); donor range p10 / p50 / p90 '
              f'{np.percentile(OR, 10):.1f} / {np.percentile(OR, 50):.1f} / {np.percentile(OR, 90):.1f} m')
        q = P[:, :3].astype(np.float64)
        rep = np.repeat(np.arange(n), count)
        s = size[rep]
        inside = (np.abs(q[:, 0]) <= s[:, 0] / 2 + 1e-4) & (np.abs(q[:, 1]) <= s[:, 1] / 2 + 1e-4) & \
            (q[:, 2] <= s[:, 2] / 2 + 1e-4) & (q[:, 2] > -s[:, 2] / 2 + CLEAR - 0.02)
        print(f'[bg_db] SANITY: {int(inside.sum())} of {len(q)} stored points inside the region and above the 0.3 m clearance '
              f'(0.02 m voxel tolerance) = {inside.mean():.4%} -> {"PASS" if inside.mean() >= 0.999 else "FAIL"}')
    print(f'[bg_db] files: {pp} {os.path.getsize(pp) / 1e6:.1f} MB, {ip} {os.path.getsize(ip) / 1e6:.2f} MB')
    db = rr.DonorDB(pp, ip)
    assert db.n == n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='mode', required=True)
    r = sp.add_parser('run')
    r.add_argument('--cfg', required=True); r.add_argument('--key', required=True); r.add_argument('--out', required=True)
    r.add_argument('--car_dir', required=True)
    r.add_argument('--shard', type=int, required=True); r.add_argument('--nshards', type=int, required=True)
    r.add_argument('--seed', type=int, default=0); r.add_argument('--max_frames', type=int, default=None)
    m = sp.add_parser('merge')
    m.add_argument('--key', required=True); m.add_argument('--dir', required=True); m.add_argument('--car_dir', required=True)
    m.add_argument('--nshards', type=int, required=True)
    a = ap.parse_args()
    {'run': run, 'merge': merge}[a.mode](a)


if __name__ == '__main__':
    main()
