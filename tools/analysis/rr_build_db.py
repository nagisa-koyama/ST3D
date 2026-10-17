"""RR v2 donor database (experiments_md 20261018_02 §6.2): every nuScenes TRAIN `car` instance of one platform,
aggregated over its track in the box frame from 27471's loader cloud. CPU; SOURCE labels only.

Per keyframe of the platform's train infos (sharded over SCENES: a track never leaves its scene): the loader's cloud
(`get_lidar_with_sweeps(i, MAX_SWEEPS)` with the per-object compensator, + SHIFT_COOR), the loader's boxes
(`gt_names == 'car'`, FILTER_MIN_POINTS_IN_GT on `num_lidar_pts`), the track id = devkit instance_token (the
compensator's `tids`). Each box's in-box points (roiaware test) go into that keyframe's box frame and into the instance's
exact 0.02 m centroid voxel accumulator; the keyframe is an observation if it held >= 1 point (relative viewing angle
stored). Size = per-dimension median of the instance's annotated sizes.

    python analysis/rr_build_db.py run --cfg <27471 cfg> --key NUSCENES_N008 --shard k --nshards N --out <dir>
    python analysis/rr_build_db.py merge --key NUSCENES_N008 --nshards N --dir <dir> [--min_points 200]
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
    scenes = sorted({comp.tok2scene[info['token']] for info in ds.infos})
    mine = set(scenes[a.shard::a.nshards])
    idx = [i for i, info in enumerate(ds.infos) if comp.tok2scene[info['token']] in mine][:a.max_frames]
    print(f'[rr_db] {a.key} shard {a.shard}/{a.nshards}: {len(mine)} of {len(scenes)} scenes, {len(idx)} of '
          f'{len(ds.infos)} keyframes; MAX_SWEEPS {max_sweeps}, SHIFT_COOR {shift.tolist()}, FILTER_MIN_POINTS_IN_GT '
          f'{fmin}', flush=True)
    acc, sizes, obs, nkf, scene_of = {}, {}, {}, {}, {}
    stats = dict(boxes=0, no_token=0, empty=0, kf=0)
    t0 = time.time()
    for n, i in enumerate(idx):
        info = ds.infos[i]
        f = comp.frames[comp.tok2frame[info['token']]]
        names = np.asarray(info['gt_names']); gb = np.asarray(info['gt_boxes'])[:, :7].astype(np.float64)
        assert len(f['tids']) == len(names) == len(gb), 'compensator tracks and info boxes disagree'
        m = names == 'car'
        if fmin:
            m &= np.asarray(info['num_lidar_pts']) > fmin - 1
        sel = np.nonzero(m)[0]
        stats['kf'] += 1
        if not len(sel):
            continue
        pts = ds.get_lidar_with_sweeps(i, max_sweeps=max_sweeps)
        pts[:, :3] += shift.astype(pts.dtype)
        xyz = pts[:, :3]
        grid = rr._BevGrid(xyz)
        for j in sel:
            stats['boxes'] += 1
            tid = f['tids'][j]
            if tid is None:
                stats['no_token'] += 1; continue
            b = gb[j].copy(); b[:3] += shift
            c = grid.query(rr.bev_corners(b[None], 0.02)[0])
            inb = c[rr.points_in_box(xyz[c], b)]
            sizes.setdefault(tid, []).append(b[3:6].copy()); nkf[tid] = nkf.get(tid, 0) + 1
            scene_of[tid] = comp.tok2scene[info['token']]
            if not len(inb):
                stats['empty'] += 1; continue
            loc = rr.to_box_frame(xyz[inb], b)
            a4 = np.c_[loc, pts[inb, 3]]
            acc.setdefault(tid, rr.VoxelAccumulator(0.02, 4)).add(a4)
            obs.setdefault(tid, []).append((rr.relative_view_angle(b), float(np.hypot(b[0], b[1])), len(inb)))
        if n % 200 == 0:
            print(f'[rr_db] {n + 1}/{len(idx)} keyframes, {len(acc)} instances, {time.time() - t0:.0f} s', flush=True)
    tids = sorted(sizes)
    pts_l, cnt, raw, size, varies, oo, on, oa, orng, onp, nk = [], [], [], [], [], [], [], [], [], [], []
    for t in tids:
        p = acc[t].points() if t in acc else np.zeros((0, 4))
        pts_l.append(p.astype(np.float32)); cnt.append(len(p)); raw.append(acc[t].n_raw if t in acc else 0)
        s = np.array(sizes[t]); size.append(np.median(s, 0)); varies.append(bool(np.ptp(s, 0).max() > 1e-6))
        ob = obs.get(t, [])
        on.append(len(ob)); oa += [o[0] for o in ob]; orng += [o[1] for o in ob]; onp += [o[2] for o in ob]
        nk.append(nkf[t])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'shard_{a.key}_{a.shard:02d}.npz', tids=np.array(tids), scene=np.array([scene_of[t] for t in tids]),
             points=np.concatenate(pts_l) if pts_l else np.zeros((0, 4), np.float32),
             count=np.array(cnt, np.int64), n_raw=np.array(raw, np.int64), size=np.array(size).reshape(-1, 3),
             size_varies=np.array(varies), obs_count=np.array(on, np.int64), obs_alpha=np.array(oa),
             obs_range=np.array(orng), obs_npts=np.array(onp, np.int64), n_keyframes=np.array(nk, np.int64),
             stats=np.array([stats['kf'], stats['boxes'], stats['no_token'], stats['empty']]))
    print(f'[rr_db] done shard {a.shard}: {len(tids)} instances, {int(np.sum(cnt))} aggregated points, stats {stats}, '
          f'{time.time() - t0:.0f} s', flush=True)


def merge(a):
    from pcdet.datasets.augmentor import relocate_render as rr
    D = Path(a.dir)
    sh = [np.load(D / f'shard_{a.key}_{k:02d}.npz') for k in range(a.nshards)]
    tids = np.concatenate([s['tids'] for s in sh])
    assert len(np.unique(tids)) == len(tids), 'an instance appears in two shards'
    count = np.concatenate([s['count'] for s in sh]); size = np.concatenate([s['size'] for s in sh])
    obs_count = np.concatenate([s['obs_count'] for s in sh]); varies = np.concatenate([s['size_varies'] for s in sh])
    n_raw = np.concatenate([s['n_raw'] for s in sh]); nk = np.concatenate([s['n_keyframes'] for s in sh])
    pts = [s['points'] for s in sh]; alpha = [s['obs_alpha'] for s in sh]; orng = [s['obs_range'] for s in sh]
    stats = np.sum([s['stats'] for s in sh], 0)
    # per-instance slices
    P = np.concatenate(pts); A = np.concatenate(alpha); OR = np.concatenate(orng)
    off = np.r_[0, np.cumsum(count)[:-1]]; ooff = np.r_[0, np.cumsum(obs_count)[:-1]]
    elig = (count >= a.min_points) & (obs_count >= 1)
    ids = np.nonzero(elig)[0]
    P_out = np.concatenate([P[off[i]:off[i] + count[i]] for i in ids]).astype(np.float32) if len(ids) else \
        np.zeros((0, 4), np.float32)
    c_out = count[ids]; o_out = np.r_[0, np.cumsum(c_out)[:-1]].astype(np.int64)
    oc_out = obs_count[ids]; oo_out = np.r_[0, np.cumsum(oc_out)[:-1]].astype(np.int64)
    a_out = np.concatenate([A[ooff[i]:ooff[i] + obs_count[i]] for i in ids]) if len(ids) else np.zeros(0)
    r_out = np.concatenate([OR[ooff[i]:ooff[i] + obs_count[i]] for i in ids]) if len(ids) else np.zeros(0)
    pp = D / f'db_{a.key}_points.npy'; ip = D / f'db_{a.key}_index.npz'
    np.save(pp, P_out)
    np.savez(ip, offset=o_out, count=c_out, size=size[ids], obs_offset=oo_out, obs_count=oc_out, obs_alpha=a_out,
             obs_range=r_out, tids=tids[ids], n_keyframes=nk[ids], n_raw=n_raw[ids])
    # database sanity: aggregated points inside the instance's box grown by 0.2 m on every side
    inside = 0; per = []
    for k in range(len(ids)):
        q = P_out[o_out[k]:o_out[k] + c_out[k], :3].astype(np.float64); s = size[ids[k]]
        m = (np.abs(q[:, 0]) <= s[0] / 2 + 0.2) & (np.abs(q[:, 1]) <= s[1] / 2 + 0.2) & (np.abs(q[:, 2]) <= s[2] / 2 + 0.2)
        inside += int(m.sum()); per.append(m.mean() if len(m) else np.nan)
    tot = int(c_out.sum())
    print(f'[rr_db] {a.key}: shards {a.nshards}; keyframes {stats[0]}, car boxes {stats[1]} (no token {stats[2]}, '
          f'0 points {stats[3]}); instances {len(tids)}, size varies within the track for {int(varies.sum())}; '
          f'eligible (>= {a.min_points} aggregated points, >= 1 observation) {len(ids)}')
    if len(ids):
        print(f'[rr_db] eligible: aggregated points per instance p10 / p50 / p90 {np.percentile(c_out, 10):.0f} / '
              f'{np.percentile(c_out, 50):.0f} / {np.percentile(c_out, 90):.0f}, total {tot}; observations per instance '
              f'median {np.median(oc_out):.0f}; keyframes per track median {np.median(nk[ids]):.0f}; raw in-box points '
              f'median {np.median(n_raw[ids]):.0f}')
        print(f'[rr_db] SANITY: {inside} of {tot} aggregated points inside the box grown by 0.2 m = {inside / tot:.4%} '
              f'(bar >= 95%) -> {"PASS" if inside / tot >= 0.95 else "FAIL"}; per-instance min {np.nanmin(per):.4f}, '
              f'p5 {np.nanpercentile(per, 5):.4f}')
    print(f'[rr_db] files: {pp} {os.path.getsize(pp) / 1e6:.1f} MB, {ip} {os.path.getsize(ip) / 1e6:.2f} MB')
    db = rr.DonorDB(pp, ip)
    assert db.n == len(ids) and (len(ids) == 0 or db.points(0).shape == (c_out[0], 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='mode', required=True)
    r = sp.add_parser('run')
    r.add_argument('--cfg', required=True); r.add_argument('--key', required=True); r.add_argument('--out', required=True)
    r.add_argument('--shard', type=int, required=True); r.add_argument('--nshards', type=int, required=True)
    r.add_argument('--max_frames', type=int, default=None, help='smoke test: first m keyframes of the shard')
    m = sp.add_parser('merge')
    m.add_argument('--key', required=True); m.add_argument('--dir', required=True)
    m.add_argument('--nshards', type=int, required=True); m.add_argument('--min_points', type=int, default=200)
    a = ap.parse_args()
    run(a) if a.mode == 'run' else merge(a)


if __name__ == '__main__':
    main()
