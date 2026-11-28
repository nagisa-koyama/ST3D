"""Check an S3 arm smoke's TARGET DUMP against the arm's declared effect (experiments_md episodes/20261116_01 §8.10;
bug_prevention_checklist rule 4: a smoke checks targets, not just that the run trains).

train_seg.py writes target_dump.json at the start of every run, in the main process, from build_datasets(cfg): per-class
point and voxel counts on 4 frames per split (engine.target_dump). This script rebuilds the same datasets on CPU, with
the arm's block and without it (the oracle's base), and checks:

  A  the job's dump equals the dump recomputed here from the same config (the job ran the declared config): eval rows
     in every field, train rows in their pre-augmentation fields (DETERMINISTIC; the voxel fields follow the random
     training augmentation);
  B  the split the arm does not touch is identical to the base, key for key, on the same fields (no leak: a
     FACTOR_SWAP arm's val is the unperturbed val, an EVAL_PERTURB config's train is the oracle's);
  C  the touched split differs from the base as the arm's operator says, frame by frame:
       DENSITY    fewer points, every class count <= the base's, and every frame thinned;
       CONE       points and class counts equal the base's points inside the evaluation's cone;
       INTENSITY  same points and class counts, same coordinates, intensities changed on most points;
       RENDER     points equal the cache record's, and class counts equal the source labels read at the cached
                  (frame, index) straight from read_frame, independently of RenderCache.load; intensities likewise.

    python3 analysis/pcseg_arm_smoke_check.py --cfg cfgs/pcseg/spunet-p1-a1-density-smoke.yaml \
        --dump /home/koyama/data/wandb/run-.../files/target_dump.json --out <dir>/arm_smoke_check.json

Run from <worktree>/tools. Exit status 1 if any check fails.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '../..'))

OPS = ('DENSITY', 'CONE', 'INTENSITY', 'RENDER')


def strip_blocks(cfg):
    """The oracle's base: the same config without FACTOR_SWAP / EVAL_PERTURB."""
    base = copy.deepcopy(cfg)
    for k in ('FACTOR_SWAP', 'EVAL_PERTURB'):
        base.DATA.pop(k, None)
    return base


def arm_block(cfg):
    """(split it touches, operator names) of a config holding exactly one of the two blocks."""
    have = [k for k in ('FACTOR_SWAP', 'EVAL_PERTURB') if cfg.DATA.get(k, None) is not None]
    if len(have) != 1:
        raise ValueError('expected exactly one of FACTOR_SWAP / EVAL_PERTURB, got %s' % have)
    ops = sorted(cfg.DATA[have[0]])
    if not ops or set(ops) - set(OPS):
        raise ValueError('unknown or empty operator set %s' % ops)
    return ('train' if have[0] == 'FACTOR_SWAP' else 'val'), ops


# A TRAINING sample is drawn after the random training augmentation (the global numpy RNG, which wandb.init and the
# logging touch between the job's seeding and its dump), so only a train row's pre-augmentation fields are
# deterministic. An eval row is deterministic in full (frame-keyed voxel draw). The job's own target_dump already
# raises if a voxel carries a label its points do not.
DETERMINISTIC = ('frame', 'points', 'ignored_points', 'point_counts')


def dump_diff(a, b, fields=None):
    """Differences between two target dumps (or two row lists), as readable strings; [] if identical. On whole dumps,
    the 'train' split is compared on DETERMINISTIC fields only; `fields` restricts a row-list comparison."""
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append('split %s only in one dump' % k)
            else:
                out += ['%s: %s' % (k, d) for d in dump_diff(a[k], b[k], split_fields(k))]
        return out
    if len(a) != len(b):
        return ['%d rows against %d' % (len(a), len(b))]
    out = []
    for ra, rb in zip(a, b):
        for k in (sorted(set(ra) | set(rb)) if fields is None else fields):
            va, vb = ra.get(k), rb.get(k)
            same = (abs(va - vb) <= 1e-12) if isinstance(va, float) and isinstance(vb, float) else va == vb
            if not same:
                out.append('%s %s: %s against %s' % (ra.get('frame'), k, va, vb))
    return out


def split_fields(split):
    return DETERMINISTIC if split == 'train' else None


def counts(seg, names):
    return {names[c]: int((seg == c).sum()) for c in range(len(names)) if (seg == c).any()}


def check_subset(arm_row, base_row):
    """DENSITY: a keep mask can only remove points, so no class count may grow."""
    errs = []
    if arm_row['points'] >= base_row['points']:
        errs.append('%s: %d points, not fewer than the base %d' % (arm_row['frame'], arm_row['points'],
                                                                   base_row['points']))
    for c, n in arm_row['point_counts'].items():
        if n > base_row['point_counts'].get(c, 0):
            errs.append('%s: class %s %d points, the base has %d' % (arm_row['frame'], c, n,
                                                                    base_row['point_counts'].get(c, 0)))
    return errs


def check_equal_counts(arm_row, expect_points, expect_counts, what):
    errs = []
    if arm_row['points'] != expect_points:
        errs.append('%s: %d points, %s gives %d' % (arm_row['frame'], arm_row['points'], what, expect_points))
    if arm_row['point_counts'] != expect_counts:
        errs.append('%s: class counts %s, %s gives %s' % (arm_row['frame'], arm_row['point_counts'], what,
                                                          expect_counts))
    return errs


def render_expectation(base_ds, cache_dir, seq, frame, names):
    """Labels and intensities of a cached render, read independently of RenderCache.load: the cache's (src_off,
    src_idx) into read_frame's points of the source frames, then the class map."""
    from pcseg.pandaset_seg import read_frame
    with np.load(os.path.join(cache_dir, seq, '%02d.npz' % frame)) as z:
        off, idx, xyz = z['src_off'].astype(int), z['src_idx'].astype(int), z['xyz']
    raw = np.full(len(idx), -999, dtype=np.int64)
    inten = np.full(len(idx), np.nan, dtype=np.float64)
    for o in np.unique(off):
        pts, lab, _ = read_frame(base_ds.root, seq, frame + int(o), base_ds.device, base_ds.pose(seq, frame + int(o)))
        sel = off == o
        raw[sel] = lab[idx[sel]]
        inten[sel] = pts[idx[sel], 3]
    if (raw == -999).any():
        raise ValueError('%s/%02d: points left without a source' % (seq, frame))
    seg = base_ds.class_map(raw)
    return len(xyz), counts(seg, names), inten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--dump', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import train_seg
    from pcseg import engine
    from pcseg.class_map import build_class_map

    cfg = train_seg.load_cfg(args.cfg)
    touched, ops = arm_block(cfg)
    names = build_class_map(cfg.DATA.CLASS_MAP).names
    arm = train_seg.build_datasets(cfg)
    base = train_seg.build_datasets(strip_blocks(cfg))
    d_job = json.load(open(args.dump))
    d_arm = engine.target_dump(arm, names)
    d_base = engine.target_dump(base, names)
    untouched = 'val' if touched == 'train' else 'train'
    res = {'cfg': args.cfg, 'dump': args.dump, 'touched_split': touched, 'ops': ops, 'checks': {}, 'frames': []}

    res['checks']['A_job_dump_reproduces'] = dump_diff(d_job, d_arm)
    res['checks']['B_untouched_split_is_the_base'] = dump_diff(d_arm[untouched], d_base[untouched],
                                                               split_fields(untouched))
    errs = []
    ds_a, ds_b = arm[touched], base[touched]
    idxs = np.linspace(0, len(ds_a) - 1, num=min(4, len(ds_a))).round().astype(int)
    spec = cfg.DATA['FACTOR_SWAP' if touched == 'train' else 'EVAL_PERTURB']
    for i, ra, rb in zip(idxs, d_arm[touched], d_base[touched]):
        row = {'frame': ra['frame'], 'points_arm': ra['points'], 'points_base': rb['points']}
        if ops == ['RENDER']:
            seq, frame = ds_a.frames[int(i)]                     # PandaSet only (P2's frames are not (seq, frame))
            n, cnt, inten = render_expectation(ds_b, spec.RENDER.CACHE, seq, frame, names)
            errs += check_equal_counts(ra, n, cnt, 'the cache read independently')
            got = ds_a.load(int(i))[1][:, 0]
            if len(got) != len(inten) or not np.allclose(got, inten, rtol=0, atol=1e-6):
                errs.append('%s: intensities differ from the source points at the cached index' % ra['frame'])
        elif ops == ['DENSITY']:
            errs += check_subset(ra, rb)
        elif ops == ['CONE']:
            coord, _, seg, _ = ds_b.load(int(i))
            m = engine.cone_mask(coord[:, :2] - ds_b.shift[:2], float(spec.CONE.FOV_DEG),
                                 float(spec.CONE.get('HEADING_DEG', 0.0)))
            errs += check_equal_counts(ra, int(m.sum()), counts(seg[m], names), "the base's in-cone points")
        elif ops == ['INTENSITY']:
            ca, ia, sa, _ = ds_a.load(int(i))
            cb, ib, sb, _ = ds_b.load(int(i))
            if not (np.array_equal(ca, cb) and np.array_equal(sa, sb)):
                errs.append('%s: coordinates or labels changed' % ra['frame'])
            changed = float(np.mean(ia[:, 0] != ib[:, 0]))
            row.update({'intensity_changed': changed, 'median_intensity_arm': float(np.median(ia)),
                        'median_intensity_base': float(np.median(ib))})
            if changed < 0.5:
                errs.append('%s: only %.1f%% of intensities changed' % (ra['frame'], 100 * changed))
        else:
            raise ValueError('no expectation coded for the operator set %s' % ops)
        row['kept'] = ra['points'] / max(rb['points'], 1)
        res['frames'].append(row)
    res['checks']['C_touched_split_as_declared'] = errs
    res['pass'] = all(not v for v in res['checks'].values())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ('cfg', 'touched_split', 'ops', 'pass')}, indent=1))
    for k, v in res['checks'].items():
        print('%s: %s' % (k, 'ok' if not v else '%d problems, first: %s' % (len(v), v[0])))
    for r in res['frames']:
        print('  ', r)
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
