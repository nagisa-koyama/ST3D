"""Derive the three geometry-alignment parameters from the data, and emit config YAML.

ST3D aligns source and target geometry with three knobs, all of which are hardcoded in this repo
and none of which were derived from the datasets actually being used:

  SHIFT_COOR           vertical shift putting the ground plane at z = 0
  SIZE_RES             Statistical Normalization: additive per-dimension size offset
  SCALE_UNIFORM_NOISE  Random Object Scaling: one isotropic multiplier per box

Run from ST3D/tools inside the container:

    singularity exec --bind /home/koyama/data/:/storage <image>.sif \
        python analysis/derive_augmentation_params.py [--target KITTI] [--frames N] [--yaml]

UDA legality, which differs between the two size knobs and is the reason to prefer one:

  * SIZE_RES needs the TARGET's mean object size, so it presumes labelled target data. It is not
    available in a strict UDA setup; it is reported here as a reference/oracle value.
  * SCALE_UNIFORM_NOISE only perturbs the source. The interval that best matches a given target
    still needs target statistics, so the `matched` rows below are an oracle too - but unlike
    SIZE_RES they turn out to be nearly identical across every source, which is what makes a
    single prior-chosen interval defensible without ever reading target labels.
"""
import argparse
import sys

import numpy as np

from domain_gap_analysis import build_platforms

CLASS_ALIASES = {
    'KITTI': {'Car': {'Car'}, 'Pedestrian': {'Pedestrian'}, 'Cyclist': {'Cyclist'}},
    'nuScenes': {'Car': {'car'}, 'Pedestrian': {'pedestrian'}, 'Cyclist': {'bicycle'}},
    'Lyft': {'Car': {'car'}, 'Pedestrian': {'pedestrian'}, 'Cyclist': {'bicycle'}},
    'PandaSet': {'Car': {'Car'}, 'Pedestrian': {'Pedestrian', 'Pedestrian with Object'},
                 'Cyclist': {'Bicycle'}},
    'Waymo': {'Car': {'Vehicle'}, 'Pedestrian': {'Pedestrian'}, 'Cyclist': {'Cyclist'}},
}
CLASSES = ('Car', 'Pedestrian', 'Cyclist')
CONFIGURED_SHIFT = {'KITTI': 1.7, 'nuScenes': 1.75, 'Lyft': 1.6, 'PandaSet': 0.3, 'Waymo': 0.0}
CONFIGURED_SIZE_RES = [-0.91, -0.49, -0.26]
CONFIGURED_ROS = [0.9, 1.1]
MIN_BOXES = 50


def isotropic_scale(dims):
    """One scalar size per box - what ROS's single multiplier acts on."""
    return np.cbrt(dims[:, 0] * dims[:, 1] * dims[:, 2])


def boxes_of(platform, info):
    """(dims Nx3, z_base N, names N) for one frame, however that platform stores them."""
    if platform.name.startswith('PandaSet'):
        fr = platform.frame(info)
        boxes, names = fr.boxes, fr.names
    elif 'gt_boxes' in info:
        boxes, names = np.asarray(info['gt_boxes']), np.asarray(info['gt_names'])
    elif 'annos' in info:
        boxes = np.asarray(info['annos']['gt_boxes_lidar'])
        names = np.asarray(info['annos']['name'])[:len(boxes)]
    else:
        return None
    if not len(boxes):
        return None
    boxes = np.asarray(boxes)
    return boxes[:, 3:6], boxes[:, 2] - boxes[:, 5] / 2, names


def collect_sizes(platform, frames):
    src = platform.sample(frames * 5) if platform.name.startswith('PandaSet') else platform.infos
    out = {c: [] for c in CLASSES}
    base = {c: [] for c in CLASSES}
    alias = CLASS_ALIASES[platform.name.split()[0]]
    for info in src:
        got = boxes_of(platform, info)
        if got is None:
            continue
        dims, z_base, names = got
        for c in CLASSES:
            m = np.isin(names, list(alias[c]))
            if m.sum():
                out[c].append(dims[m])
                base[c].append(z_base[m])
    return ({c: (np.concatenate(v) if v else np.zeros((0, 3))) for c, v in out.items()},
            {c: (np.concatenate(v) if v else np.zeros(0)) for c, v in base.items()})


def road_z(platform, frames):
    """10th percentile of z in a 3-15 m annulus - stable where the modal z is not."""
    vals = []
    for info in platform.sample(frames):
        pts = platform.frame(info).points
        r = np.linalg.norm(pts[:, :2], axis=1)
        m = (r > 3) & (r < 15)
        if m.sum() > 500:
            vals.append(np.percentile(pts[m, 2], 10))
    return float(np.median(vals)) if vals else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', default='KITTI')
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--yaml', action='store_true', help='emit pasteable config blocks')
    args = ap.parse_args()

    P = build_platforms()
    if args.target not in P:
        sys.exit('unknown target %r; choose from %s' % (args.target, sorted(P)))
    sizes, bases, roads = {}, {}, {}
    for name, plat in P.items():
        sizes[name], bases[name] = collect_sizes(plat, args.frames)
        roads[name] = road_z(plat, args.frames)
    tgt = sizes[args.target]
    sources = [n for n in P if n != args.target]

    print('=' * 94)
    print('1. SHIFT_COOR   - lift each platform so Car box bases sit on z = 0')
    print('=' * 94)
    print(f"{'platform':26s} {'road z':>9s} {'Car base z':>12s} {'derived':>9s} {'configured':>11s} "
          f"{'delta':>7s}")
    shift = {}
    for name in P:
        b = bases[name]['Car']
        base_z = float(np.median(b)) if len(b) else float('nan')
        shift[name] = -base_z
        cfg = CONFIGURED_SHIFT.get(name.split()[0], 0.0)
        print(f'{name:26s} {roads[name]:9.2f} {base_z:12.2f} {-base_z:9.2f} {cfg:11.2f} '
              f'{-base_z - cfg:+7.2f}')
    print('\n  Both estimators are reported because they disagree by up to 0.3 m: the road is a low')
    print('  percentile (biased low), the box base is what anchor_bottom_heights: [0] must match')
    print('  (authoritative, but a median over vehicles standing at different elevations).')

    print('\n' + '=' * 94)
    print('2. SIZE_RES  (Statistical Normalization)  = target mean - source mean, per class')
    print('   NOT UDA-legal: requires the target\'s labelled object sizes.')
    print('=' * 94)
    print(f"{'source':26s} {'class':11s} {'d_length':>9s} {'d_width':>8s} {'d_height':>9s} "
          f"{'n_src':>8s}")
    size_res = {}
    for c in CLASSES:
        if len(tgt[c]) < MIN_BOXES:
            continue
        t = tgt[c].mean(0)
        for name in sources:
            if len(sizes[name][c]) < MIN_BOXES:
                continue
            d = t - sizes[name][c].mean(0)
            size_res[(name, c)] = d
            print(f'{name:26s} {c:11s} {d[0]:9.2f} {d[1]:8.2f} {d[2]:9.2f} {len(sizes[name][c]):8d}')
    print(f'\n  repo hardcodes {CONFIGURED_SIZE_RES} for every pair AND every class;')
    print('  normalize_object_size receives no class names, so it cannot be class-aware today.')

    print('\n' + '=' * 94)
    print('3. SCALE_UNIFORM_NOISE  (Random Object Scaling)  - one isotropic multiplier per box')
    print('=' * 94)
    print(f"{'source':26s} {'class':11s} {'src median':>11s} {'matched (oracle)':>19s} {'centre':>7s}")
    ros = {}
    for c in CLASSES:
        if len(tgt[c]) < MIN_BOXES:
            continue
        ts = isotropic_scale(tgt[c])
        lo_t, hi_t = np.percentile(ts, [5, 95])
        for name in sources:
            if len(sizes[name][c]) < MIN_BOXES:
                continue
            med = float(np.median(isotropic_scale(sizes[name][c])))
            ros[(name, c)] = (lo_t / med, hi_t / med)
            print(f'{name:26s} {c:11s} {med:11.3f} '
                  f'{"[%.2f, %.2f]" % (lo_t / med, hi_t / med):>19s} '
                  f'{float(np.median(ts)) / med:7.2f}')

    print('\n  --- strictly UDA-legal check: what the SOURCE POOL alone would suggest ---')
    for c in CLASSES:
        meds = [float(np.median(isotropic_scale(sizes[n][c]))) for n in sources
                if len(sizes[n][c]) >= MIN_BOXES]
        if len(meds) < 3 or len(tgt[c]) < MIN_BOXES:
            continue
        meds = np.array(meds)
        ref = float(np.median(meds))
        actual = float(np.median(isotropic_scale(tgt[c]))) / ref
        print(f'  {c:11s} source-only [{meds.min() / ref:.2f}, {meds.max() / ref:.2f}]   '
              f'target actually at {actual:.2f}')
    print('\n  If the source-only interval does not reach the target, the source pool carries no')
    print('  signal about it - every source here is a large-vehicle domain. A target-free interval')
    print('  must then encode a PRIOR (e.g. "the target may be a smaller-vehicle region"), which is')
    print('  the justification UADA3D gives, not a measured spread.')

    if args.yaml:
        print('\n' + '=' * 94)
        print('Config blocks')
        print('=' * 94)
        for name in sources:
            if not np.isfinite(shift[name]):
                continue
            print(f'\n# --- {name} as source, {args.target} as target ---')
            print(f'SHIFT_COOR: [0.0, 0.0, {shift[name]:.2f}]')
            car = size_res.get((name, 'Car'))
            if car is not None:
                print('DATA_AUGMENTOR:')
                print('    AUG_CONFIG_LIST:')
                print('        - NAME: normalize_object_size   # oracle: needs target labels')
                print(f'          SIZE_RES: [{car[0]:.2f}, {car[1]:.2f}, {car[2]:.2f}]')
                lo, hi = ros.get((name, 'Car'), (np.nan, np.nan))
                print('        - NAME: random_object_scaling')
                print(f'          SCALE_UNIFORM_NOISE: [{lo:.2f}, {hi:.2f}]   '
                      f'# matched; prior-safe choice is [0.75, 1.00]')
        print(f'\n# repo default ROS is {CONFIGURED_ROS}, which is both too narrow and centred too')
        print('# high - it barely overlaps the interval any of these sources actually needs.')


if __name__ == '__main__':
    main()
