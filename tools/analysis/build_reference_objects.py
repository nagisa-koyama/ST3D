"""Build a PPCG reference-object library from SOURCE ground truth.

DALI(Points) draws its clean reference objects from source-domain GT boxes carrying at least 300
interior returns, which is what lets the Points variant avoid the CAD variant's mesh library,
trimesh and custom CUDA op for 0.07-0.25 AP. This is Tier D2.1 of the plan in
experiments_md/20260922_04 section 2.4, ported from upstream `tools/ppcg_points.py`.

Using SOURCE labels keeps it UDA-legal: no target annotation is read at any point.

TWO PATHS, because the plan's assumption held for only two of five sources:

  --from_gt_database   read an existing OpenPCDet gt_database + *_dbinfos_*.pkl. Points there are
                       already centred on their box, so this really is bookkeeping - seconds, no
                       dataset pass. Available for KITTI and PandaSet ONLY.
  (default)            iterate the source dataset and cut the boxes out directly, which is what
                       nuScenes, Lyft and Waymo need because no gt_database was ever built for
                       them here. Minutes to hours, CPU only.

Neither path needs a GPU or a checkpoint - the library is built from labels, not predictions.

Usage, from ST3D/tools inside the container:

    python3 analysis/build_reference_objects.py \
        --cfg_file cfgs/da-ieee-access/centerpoint-sourceonly-kitti.yaml \
        --class_name car --out /storage/ppcg_reference_kitti_car.npz \
        --from_gt_database /storage/kitti/kitti_dbinfos_train.pkl

Note the sys.path handling below rather than `import _init_path`: Python puts the *script's own*
directory on sys.path, which is analysis/, not tools/ - so the usual first-line import is not
available here. Getting this wrong silently resolves `pcdet` to the stale editable install baked
into the .sif at /code/ST3D instead of this checkout.
"""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent))

import argparse  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils  # noqa: E402
from pcdet.utils import common_utils, ppcg_utils  # noqa: E402
from ptsn_search import source_configs  # noqa: E402

TH_POINTS = 300


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='the config whose SOURCE the library is built from')
    parser.add_argument('--out', type=str, required=True, help='destination .npz')
    parser.add_argument('--class_name', type=str, default='car',
                        help='substring selecting the class (case-insensitive)')
    parser.add_argument('--min_points', type=int, default=TH_POINTS,
                        help="minimum interior returns for an object to be a reference (DALI's 300)")
    parser.add_argument('--from_gt_database', type=str, default=None,
                        help='an OpenPCDet *_dbinfos_*.pkl; skips the dataset pass entirely')
    parser.add_argument('--frames', type=int, default=10000,
                        help='dataset-pass path: source frames to scan')
    parser.add_argument('--max_objects', type=int, default=None,
                        help='stop once this many references have been collected')
    return parser.parse_args()


def matches(name, needle):
    return needle.lower() in str(name).lower()


def describe(box, points, name):
    """The record RC-PPCG's matching step reads back.

    `size` is the isotropic extent, cbrt(dx*dy*dz). Upstream computes
    `(box[0]*box[1]*box[2])**(1/3)` - the cube root of the product of the box's CENTRE
    COORDINATES, which is not a size at all and goes NaN whenever that product is negative. It is
    inert upstream because RC-PPCG's matcher never reads the field, but the field is stored, so
    storing a correct one costs nothing and removes a trap.
    """
    box = np.asarray(box, dtype=np.float64).reshape(7)
    return {
        'points': points,
        'box': box,
        'direction': float(np.arctan2(box[1], box[0])),   # azimuth of the object from the sensor
        'orientation': float(box[6]),                      # its heading
        'numpts': len(points),
        'size': float(np.cbrt(box[3] * box[4] * box[5])),
        'name': name,
    }


def from_gt_database(args, logger):
    """Read an existing gt_database. Its points are centred on the box but NOT heading-aligned."""
    dbinfos_path = Path(args.from_gt_database)
    root = dbinfos_path.parent
    with open(dbinfos_path, 'rb') as f:
        dbinfos = pickle.load(f)

    classes = [c for c in dbinfos if matches(c, args.class_name)]
    assert classes, ('--class_name %r matches none of %s'
                     % (args.class_name, sorted(dbinfos)))
    logger.info('gt_database classes matched: %s' % classes)

    objects, all_counts = [], []
    for cls in classes:
        for info in dbinfos[cls]:
            all_counts.append(int(info['num_points_in_gt']))
            if info['num_points_in_gt'] < args.min_points:
                continue
            points = np.fromfile(str(root / info['path']), dtype=np.float32).reshape(-1, 4)
            box = np.asarray(info['box3d_lidar'], dtype=np.float64)[:7]
            # Already translated to the box centre by create_groundtruth_database, never rotated.
            centred = np.concatenate([points[:, 0:3], points[:, 3:]], axis=1).astype(np.float64)
            centred[:, 0:3] = centred[:, 0:3] @ ppcg_utils.rotation_z(-box[6]).T
            objects.append(describe(box, centred.astype(np.float32),
                                    '%s_%s' % (info.get('image_idx', '?'), info.get('gt_idx', '?'))))
            if args.max_objects and len(objects) >= args.max_objects:
                return objects, all_counts
    return objects, all_counts


def from_dataset(args, config, logger):
    """Cut the boxes out of the source cloud directly - what nuScenes, Lyft and Waymo need."""
    objects, all_counts = [], []
    for source_cfg in source_configs(config):
        source_set, _, _ = build_dataloader(
            dataset_cfg=source_cfg, class_names=config.CLASS_NAMES, batch_size=1,
            dist=False, workers=0, logger=logger, training=False,
            model_ontology=config.get('ONTOLOGY', None))
        logger.info('scanning %s: %d frames available'
                    % (source_cfg.get('DATASET', '?'), len(source_set)))
        for idx in range(min(args.frames, len(source_set))):
            sample = source_set[idx]
            if 'gt_boxes' not in sample or len(sample['gt_boxes']) == 0:
                continue
            boxes = np.asarray(sample['gt_boxes'], dtype=np.float64)
            names = np.asarray(sample.get('gt_names', []))
            points = np.asarray(sample['points'], dtype=np.float32)
            keep = (np.array([matches(n, args.class_name) for n in names])
                    if len(names) == len(boxes) else np.ones(len(boxes), dtype=bool))
            if not keep.any():
                continue
            wanted = boxes[keep]
            masks = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(wanted[:, 0:7])).numpy()
            for b in range(len(wanted)):
                inside = points[masks[b] == 1]
                all_counts.append(len(inside))
                if len(inside) < args.min_points:
                    continue
                canonical = ppcg_utils.canonicalize_object_points(inside, wanted[b])
                objects.append(describe(wanted[b][:7], canonical.astype(np.float32),
                                        '%s_%d' % (sample.get('frame_id', idx), b)))
                if args.max_objects and len(objects) >= args.max_objects:
                    return objects, all_counts
            if idx % 500 == 0 and idx:
                logger.info('  %d frames, %d references so far' % (idx, len(objects)))
    return objects, all_counts


def report_yield(all_counts, min_points):
    """What the threshold costs on THIS source.

    DALI's TH_POINTS = 300 is a constant calibrated on KITTI and Waymo, both 64-beam. A sparse
    source clears it far less often - and the library is what every regenerated box is built
    from, so a thin one means RC-PPCG repeatedly reaches for a poorly-matched reference. Printing
    the distribution makes the threshold a decision rather than an inherited default.
    """
    if not all_counts:
        return
    counts = np.asarray(all_counts)
    percentiles = np.percentile(counts, [50, 75, 90, 95, 99]).astype(int)
    print('  interior returns per box of this class, over %d boxes scanned:' % len(counts))
    print('    p50 %d   p75 %d   p90 %d   p95 %d   p99 %d'
          % tuple(percentiles))
    print('    yield at a threshold of  100: %d   200: %d   300: %d   (in use: %d -> %d)'
          % ((counts >= 100).sum(), (counts >= 200).sum(), (counts >= 300).sum(),
             min_points, (counts >= min_points).sum()))


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    if args.from_gt_database:
        objects, all_counts = from_gt_database(args, logger)
        source = 'gt_database:%s' % Path(args.from_gt_database).name
    else:
        objects, all_counts = from_dataset(args, cfg, logger)
        source = 'dataset_pass:%s' % ','.join(
            c.get('DATASET', '?') for c in source_configs(cfg))

    print('\n' + '=' * 94)
    print('PPCG reference library   cfg=%s' % Path(args.cfg_file).stem)
    print('  built from       %s' % source)
    print('  class            %r   min_points %d' % (args.class_name, args.min_points))
    print('=' * 94)

    report_yield(all_counts, args.min_points)

    if not objects:
        print('NO REFERENCE OBJECTS. Either the class matched nothing, or no box in the scanned')
        print('frames carried %d interior returns. Lower --min_points or raise --frames.'
              % args.min_points)
        return 1

    numpts = np.array([o['numpts'] for o in objects])
    lwh = np.array([o['box'][3:6] for o in objects])
    print('  objects          %d' % len(objects))
    print('  points/object    min %d  median %d  max %d'
          % (numpts.min(), int(np.median(numpts)), numpts.max()))
    print('  extent l/w/h     mean %s' % np.round(lwh.mean(axis=0), 3).tolist())
    print('  azimuth coverage %d of 36 ten-degree bins occupied'
          % len(np.unique(np.floor(np.degrees(
              [o['direction'] for o in objects]) / 10).astype(int))))

    ppcg_utils.save_reference_objects(
        args.out, objects,
        metadata={'cfg': args.cfg_file, 'source': source, 'class_name': args.class_name,
                  'min_points': args.min_points, 'frames_scanned': args.frames,
                  'boxes_scanned': len(all_counts), 'objects': len(objects)})
    size_mb = Path(args.out).stat().st_size / 1e6
    print('\n  wrote %s (%.1f MB, one packed file rather than %d small ones)'
          % (args.out, size_mb, len(objects)))

    # Matching picks on direction AND orientation, so a library that only ever saw objects from
    # one bearing cannot serve a pseudo box seen from another - it would return a reference whose
    # self-occlusion pattern is wrong, which is the one thing RC-PPCG is supposed to get right.
    if len(objects) < 200:
        print('  WARNING: a library this small will often have no well-matched reference for a '
              'given\n  (direction, orientation) pair. Raise --frames or lower --min_points.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
