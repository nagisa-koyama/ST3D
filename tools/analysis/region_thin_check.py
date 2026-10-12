"""CPU check before the inside / outside line-cut eval (experiments_md 20261010_04 §5.0a). No target label is read.

Builds the EVALUATION dataset of the first config and, on N strided Waymo val frames, runs every given config's
EVAL_TOP_THIN (with REGION) through the dataset's own hook (`get_lidar_top_thin`), reporting:
- that every evaluation frame id is in the REGION box file;
- boxes per frame, and the share of TOP points inside the enlarged boxes;
- points per frame as scored (after the no-label-zone filter) in each arm, against the unthinned frame;
- the count match per side: TOP points kept by 'rows' and 'random' are equal before the filter (checked exactly,
  through region_top_thin_points on the filter-free frame), and close after it.

    python analysis/region_thin_check.py <frames> <cfg> [<cfg> ...]
"""
import sys
import zlib

import numpy as np

sys.path.insert(0, '.')
import _init_path  # noqa
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.waymo.waymo_region_thin import (enlarge_boxes, points_in_any_box, region_boxes,
                                                    region_top_thin_points)
from pcdet.utils import common_utils


def load(path):
    c = EasyDict()
    cfg_from_yaml_file(path, c)
    return c


def main():
    n = int(sys.argv[1])
    paths = sys.argv[2:]
    cfgs = [load(p) for p in paths]
    arms = [c.DATA_CONFIG_TAR.EVAL_TOP_THIN for c in cfgs]
    box_file = arms[0].REGION.BOX_FILE
    assert all(a.REGION.BOX_FILE == box_file for a in arms), 'the arms use different box files'
    ds, _, _ = build_dataloader(dataset_cfg=cfgs[0].DATA_CONFIG_TAR, class_names=cfgs[0].CLASS_NAMES, batch_size=1,
                                dist=False, workers=0, logger=common_utils.create_logger(), training=False,
                                model_ontology=cfgs[0].get('ONTOLOGY'))
    table_entry = ds._region_boxes(box_file, ds.infos[0]['frame_id'])   # loads the table once
    missing = [i['frame_id'] for i in ds.infos if i['frame_id'] not in ds._region_box_table]
    print('evaluation frames %d, box file %s: %d frames, meta %s; missing %d'
          % (len(ds.infos), box_file, len(ds._region_box_table) - 1, ds._region_box_table.get('_meta'), len(missing)))
    assert not missing, 'frames missing from the box file, e.g. %s' % missing[:3]
    del table_entry

    step = max(1, len(ds.infos) // n)
    stats = {p: [] for p in paths}
    inside_share, nboxes, exact = [], [], []
    for k, info in enumerate(ds.infos[::step][:n]):
        pc = info['point_cloud']
        full = ds.get_lidar(pc['lidar_sequence'], pc['sample_idx'], info.get('num_points_of_each_lidar', None))
        entry = ds._region_boxes(box_file, info['frame_id'])
        raw = ds._raw_frame(info)
        counts = np.asarray(info['num_points_of_each_lidar']).astype(np.int64)
        n_top = int(counts[0])
        a0 = arms[0]
        boxes = enlarge_boxes(region_boxes(entry), a0.REGION.MARGIN_XY, a0.REGION.MARGIN_Z)
        inside = points_in_any_box(raw[:n_top, :3], boxes)
        inside_share.append(inside.mean() if n_top else np.nan)
        nboxes.append(len(boxes))
        rng_seed = zlib.crc32(str(info['frame_id']).encode())
        for p, a in zip(paths, arms):
            pts = ds.get_lidar_top_thin(info, a, np.random.default_rng(rng_seed))
            stats[p].append((len(pts), len(full)))
        # exact count match before the NLZ filter, per side
        calib = ds._top_calibration(a0.TOP_CALIB)[pc['lidar_sequence']]
        free = raw.copy()
        free[:, 5] = -1
        for side in ('inside', 'outside'):
            r = region_top_thin_points(free, counts, calib, 'rows', entry, side, a0.REGION.MARGIN_XY,
                                       a0.REGION.MARGIN_Z, a0.get('STRIDE', 2))
            q = region_top_thin_points(free, counts, calib, 'random', entry, side, a0.REGION.MARGIN_XY,
                                       a0.REGION.MARGIN_Z, a0.get('STRIDE', 2), rng=np.random.default_rng(rng_seed))
            exact.append(len(r) == len(q))
        if (k + 1) % 10 == 0:
            print('  %d frames' % (k + 1), flush=True)

    print('\nframes checked %d; boxes per frame %.1f (median %d); TOP share inside the enlarged boxes %.4f (median %.4f)'
          % (len(nboxes), np.mean(nboxes), np.median(nboxes), np.nanmean(inside_share), np.nanmedian(inside_share)))
    print('count match before the no-label-zone filter (rows vs random, per side and frame): %d / %d exact'
          % (sum(exact), len(exact)))
    print('%-75s %12s %12s %9s' % ('config', 'pts/frame', 'unthinned', 'kept'))
    for p in paths:
        s = np.array(stats[p], float)
        print('%-75s %12.0f %12.0f %9.4f' % (p.split('/')[-1], s[:, 0].mean(), s[:, 1].mean(), s[:, 0].sum() / s[:, 1].sum()))
    assert all(exact), 'rows and random keep different counts before the filter'


if __name__ == '__main__':
    main()
