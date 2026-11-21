"""Item 51 (experiments_md 20261003_05 §5), row D: checks of DTS's two target views (DTS_VIEWS) on real KITTI train frames
before any GPU run. CPU only. Ground-truth boxes stand in for pseudo-labels (USE_PSEUDO_LABEL off): a loader check, no
training and no reading.

    python3 analysis/item51_dts_views_check.py [--frames 40] [--json OUT]

Per frame, with the same np.random seed for every arm:
  1. the stored order gives exactly 64 rings (kitti_rings.rings_from_order);
  2. identity: KEEP_PROB 1 and no teacher mask -> both views hold exactly the base pipeline's points (as row sets) and
     its gt_boxes (the views are split after ONE augmentation draw);
  3. at 0.75 / 0.75 both views keep the base's gt_boxes exactly, each view's points are a subset of the base's, the two
     masks differ, and the kept share of points is reported;
  4. collate_batch returns [student batch, teacher batch] with the same frame ids.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
D = 'cfgs/da-ieee-access/item51/centerpoint-dts-nuscenes2kitti.yaml'


def build(views):
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    from pcdet.utils import common_utils
    cfg = cfg_from_yaml_file(D, EasyDict())
    block = copy.deepcopy(cfg.DATA_CONFIG_TAR)
    block.USE_PSEUDO_LABEL = False
    if views is None:
        block.pop('DTS_VIEWS')
    else:
        block.DTS_VIEWS.KEEP_PROB, block.DTS_VIEWS.TEACHER_KEEP_PROB = views
    return KittiDataset(dataset_cfg=block, class_names=cfg.CLASS_NAMES, training=True, root_path=None,
                        logger=common_utils.create_logger())


def rows(points):
    return set(map(tuple, np.round(points.astype(np.float64), 6).tolist()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', type=int, default=40)
    p.add_argument('--json', default=None)
    args = p.parse_args()
    os.chdir(TOOLS)
    import _init_path  # noqa: F401
    from pcdet.datasets.kitti.kitti_rings import rings_from_order
    from pcdet.datasets.dataset import DatasetTemplate
    base, ident, dts = build(None), build((1.0, None)), build((0.75, 0.75))
    idx = np.linspace(0, len(base.kitti_infos) - 1, args.frames).astype(int)
    out, fails = [], []
    for k in idx:
        k = int(k)
        sample = base.kitti_infos[k]['point_cloud']['lidar_idx']
        n_rings = int(rings_from_order(base.get_lidar(sample)[:, :3])[0].max()) + 1
        np.random.seed(k)
        b = base[k]
        np.random.seed(k)
        s1, t1 = ident[k]
        np.random.seed(k)
        s, t = dts[k]
        rb = rows(b['points'])
        ok2 = rows(s1['points']) == rb and rows(t1['points']) == rb and np.array_equal(s1['gt_boxes'], b['gt_boxes']) \
            and np.array_equal(t1['gt_boxes'], b['gt_boxes'])
        rs, rt = rows(s['points']), rows(t['points'])
        ok3 = (np.array_equal(s['gt_boxes'], b['gt_boxes']) and np.array_equal(t['gt_boxes'], b['gt_boxes'])
               and rs <= rb and rt <= rb and rs != rt)
        r = dict(index=k, frame=sample, rings=n_rings, base=len(b['points']), student=len(s['points']),
                 teacher=len(t['points']), rings_ok=n_rings == 64, identity=bool(ok2), views=bool(ok3))
        out.append(r)
        if not (r['rings_ok'] and ok2 and ok3):
            fails.append(r)
        print(r, flush=True)
    np.random.seed(0)
    batch = DatasetTemplate.collate_batch([dts[int(idx[0])], dts[int(idx[1])]])
    ok4 = isinstance(batch, list) and len(batch) == 2 and list(batch[0]['frame_id']) == list(batch[1]['frame_id'])
    share_s = np.mean([r['student'] / r['base'] for r in out])
    share_t = np.mean([r['teacher'] / r['base'] for r in out])
    print('frames %d, failures %d, collate %s; kept share of points: student %.3f, teacher %.3f'
          % (len(out), len(fails), ok4, share_s, share_t))
    if args.json:
        json.dump(dict(rows=out, failures=fails, collate=bool(ok4)), open(args.json, 'w'), indent=1)
    sys.exit(0 if not fails and ok4 else 1)


if __name__ == '__main__':
    main()
