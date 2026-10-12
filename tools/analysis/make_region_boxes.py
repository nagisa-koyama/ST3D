"""Write the EVAL_TOP_THIN.REGION box file from a model's saved predictions (experiments_md 20261010_04 §5.0a).

Label-free: reads only a `result.pkl` of predictions (a legal model scored on the target), keeps one class above a fixed
score, and writes {frame_id: {'boxes': (M, 7) float32, 'labels': (M,) int8 = 1}} plus a '_meta' entry. No target label
is read. For §5.0a: 27704's Waymo-val predictions (rescore 27803, epoch 20), Car, score >= 0.180.

    python analysis/make_region_boxes.py <result.pkl> <out.pkl> [class=Car] [min_score=0.18]
"""
import os
import pickle
import sys

import numpy as np


def main():
    src, out = sys.argv[1], sys.argv[2]
    cls = sys.argv[3] if len(sys.argv) > 3 else 'Car'
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.18
    with open(src, 'rb') as f:
        preds = pickle.load(f)
    table, n = {}, []
    for p in preds:
        m = (np.asarray(p['name']) == cls) & (np.asarray(p['score']) >= thr)
        boxes = np.asarray(p['boxes_lidar'], dtype=np.float32).reshape(-1, 7)[m] if len(p['name']) else np.zeros((0, 7), np.float32)
        table[str(p['frame_id'])] = {'boxes': boxes, 'labels': np.ones(len(boxes), dtype=np.int8)}
        n.append(len(boxes))
    assert len(table) == len(preds), 'duplicate frame ids in %s' % src
    table['_meta'] = {'source': os.path.abspath(src), 'class': cls, 'min_score': thr, 'frames': len(preds),
                      'boxes_per_frame': float(np.mean(n))}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(table, f)
    print('%d frames, %.2f %s boxes per frame (min %d, max %d) at score >= %.3f -> %s'
          % (len(preds), np.mean(n), cls, np.min(n), np.max(n), thr, out))


if __name__ == '__main__':
    main()
