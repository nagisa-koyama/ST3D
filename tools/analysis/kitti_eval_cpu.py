"""The OFFICIAL KITTI evaluator on CPU, optionally over a subset of frames.

`kitti_object_eval_python/eval.py` is run unmodified; only its rotated-rectangle IoU kernel
(`rotate_iou_gpu_eval`, a numba.cuda kernel that cannot even be imported without a GPU) is replaced
by pcdet's CPU BEV IoU op. That reproduces the logged AP to the last printed digit on BEV (job 26717:
84.8663 / 76.5629 / 75.7683 both ways) and to ~0.03-0.08 on 3D, where the kernel's float32
intersection is recovered from the IoU. Unlike `gates/per_range_ap.py`, which re-implements the
protocol and lands within 1.1-1.4 AP, nothing about difficulty levels, ignored classes or the
40-point interpolation is re-derived here.

The evaluator calls the kernel with criterion -1 (BEV IoU) and 2 (raw intersection, for 3D) only;
`rotate_iou_cpu` refuses 0 and 1 rather than guess. Its equivalence to the CUDA kernel is pinned by
tests/test_kitti_eval_cpu.py, which runs the real kernel under numba's CUDA simulator.

    python kitti_eval_cpu.py <label>=<result.pkl> [...] [--split-ground -1.78]

`--split-ground Z` also scores each result on the frames whose road 4-8 m ahead (5th percentile of
point z in a 4 m corridor, raw LiDAR frame) sits below / above Z. The estimate reads points only, no
labels. It exists for experiments_md/20260930_07: the KITTI oracle loses cars on downhill frames.
"""
import argparse
import copy
import importlib.util
import pickle
import sys
import types
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu  # noqa: E402

INFOS = TOOLS.parent / 'data/kitti/kitti_infos_val.pkl'
VELODYNE = TOOLS.parent / 'data/kitti/training/velodyne'
_PACKAGE = 'pcdet.datasets.kitti.kitti_object_eval_python'


def rotate_iou_cpu(boxes, qboxes, criterion=-1):
    """Drop-in for `rotate_iou_gpu_eval`: boxes are [N, 5] = (x, z, l, w, ry) in the camera BEV.

    The CUDA kernel rotates corners by -ry (clockwise); pcdet's op rotates by +heading, so the angle
    is negated. Getting that sign wrong still yields plausible IoUs and costs ~2 AP.
    """
    if criterion not in (-1, 2):
        raise ValueError(f'criterion {criterion}: the evaluator only uses -1 and 2')

    def to_lidar7(b):
        out = np.zeros((len(b), 7), np.float32)
        out[:, 0], out[:, 1], out[:, 3], out[:, 4] = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        out[:, 5] = 1.0
        out[:, 6] = -b[:, 4]
        return out

    b = np.asarray(boxes, np.float32)
    q = np.asarray(qboxes, np.float32)
    if len(b) == 0 or len(q) == 0:
        return np.zeros((len(b), len(q)), np.float32)
    iou = boxes_bev_iou_cpu(to_lidar7(b), to_lidar7(q))
    if criterion == -1:
        return iou
    area_b = (b[:, 2] * b[:, 3])[:, None]
    area_q = (q[:, 2] * q[:, 3])[None, :]
    return iou * (area_b + area_q) / (1.0 + iou)  # the intersection, which is what criterion 2 returns


def load_official_eval():
    """eval.py under a private module name, with the IoU kernel swapped for `rotate_iou_cpu`.

    Private name for the reason tests/test_kitti_ap_truncation.py gives: binding the package's
    `eval` attribute would defeat test_kitti_eval_class_mapping.py's stub in the same process.
    """
    rotate_iou = _PACKAGE + '.rotate_iou'
    if rotate_iou not in sys.modules:  # the real module compiles a CUDA kernel at import
        stub = types.ModuleType(rotate_iou)
        stub.rotate_iou_gpu_eval = rotate_iou_cpu
        sys.modules[rotate_iou] = stub
    spec = importlib.util.spec_from_file_location(
        _PACKAGE + '._eval_cpu', TOOLS.parent / 'pcdet/datasets/kitti/kitti_object_eval_python/eval.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.rotate_iou_gpu_eval = rotate_iou_cpu  # the name eval.py's functions look up at call time
    return module


def evaluate(result_pkl, keep_frame=None, classes=('Car',), infos_pkl=INFOS, official=None):
    """Official AP dict over the frames `keep_frame(info)` accepts (all when None)."""
    official = official or load_official_eval()
    dets = {d['frame_id']: d for d in pickle.load(open(result_pkl, 'rb'))}
    gt, dt = [], []
    for info in pickle.load(open(infos_pkl, 'rb')):
        if keep_frame is not None and not keep_frame(info):
            continue
        gt.append(copy.deepcopy(info['annos']))
        dt.append(copy.deepcopy(dets[info['point_cloud']['lidar_idx']]))
    _, ap = official.get_official_eval_result(gt, dt, list(classes))
    return len(gt), ap


def road_height_ahead(frame_id):
    """5th percentile of point z, 4-8 m straight ahead in a 4 m corridor; raw LiDAR frame, no labels."""
    pts = np.fromfile(VELODYNE / f'{frame_id}.bin', dtype=np.float32).reshape(-1, 4)
    ahead = pts[(np.abs(pts[:, 1]) < 2) & (pts[:, 0] > 4) & (pts[:, 0] < 8)]
    return float(np.percentile(ahead[:, 2], 5)) if len(ahead) > 20 else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', nargs='+', help='<label>=<result.pkl>')
    parser.add_argument('--split-ground', type=float, default=None, metavar='Z')
    args = parser.parse_args()
    official = load_official_eval()
    ground = None
    if args.split_ground is not None:
        ids = [i['point_cloud']['lidar_idx'] for i in pickle.load(open(INFOS, 'rb'))]
        ground = {fid: road_height_ahead(fid) for fid in ids}
        low = sum(g < args.split_ground for g in ground.values())
        print(f'frames with the road ahead below {args.split_ground} m: {low} of {len(ground)}')
    arms = [('all', None)]
    if ground is not None:
        z = args.split_ground
        arms += [(f'road >= {z}', lambda i: not ground[i['point_cloud']['lidar_idx']] < z),
                 (f'road < {z}', lambda i: ground[i['point_cloud']['lidar_idx']] < z)]
    for item in args.results:
        label, path = item.split('=', 1)
        cells = []
        for name, keep in arms:
            n, ap = evaluate(path, keep, official=official)
            cells.append(f'{name}: BEV {ap["Car_bev/easy_R40"]:.2f} / {ap["Car_bev/moderate_R40"]:.2f} / '
                         f'{ap["Car_bev/hard_R40"]:.2f}  3D mod {ap["Car_3d/moderate_R40"]:.2f}  (n={n})')
        print(f'{label}\n  ' + '\n  '.join(cells), flush=True)


if __name__ == '__main__':
    main()
