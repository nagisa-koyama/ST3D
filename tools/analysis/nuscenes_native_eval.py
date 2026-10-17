"""nuScenes-native detection metrics for a model trained in the KITTI class vocabulary, from a STORED result.pkl.

Why a separate script: `NuScenesDataset.nuscene_eval` (the `EVAL_METRIC: nuscenes` path) raises KeyError on
KITTI-capitalised class names (`Car`, `Pedestrian`, `Cyclist`), both in `transform_det_annos_to_nusc_annos`
(`cls_attr_dist[name]`) and in `format_nuscene_results` (`metrics['label_aps']['Car']`). This script does what
`nuscene_eval` does - the same `transform_det_annos_to_nusc_annos`, the same `meta`, the devkit's `NuScenesEval`
under `detection_cvpr_2019`, `eval_set='val'` - after renaming the detections into nuScenes detection names, and
then summarises over the MODEL's classes only. Nothing in pcdet is changed.

For each row it also recomputes the KITTI-metric Car BEV AP_R40 at IoU 0.7 from the SAME detections, through
`NuScenesDataset.evaluation(eval_metric='kitti')` with the official evaluator on CPU (the CUDA rotated-IoU kernel
swapped as in `kitti_eval_cpu.py`), and compares it with the value the run logged (validity check, |d| <= 0.1).

Class names: model -> nuScenes through `ontology_mapping.get_ontology_mapping('kitti', 'nuscenes')`
(Car -> car, Pedestrian -> pedestrian, Cyclist -> bicycle). Anything else raises: an ontology-prefixed name,
a name that maps outside the devkit's ten detection names, two model classes on one nuScenes class.

Reported per row (experiments_md 20261017_12 §2):
  - AP per model class (devkit: mean over centre-distance thresholds 0.5 / 1 / 2 / 4 m), and per threshold;
  - mAP_k over the model's k classes (NOT the devkit's ten-class mean);
  - ATE / ASE / AOE per class and their means over the k classes;
  - NDS-det = (5 mAP_k + sum over ATE, ASE, AOE of (1 - min(1, mTP_k))) / 8: the NDS formula restricted to the
    TP errors the model predicts. mAVE / mAAE are excluded: the boxes carry no velocity (7-dim, the converter
    writes zero velocity) and the attribute is the class prior from `cls_attr_dist`, so both measure the
    dataset's motion statistics, not the model. The devkit's own ten-class mAP / NDS / mAVE / mAAE are kept in the
    JSON for completeness and are not a reported number.

    python nuscenes_native_eval.py --out <dir> --row <label> <wandb_run_id> <epoch> <error_log> [--row ...]
    python nuscenes_native_eval.py --out <dir> --gt-self-test

`--gt-self-test` feeds the val GT itself (car / pedestrian / bicycle from the infos, in the LiDAR frame, score 1)
through the same renaming and conversion: AP must be 100 and ATE / ASE / AOE ~0. It reads target labels to test
the EVALUATOR; it is not a method and selects nothing.
"""
import argparse
import copy
import glob
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS / 'analysis')); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402  (the live repo's pcdet, not the image's stale copy)

WANDB = Path('/home/koyama/data/wandb')
MAX_BOXES = 500  # detection_cvpr_2019 max_boxes_per_sample; the devkit asserts it
TP_REPORTED = ('trans_err', 'scale_err', 'orient_err')
EVAL_VERSION = 'detection_cvpr_2019'


# ----------------------------------------------------------------------------------------------- pure helpers
def model_to_nusc_names(class_names):
    """{model class: nuScenes detection name} for the model's own CLASS_NAMES; raises on anything ambiguous."""
    from pcdet.utils.ontology_mapping import get_ontology_mapping
    from nuscenes.eval.detection.constants import DETECTION_NAMES
    kitti_to_nusc = get_ontology_mapping('kitti', 'nuscenes')
    out = {}
    for name in class_names:
        if ':' in name:
            raise ValueError(f'ontology-prefixed class {name!r}: this script handles the plain KITTI vocabulary only')
        if name not in kitti_to_nusc:
            raise ValueError(f'class {name!r} has no kitti -> nuscenes mapping')
        target = kitti_to_nusc[name]
        if target not in DETECTION_NAMES:
            raise ValueError(f'class {name!r} maps to {target!r}, not one of the devkit detection names')
        out[name] = target
    if len(set(out.values())) != len(out):
        raise ValueError(f'two model classes map to one nuScenes class: {out}')
    return out


def check_and_rename(det_annos, class_names, name_map, max_boxes=MAX_BOXES):
    """Validated, renamed deep copy of `det_annos`, plus counters.

    Every box must carry a label in 1..len(class_names) (a 0 / negative / -99 label would have WRAPPED in
    `np.array(class_names)[pred_labels - 1]` when the result was written, so it is refused rather than trusted),
    a name equal to class_names[label - 1], a finite score, a finite 7- or 9-dim box with positive size.
    A sample with more than `max_boxes` boxes keeps its `max_boxes` highest-scoring ones (the devkit limit).
    """
    names_arr = np.array(class_names)
    out, n_capped, n_boxes = [], 0, 0
    for det in det_annos:
        d = copy.deepcopy(det)
        n = len(d['name'])
        if not (len(d['score']) == len(d['pred_labels']) == len(d['boxes_lidar']) == n):
            raise ValueError(f"{d.get('frame_id')}: field lengths disagree")
        if 'metadata' not in d or 'token' not in d['metadata']:
            raise ValueError(f"{d.get('frame_id')}: no metadata token")
        if n:
            labels = np.asarray(d['pred_labels'])
            if not np.all(np.equal(np.mod(labels, 1), 0)):
                raise ValueError(f"{d.get('frame_id')}: non-integer labels")
            labels = labels.astype(np.int64)
            if labels.min() < 1 or labels.max() > len(class_names):
                raise ValueError(f"{d.get('frame_id')}: label outside 1..{len(class_names)}: "
                                 f"{sorted(set(labels.tolist()))}")
            if not np.array_equal(np.asarray(d['name']), names_arr[labels - 1]):
                raise ValueError(f"{d.get('frame_id')}: names disagree with class_names[label - 1]")
            boxes = np.asarray(d['boxes_lidar'])
            if boxes.ndim != 2 or boxes.shape[1] not in (7, 9):
                raise ValueError(f"{d.get('frame_id')}: boxes_lidar shape {boxes.shape}")
            if not (np.isfinite(boxes).all() and np.isfinite(d['score']).all()):
                raise ValueError(f"{d.get('frame_id')}: non-finite box or score")
            if (boxes[:, 3:6] <= 0).any():
                raise ValueError(f"{d.get('frame_id')}: non-positive box size")
            if n > max_boxes:
                keep = np.argsort(-np.asarray(d['score']), kind='stable')[:max_boxes]
                for k in ('name', 'score', 'pred_labels', 'boxes_lidar'):
                    d[k] = np.asarray(d[k])[keep]
                n_capped += 1
        d['name'] = np.array([name_map[x] for x in d['name']], dtype=object) if len(d['name']) else np.zeros(0)
        # float64: the converter json-dumps box.score (a numpy float32 is not JSON-serialisable)
        d['score'] = np.asarray(d['score'], dtype=np.float64)
        d['boxes_lidar'] = np.asarray(d['boxes_lidar'], dtype=np.float64)
        n_boxes += len(d['name'])
        out.append(d)
    return out, {'samples': len(out), 'boxes': n_boxes, 'samples_capped_at_500': n_capped}


def nds_det(map_k, tp_means):
    """NDS restricted to ATE / ASE / AOE: (5 mAP + sum(1 - min(1, err))) / 8."""
    return (5.0 * map_k + sum(1.0 - min(1.0, tp_means[k]) for k in TP_REPORTED)) / (5.0 + len(TP_REPORTED))


def summarise(metrics, nusc_classes):
    """Reported numbers from a devkit metrics_summary dict, over `nusc_classes` only."""
    per_class = {}
    for c in nusc_classes:
        tp = {k: float(metrics['label_tp_errors'][c][k]) for k in TP_REPORTED}
        if not all(np.isfinite(v) for v in tp.values()):
            raise ValueError(f'non-finite TP error for {c}: {tp}')
        per_class[c] = {
            'AP': 100.0 * float(metrics['mean_dist_aps'][c]),
            'AP_by_dist': {k: 100.0 * float(v) for k, v in metrics['label_aps'][c].items()},
            **{k: tp[k] for k in TP_REPORTED},
        }
    map_k = float(np.mean([metrics['mean_dist_aps'][c] for c in nusc_classes]))
    tp_means = {k: float(np.mean([per_class[c][k] for c in nusc_classes])) for k in TP_REPORTED}
    return {
        'classes': list(nusc_classes),
        'per_class': per_class,
        'mAP_k': 100.0 * map_k,
        'mATE_k': tp_means['trans_err'], 'mASE_k': tp_means['scale_err'], 'mAOE_k': tp_means['orient_err'],
        'NDS_det': 100.0 * nds_det(map_k, tp_means),
        'devkit_10class_not_reported': {
            'mAP': 100.0 * float(metrics['mean_ap']), 'NDS': 100.0 * float(metrics['nd_score']),
            'tp_errors': {k: float(v) for k, v in metrics['tp_errors'].items()},
        },
    }


def parse_logged_car_bev(text, epoch):
    """Car BEV AP_R40 @ IoU 0.7 (the three columns) of the evaluation of `epoch` in a training log's text."""
    text = text.replace('\r', '\n')
    starts = [m.end() for m in re.finditer(rf'EPOCH {epoch} of \S+ EVALUATION start', text)]
    if not starts:
        raise ValueError(f'no evaluation of epoch {epoch} in the log')
    tail = text[starts[-1]:]
    m = re.search(r'Car AP_R40@0\.70, 0\.70, 0\.70:\s*\nbbox AP:[^\n]*\nbev\s+AP:([0-9.]+), ([0-9.]+), ([0-9.]+)', tail)
    if not m:
        raise ValueError(f'no Car AP_R40@0.70 block after the epoch {epoch} evaluation start')
    res = re.search(r'Result is save to (\S+)', tail)
    return [float(x) for x in m.groups()], (res.group(1) if res else None)


# ---------------------------------------------------------------------------------------------- run-level work
def run_files(run_id, epoch):
    d = glob.glob(str(WANDB / f'run-*-{run_id}'))
    if len(d) != 1:
        raise FileNotFoundError(f'W&B run dir for {run_id}: {d}')
    files = Path(d[0]) / 'files'
    return files / f'eval/eval_with_train/epoch_{epoch}/val/result.pkl', files / 'config.yaml', files


def resolved_config(config_yaml):
    import yaml
    from easydict import EasyDict
    raw = yaml.safe_load(open(config_yaml))
    return EasyDict({k: (v['value'] if isinstance(v, dict) and 'value' in v else v) for k, v in raw.items()})


def target_dataset_shell(data_cfg, logger=None):
    """A NuScenesDataset carrying only what `evaluation(eval_metric='kitti')` reads: cfg, val infos, calib flags.

    The infos are read exactly as `include_nuscenes_data('test')` reads them, platform filter included.
    """
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    ds = NuScenesDataset.__new__(NuScenesDataset)
    ds.dataset_cfg = data_cfg
    ds.root_path = (TOOLS / data_cfg.DATA_PATH).resolve() / data_cfg.VERSION
    ds.infos = []
    for info_path in data_cfg.INFO_PATH['test']:
        with open(ds.root_path / info_path, 'rb') as f:
            ds.infos.extend(pickle.load(f))
    ds.infos = ds.filter_by_platform(ds.infos)
    ds.draw_conf_calib_curve = data_cfg.get('DRAW_CONF_CALIB_CURVE', False)
    ds.run_conf_calib = data_cfg.get('RUN_CONF_CALIB', False)
    return ds


def install_cpu_kitti_kernel():
    import kitti_eval_cpu
    kitti_eval_cpu.load_official_eval()  # puts the CPU stub at ...kitti_object_eval_python.rotate_iou
    import pcdet.datasets.kitti.kitti_object_eval_python.eval as official
    official.rotate_iou_gpu_eval = kitti_eval_cpu.rotate_iou_cpu
    return official


def kitti_metric(ds, det_annos, class_names):
    if len(det_annos) != len(ds.infos):
        raise ValueError(f'{len(det_annos)} detections vs {len(ds.infos)} infos')
    for d, info in zip(det_annos, ds.infos):  # kitti_eval pairs them BY INDEX
        if d['metadata']['token'] != info['token']:
            raise ValueError(f"order mismatch at {d['frame_id']}")
    _, ap = ds.evaluation(copy.deepcopy(det_annos), list(class_names), eval_metric='kitti')
    return {k: float(ap[k]) for k in ('Car_bev/easy_R40', 'Car_bev/moderate_R40', 'Car_bev/hard_R40',
                                      'Car_3d/easy_R40', 'Car_3d/moderate_R40')}


def numpy_float_alias():
    """nuscenes-devkit 1.0.5 `accumulate` calls `.astype(np.float)`, an alias numpy 1.24 removed (the image has 1.26.4).

    numpy's own message: "use `float` by itself. Doing this will not modify any behavior". Restored for this process
    only, so `NuScenesEval` can run at all (it raised AttributeError in the first self-test, job 29356).
    """
    if 'float' not in np.__dict__:
        np.float = float
    return np.float


def devkit_eval(nusc, renamed, out_dir, keep_json=False):
    numpy_float_alias()
    from pcdet.datasets.nuscenes import nuscenes_utils
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    out_dir.mkdir(parents=True, exist_ok=True)
    nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(renamed, nusc)
    nusc_annos['meta'] = {'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False,
                          'use_external': False}  # as nuscene_eval writes it
    res_path = out_dir / 'results_nusc.json'
    with open(res_path, 'w') as f:
        json.dump(nusc_annos, f)
    nusc_eval = NuScenesEval(nusc, config=config_factory(EVAL_VERSION), result_path=str(res_path), eval_set='val',
                             output_dir=str(out_dir), verbose=True)
    nusc_eval.main(plot_examples=0, render_curves=False)
    if not keep_json:
        os.remove(res_path)
    with open(out_dir / 'metrics_summary.json') as f:
        return json.load(f)


def gt_as_detections(ds, nusc_to_model):
    """Val GT of the classes the model predicts, as det_annos in the model vocabulary (score 1)."""
    model_names = list(nusc_to_model.values())
    dets = []
    for info in ds.infos:
        keep = np.array([n in nusc_to_model for n in info['gt_names']], dtype=bool)
        names = np.array([nusc_to_model[n] for n in info['gt_names'][keep]])
        dets.append({
            'name': names, 'score': np.ones(len(names)),
            'boxes_lidar': np.asarray(info['gt_boxes'])[keep, :7].astype(np.float64),
            'pred_labels': np.array([model_names.index(n) + 1 for n in names], dtype=np.int64),
            'frame_id': Path(info['lidar_path']).name, 'metadata': {'token': info['token']},
        })
    return dets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--row', nargs=4, action='append', default=[], metavar=('LABEL', 'WANDB_RUN', 'EPOCH', 'LOG'))
    p.add_argument('--gt-self-test', action='store_true')
    p.add_argument('--config-from', default=None, help='W&B run whose config.yaml defines the self-test target')
    p.add_argument('--keep-json', action='store_true')
    p.add_argument('--strict', action='store_true', help='exit non-zero if any row fails the validity check')
    args = p.parse_args()
    from nuscenes.nuscenes import NuScenes
    install_cpu_kitti_kernel()

    nusc, summary_rows = None, []
    jobs = []
    if args.gt_self_test:
        jobs.append(('gt_self_test', args.config_from, None, None))
    jobs += [tuple(r) for r in args.row]
    for label, run_id, epoch, log in jobs:
        t0 = time.time()
        out_dir = args.out / label
        if label == 'gt_self_test':
            cfg = resolved_config(run_files(run_id, 0)[1])
        else:
            result_pkl, cfg_yaml, _ = run_files(run_id, epoch)
            cfg = resolved_config(cfg_yaml)
        data_cfg = cfg.DATA_CONFIG_TAR if 'DATA_CONFIG_TAR' in cfg else cfg.DATA_CONFIG
        if data_cfg.DATASET != 'NuScenesDataset' or cfg.MODEL.POST_PROCESSING.EVAL_METRIC != 'kitti':
            raise ValueError(f'{label}: not a nuScenes target scored with the KITTI metric')
        class_names = list(cfg.CLASS_NAMES)
        name_map = model_to_nusc_names(class_names)
        ds = target_dataset_shell(data_cfg)
        if nusc is None:
            nusc = NuScenes(version=data_cfg.VERSION, dataroot=str(ds.root_path), verbose=True)
        row = {'label': label, 'wandb_run': run_id, 'epoch': epoch, 'class_names': class_names,
               'name_map': name_map, 'eval_version': EVAL_VERSION}
        if label == 'gt_self_test':
            det_annos = gt_as_detections(ds, {v: k for k, v in name_map.items()})
        else:
            det_annos = pickle.load(open(result_pkl, 'rb'))
            row['result_pkl'] = str(result_pkl)
            row['result_pkl_mtime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result_pkl.stat().st_mtime))
            logged, logged_dir = parse_logged_car_bev(Path(log).read_text(errors='replace'), epoch)
            row['log'] = log
            row['logged_car_bev_r40'] = logged
            row['logged_result_dir'] = logged_dir
            expected_dir = str(result_pkl.parent).replace('/home/koyama/data/', '/storage/')
            row['logged_result_dir_matches'] = (logged_dir == expected_dir)
            km = kitti_metric(ds, det_annos, class_names)
            row['kitti_recomputed'] = km
            row['kitti_delta_bev'] = km['Car_bev/easy_R40'] - logged[0]
            row['validity_pass'] = bool(abs(row['kitti_delta_bev']) <= 0.1 and row['logged_result_dir_matches'])
            print(f"[{label}] KITTI-metric Car BEV R40 recomputed {km['Car_bev/easy_R40']:.4f} vs logged "
                  f"{logged[0]:.4f} (d {row['kitti_delta_bev']:+.4f}); result dir matches log: "
                  f"{row['logged_result_dir_matches']}; PASS={row['validity_pass']}", flush=True)
        renamed, counts = check_and_rename(det_annos, class_names, name_map)
        row['counts'] = counts
        metrics = devkit_eval(nusc, renamed, out_dir, keep_json=args.keep_json)
        row['native'] = summarise(metrics, [name_map[c] for c in class_names])
        if label == 'gt_self_test':  # GT as detections: AP 100, centre / size exact, yaw within 0.05 rad
            pc = row['native']['per_class']
            row['validity_pass'] = bool(all(pc[c]['AP'] > 99.9 and pc[c]['trans_err'] < 0.01 and
                                            pc[c]['scale_err'] < 0.01 and pc[c]['orient_err'] < 0.05 for c in pc))
            print(f'[gt_self_test] PASS={row["validity_pass"]}', flush=True)
        row['seconds'] = round(time.time() - t0, 1)
        with open(out_dir / 'row_summary.json', 'w') as f:
            json.dump(row, f, indent=2)
        n = row['native']
        print(f"[{label}] native: " + ', '.join(f"{c} AP {n['per_class'][c]['AP']:.2f}" for c in n['classes'])
              + f"; mAP_k {n['mAP_k']:.2f}; mATE {n['mATE_k']:.3f} mASE {n['mASE_k']:.3f} mAOE {n['mAOE_k']:.3f};"
              f" NDS-det {n['NDS_det']:.2f}; {counts}; {row['seconds']} s", flush=True)
        summary_rows.append(row)
    with open(args.out / f'summary_{int(time.time())}.json', 'w') as f:
        json.dump(summary_rows, f, indent=2)
    failed = [r['label'] for r in summary_rows if r.get('validity_pass') is False]
    if args.strict and failed:
        sys.exit(f'validity check failed for {failed}')  # non-zero exit: an afterok successor does not start


if __name__ == '__main__':
    main()
