"""nuScenes-native metrics for a stored result.pkl named by PATH, with the provenance check done against the log block
that wrote it.

Why (experiments_md 20261024_04 item 2): `nuscenes_native_eval.py` (20261017_12, unchanged here) finds a row's
`result.pkl` as `<W&B run>/files/eval/eval_with_train/epoch_N/val/` and parses the training log's in-run evaluation.
The IA-SSD rows' second scorings were eval-only jobs (`analysis/eval_checkpoint.sh`) whose predictions sit under
`output/<cfg>/.../eval/epoch_N/val/<tag>/`, and one log can hold two scorings (job 28020, 28130). So each row here
names its `result.pkl`, the config.yaml of the W&B run that produced it, and the job's error log. The logged value
is taken from the ONE evaluation block (between `EVALUATION start` and its `Result is save to <dir>`) whose
directory is the pickle's directory (`/storage/` read as `/home/koyama/data/`); zero or two such blocks raise.

Everything else is `nuscenes_native_eval.py`'s: the class renaming and its refusals (sentinel labels, malformed
boxes, the 500-box cap), the devkit run (`detection_cvpr_2019`, val), mAP_k over the model's classes, NDS-det, and
the KITTI-metric Car BEV R40 @ IoU 0.7 recomputed from the same detections on CPU.

Validity (20261024_04 item 2, T05 §5 item 4): |recomputed - logged| Car BEV R40 @0.7 <= TOL (default 0.02), and
exactly one log block names the pickle's directory. A failing row is still written, flagged, and is not read.

    python analysis/nuscenes_native_eval_pkl.py --out <dir> [--tol 0.02] [--strict] \
        --row <label> <result.pkl> <config.yaml> <error_log> [--row ...]
"""
import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / 'analysis'))

HOST_DATA, CONTAINER_DATA = '/home/koyama/data/', '/storage/'
BLOCKS = (('bev_07', '3d_07', 'Car AP_R40@0.70, 0.70, 0.70:'), ('bev_05', '3d_05', 'Car AP_R40@0.70, 0.50, 0.50:'))


def host_path(p):
    p = str(p).rstrip('/')
    return HOST_DATA + p[len(CONTAINER_DATA):] if p.startswith(CONTAINER_DATA) else p


def eval_blocks(text):
    """Every evaluation block of a log: Car R40 BEV / 3D at IoU 0.7 and 0.5 (first column), with 1-based line numbers
    of the `Car AP_R40@...:` header lines in the '\\r' -> '\\n' converted text, and the block's result directory."""
    lines = text.replace('\r', '\n').split('\n')
    out, cur = [], None
    for i, line in enumerate(lines):
        if 'EVALUATION start' in line:
            cur = {'start_line': i + 1}
        if cur is None:
            continue
        for kb, k3, header in BLOCKS:
            if line.startswith(header):
                mb = re.match(r'bev\s+AP:\s*([0-9.]+),', lines[i + 2])
                m3 = re.match(r'3d\s+AP:\s*([0-9.]+),', lines[i + 3])
                if not (mb and m3):
                    raise ValueError(f'malformed block after line {i + 1}')
                cur[kb], cur[k3], cur[f'{kb}_header_line'] = float(mb.group(1)), float(m3.group(1)), i + 1
        m = re.search(r'Result is save to (\S+)', line)
        if m:
            cur['result_dir'] = host_path(m.group(1))
            cur['result_line'] = i + 1
            out.append(cur)
            cur = None
    return out


def logged_block_for(text, result_pkl):
    """The single evaluation block whose result directory holds `result_pkl`; raises on zero or several."""
    want = host_path(Path(result_pkl).parent)
    hits = [b for b in eval_blocks(text) if b['result_dir'] == want]
    if len(hits) != 1:
        raise ValueError(f'{len(hits)} evaluation blocks name {want}')
    if 'bev_07' not in hits[0]:
        raise ValueError(f'no Car AP_R40@0.70, 0.70, 0.70 block for {want}')
    return hits[0]


def validity(recomputed_bev, logged_bev, tol):
    """(difference, pass); inclusive at the tolerance (1e-9 absorbs the float error of 4-decimal log values)."""
    d = recomputed_bev - logged_bev
    return d, bool(abs(d) <= tol + 1e-9)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--row', nargs=4, action='append', required=True, metavar=('LABEL', 'RESULT_PKL', 'CONFIG', 'LOG'))
    p.add_argument('--tol', type=float, default=0.02)
    p.add_argument('--strict', action='store_true', help='exit non-zero if any row fails the validity check')
    args = p.parse_args()
    import nuscenes_native_eval as nne   # imports _init_path (the live pcdet) on load
    from nuscenes.nuscenes import NuScenes
    nne.install_cpu_kitti_kernel()
    nusc, rows = None, []
    for label, result_pkl, config_yaml, log in args.row:
        t0 = time.time()
        out_dir = args.out / label
        cfg = nne.resolved_config(config_yaml)
        data_cfg = cfg.DATA_CONFIG_TAR if 'DATA_CONFIG_TAR' in cfg else cfg.DATA_CONFIG
        if data_cfg.DATASET != 'NuScenesDataset' or cfg.MODEL.POST_PROCESSING.EVAL_METRIC != 'kitti':
            raise ValueError(f'{label}: not a nuScenes target scored with the KITTI metric')
        class_names = list(cfg.CLASS_NAMES)
        name_map = nne.model_to_nusc_names(class_names)
        ds = nne.target_dataset_shell(data_cfg)
        if nusc is None:
            nusc = NuScenes(version=data_cfg.VERSION, dataroot=str(ds.root_path), verbose=True)
        block = logged_block_for(Path(log).read_text(errors='replace'), result_pkl)
        det_annos = pickle.load(open(result_pkl, 'rb'))
        km = nne.kitti_metric(ds, det_annos, class_names)
        d, ok = validity(km['Car_bev/easy_R40'], block['bev_07'], args.tol)
        row = {'label': label, 'result_pkl': str(result_pkl), 'config': str(config_yaml), 'log': str(log),
               'result_pkl_mtime': time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(Path(result_pkl).stat().st_mtime)),
               'class_names': class_names, 'name_map': name_map, 'eval_version': nne.EVAL_VERSION,
               'logged': block, 'kitti_recomputed': km, 'kitti_delta_bev': d,
               'kitti_delta_3d': km['Car_3d/easy_R40'] - block['3d_07'], 'tol': args.tol, 'validity_pass': ok}
        print(f"[{label}] KITTI-metric Car BEV R40 @0.7 recomputed {km['Car_bev/easy_R40']:.4f} vs logged "
              f"{block['bev_07']:.4f} (log line {block['bev_07_header_line']}; d {d:+.4f}; tol {args.tol}); "
              f"3D d {row['kitti_delta_3d']:+.4f}; PASS={ok}", flush=True)
        renamed, counts = nne.check_and_rename(det_annos, class_names, name_map)
        row['counts'] = counts
        row['native'] = nne.summarise(nne.devkit_eval(nusc, renamed, out_dir), [name_map[c] for c in class_names])
        row['seconds'] = round(time.time() - t0, 1)
        with open(out_dir / 'row_summary.json', 'w') as f:
            json.dump(row, f, indent=2)
        n = row['native']
        print(f"[{label}] native: " + ', '.join(f"{c} AP {n['per_class'][c]['AP']:.2f}" for c in n['classes'])
              + f"; mAP_k {n['mAP_k']:.2f}; mATE {n['mATE_k']:.3f} mASE {n['mASE_k']:.3f} mAOE {n['mAOE_k']:.3f};"
              f" NDS-det {n['NDS_det']:.2f}; {counts}; {row['seconds']} s", flush=True)
        rows.append(row)
    with open(args.out / f'summary_{int(time.time())}.json', 'w') as f:
        json.dump(rows, f, indent=2)
    failed = [r['label'] for r in rows if not r['validity_pass']]
    if args.strict and failed:
        sys.exit(f'validity check failed for {failed}')


if __name__ == '__main__':
    main()
