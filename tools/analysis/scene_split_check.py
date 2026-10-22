"""SCENE pre-launch sanity (experiments_md/20261022_06 §3): the nuScenes two-city configs, before any GPU row.

    singularity exec --bind /home/koyama/data/:/storage <sif> python3 analysis/scene_split_check.py   (from ST3D/tools, CPU)

1. Every SCENE config resolves (cfg_from_yaml_file) and its resolved keys are what the design says: source VEHICLE,
   eval VEHICLE, MAX_SWEEPS 1, the oracle's schedule (NUM_EPOCHS 20, BATCH_SIZE_PER_GPU 3, LR 0.003), no target label
   path in a source-only row (no SELF_TRAIN, no USE_PSEUDO_LABEL), the PL row with USE_PSEUDO_LABEL and a frozen
   single-pass teacher. Any mismatch raises.
2. The split is really one city: the 10-sweep TRAIN and VAL infos are mapped to their log (sample -> scene -> log,
   as nuscenes_dataset.filter_by_platform does) and counted per vehicle and per location, with Car / Pedestrian /
   Cyclist labels per frame (dataset description, reported). Iterations per epoch at global batch 6 (ceil, as
   len(DataLoader) without drop_last) and the 20-epoch step count follow.
3. Mounting per vehicle: ref_from_car (vehicle -> LIDAR_TOP) of every info, inverted to the sensor pose in the vehicle
   frame; distinct translations (mm) and Euler angles (0.01 deg) per vehicle, with counts.
No point cloud is read. Exit code 0 only if every assertion holds.
"""
import collections
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _init_path  # noqa: F401,E402  (must precede pcdet)
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402

CFG = 'cfgs/da-ieee-access/'
ROWS = {  # file -> (source vehicle, eval vehicle, pseudo-labelling)
    'centerpoint-sourceonly-nuscenes-boston2boston.yaml': ('n008', 'n008', False),
    'centerpoint-sourceonly-nuscenes-singapore2singapore.yaml': ('n015', 'n015', False),
    'centerpoint-sourceonly-nuscenes-boston2singapore.yaml': ('n008', 'n015', False),
    'centerpoint-sourceonly-nuscenes-singapore2boston.yaml': ('n015', 'n008', False),
    'centerpoint-st3d-nuscenes-boston2singapore.yaml': ('n008', 'n015', True),
}
DATA = Path('../data/nuscenes/v1.0-trainval')
META = DATA / 'v1.0-trainval'
GLOBAL_BATCH = 6
CITY = {'n008': 'Boston', 'n015': 'Singapore'}


def check_configs():
    ok = True
    for name, (src, tgt, pl) in ROWS.items():
        cfg = EasyDict()
        cfg_from_yaml_file(CFG + name, cfg)
        d, t, o = cfg.DATA_CONFIG, cfg.DATA_CONFIG_TAR, cfg.OPTIMIZATION
        checks = {
            'source VEHICLE': (d.get('VEHICLE'), src),
            'eval VEHICLE': (t.get('VEHICLE'), tgt),
            'source LOCATION absent': (d.get('LOCATION'), None),
            'eval LOCATION absent': (t.get('LOCATION'), None),
            'source DATASET': (d.DATASET, 'NuScenesDataset'),
            'eval DATASET': (t.DATASET, 'NuScenesDataset'),
            'source MAX_SWEEPS': (d.MAX_SWEEPS, 1),
            'eval MAX_SWEEPS': (t.MAX_SWEEPS, 1),
            'source split': (dict(d.DATA_SPLIT), {'train': 'train', 'test': 'val'}),
            'eval split': (dict(t.DATA_SPLIT), {'train': 'train', 'test': 'val'}),
            'source train infos': (list(d.INFO_PATH['train']), ['nuscenes_infos_10sweeps_train.pkl']),
            'eval test infos': (list(t.INFO_PATH['test']), ['nuscenes_infos_10sweeps_val.pkl']),
            'SHIFT_COOR': (list(d.SHIFT_COOR), [0.0, 0.0, 1.75]),
            'eval SHIFT_COOR': (list(t.SHIFT_COOR), [0.0, 0.0, 1.75]),
            'features': (list(d.POINT_FEATURE_ENCODING.used_feature_list), ['x', 'y', 'z']),
            'NUM_EPOCHS': (o.NUM_EPOCHS, 20),
            'BATCH_SIZE_PER_GPU': (o.BATCH_SIZE_PER_GPU, 3),
            'LR': (o.LR, 0.003),
            'OPTIMIZER': (o.OPTIMIZER, 'adam_onecycle'),
            'NUM_EPOCHS_TO_EVAL': (o.NUM_EPOCHS_TO_EVAL, 1),
            'CLASS_NAMES': (list(cfg.CLASS_NAMES), ['Car', 'Pedestrian', 'Cyclist']),
            'ONTOLOGY': (cfg.ONTOLOGY, 'kitti'),
            'MODEL': (cfg.MODEL.NAME, 'CenterPoint'),
            'VFE': (cfg.MODEL.VFE.NAME, 'MeanVFE'),
            'head SCORE_THRESH': (cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH, 0.0001),
            'model SCORE_THRESH': (cfg.MODEL.POST_PROCESSING.SCORE_THRESH, 0.0001),
            'eval USE_PSEUDO_LABEL': (bool(t.get('USE_PSEUDO_LABEL', False)), pl),
            'SELF_TRAIN present': ('SELF_TRAIN' in cfg, pl),
            'source DATA_PROCESSOR resolved': (len(d.DATA_PROCESSOR) >= 3, True),
            'eval DATA_PROCESSOR resolved': (len(t.DATA_PROCESSOR) >= 3, True),
            'POINT_CLOUD_RANGE': (list(d.POINT_CLOUD_RANGE), [-75.2, -75.2, -2, 75.2, 75.2, 4]),
        }
        if pl:
            st = cfg.SELF_TRAIN
            checks.update({
                'FROZEN_TEACHER_SINGLE_PASS': (st.FROZEN_TEACHER_SINGLE_PASS, True),
                'EPOCH_FOLLOWS': (st.EPOCH_FOLLOWS, 'source'),
                'UPDATE_PSEUDO_LABEL': (list(st.UPDATE_PSEUDO_LABEL), [0]),
                'PROG_AUG': (list(st.PROG_AUG.UPDATE_AUG), [5, 10, 15]),
                'no DSNORM': (st.get('DSNORM', False), False),
                'no TEACHER_ADABN': (st.get('TEACHER_ADABN', None), None),
                'teacher VFE': (st.MODEL_TEACHER.VFE.NAME, 'MeanVFE'),
                'teacher head floor': (st.MODEL_TEACHER.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH, 0.0001),
                'cuts still placeholders': (list(st.SCORE_THRESH) == [1.01, 1.01, 1.01], 'see stage 2'),
            })
        print(f'\n=== {name}')
        for k, (got, want) in checks.items():
            if want == 'see stage 2':
                print(f'  {k}: {got}  (filled by the session after stage 2)')
                continue
            flag = 'ok' if got == want else 'MISMATCH'
            ok &= got == want
            print(f'  {k}: {got} {"" if flag == "ok" else f"!= {want} {flag}"}')
        # the source-only rows must carry nothing that touches target labels at training time
        if not pl:
            for key in ('SELF_TRAIN', 'DATA_CONFIGS', 'MODEL_TEACHER'):
                if key in cfg:
                    print(f'  {key} present in a source-only row: MISMATCH'); ok = False
    return ok


def sample_to_log():
    logs = {l['token']: l for l in json.load(open(META / 'log.json'))}
    scene_log = {s['token']: logs[s['log_token']] for s in json.load(open(META / 'scene.json'))}
    return {s['token']: scene_log[s['scene_token']] for s in json.load(open(META / 'sample.json'))}


def euler_deg(R):
    # ZYX (yaw, pitch, roll) of a rotation matrix, degrees
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    pitch = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def check_split():
    log_of = sample_to_log()
    classes = {'car': 'Car', 'pedestrian': 'Pedestrian', 'bicycle': 'Cyclist'}
    mount = {v: collections.Counter() for v in CITY}
    summary = {}
    for split in ('train', 'val'):
        infos = pickle.load(open(DATA / f'nuscenes_infos_10sweeps_{split}.pkl', 'rb'))
        per_vehicle = collections.Counter()
        per_location = collections.Counter()
        labels = {v: collections.Counter() for v in CITY}
        missing = 0
        for info in infos:
            log = log_of.get(info['token'])
            if log is None:
                missing += 1
                continue
            v = log['vehicle']
            per_vehicle[v] += 1
            per_location[(v, log['location'])] += 1
            for n in info['gt_names']:
                if n in classes:
                    labels[v][classes[n]] += 1
            M = np.asarray(info['ref_from_car'], dtype=np.float64)
            inv = np.linalg.inv(M)  # sensor pose in the vehicle frame
            t = tuple(np.round(inv[:3, 3] * 1000).astype(int))  # mm
            ang = tuple(np.round(euler_deg(inv[:3, :3]), 2))
            mount[v][(t, ang)] += 1
        print(f'\n=== {split}: {len(infos)} infos, {missing} without a log record')
        for v in sorted(per_vehicle):
            n = per_vehicle[v]
            it = math.ceil(n / GLOBAL_BATCH)
            print(f'  {v} ({CITY[v]}): {n} frames'
                  + (f'; {it} it/epoch at global batch {GLOBAL_BATCH}; 20 epochs = {20 * it} steps' if split == 'train' else '')
                  + '; per frame ' + ', '.join(f'{c} {labels[v][c] / n:.2f}' for c in ('Car', 'Pedestrian', 'Cyclist')))
            summary[(split, v)] = n
        for (v, loc), n in sorted(per_location.items()):
            print(f'    {v} {loc}: {n}')
        assert missing == 0, 'infos without a log record'
        assert set(per_vehicle) == {'n008', 'n015'}, per_vehicle
        for (v, loc) in per_location:
            assert (v == 'n008') == loc.startswith('boston'), f'vehicle/city mapping not exact: {v} {loc}'
    print('\n=== mounting (LIDAR_TOP pose in the vehicle frame from ref_from_car; translation mm, yaw/pitch/roll deg ZYX)')
    for v in sorted(mount):
        print(f'  {v} ({CITY[v]}): {len(mount[v])} distinct calibrations over {sum(mount[v].values())} infos')
        for (t, ang), n in mount[v].most_common(6):
            print(f'    t = {t} mm, (yaw, pitch, roll) = {ang} deg: {n} infos')
    return summary


if __name__ == '__main__':
    ok = check_configs()
    summary = check_split()
    print('\n=== SCENE pre-launch sanity:', 'ALL PASS' if ok else 'FAILED')
    sys.exit(0 if ok else 1)
