"""P2-B condition oracles (ANALYSIS; never a method): pre-launch checks of one arm config (bug-prevention rule 4).

Design: experiments_md T05 §5 "PHASE 2, design FIXED 12:04" (af7680c); report experiments_md 20261028_03.

    python analysis/condoracle_check.py <arm cfg> <arm key in arms.json> <oracle train log> [<oracle eval log>]

Run inside the container from ST3D/tools (`import _init_path` first, so the live pcdet is tested, not the image's).
Checks, each printed with PASS / FAIL:
1. Recipe: the arm's resolved config against the oracle run's LOGGED resolved config (log_config_to_file lines of
   its W&B log, i.e. the config as it ran at its recorded commit). Allowed differences only: the train frame set key
   (DATA_SPLIT.train or INFO_PATH.train) and OPTIMIZATION.NUM_EPOCHS. DATA_CONFIG_TAR is compared against the eval
   log when given (Waymo: the oracle was trained with a nuScenes eval target and scored by 27179 on Waymo val).
2. No object from outside the arm's frames: no gt_sampling in AUG_CONFIG_LIST or in the live augmentor queue,
   MAX_SWEEPS 1 (no other frame's points).
3. The training dataset as the loader builds it: its length equals arms.json's expectation; its sequence / scene set
   equals the arm's list and is disjoint from val; steps per epoch at 2 ranks x 3 and total steps.
4. The evaluation dataset is the FULL val (Waymo 7,998 at interval 5; nuScenes 6,019).
5. Stored GT per model class over all arm frames (raw and >= MIN points), and served GT per class in one collated
   training batch (batch 3, as one rank sees it) and over 24 frames, against the stored labels of the same frames.
"""
import json, math, os, re, sys
from collections import Counter
import numpy as np
sys.path.insert(0, '.')
import _init_path  # noqa: F401  (live pcdet, allocator guard)
import torch
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

ARMS = '/home/koyama/data/cond_oracle_20261028/arms.json'
# set by train.py at run time, or the arm file's own top-level _BASE_CONFIG_ (nested _BASE_CONFIG_ keys ARE compared)
IGNORE_KEYS = {'ROOT_DIR', 'LOCAL_RANK', 'TAG', 'EXP_GROUP_PATH', '_BASE_CONFIG_'}
fails = []


def check(ok, msg):
    print(('  PASS ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


def flatten(d, pre='cfg'):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, pre + '.' + k))
        else:
            out[pre + '.' + k] = str(v)
    return out


def logged_cfg(path):
    out = {}
    for line in open(path, errors='replace'):
        m = re.search(r'config\.py \d+ +INFO\]  (cfg\.[^:]+): (.*)$', line.rstrip('\n'))
        if m:
            out[m.group(1)] = m.group(2)
    return out


def compare(resolved, logged, prefix_filter, allowed, label):
    keys = {k for k in set(resolved) | set(logged) if prefix_filter(k) and k.split('.')[1] not in IGNORE_KEYS}
    diffs = sorted(k for k in keys if resolved.get(k) != logged.get(k))
    unexpected = [k for k in diffs if k not in allowed]
    for k in diffs:
        print(f'    {"allowed" if k in allowed else "UNEXPECTED"} {k}: oracle {logged.get(k)!r} -> arm {resolved.get(k)!r}')
    check(not unexpected, f'{label}: {len(keys)} keys compared, {len(diffs)} differ, unexpected {unexpected}')


def main():
    cfg_file, arm, train_log = sys.argv[1], sys.argv[2], sys.argv[3]
    eval_log = sys.argv[4] if len(sys.argv) > 4 else None
    exp = json.load(open(ARMS))[arm]
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg)
    dcfg, tcfg = cfg.DATA_CONFIG, cfg.DATA_CONFIG_TAR
    waymo = dcfg.DATASET == 'WaymoDataset'
    print(f'== {arm}: {cfg_file}')

    print('1. recipe against the oracle run\'s logged config')
    res = flatten(cfg)
    frame_key = 'cfg.DATA_CONFIG.DATA_SPLIT.train' if waymo else 'cfg.DATA_CONFIG.INFO_PATH.train'
    allowed = {frame_key, 'cfg.OPTIMIZATION.NUM_EPOCHS'}
    compare(res, logged_cfg(train_log), lambda k: not (eval_log and k.startswith('cfg.DATA_CONFIG_TAR.')),
            allowed, f'train log {os.path.basename(train_log)}')
    if eval_log:
        compare(res, logged_cfg(eval_log), lambda k: k.startswith('cfg.DATA_CONFIG_TAR.'), set(),
                f'DATA_CONFIG_TAR vs eval log {os.path.basename(eval_log)}')
    print(f'    arm frame set: {frame_key} = {res[frame_key]}; NUM_EPOCHS {cfg.OPTIMIZATION.NUM_EPOCHS}')

    print('2. no object from outside the arm\'s frames')
    aug = dcfg.DATA_AUGMENTOR
    active = [a.NAME for a in aug.AUG_CONFIG_LIST if a.NAME not in aug.DISABLE_AUG_LIST]
    check('gt_sampling' not in [a.NAME for a in aug.AUG_CONFIG_LIST], f'no gt_sampling entry in AUG_CONFIG_LIST; active {active}')
    check((dcfg.get('MAX_SWEEPS', 1) or 1) == 1, f'MAX_SWEEPS {dcfg.get("MAX_SWEEPS", None)} (one frame per sample)')

    print('3. training dataset as the loader builds it')
    logger = common_utils.create_logger()
    ds, loader, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=3, dist=False,
                                     workers=0, logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY'))
    ds = loader.dataset.dataset if hasattr(loader.dataset, 'dataset') else ds
    queue = [getattr(f, 'func', f).__name__ if hasattr(getattr(f, 'func', f), '__name__') else str(f)
             for f in ds.data_augmentor.data_augmentor_queue]
    check(not any('gt_sampling' in q or 'DataBaseSampler' in q for q in queue), f'live augmentor queue {queue}')
    n = len(ds)
    want = exp['frames_expected_at_interval2_own_list'] if waymo else exp['keyframes']
    check(n == want, f'len(dataset) {n} == expected {want}')
    if waymo:
        seqs = {i['point_cloud']['lidar_sequence'] for i in ds.infos}
        check(seqs == set(exp['list']), f'{len(seqs)} sequences == the arm list ({len(exp["list"])})')
        val = {os.path.splitext(l.strip())[0] for l in open(ds.root_path / 'ImageSets' / 'val.txt') if l.strip()}
        check(not (seqs & val), 'disjoint from ImageSets/val.txt')
        names = lambda i: (i['annos']['name'], i['annos'].get('num_points_in_gt'))
        min_pts = dcfg.get('MIN_POINTS_OF_GT', 1)
    else:
        log_of = ds.sample_token_to_log()
        meta = ds.root_path / ds.dataset_cfg.VERSION
        scene_name = {s['token']: s['name'] for s in json.load(open(meta / 'scene.json'))}
        scene_of = {s['token']: scene_name[s['scene_token']] for s in json.load(open(meta / 'sample.json'))}
        scenes = {scene_of[i['token']] for i in ds.infos}
        check(scenes == set(exp['list']), f'{len(scenes)} scenes == the arm list ({len(exp["list"])})')
        check({log_of[i['token']]['location'] for i in ds.infos} == {'boston-seaport'}, 'every frame is boston-seaport')
        import pickle
        val_scenes = {scene_of[i['token']] for i in pickle.load(open(ds.root_path / 'nuscenes_infos_10sweeps_val.pkl', 'rb'))}
        check(not (scenes & val_scenes), f'disjoint from the {len(val_scenes)} val scenes')
        names = lambda i: (i['gt_names'], i.get('num_lidar_pts'))
        min_pts = dcfg.get('FILTER_MIN_POINTS_IN_GT', 1)
    it = math.ceil(math.ceil(n / 2) / 3)
    print(f'    steps per epoch at 2 ranks x 3 (DistributedSampler, drop_last False): {it}; '
          f'x {cfg.OPTIMIZATION.NUM_EPOCHS} epochs = {it * cfg.OPTIMIZATION.NUM_EPOCHS} steps')

    print('4. evaluation dataset')
    tds, _, _ = build_dataloader(dataset_cfg=tcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                 logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    check(len(tds) == (7998 if waymo else 6019), f'full val {len(tds)} frames')

    print('5. labels: stored over the arm, served in one batch and over 24 frames')
    m = ds.map_ontology_dataset_to_model
    to_cls = lambda nm: m.get(nm, None)
    C = list(cfg.CLASS_NAMES)

    def stored(info):
        nm, pts = names(info)
        nm = np.asarray(nm)
        keep = np.ones(len(nm), bool) if pts is None else (np.asarray(pts) >= min_pts)
        raw = Counter(to_cls(x) for x in nm)
        kept = Counter(to_cls(x) for x in nm[keep])
        return raw, kept
    tot_raw, tot_kept = Counter(), Counter()
    for info in ds.infos:
        r, k = stored(info)
        tot_raw.update(r); tot_kept.update(k)
    print('    stored over all', n, 'frames (raw / >= %d pts):' % min_pts,
          {c: (tot_raw[c], tot_kept[c]) for c in C}, '| other names ->', {k: v for k, v in tot_raw.items() if k not in C})
    check(all(tot_kept[c] > 0 for c in C), 'every class has stored boxes')

    torch.manual_seed(0); np.random.seed(0)
    batch_idx = [0, n // 2, n - 1]
    batch = ds.collate_batch([ds[i] for i in batch_idx])
    lab = batch['gt_boxes'][..., 7].astype(int)
    served = {c: int((lab == j + 1).sum()) for j, c in enumerate(C)}
    st = Counter()
    for i in batch_idx:
        st.update(stored(ds.infos[i])[1])
    print(f'    one batch (frames {batch_idx}, gt_boxes {batch["gt_boxes"].shape}, points {batch["points"].shape}): '
          f'served {served} vs stored >= {min_pts} pts {dict((c, st[c]) for c in C)}')
    check(int((lab < 0).sum()) == 0, 'no negative label in the batch')
    srv, sto = Counter(), Counter()
    for i in np.linspace(0, n - 1, 24).astype(int):
        d = ds[int(i)]
        ll = d['gt_boxes'][:, 7].astype(int)
        for j, c in enumerate(C):
            srv[c] += int((ll == j + 1).sum())
        check_neg = int((ll <= 0).sum())
        if check_neg:
            check(False, f'frame {i}: {check_neg} labels <= 0 served')
        sto.update(stored(ds.infos[int(i)])[1])
    ratio = {c: (srv[c], sto[c], round(srv[c] / max(sto[c], 1), 3)) for c in C}
    print('    24 frames, served / stored / ratio:', ratio)
    check(all(0.8 <= r[2] <= 1.0 for c, r in ratio.items() if r[1] >= 20),
          'served/stored in [0.8, 1.0] for every class with >= 20 stored boxes (range filter after augmentation)')
    print(f'== {arm}: {"ALL PASS" if not fails else str(len(fails)) + " FAIL"}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
