"""O_f, the frame-matched Waymo oracle twin (ANALYSIS; never a method): pre-launch checks (bug-prevention rule 4).

Design: experiments_md T05 §5 item 23 "FRAME-MATCHED TWIN O_f, design FIXED" (e9bb63b); report experiments_md 20261028_04.

    python analysis/oraclef_check.py

Run inside the container from ST3D/tools (`import _init_path` first, so the live pcdet is tested, not the image's).
Reuses condoracle_check.py's recipe comparison (flatten / logged_cfg / compare / check), imported, not edited.
Checks, each printed with PASS / FAIL:
1. Recipe: O_f's resolved config against the oracle 26398's LOGGED resolved config (its W&B log, i.e. the config as it
   ran at ST3D 08518ae). Allowed differences only: DATA_CONFIG.SAMPLED_INTERVAL.train (2 -> 10) and
   OPTIMIZATION.NUM_EPOCHS (7 -> 35). DATA_CONFIG_TAR against 27179's log (the scoring of the oracle on Waymo val): none.
   Beside: against the base yaml waymo2waymo resolved now: the same two keys only.
2. No gt_sampling (config and live augmentor queue); MAX_SWEEPS 1.
3. The training dataset as the loader builds it: 15,809 frames; frame ids, in order, equal PL-GT's frozen file
   (/home/koyama/data/plgt_20261027/waymo_train_si10_gt_as_pseudo_labels.pkl); a subset of the oracle's 79,041 (built
   with the oracle's config) and exactly its every 5th; per-frame stored labels (>= 1 point, model classes) equal the
   frozen file's per frame and in total; len(loader) at batch 6 on one process = steps per epoch; x 35 = total steps.
4. The evaluation dataset is the full Waymo val at interval 5 (7,998 frames).
5. One collated batch of 6 frames (one GPU x 6): served labels against stored, no label <= 0, and CenterHead
   assign_targets on the collated batch: heatmap peaks per class = distinct positive centre pixels per frame,
   regression targets = positives, no ignored pixel.
"""
import math, os, pickle, sys
from collections import Counter
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import condoracle_check as cc  # noqa: E402  (imports _init_path first: live pcdet, allocator guard)
import torch  # noqa: E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models.dense_heads.center_head import CenterHead  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

CFG = 'cfgs/da-ieee-access/centerpoint-oraclef-waymo-interval10.yaml'
BASE = 'cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml'
ORACLE_LOG = '/home/koyama/data/wandb/run-20260928_082639-y6p61n0x/files/log_train_20260928-082644.txt'  # 26398
EVAL_LOG = 'logs/error_27179_evWa2Wa.txt'                                                                 # 27179
PLGT = '/home/koyama/data/plgt_20261027/waymo_train_si10_gt_as_pseudo_labels.pkl'
WANT_FRAMES, ORACLE_FRAMES, EPOCHS, BATCH = 15809, 79041, 35, 6
check = cc.check


def load(path):
    c = EasyDict()
    cfg_from_yaml_file(path, c)
    return c


def train_set(c, logger, batch_size):
    ds, loader, _ = build_dataloader(dataset_cfg=c.DATA_CONFIG, class_names=c.CLASS_NAMES, batch_size=batch_size,
                                     dist=False, workers=0, logger=logger, training=True,
                                     model_ontology=c.get('ONTOLOGY'))
    return (loader.dataset.dataset if hasattr(loader.dataset, 'dataset') else ds), loader


def stored_labels(info, to_cls, names, min_pts):
    a = info['annos']
    nm, pts = np.asarray(a['name']), np.asarray(a['num_points_in_gt'])
    keep = pts >= min_pts
    return Counter(to_cls.get(x, None) for x in nm[keep] if to_cls.get(x, None) in names)


def main():
    cfg = load(CFG)
    C = list(cfg.CLASS_NAMES)
    logger = common_utils.create_logger()
    print(f'== O_f: {CFG}')

    print("1. recipe against the oracle run's logged config")
    res = cc.flatten(cfg)
    allowed = {'cfg.DATA_CONFIG.SAMPLED_INTERVAL.train', 'cfg.OPTIMIZATION.NUM_EPOCHS'}
    cc.compare(res, cc.logged_cfg(ORACLE_LOG), lambda k: not k.startswith('cfg.DATA_CONFIG_TAR.'), allowed,
               f'train log {os.path.basename(ORACLE_LOG)} (26398)')
    cc.compare(res, cc.logged_cfg(EVAL_LOG), lambda k: k.startswith('cfg.DATA_CONFIG_TAR.'), set(),
               f'DATA_CONFIG_TAR vs eval log {os.path.basename(EVAL_LOG)} (27179)')
    base = cc.flatten(load(BASE))
    cc.compare(res, base, lambda k: True, allowed, f'base yaml {os.path.basename(BASE)} resolved now')
    si = cfg.DATA_CONFIG.SAMPLED_INTERVAL
    check(int(si['train']) == 10 and int(si['test']) == 5 and cfg.OPTIMIZATION.NUM_EPOCHS == EPOCHS,
          f'SAMPLED_INTERVAL train {si["train"]} / test {si["test"]}; NUM_EPOCHS {cfg.OPTIMIZATION.NUM_EPOCHS}')

    print("2. no gt_sampling; one frame per sample")
    dcfg = cfg.DATA_CONFIG
    aug = dcfg.DATA_AUGMENTOR
    active = [a.NAME for a in aug.AUG_CONFIG_LIST if a.NAME not in aug.DISABLE_AUG_LIST]
    check('gt_sampling' not in [a.NAME for a in aug.AUG_CONFIG_LIST], f'no gt_sampling in AUG_CONFIG_LIST; active {active}')
    check((dcfg.get('MAX_SWEEPS', 1) or 1) == 1, f'MAX_SWEEPS {dcfg.get("MAX_SWEEPS", None)} (one frame per sample)')

    print('3. training dataset as the loader builds it')
    ds, loader = train_set(cfg, logger, BATCH)
    queue = [getattr(getattr(f, 'func', f), '__name__', str(f)) for f in ds.data_augmentor.data_augmentor_queue]
    check(not any('gt_sampling' in q or 'DataBaseSampler' in q for q in queue), f'live augmentor queue {queue}')
    n = len(ds)
    check(n == WANT_FRAMES, f'len(dataset) {n} == {WANT_FRAMES}')
    fids = [i['frame_id'] for i in ds.infos]
    check(len(set(fids)) == n, f'{len(set(fids))} distinct frame ids')
    plgt = pickle.load(open(PLGT, 'rb'))
    pl_ids = list(plgt['labels'].keys())
    print(f'    PL-GT file: {len(pl_ids)} frames, meta sampled_interval_train {plgt["meta"]["sampled_interval_train"]}, '
          f'written at ST3D {plgt["meta"]["st3d_head"]}; first 3 {pl_ids[:3]}')
    print(f'    O_f loader: first 3 {fids[:3]}; last {fids[-1]}')
    check(fids == pl_ids, 'O_f frame ids == PL-GT frozen file frame ids, same order, all frames')
    check(fids[:5] == pl_ids[:5], 'first five frame ids equal')
    steps = len(loader)
    check(steps == math.ceil(n / BATCH) == 2635, f'len(loader) at batch {BATCH}, one process: {steps} steps per epoch '
          f'(ceil({n} / {BATCH}) = {math.ceil(n / BATCH)})')
    print(f'    total steps {steps} x {EPOCHS} = {steps * EPOCHS} (oracle 13,174 x 7 = 92,218); '
          f'frame-passes {n * EPOCHS} (oracle 553,287)')

    to_cls = ds.map_ontology_dataset_to_model
    min_pts = dcfg.get('MIN_POINTS_OF_GT', 1)
    tot, mism = Counter(), 0
    for info in ds.infos:
        st = stored_labels(info, to_cls, C, min_pts)
        tot.update(st)
        lab = plgt['labels'][info['frame_id']]['gt_boxes'][:, 7].astype(int)
        pl = Counter(C[l - 1] for l in lab if l > 0)
        mism += int(pl != st)
    print(f'    stored labels over the {n} frames (>= {min_pts} point, model classes): {dict((c, tot[c]) for c in C)}; '
          f'PL-GT meta counts {plgt["meta"]["counts"]}')
    check(dict((c, tot[c]) for c in C) == dict(plgt['meta']['counts']), 'total per-class labels == the PL-GT file')
    check(mism == 0, f'per-frame per-class labels == the PL-GT file in every frame ({mism} frames differ)')
    del plgt

    oc = load(BASE)
    ods, _ = train_set(oc, logger, BATCH)
    ofids = [i['frame_id'] for i in ods.infos]
    check(len(ofids) == ORACLE_FRAMES, f"oracle config's train set {len(ofids)} == {ORACLE_FRAMES}")
    check(set(fids) <= set(ofids), "O_f's frames are a subset of the oracle's")
    check(fids == ofids[::5], "O_f's frames == every 5th of the oracle's (interval 10 = interval 2 x 5, one list)")
    del ods, ofids

    print('4. evaluation dataset')
    tds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=C, batch_size=1, dist=False, workers=0,
                                 logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY'))
    check(len(tds) == 7998, f'full Waymo val at interval {cfg.DATA_CONFIG_TAR.SAMPLED_INTERVAL["test"]}: {len(tds)} frames')
    del tds

    print('5. one collated batch of 6 (one GPU x 6): labels and assigned targets')
    torch.manual_seed(0); np.random.seed(0)
    idx = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]
    batch = ds.collate_batch([ds[i] for i in idx])
    gt = batch['gt_boxes']
    lab = gt[..., 7].astype(int)
    served = {c: int((lab == j + 1).sum()) for j, c in enumerate(C)}
    st = Counter()
    for i in idx:
        st.update(stored_labels(ds.infos[i], to_cls, C, min_pts))
    print(f'    frames {idx}; gt_boxes {gt.shape}, points {batch["points"].shape}; served {served} vs stored '
          f'{dict((c, st[c]) for c in C)}')
    check(int((lab < 0).sum()) == 0, 'no negative label in the batch (padding rows are 0)')
    check(all(served[c] <= st[c] for c in C) and served['Car'] >= 0.8 * st['Car'],
          'served <= stored per class, Car >= 0.8 (range filter after augmentation)')

    head = CenterHead.__new__(CenterHead); torch.nn.Module.__init__(head)
    head.model_cfg = cfg.MODEL.DENSE_HEAD
    head.class_names = C; head.class_names_each_head = [C]
    pcr = np.array(dcfg.POINT_CLOUD_RANGE, dtype=np.float32)
    head.point_cloud_range = pcr
    head.voxel_size = [p for p in dcfg.DATA_PROCESSOR if p.NAME == 'transform_points_to_voxels'][0].VOXEL_SIZE
    stride = cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
    side = int(round((pcr[3] - pcr[0]) / head.voxel_size[0] / stride))
    t = head.assign_targets(torch.from_numpy(gt).float().clone(), feature_map_size=[side, side])
    hm = t['heatmaps'][0]                                  # (B, C, side, side)
    cell = head.voxel_size[0] * stride
    exp = Counter()
    for b in range(gt.shape[0]):
        seen = set()
        for box, l in zip(gt[b], lab[b]):
            if l > 0 and box[3] > 0 and box[4] > 0:
                x = min(max((box[0] - pcr[0]) / cell, 0), side - 0.5)
                y = min(max((box[1] - pcr[1]) / cell, 0), side - 0.5)
                seen.add((l - 1, int(y), int(x)))
        exp.update(C[c] for c, _, _ in seen)
    peaks = {c: int(hm[:, j].eq(1).sum()) for j, c in enumerate(C)}
    print(f'    heatmap {tuple(hm.shape)}: peaks {peaks} vs distinct positive centre pixels {dict((c, exp[c]) for c in C)}; '
          f'input positives {served}')
    check(all(peaks[c] == exp[c] for c in C), 'heatmap peaks per class == distinct positive centre pixels per frame')
    reg = int(t['masks'][0].sum())
    check(reg == sum(served.values()), f'regression targets {reg} == input positives {sum(served.values())}')
    check(t['heatmap_masks'][0] is None, 'no ignored pixel (heatmap_masks None: no negative label)')

    print(f'== O_f: {"ALL PASS" if not cc.fails else str(len(cc.fails)) + " FAIL"}')
    return 1 if cc.fails else 0


if __name__ == '__main__':
    sys.exit(main())
