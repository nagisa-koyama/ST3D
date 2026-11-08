"""Pre-GPU checks of R7r, the random label-drop twin of the label-range swap R7 (TRAIN_FACTOR_SWAP.GT_IGNORE_RANDOM;
experiments_md 20261003_05 §5 item 42c, report 20261108_07; bug_prevention_checklist rules 1 / 4 / 8). For one target,
the oracle, R7 and R7r configs go through the TRAINING loader on strided TRAIN frames and the script checks:
  1. config: R7r = R7 with GT_IGNORE_BY_RANGE replaced by GT_IGNORE_RANDOM, against today's R7 file AND both R7 jobs'
     LOGGED configs (train.py's `cfg.<path>: <value>` lines); R7r = the oracle plus that one key; KEEP = R7's kept share
     of Car supervision from its stats JSON, rounded to 4 decimals; CLASS Car; no processor recount (drop_empty_gt_boxes);
  2. identity (rule 8): R7r at KEEP 1 against the oracle on IDENTITY_FRAMES strided frames WITH augmentation, each frame
     seeded alike: points and labelled boxes exact; R7r at its real KEEP on the same frames, augmentation on: points and
     box geometry exact, labels differ only Car -> -99 (frames with a non-class box before a class box counted, the order
     that misaligned gt_classes before ST3D 12b5147);
  3. kept share of Car supervision on SHARE_FRAMES strided frames (augmentation off; the boxes that reach the head): R7r
     within +-0.01 of KEEP; per 10 m range bin FLAT, i.e. every bin with >= MIN_BIN Car boxes within 3 binomial SE of
     KEEP; R7's on the same frames beside it, each bin against R7's KEEP[b] (R7 is not flat by design);
  4. one IGNORE set for both training seeds: the arm's gt_classes identical under np.random seeds 666 / 667 and in a
     second dataset instance;
  5. targets (rule 4): one collated batch through CenterHead.assign_targets: per-class positives and peaks, ignore boxes,
     ignored px; every -99 box was an oracle Car positive; no -99 centre is a peak and its pixel is masked in all classes;
     Pedestrian / Cyclist heatmaps identical to the oracle's; fewer regression targets by at most the ignored count; the
     masked heatmap loss is unchanged by any prediction on the ignored footprint (neither a positive nor a negative);
  6. evaluation-inert: DATA_CONFIG_TAR identical to the oracle's and its frame 0 identical; R7r's training block built for
     evaluation identical at frame 0 (the key is training-gated).
Any failed check exits 1. ANALYSIS (target TRAIN labels are read); CPU; run as a Slurm CPU job from ST3D/tools:

    python analysis/item42c_random_drop_check.py --target waymo|pandaset|lyft64

--full-pass (added 2026-11-08 23:09 JST after the 300-frame flatness check failed in one PandaSet bin; report 20261108_07
§2.3) instead runs R7r and R7 over EVERY training frame (augmentation off, --workers processes) and reports the kept share
of Car supervision that each training row actually sees, in total and per 10 m bin. A prepared box of R7 / R7r is a Car
iff its label is 1 or -99 (non-class placeholders are removed by the class filter; checked in 5 above). Criteria, fixed
before it ran: R7r's total within +-0.005 of KEEP; every R7r bin with >= MIN_BIN boxes within 3 binomial SE of KEEP; R7
and R7r see the same Car box population per bin.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent)); sys.path.insert(0, str(TOOLS / 'analysis'))
os.chdir(TOOLS)
import _init_path  # noqa: E402,F401
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.datasets.waymo import waymo_factor_swap as wfs  # noqa: E402
from pcdet.models.dense_heads.center_head import CenterHead  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402
import factorswap_pl_config_check as C  # noqa: E402  (flat / logged / diff of item 34)

A = 'cfgs/da-ieee-access-analysis/'
TARGETS = {
    'waymo': dict(oracle='cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml',
                  r7=A + 'factorswap-n2w/centerpoint-factorswap-n2w-r7-labelrange.yaml',
                  r7r=A + 'factorswap-n2w/centerpoint-factorswap-n2w-r7r-labelrandom.yaml', jobs=(31621, 31812),
                  stats=('/home/koyama/data/factorswap_n2w/stats.json', 'kept_share_of_waymo_cars')),
    'pandaset': dict(oracle='cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml',
                     r7=A + 'factorswap-pandaset/centerpoint-factorswap-pandaset-r7-labelrange.yaml',
                     r7r=A + 'factorswap-pandaset/centerpoint-factorswap-pandaset-r7r-labelrandom.yaml', jobs=(31981, 32125),
                     stats=('/home/koyama/data/factorswap_pandaset/stats.json', 'kept_share_of_target_cars')),
    'lyft64': dict(oracle=A + 'factorswap-lyft64/centerpoint-factorswap-lyft64-oracle.yaml',
                   r7=A + 'factorswap-lyft64/centerpoint-factorswap-lyft64-r7-labelrange.yaml',
                   r7r=A + 'factorswap-lyft64/centerpoint-factorswap-lyft64-r7r-labelrandom.yaml', jobs=(32083, 32127),
                   stats=('/home/koyama/data/factorswap_lyft/stats_lyft64.json', 'kept_share_of_target_cars')),
}
P = 'DATA_CONFIG.TRAIN_FACTOR_SWAP.'
IDENTITY_FRAMES = 48
SHARE_FRAMES = 300
SEED_FRAMES = 20
BATCH_FRAMES = 6
MIN_BIN = 100
IGN = common_utils.IGNORE_CLASS_LABEL
logger = common_utils.create_logger()
FAILS = []


def check(ok, msg):
    print('    [%s] %s' % ('ok' if ok else 'FAIL', msg), flush=True)
    if not ok:
        FAILS.append(msg)


def load(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return cfg_from_yaml_file(path, EasyDict())


def build(cfg, dcfg, training=True, aug_off=True):
    with contextlib.redirect_stdout(io.StringIO()):
        ds, _, _ = build_dataloader(dataset_cfg=dcfg, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                    logger=logger, training=training, model_ontology=cfg.get('ONTOLOGY'))
    if training and aug_off:
        ds.data_augmentor.data_augmentor_queue = []
    return ds


def item(ds, i, seed=0):
    np.random.seed(seed)
    t = time.time()
    d = ds[i]
    return d, time.time() - t


def raw_input(ds, i, seed=0):
    """The loader's input_dict for frame i before prepare_data (the swap applied)."""
    keep = ds.prepare_data
    ds.prepare_data = lambda data_dict: data_dict
    try:
        np.random.seed(seed)
        return ds[i]
    finally:
        ds.prepare_data = keep


def frame_key(ds, i):
    if hasattr(ds, 'pandaset_infos'):
        info = ds.pandaset_infos[i]
        return '%s/%02d' % (info['sequence'], int(info['frame_idx']))
    if 'frame_id' in ds.infos[i]:
        return ds.infos[i]['frame_id']
    return Path(ds.infos[i]['lidar_path']).stem


def digest(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def passes(cfg, frames, aug_off, seed_of=lambda i: 0, keep_items=()):
    """One pass of a config's training loader over `frames`: boxes (with labels), a points digest, the time per frame."""
    ds = build(cfg, cfg.DATA_CONFIG, aug_off=aug_off)
    out = {}
    for i in frames:
        d, t = item(ds, i, seed_of(i))
        out[i] = {'boxes': np.array(d['gt_boxes'], copy=True), 'pts': digest(d['points']), 't': t,
                  'key': frame_key(ds, i)}
        if i in keep_items:
            out[i]['item'] = d
    return ds, out


def nonclass_first(names_model, class_names):
    seen_other = False
    for n in names_model:
        if n not in class_names:
            seen_other = True
        elif seen_other:
            return True
    return False


# ------------------------------------------------------------------------------------------------- 1. configs
def config_checks(T, r7r_cfg):
    print('=' * 118 + '\n1. CONFIG: R7r = R7 with the arm swapped (today\'s file and both R7 jobs\' logged configs)')
    r7r, r7, o = C.flat(load(T['r7r'])), C.flat(load(T['r7'])), C.flat(load(T['oracle']))
    want = [P + 'GT_IGNORE_BY_RANGE.CLASS', P + 'GT_IGNORE_BY_RANGE.EDGES', P + 'GT_IGNORE_BY_RANGE.KEEP',
            P + 'GT_IGNORE_RANDOM.CLASS', P + 'GT_IGNORE_RANDOM.KEEP']
    got = C.diff(r7r, r7)
    check(set(got) == set(want), 'R7r vs R7 (today): differs in %s' % got)
    for j in T['jobs']:
        lg = C.logged(j)
        got = C.diff(r7r, lg)
        check(set(got) == set(want), 'R7r vs R7 job %d LOGGED config: differs in %s' % (j, got))
        print('      job %d logged R7: CLASS %s, EDGES %s, KEEP %s' % (j, lg.get(P + 'GT_IGNORE_BY_RANGE.CLASS'),
                                                                   lg.get(P + 'GT_IGNORE_BY_RANGE.EDGES'),
                                                                   lg.get(P + 'GT_IGNORE_BY_RANGE.KEEP')))
    got = C.diff(r7r, o)
    check(set(got) == set(want[3:]), 'R7r vs the oracle (today): differs in %s' % got)
    path, key = T['stats']
    k_stats = json.load(open(path))['label_range'][key]
    k = float(r7r[P + 'GT_IGNORE_RANDOM.KEEP'])
    print('      R7 kept share (%s in %s): %.7f; R7r KEEP %.4f' % (key, path, k_stats, k))
    check(k == round(k_stats, 4) and r7r[P + 'GT_IGNORE_RANDOM.CLASS'] == 'Car',
          'KEEP = round(R7 kept share, 4) = %.4f, CLASS Car' % round(k_stats, 4))
    procs = [p.NAME for p in r7r_cfg.DATA_CONFIG.DATA_PROCESSOR]
    augs = [a.NAME for a in r7r_cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST
            if a.NAME not in r7r_cfg.DATA_CONFIG.DATA_AUGMENTOR.get('DISABLE_AUG_LIST', [])]
    print('      processor %s; augmentor %s' % (procs, augs))
    check('drop_empty_gt_boxes' not in procs, 'no processor recount (an empty -99 box would be kept where the oracle '
                                              'drops the positive; the comparisons below assume equal box sets)')
    return k


# ------------------------------------------------------------------------------------------------- 2. identity
def identity_checks(T, o_cfg, r_cfg, k, n):
    print('=' * 118 + '\n2. IDENTITY (rule 8): augmentation ON, each frame seeded alike, %d strided TRAIN frames' % IDENTITY_FRAMES)
    idx = list(np.linspace(0, n - 1, IDENTITY_FRAMES).astype(int))
    ds, base = passes(o_cfg, idx, aug_off=False, seed_of=int)
    nf = sum(nonclass_first(wfs.model_names(raw_input(ds, i)['gt_names'], ds.map_ontology_dataset_to_model),
                            o_cfg.CLASS_NAMES) for i in idx)
    del ds
    i_cfg = copy.deepcopy(r_cfg)
    i_cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP.GT_IGNORE_RANDOM.KEEP = 1.0
    _, ident = passes(i_cfg, idx, aug_off=False, seed_of=int)
    n_same = sum(base[i]['pts'] == ident[i]['pts'] and np.array_equal(base[i]['boxes'], ident[i]['boxes']) for i in idx)
    print('  KEEP 1.0: %d of %d frames identical to the oracle in points AND labelled boxes (%d with a non-class box '
          'before a class box)' % (n_same, len(idx), nf))
    check(n_same == len(idx), 'R7r at KEEP 1 reproduces the oracle exactly')
    check(nf > 0, 'the identity frames include non-class boxes before class boxes')
    _, real = passes(r_cfg, idx, aug_off=False, seed_of=int)
    n_only = n_car = n_ign = 0
    for i in idx:
        b, s = base[i]['boxes'], real[i]['boxes']
        same_geo = base[i]['pts'] == real[i]['pts'] and b.shape == s.shape and np.array_equal(b[:, :-1], s[:, :-1])
        bl, sl = b[:, -1].astype(int), s[:, -1].astype(int) if same_geo else None
        ok = same_geo and bool(np.all((bl == sl) | ((bl == 1) & (sl == IGN))))
        n_only += ok
        if same_geo:
            n_car += int((bl == 1).sum()); n_ign += int(((bl == 1) & (sl == IGN)).sum())
    print('  KEEP %.4f: %d of %d frames with the oracle\'s points and box geometry and only Car -> -99; %d of %d Car '
          'boxes ignored (kept %.3f)' % (k, n_only, len(idx), n_ign, n_car, 1 - n_ign / max(n_car, 1)))
    check(n_only == len(idx), 'the real KEEP changes only Car labels, to -99, with augmentation on')


# ------------------------------------------------------------------------------------------------- 3. kept share
def share_checks(T, o_cfg, r_cfg, r7_cfg, k, n):
    print('=' * 118 + '\n3. KEPT SHARE of Car supervision, %d strided TRAIN frames, augmentation off' % SHARE_FRAMES)
    idx = list(np.linspace(0, n - 1, SHARE_FRAMES).astype(int))
    batch = idx[::SHARE_FRAMES // BATCH_FRAMES][:BATCH_FRAMES]
    ds_o, base = passes(o_cfg, idx, aug_off=True, keep_items=batch)
    del ds_o                                  # (Waymo's train infos are 2.7 GB: at most two loaders alive at once)
    ds_r, rr = passes(r_cfg, idx, aug_off=True, keep_items=batch)
    _, r7 = passes(r7_cfg, idx, aug_off=True)
    edges = list(r7_cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP.GT_IGNORE_BY_RANGE.EDGES)
    keep7 = np.asarray(r7_cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP.GT_IGNORE_BY_RANGE.KEEP, dtype=np.float64)
    nb = len(edges) - 1
    n_car, k_r, k_7 = np.zeros(nb, int), np.zeros(nb, int), np.zeros(nb, int)
    bad = 0
    for i in idx:
        b, s, t = base[i]['boxes'], rr[i]['boxes'], r7[i]['boxes']
        if not (base[i]['pts'] == rr[i]['pts'] == r7[i]['pts'] and b.shape == s.shape == t.shape
                and np.array_equal(b[:, :-1], s[:, :-1]) and np.array_equal(b[:, :-1], t[:, :-1])):
            bad += 1
            continue
        bl, sl, tl = (x[:, -1].astype(int) for x in (b, s, t))
        if not (np.all((bl == sl) | ((bl == 1) & (sl == IGN))) and np.all((bl == tl) | ((bl == 1) & (tl == IGN)))):
            bad += 1
            continue
        car = bl == 1
        rb = np.clip(np.searchsorted(edges, np.hypot(b[car, 0], b[car, 1]), side='right') - 1, 0, nb - 1)
        n_car += np.bincount(rb, minlength=nb)
        k_r += np.bincount(rb[sl[car] == 1], minlength=nb)
        k_7 += np.bincount(rb[tl[car] == 1], minlength=nb)
    check(bad == 0, 'oracle, R7 and R7r: same points and box geometry on every frame, labels differ only Car -> -99 '
                    '(%d frames off)' % bad)
    tot = n_car.sum()
    share_r, share_7 = k_r.sum() / max(tot, 1), k_7.sum() / max(tot, 1)
    se = np.sqrt(k * (1 - k) / max(tot, 1))
    print('  %d Car boxes reach the head on these frames; kept: R7r %.4f (KEEP %.4f, binomial SE %.4f), R7 %.4f'
          % (tot, share_r, k, se, share_7))
    print('  oracle / R7r / R7 loader time per frame (median, s): %.3f / %.3f / %.3f'
          % tuple(np.median([d[i]['t'] for i in idx]) for d in (base, rr, r7)))
    check(abs(share_r - k) <= 0.01, 'R7r kept share %.4f within +-0.01 of KEEP %.4f' % (share_r, k))
    print('  %-10s %7s %8s %8s %8s %8s' % ('bin (m)', 'n Car', 'R7r', '|dev|/SE', 'R7', 'R7 KEEP'))
    flat_ok, r7_ok = True, True
    for j in range(nb):
        sr = k_r[j] / n_car[j] if n_car[j] else float('nan')
        s7 = k_7[j] / n_car[j] if n_car[j] else float('nan')
        se_j = np.sqrt(k * (1 - k) / n_car[j]) if n_car[j] else float('nan')
        se7 = np.sqrt(keep7[j] * (1 - keep7[j]) / n_car[j]) if n_car[j] else float('nan')
        z = abs(sr - k) / se_j if n_car[j] else float('nan')
        print('  %4g-%-5g %7d %8.4f %8.2f %8.4f %8.4f' % (edges[j], edges[j + 1], n_car[j], sr, z, s7, keep7[j]))
        if n_car[j] >= MIN_BIN:
            flat_ok &= bool(z <= 3)
            r7_ok &= bool(abs(s7 - keep7[j]) <= max(3 * se7, 1e-9))
    trusted = n_car >= MIN_BIN
    spread_r = np.ptp((k_r / np.maximum(n_car, 1))[trusted])
    spread_7 = np.ptp((k_7 / np.maximum(n_car, 1))[trusted])
    print('  spread of the per-bin share over bins with >= %d boxes: R7r %.3f, R7 %.3f' % (MIN_BIN, spread_r, spread_7))
    check(flat_ok, 'R7r flat over range: every bin with >= %d Car boxes within 3 binomial SE of KEEP' % MIN_BIN)
    check(r7_ok, 'R7 (contrast) follows its KEEP per bin: every bin with >= %d boxes within 3 SE of KEEP[b]' % MIN_BIN)
    check(spread_7 > 0.3, 'R7 is not flat (per-bin spread %.3f > 0.3)' % spread_7)
    return ds_r, base, rr, batch


# ------------------------------------------------------------------------------------------------- 4. seeds
def seed_checks(r_cfg, ds_r, n):
    print('=' * 118 + '\n4. ONE IGNORE SET for both training seeds (%d strided frames, the arm\'s gt_classes)' % SEED_FRAMES)
    idx = list(np.linspace(0, n - 1, SEED_FRAMES).astype(int) + 1)
    idx = [min(i, n - 1) for i in idx]
    other = build(r_cfg, r_cfg.DATA_CONFIG, aug_off=False)
    same = n_ign = 0
    for i in idx:
        a = raw_input(ds_r, i, seed=666)['gt_classes']
        b = raw_input(ds_r, i, seed=667)['gt_classes']
        c = raw_input(other, i, seed=667)['gt_classes']
        same += bool(np.array_equal(a, b) and np.array_equal(a, c))
        n_ign += int((a == IGN).sum())
    print('  %d of %d frames: gt_classes identical under seeds 666 / 667 and in a second dataset instance (augmentation '
          'on); %d -99 labels (Car ignores and non-class placeholders)' % (same, len(idx), n_ign))
    check(same == len(idx), 'the IGNORE set does not depend on the global numpy state or the dataset instance')


# ------------------------------------------------------------------------------------------------- 5. targets
def head_for(cfg, ds):
    head = CenterHead.__new__(CenterHead)
    torch.nn.Module.__init__(head)
    head.model_cfg = cfg.MODEL.DENSE_HEAD
    head.class_names = list(cfg.CLASS_NAMES)
    head.class_names_each_head = [[c for c in names if c in cfg.CLASS_NAMES]
                                  for names in cfg.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD]
    head.point_cloud_range = np.asarray(ds.point_cloud_range, dtype=np.float32)
    head.voxel_size = ds.voxel_size
    head.separate_head_cfg = cfg.MODEL.DENSE_HEAD.SEPARATE_HEAD_CFG
    head.build_losses()
    stride = cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
    return head, list((ds.grid_size[:2] // stride)[::-1]), stride


def targets(cfg, ds, dicts):
    head, fmap, stride = head_for(cfg, ds)
    assert len(head.class_names_each_head) == 1, 'one head expected (the oracle recipe)'
    gt = ds.collate_batch(dicts)['gt_boxes']
    t = head.assign_targets(torch.from_numpy(gt).float(), feature_map_size=fmap)
    hm, mask = t['heatmaps'][0], t['heatmap_masks'][0]
    pos = {c: int((gt[:, :, -1] == k + 1).sum()) for k, c in enumerate(cfg.CLASS_NAMES)}
    peaks = {c: int(hm[:, j].eq(1).sum()) for j, c in enumerate(head.class_names_each_head[0])}
    return {'gt': gt, 'hm': hm, 'mask': mask, 'pos': pos, 'peaks': peaks, 'ign': int((gt[:, :, -1] < 0).sum()),
            'ign_px': 0 if mask is None else int((mask == 0).sum()), 'reg': int(t['masks'][0].sum()), 'stride': stride,
            'ds': ds, 'head': head}


def centre_px(t, box):
    pcr, vs = t['ds'].point_cloud_range, t['ds'].voxel_size
    return int((box[1] - pcr[1]) / vs[1] / t['stride']), int((box[0] - pcr[0]) / vs[0] / t['stride'])


def target_checks(o_cfg, ds_r, base, rr, batch):
    """Both batches are collated and assigned with R7r's dataset object (collate_batch, range, voxel and grid are the
    oracle's: the configs differ in TRAIN_FACTOR_SWAP only, checked in 1)."""
    print('=' * 118 + '\n5. TARGETS (rule 4): one collated batch of %d frames %s' % (len(batch), [base[i]['key'] for i in batch]))
    to = targets(o_cfg, ds_r, [base[i]['item'] for i in batch])
    tr = targets(o_cfg, ds_r, [rr[i]['item'] for i in batch])
    for name, t in (('oracle', to), ('R7r', tr)):
        print('  %-6s: positive-label boxes %s, peaks %s, ignore-label boxes %d, ignored px %d, regression targets %d'
              % (name, t['pos'], t['peaks'], t['ign'], t['ign_px'], t['reg']))
        check(all(t['peaks'][c] <= t['pos'][c] for c in t['pos']), '%s: no heatmap peak without a positive box' % name)
    go, gr = to['gt'], tr['gt']
    check(go.shape == gr.shape and np.array_equal(go[..., :-1], gr[..., :-1]), 'batch box geometry identical')
    lo, lr = go[..., -1].astype(int), gr[..., -1].astype(int)
    n_ign = int((lr == IGN).sum())
    check(to['ign'] == 0, 'the oracle batch carries no ignore label (non-class placeholders are filtered)')
    check(bool(np.all((lr != IGN) | (lo == 1))) and n_ign == to['pos']['Car'] - tr['pos']['Car'] and n_ign > 0,
          'every -99 box (%d) was an oracle Car positive (%d -> %d Car positives)' % (n_ign, to['pos']['Car'],
                                                                                    tr['pos']['Car']))
    names = tr['head'].class_names_each_head[0]
    for j, c in enumerate(names):
        if c != 'Car':
            check(torch.equal(to['hm'][:, j], tr['hm'][:, j]), '%s heatmap identical to the oracle\'s' % c)
    check(to['reg'] - n_ign <= tr['reg'] <= to['reg'], 'regression targets %d -> %d: only ignored Cars lose theirs'
          % (to['reg'], tr['reg']))
    hm, mask = tr['hm'], tr['mask']
    check(mask is not None and tr['ign_px'] > 0, 'the ignored footprint is non-empty (%d px)' % tr['ign_px'])
    H, W = hm.shape[-2:]
    off = n_checked = n_bad = 0
    for b in range(gr.shape[0]):
        positives = {centre_px(tr, g) for g in gr[b] if g[-1] > 0}
        for g in gr[b]:
            if g[-1] >= 0:
                continue
            cy, cx = centre_px(tr, g)
            if not (0 <= cy < H and 0 <= cx < W):
                off += 1
                continue
            if (cy, cx) in positives:
                continue
            n_checked += 1
            n_bad += not (bool((hm[b, :, cy, cx] < 1).all()) and bool((mask[b, cy, cx] == 0).all()))
    check(n_checked > 0 and n_bad == 0, '%d -99 centres on the grid: no peak, pixel masked in every class (%d centres '
                                        'beyond the heatmap, counted not indexed)' % (n_checked, off))
    # neither positive nor negative: the masked heatmap loss ignores any prediction on the footprint
    head = tr['head']
    ignored = mask == 0
    pred = torch.full_like(hm, 0.1)
    pred2 = pred.clone()
    car = names.index('Car')
    p2 = pred2[:, car]
    p2[ignored] = 0.95
    pred2[:, car] = p2
    l1, l2 = head.hm_loss_func(pred, hm, mask=mask), head.hm_loss_func(pred2, hm, mask=mask)
    u1, u2 = head.hm_loss_func(pred, hm, mask=None), head.hm_loss_func(pred2, hm, mask=None)
    print('  heatmap loss, Car predicted 0.1 -> 0.95 on the ignored footprint: masked %.6f -> %.6f; unmasked (as if '
          'negatives) %.6f -> %.6f' % (float(l1), float(l2), float(u1), float(u2)))
    check(torch.equal(l1, l2) and not torch.equal(u1, u2), 'the ignored footprint carries zero heatmap-loss weight')


# ------------------------------------------------------------------------------------------------- 6. evaluation
def eval_checks(o_cfg, r_cfg):
    print('=' * 118 + '\n6. EVALUATION-INERT')
    check(r_cfg.DATA_CONFIG_TAR == o_cfg.DATA_CONFIG_TAR, 'DATA_CONFIG_TAR identical to the oracle\'s')
    a, _ = item(build(o_cfg, o_cfg.DATA_CONFIG_TAR, training=False), 0)
    b, _ = item(build(r_cfg, r_cfg.DATA_CONFIG_TAR, training=False), 0)
    check(np.array_equal(a['points'], b['points']) and np.array_equal(a['gt_boxes'], b['gt_boxes']),
          'evaluation (DATA_CONFIG_TAR) frame 0 identical (%s)' % a.get('frame_id'))
    a, _ = item(build(o_cfg, o_cfg.DATA_CONFIG, training=False), 0)
    b, _ = item(build(r_cfg, r_cfg.DATA_CONFIG, training=False), 0)
    same = (np.array_equal(a['points'], b['points']) and np.array_equal(a.get('gt_boxes'), b.get('gt_boxes'))
            and set(a) == set(b) and 'gt_classes' not in b)
    check(same, 'R7r\'s training block built for evaluation: frame 0 identical, no gt_classes (key training-gated)')


# ------------------------------------------------------------------------------------------------- --full-pass
_DS, _EDGES = None, None


def _init_worker():
    torch.set_num_threads(1)


def _count_chunk(idx):
    """(Car boxes per bin, kept per bin, frames with a label outside {1, 2, 3, -99}) over frames idx of _DS."""
    nb = len(_EDGES) - 1
    n_car, kept, bad = np.zeros(nb, np.int64), np.zeros(nb, np.int64), 0
    for i in idx:
        np.random.seed(0)
        b = _DS[int(i)]['gt_boxes']
        lab = b[:, -1].astype(int)
        bad += int(not np.isin(lab, [1, 2, 3, IGN]).all())
        car = (lab == 1) | (lab == IGN)
        rb = np.clip(np.searchsorted(_EDGES, np.hypot(b[car, 0], b[car, 1]), side='right') - 1, 0, nb - 1)
        n_car += np.bincount(rb, minlength=nb)
        kept += np.bincount(rb[lab[car] == 1], minlength=nb)
    return n_car, kept, bad


def full_counts(cfg, edges, workers):
    import multiprocessing as mp
    global _DS, _EDGES
    _DS, _EDGES = build(cfg, cfg.DATA_CONFIG, aug_off=True), list(edges)
    n = len(_DS)
    chunks = np.array_split(np.arange(n), workers * 32)
    t = time.time()
    with mp.get_context('fork').Pool(workers, initializer=_init_worker) as pool:
        res = pool.map(_count_chunk, chunks, chunksize=1)
    _DS = None
    n_car = sum(r[0] for r in res)
    kept = sum(r[1] for r in res)
    bad = sum(r[2] for r in res)
    print('  %s: %d frames in %.0f s' % (cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP and list(cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP)[0],
                                        n, time.time() - t), flush=True)
    return n_car, kept, bad, n


def full_pass(T, r_cfg, r7_cfg, workers):
    print('=' * 118 + '\nFULL PASS: R7r and R7 over every TRAIN frame, augmentation off, %d workers' % workers)
    k = float(r_cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP.GT_IGNORE_RANDOM.KEEP)
    edges = list(r7_cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP.GT_IGNORE_BY_RANGE.EDGES)
    keep7 = np.asarray(r7_cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP.GT_IGNORE_BY_RANGE.KEEP, dtype=np.float64)
    nr, kr, br, fr = full_counts(r_cfg, edges, workers)
    n7, k7, b7, f7 = full_counts(r7_cfg, edges, workers)
    check(fr == f7, 'R7r and R7 train on the same %d frames' % fr)
    check(br == 0 and b7 == 0, 'every prepared label in {1, 2, 3, -99} (%d / %d frames off)' % (br, b7))
    check(np.array_equal(nr, n7), 'R7 and R7r see the same Car box population per bin (%d boxes)' % nr.sum())
    tot = nr.sum()
    sr, s7 = kr.sum() / tot, k7.sum() / tot
    path, key = T['stats']
    k_stats = json.load(open(path))['label_range'][key]
    print('  kept share of Car supervision over the whole training set: R7r %.4f (KEEP %.4f, SE %.4f), R7 %.4f '
          '(its expected share %.4f); supervised Cars R7r %d, R7 %d, oracle %d'
          % (sr, k, np.sqrt(k * (1 - k) / tot), s7, k_stats, kr.sum(), k7.sum(), tot))
    check(abs(sr - k) <= 0.005, 'R7r total kept share %.4f within +-0.005 of KEEP %.4f' % (sr, k))
    print('  %-10s %8s %8s %8s %8s %8s' % ('bin (m)', 'n Car', 'R7r', '|dev|/SE', 'R7', 'R7 KEEP'))
    zs = []
    for j in range(len(edges) - 1):
        z = abs(kr[j] / nr[j] - k) / np.sqrt(k * (1 - k) / nr[j]) if nr[j] else float('nan')
        print('  %4g-%-5g %8d %8.4f %8.2f %8.4f %8.4f' % (edges[j], edges[j + 1], nr[j], kr[j] / max(nr[j], 1), z,
                                                        k7[j] / max(n7[j], 1), keep7[j]))
        if nr[j] >= MIN_BIN:
            zs.append(z)
    chi = float(np.sum(np.square(zs)))
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(chi, len(zs)))
    except ImportError:
        p = float('nan')
    print('  sum of squared z over %d bins (chi-square against KEEP, %d df): %.2f, p = %.3f' % (len(zs), len(zs), chi, p))
    check(all(z <= 3 for z in zs), 'R7r flat over range on the whole training set: every bin within 3 SE of KEEP')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=sorted(TARGETS))
    ap.add_argument('--full-pass', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    T = TARGETS[args.target]
    o_cfg, r7_cfg, r_cfg = load(T['oracle']), load(T['r7']), load(T['r7r'])
    if args.full_pass:
        full_pass(T, r_cfg, r7_cfg, args.workers)
        print('=' * 118)
        if FAILS:
            print('FAILED CHECKS (%d):' % len(FAILS))
            for f in FAILS:
                print('  ' + f)
            raise SystemExit(1)
        print('FULL PASS CHECKS PASSED (%s)' % args.target)
        return
    print('item 42c R7r checks, target %s\n  oracle %s\n  R7     %s\n  R7r    %s\n  R7r TRAIN_FACTOR_SWAP %s'
          % (args.target, T['oracle'], T['r7'], T['r7r'], json.loads(json.dumps(r_cfg.DATA_CONFIG.TRAIN_FACTOR_SWAP))))
    k = config_checks(T, r_cfg)
    n = len(build(o_cfg, o_cfg.DATA_CONFIG))
    print('  %d TRAIN frames in the oracle\'s training loader' % n)
    identity_checks(T, o_cfg, r_cfg, k, n)
    ds_r, base, rr, batch = share_checks(T, o_cfg, r_cfg, r7_cfg, k, n)
    seed_checks(r_cfg, ds_r, n)
    target_checks(o_cfg, ds_r, base, rr, batch)
    del ds_r, base, rr
    eval_checks(o_cfg, r_cfg)
    print('=' * 118)
    if FAILS:
        print('FAILED CHECKS (%d):' % len(FAILS))
        for f in FAILS:
            print('  ' + f)
        raise SystemExit(1)
    print('ALL ITEM-42c CHECKS PASSED (%s)' % args.target)


if __name__ == '__main__':
    main()
