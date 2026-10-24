"""Pre-launch smoke of the in-head DANN on a REAL CenterPoint, real batches (experiments_md/20261022_06 §9, PL S->B + DANN).

    python analysis/dann_grad_smoke.py <cfg> <student init ckpt> <ps_label cut pkl> [n_trials]

Builds the row's student exactly as train.py does (build_network on cfg.MODEL, non-strict load of the init checkpoint, plain
model, no DSNorm unless the config asks for it), and the teacher from SELF_TRAIN.MODEL_TEACHER. Takes one source batch and
one pseudo-labelled target batch at BATCH_SIZE_PER_GPU, stamps domain_label 0 / 1 as train_st_utils does, runs both forwards
through model_fn_decorator, and checks, per trial:

  1. the domain loss exists on both halves and is finite; the discriminator accuracy per half;
  2. the gradient of the weighted domain loss (0.5 src + 0.5 tar, as train_st_utils sums it) reaches a backbone_3d and a
     backbone_2d parameter NON-ZERO, and is the NEGATIVE of the gradient the same loss gives with the GRL bypassed (cosine
     near -1, norm ratio near 1); the discriminator's own gradient is NOT reversed (cosine near +1). This is the check the
     2026-09-26 UADA3D conditional-GRL defect (lambda 0 -> zero gradient, ST3D 1247338) would have failed;
  3. the domain loss sends nothing to parameters downstream of the tap (dense_head.shared_conv), and the detection loss
     sends nothing to the discriminator;
  4. the target batch's pseudo-labels (gt_boxes, as filled from the pkl) are identical before and after both forwards, and
     the dataset's pseudo-label dict is unchanged for those frames;
  5. BN: every normalisation module of the student and the teacher, by type (plain BatchNorm1d / 2d expected; no DSNorm).

Prints `checks: ALL PASS` or the failures. Needs a GPU (spconv); reads the live checkout (run it from ST3D/tools).
"""
import copy
import math
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
import _init_path  # noqa: F401,E402  (installs the allocator guard and points pcdet at this checkout)
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, model_fn_decorator  # noqa: E402
from pcdet.models.model_utils import dann_utils  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

cfg_file, ckpt, ps_path = sys.argv[1], sys.argv[2], sys.argv[3]
n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else 3
cfg = EasyDict()
cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger()
common_utils.set_random_seed(666)
fails = []


def check(ok, msg):
    print(('  PASS ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


# ------------------------------------------------------------------ datasets (as train.py builds them)
data_cfg = cfg.DATA_CONFIG if cfg.get('DATA_CONFIG', None) else list(cfg.DATA_CONFIGS.values())[0]
src_names = cfg.SELF_TRAIN.MODEL_TEACHER.get('CLASS_NAMES', cfg.CLASS_NAMES)
bs = int(cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU)
src_ds, _, _ = build_dataloader(dataset_cfg=data_cfg, class_names=src_names, batch_size=bs, dist=False, workers=0,
                                logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY', None))
tar_ds, _, _ = build_dataloader(cfg.DATA_CONFIG_TAR, cfg.CLASS_NAMES, bs, False, workers=0, logger=logger,
                                training=True, model_ontology=cfg.get('ONTOLOGY', None))
ps = pickle.load(open(ps_path, 'rb'))
tar_ds.set_pseudo_labels(ps)
print(f'source {len(src_ds)} frames, target {len(tar_ds)} frames, pseudo-labelled frames {len(ps)}, batch {bs} per half')

# ------------------------------------------------------------------ models
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=tar_ds)   # train.py: target_set under SELF_TRAIN
model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
teacher = build_network(model_cfg=cfg.SELF_TRAIN.MODEL_TEACHER, num_class=len(src_names), dataset=src_ds)
model.cuda().train()
disc = model.dense_head.domain_discriminator
check(disc is not None, 'student CenterHead built a domain discriminator (dann_weight %s)'
      % cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.get('dann_weight'))
check(teacher.dense_head.domain_discriminator is None, 'teacher has no discriminator (frozen, never sees a domain label)')
alpha = float(model.dense_head.dann_grl_coeff)
check(alpha == dann_utils.grl_coefficient(cfg.MODEL.DENSE_HEAD.LOSS_CONFIG),
      f'GRL coefficient {alpha} (LOSS_CONFIG.DANN_GRL_COEFF {cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.get("DANN_GRL_COEFF", "absent -> 1.0")}), constant, no ramp')


def norm_types(m):
    out = {}
    for _, mod in m.named_modules():
        n = type(mod).__name__
        if 'Norm' in n:
            out[n] = out.get(n, 0) + 1
    return out


st, tt = norm_types(model), norm_types(teacher)
print(f'  student norm modules {st}; teacher {tt}; SELF_TRAIN.DSNORM {cfg.SELF_TRAIN.get("DSNORM", None)}, '
      f'TEACHER_ADABN {cfg.SELF_TRAIN.get("TEACHER_ADABN", None)}')
check(set(st) <= {'BatchNorm1d', 'BatchNorm2d'} and set(tt) <= {'BatchNorm1d', 'BatchNorm2d'}
      and not cfg.SELF_TRAIN.get('DSNORM', None), 'plain shared BN (no DSNorm), as in 30243')

named = dict(model.named_parameters())
bb3 = [k for k in named if k.startswith('backbone_3d.')]
bb2 = [k for k in named if k.startswith('backbone_2d.') and k.endswith('weight') and named[k].dim() == 4]
probe = {'backbone_3d first': bb3[0], 'backbone_3d last': bb3[-1], 'backbone_2d first conv': bb2[0],
         'backbone_2d last conv': bb2[-1], 'discriminator conv1': 'dense_head.domain_discriminator.conv1.weight',
         'discriminator classifier': 'dense_head.domain_discriminator.classifier.weight',
         'shared_conv (downstream of the tap)': [k for k in named if k.startswith('dense_head.shared_conv')][0]}
print('  probes:', probe)
model_func = model_fn_decorator()
rng = np.random.RandomState(666)
_real_domain_predictions = dann_utils.domain_predictions


def forwards(src_np, tar_np, grl):
    dann_utils.domain_predictions = _real_domain_predictions if grl else (lambda d, f, *a, **k: d(f))
    try:
        s, t = copy.deepcopy(src_np), copy.deepcopy(tar_np)
        s['domain_label'] = 0
        t['domain_label'] = 1
        ls, tbs, _, ds_ = model_func(model, s)
        lt, tbt, _, dt_ = model_func(model, t)
    finally:
        dann_utils.domain_predictions = _real_domain_predictions
    return ls, lt, ds_, dt_, tbs, tbt, t


def grads(loss, keys):
    g = torch.autograd.grad(loss, [named[probe[k]] for k in keys], retain_graph=True, allow_unused=True)
    return dict(zip(keys, g))


for trial in range(n_trials):
    si = rng.choice(len(src_ds), bs, replace=False)
    ti = rng.choice(len(tar_ds), bs, replace=False)
    src_np = src_ds.collate_batch([src_ds[int(i)] for i in si])
    tar_np = tar_ds.collate_batch([tar_ds[int(i)] for i in ti])
    frames = [str(f) for f in tar_np['frame_id']]
    ps_before = {f: copy.deepcopy(ps[f]) for f in frames if f in ps}
    gt_before = tar_np['gt_boxes'].copy()
    lab = gt_before[:, :, 7].astype(int)
    print(f'trial {trial}: source idx {si.tolist()}, target frames {frames}')
    print(f'  target batch labels: positives {[int((lab == c + 1).sum()) for c in range(len(cfg.CLASS_NAMES))]}, '
          f'ignore band {[int((lab == -(c + 1)).sum()) for c in range(len(cfg.CLASS_NAMES))]}, padding {int((lab == 0).sum())}')

    model.zero_grad(set_to_none=True)
    ls, lt, ds_, dt_, tbs, tbt, t_after = forwards(src_np, tar_np, grl=True)
    check(ds_ is not None and dt_ is not None, f'domain loss returned on both halves (src {None if ds_ is None else float(ds_):.5f}, '
          f'tar {None if dt_ is None else float(dt_):.5f})')
    if ds_ is None or dt_ is None:
        continue
    dann = 0.5 * ds_ + 0.5 * dt_   # train_st_utils: 0.5 per half, summed
    acc_s, acc_t = float(tbs['domain_preds_accuracy']), float(tbt['domain_preds_accuracy'])
    print(f'  dann_loss (0.5 src + 0.5 tar) {float(dann):.5f}; at chance it would be ln2 / {bs} x '
          f'{cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.dann_weight} = '
          f'{math.log(2) / bs * cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.dann_weight:.5f}; '
          f'accuracy src {acc_s:.3f}, tar {acc_t:.3f}; det loss src {float(ls):.3f}, tar {float(lt):.3f}')
    check(math.isfinite(float(dann)) and float(dann) > 0, 'domain loss finite and positive')
    keys = list(probe)
    g_rev = grads(dann, keys)
    g_det = grads(ls + lt, keys)
    model.zero_grad(set_to_none=True)
    ls2, lt2, ds2, dt2, _, _, _ = forwards(src_np, tar_np, grl=False)
    g_plain = grads(0.5 * ds2 + 0.5 * dt2, keys)
    check(abs(float(ds2) - float(ds_)) < 1e-4 * max(1.0, abs(float(ds_))) and abs(float(dt2) - float(dt_)) < 1e-4 * max(1.0, abs(float(dt_))),
          f'the GRL-bypassed forward reproduces the domain loss ({float(ds2):.6f} / {float(dt2):.6f})')
    for k in keys:
        a, b, d = g_rev[k], g_plain[k], g_det[k]
        if k.startswith('shared_conv'):
            check(a is None or float(a.abs().sum()) == 0.0, f'{k}: no domain-loss gradient (tap is upstream of it)')
            continue
        if k.startswith('discriminator'):
            check(d is None or float(d.abs().sum()) == 0.0, f'{k}: the detection loss sends it nothing')
            cos = torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()
            check(float(a.abs().sum()) > 0 and cos > 0.999, f'{k}: domain-loss gradient |g| {a.norm():.3e}, NOT reversed '
                  f'(cosine with GRL-bypassed {cos:+.6f})')
            continue
        cos = torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()
        ratio = (a.norm() / b.norm()).item() if b.norm() > 0 else float('nan')
        rel = (a.norm() / d.norm()).item() if d is not None and d.norm() > 0 else float('nan')
        check(float(a.abs().sum()) > 0 and cos < -0.999 and abs(ratio / alpha - 1) < 1e-2,
              f'{k} ({probe[k]}): domain-loss gradient |g| {a.norm():.3e} non-zero, REVERSED (cosine with GRL-bypassed '
              f'{cos:+.6f}, norm ratio {ratio:.4f} = alpha x {ratio / alpha:.4f}); |g_dann| / |g_det| {rel:.3e}')
    # both gradient norms over EVERY parameter upstream of the discriminator input (backbone_3d + backbone_2d)
    up = [p for n, p in model.named_parameters() if n.startswith('backbone_3d.') or n.startswith('backbone_2d.')]
    model.zero_grad(set_to_none=True)
    ls3, lt3, ds3, dt3, _, _, _ = forwards(src_np, tar_np, grl=True)
    gd = torch.autograd.grad(ls3, up, retain_graph=True, allow_unused=True)
    gm = torch.autograd.grad(0.5 * ds3 + 0.5 * dt3, up, allow_unused=True)
    nd = math.sqrt(sum(float(g.double().pow(2).sum()) for g in gd if g is not None))
    nm = math.sqrt(sum(float(g.double().pow(2).sum()) for g in gm if g is not None))
    print(f'  upstream ({len(up)} tensors): |g_src_det| {nd:.4e}, |g_dom| (through the GRL at alpha {alpha:g}) {nm:.4e}, '
          f'ratio |g_src_det| / |g_dom| {nd / nm:.4g}')
    check(nm > 0 and math.isfinite(nm), 'domain gradient on the upstream parameters non-zero and finite')
    # pseudo-labels untouched
    gt_after = t_after['gt_boxes'].detach().cpu().numpy() if torch.is_tensor(t_after['gt_boxes']) else t_after['gt_boxes']
    check(np.array_equal(gt_before.astype(np.float32), gt_after.astype(np.float32)),
          'target batch gt_boxes (pseudo-labels) identical before and after both forwards')
    same = all(np.array_equal(np.asarray(ps_before[f]['gt_boxes']), np.asarray(ps[f]['gt_boxes'])) for f in ps_before)
    check(len(ps_before) == len(frames) and same, f'dataset pseudo-label dict unchanged for the {len(ps_before)} batch frames')
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

print('checks:', 'ALL PASS' if not fails else f'{len(fails)} FAIL')
for f in fails:
    print('   ', f)
sys.exit(0 if not fails else 1)
