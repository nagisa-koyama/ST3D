"""alpha_M for the matched-weight DANN row (experiments_md 20261022_06 §10; design T05 §5, fixed 2026-10-24 10:13 JST).

    python analysis/dann_grad_ratio.py <cfg (30306's, alpha = 1)> <init ckpt> <ps_label_e0.pkl of 30306> <n_iters> <out.json>

The rule, as fixed: on the init checkpoint, 200 training iterations of 30306's setup with alpha = 1; per iteration
r = ||g_src_det|| / ||g_dom|| over the parameters UPSTREAM of the discriminator input (every parameter that feeds
spatial_features_2d: backbone_3d and backbone_2d here; MeanVFE and HeightCompression have none); alpha_M = median(r), two
significant figures, fixed for the whole run.

- g_src_det: gradient of the source detection loss (SRC.LOSS_WEIGHT x loss on the Singapore batch).
- g_dom: gradient of the domain loss as train_st_utils sums it (0.5 dann_src + 0.5 dann_tar), through the GRL at alpha = 1.
- 30306 ran 2 GPUs x 3 frames under DDP, whose reduced gradient is the MEAN of the two ranks' gradients. Emulated on one GPU,
  exactly: each iteration draws two per-rank batches of 3 (source + pseudo-labelled target each), computes each rank's
  gradients separately (BN statistics per 3-frame forward, as per rank), averages them, and r is taken on the averages.
  Per-rank r is recorded too (disclosure).
- The iterations TRAIN, as 30306's first 200 did: the summed loss (src + tar + 0.5 dann_src + 0.5 dann_tar, averaged over the
  two ranks) is back-propagated, clipped at OPTIMIZATION.GRAD_NORM_CLIP and stepped with 30306's optimizer and OneCycle plan
  (2,073 x 20 = 41,460 steps, lr_scheduler.step(it) as the loop does). The discriminator starts from random init as in 30306.
- The target batches carry 30306's own first-pass pseudo-labels (ps_label_e0.pkl, frozen teacher, one pass).
Reads no target label. Needs a GPU. Run from ST3D/tools (reads the live checkout).
"""
import json
import math
import pickle
import sys

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, '.')
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, model_fn_decorator  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402
from train_utils.optimization import build_optimizer, build_scheduler  # noqa: E402

cfg_file, ckpt, ps_path, n_iters, out_json = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
cfg = EasyDict()
cfg_from_yaml_file(cfg_file, cfg)
lc = cfg.MODEL.DENSE_HEAD.LOSS_CONFIG
assert 'dann_weight' in lc.LOSS_WEIGHTS, 'needs the DANN row config'
assert float(lc.get('DANN_GRL_COEFF', 1.0)) == 1.0, 'the rule measures at alpha = 1 (30306\'s config)'
logger = common_utils.create_logger()
common_utils.set_random_seed(666)

data_cfg = cfg.DATA_CONFIG if cfg.get('DATA_CONFIG', None) else list(cfg.DATA_CONFIGS.values())[0]
src_names = cfg.SELF_TRAIN.MODEL_TEACHER.get('CLASS_NAMES', cfg.CLASS_NAMES)
bs = int(cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU)
src_ds, src_loader, _ = build_dataloader(dataset_cfg=data_cfg, class_names=src_names, batch_size=bs, dist=False,
                                         workers=4, logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY', None))
tar_ds, tar_loader, _ = build_dataloader(cfg.DATA_CONFIG_TAR, cfg.CLASS_NAMES, bs, False, workers=4, logger=logger,
                                         training=True, model_ontology=cfg.get('ONTOLOGY', None))
ps = pickle.load(open(ps_path, 'rb'))
tar_ds.set_pseudo_labels(ps)
print(f'source {len(src_ds)} frames, target {len(tar_ds)} frames, pseudo-labelled {len(ps)}; {bs} frames per rank-batch')

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=tar_ds)
model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
model.cuda().train()
assert model.dense_head.domain_discriminator is not None and model.dense_head.dann_grl_coeff == 1.0
optimizer = build_optimizer(model, cfg.OPTIMIZATION)
steps_per_epoch = 2073   # 30306: epoch length follows the source, 2,073 x 20 = 41,460 steps
lr_scheduler, _ = build_scheduler(optimizer, total_iters_each_epoch=steps_per_epoch,
                                  total_epochs=cfg.OPTIMIZATION.NUM_EPOCHS, last_epoch=-1, optim_cfg=cfg.OPTIMIZATION)
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
up = [(n, p) for n, p in named if n.startswith('backbone_3d.') or n.startswith('backbone_2d.')]
others_upstream = [n for n, _ in named if n.startswith('vfe.') or n.startswith('map_to_bev')]
assert not others_upstream, others_upstream
up_params = [p for _, p in up]
print(f'upstream parameters: {len(up)} tensors, {sum(p.numel() for p in up_params):,} values '
      f'(backbone_3d {sum(p.numel() for n, p in up if n.startswith("backbone_3d.")):,}, '
      f'backbone_2d {sum(p.numel() for n, p in up if n.startswith("backbone_2d.")):,})')
model_func = model_fn_decorator()
w_src = float(cfg.SELF_TRAIN.SRC.get('LOSS_WEIGHT', 1.0))
w_tar = float(cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0))


def flat(gs):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g, p in zip(gs, up_params)])


src_it, tar_it = iter(src_loader), iter(tar_loader)


def nxt(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


rec = []
for it in range(n_iters):
    lr_scheduler.step(it)
    optimizer.zero_grad(set_to_none=True)
    g_det_sum = g_dom_sum = None
    per_rank = []
    acc = []
    dl = []
    for rank in range(2):
        s, src_it = nxt(src_it, src_loader)
        t, tar_it = nxt(tar_it, tar_loader)
        s['domain_label'] = 0
        t['domain_label'] = 1
        ls, tbs, _, ds_ = model_func(model, s)
        lt, tbt, _, dt_ = model_func(model, t)
        ls, lt = w_src * ls, w_tar * lt
        dom = 0.5 * ds_ + 0.5 * dt_
        g_det = flat(torch.autograd.grad(ls, up_params, retain_graph=True, allow_unused=True))
        g_dom = flat(torch.autograd.grad(dom, up_params, retain_graph=True, allow_unused=True))
        per_rank.append(float(g_det.norm() / g_dom.norm()))
        g_det_sum = g_det / 2 if g_det_sum is None else g_det_sum + g_det / 2
        g_dom_sum = g_dom / 2 if g_dom_sum is None else g_dom_sum + g_dom / 2
        ((ls + lt + dom) / 2).backward()   # DDP's reduced gradient = mean over the two ranks
        acc.append((float(tbs['domain_preds_accuracy']), float(tbt['domain_preds_accuracy'])))
        dl.append(float(dom))
        del ls, lt, ds_, dt_, dom, g_det, g_dom
    n_det, n_dom = float(g_det_sum.norm()), float(g_dom_sum.norm())
    r = n_det / n_dom
    total_norm = float(clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP))
    optimizer.step()
    rec.append(dict(it=it, r=r, g_src_det=n_det, g_dom=n_dom, r_rank=per_rank, dann=float(np.mean(dl)),
                    acc_src=float(np.mean([a[0] for a in acc])), acc_tar=float(np.mean([a[1] for a in acc])),
                    total_norm_before_clip=total_norm, lr=float(optimizer.param_groups[0]['lr'])))
    if it % 10 == 0 or it == n_iters - 1:
        print(f'it {it:3d}: r {r:.4g} (|g_src_det| {n_det:.4g}, |g_dom| {n_dom:.4g}; per rank {per_rank[0]:.4g} / '
              f'{per_rank[1]:.4g}); dann {np.mean(dl):.5f}; acc src {rec[-1]["acc_src"]:.3f} tar {rec[-1]["acc_tar"]:.3f}; '
              f'grad norm {total_norm:.3f}; lr {rec[-1]["lr"]:.2e}', flush=True)
    del g_det_sum, g_dom_sum
    torch.cuda.empty_cache()

r = np.array([x['r'] for x in rec])
q = np.quantile(r, [0, 0.25, 0.5, 0.75, 1.0])
med = float(np.median(r))
alpha_m = float('%.2g' % med)
rr = np.array([x['r_rank'] for x in rec]).reshape(-1)
summary = dict(n_iters=n_iters, r_min=q[0], r_q1=q[1], r_median=med, r_q3=q[3], r_max=q[4], alpha_M=alpha_m,
               r_per_rank_median=float(np.median(rr)), r_first10_median=float(np.median(r[:10])),
               r_last50_median=float(np.median(r[-50:])), cfg=cfg_file, ckpt=ckpt, ps_label=ps_path)
print('r over %d iterations: min %.4g, q1 %.4g, median %.4g, q3 %.4g, max %.4g' % (n_iters, q[0], q[1], med, q[3], q[4]))
print('per-rank r median %.4g; first 10 iterations median %.4g; last 50 median %.4g'
      % (summary['r_per_rank_median'], summary['r_first10_median'], summary['r_last50_median']))
print('alpha_M = %s (median, two significant figures)' % ('%.2g' % med))
json.dump(dict(summary=summary, per_iteration=rec), open(out_json, 'w'), indent=1)
print('wrote', out_json)
