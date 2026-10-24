"""Pre-launch smoke of UADA3D (Discriminator2, conditional) on a REAL DACenterPoint, real batches (experiments_md/20261022_06 §10,
row 3, UADA3D at GRL 1.0 on S->B).

    python analysis/uada3d_grad_smoke.py <cfg> <init ckpt> [n_trials]

Builds the row's model as adaptive_train.py does (build_network on cfg.MODEL, non-strict load of the init checkpoint), takes a
Singapore source batch and an UNSUPERVISED Boston target batch (3 frames each, the per-rank half-batch), stamps `domain` 0 / 1
and `grl_coeff` = OPTIMIZATION.GRL_LAMBDA exactly as train_utils_adaptive.py does, runs DACenterPoint's module list, and
checks per trial:
  1. the conditional discriminator losses on both halves are finite (chance for LeastSquares at sigmoid 0.5: 0.25 per class);
     the discriminator's per-cell accuracy per class (pred > 0.5 <-> target domain);
  2. both gradient norms over the parameters UPSTREAM of the GRL (backbone_3d, backbone_2d and the CenterHead, whose class and
     box maps the conditional discriminator reads): |g_src_det| (source detection loss) and |g_dom| (sum of the two halves'
     discriminator losses, the loop's weight 1.0 each), through the GRL at the configured coefficient;
  3. GRL sign: the same gradient with grl_coeff = -lambda (GradientReversal: grad_in = -lambda x grad_out, so -lambda bypasses
     it) has cosine -1 and norm ratio 1 on backbone and head parameters; the discriminator's own gradient is NOT reversed;
  4. the target batch carries no label (UNSUPERVISED), CenterHead assigns no target for it (domain 1), and DACenterPoint takes
     no detection loss on it;
  5. BN: plain BatchNorm modules (no DSNorm).
Prints `checks: ALL PASS` or the failures. Needs a GPU. Run from ST3D/tools (reads the live checkout).
"""
import copy
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
import _init_path  # noqa: F401,E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

cfg_file, ckpt = sys.argv[1], sys.argv[2]
n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 3
cfg = EasyDict()
cfg_from_yaml_file(cfg_file, cfg)
logger = common_utils.create_logger()
common_utils.set_random_seed(666)
fails = []


def check(ok, msg):
    print(('  PASS ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


lam = float(cfg.OPTIMIZATION.get('GRL_LAMBDA', 0.1))
check(cfg.MODEL.NAME == 'DACenterPoint' and cfg.MODEL.DISCRIMINATOR.NAME == 'Discriminator2'
      and 'CONDITIONAL_ADAPTATION' in cfg.MODEL.DISCRIMINATOR, 'DACenterPoint + Discriminator2 conditional adaptation')
check(lam == 1.0, f'OPTIMIZATION.GRL_LAMBDA = {lam} (the fixed design: 1.0)')
check(bool(cfg.DATA_CONFIG_TAR.get('UNSUPERVISED', False)), 'target UNSUPERVISED')

src_ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=3, dist=False, workers=0,
                                logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY', None))
tar_ds, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, batch_size=3, dist=False,
                                workers=0, logger=logger, training=True, model_ontology=cfg.get('ONTOLOGY', None))
print(f'source {len(src_ds)} frames, target {len(tar_ds)} frames')
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=src_ds)
model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
model.cuda().train()
disc = model.discriminator
check(disc is not None and disc.conditional and not disc.marginal, 'discriminator built, conditional only')
norms = {}
for n, m in model.named_modules():
    if 'Norm' in type(m).__name__:
        norms[type(m).__name__] = norms.get(type(m).__name__, 0) + 1
print('  norm modules', norms)
check(set(norms) <= {'BatchNorm1d', 'BatchNorm2d'}, 'plain BatchNorm (no DSNorm)')

named = dict(model.named_parameters())
up_keys = [k for k in named if k.startswith(('backbone_3d.', 'backbone_2d.', 'dense_head.'))]
disc_keys = [k for k in named if k.startswith('discriminator.')]
up = [named[k] for k in up_keys]
dp = [named[k] for k in disc_keys]
print(f'  upstream of the GRL: {len(up)} tensors ({sum(p.numel() for p in up):,} values); discriminator {len(dp)} tensors')


def run(batch_np, domain, grl):
    b = copy.deepcopy(batch_np)
    b['domain'] = domain
    b['grl_coeff'] = grl
    load_data_to_gpu(b)
    model.dense_head.forward_ret_dict.pop('target_dicts', None)
    for m in model.module_list:
        b = m(b)
    det = model.dense_head.get_loss()[0] if domain == 0 else None
    has_targets = 'target_dicts' in model.dense_head.forward_ret_dict
    dl, tb = model.discriminator.get_loss({})
    preds = [torch.sigmoid(p) if p.min() < 0 or p.max() > 1 else p for p in model.discriminator.forward_ret_dict['cond_preds']]
    acc = [float(((p > 0.5) == bool(domain)).float().mean()) for p in preds]
    return det, dl, tb, acc, has_targets, b


def flat(gs, ps):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1).double() for g, p in zip(gs, ps)])


rng = np.random.RandomState(666)
for trial in range(n_trials):
    si = rng.choice(len(src_ds), 3, replace=False)
    ti = rng.choice(len(tar_ds), 3, replace=False)
    s_np = src_ds.collate_batch([src_ds[int(i)] for i in si])
    t_np = tar_ds.collate_batch([tar_ds[int(i)] for i in ti])
    tgt_gt = t_np.get('gt_boxes', None)
    n_tgt_gt = 0 if tgt_gt is None else int((np.abs(tgt_gt[..., :7]).sum(-1) > 0).sum())
    print(f'trial {trial}: source idx {si.tolist()}, target idx {ti.tolist()}; source GT boxes '
          f'{int((s_np["gt_boxes"][..., 7] > 0).sum())}, target GT boxes in the batch {n_tgt_gt}')
    check(n_tgt_gt == 0, 'target batch carries no label (UNSUPERVISED)')
    out = {}
    for name, g in (('grl', lam), ('bypass', -lam)):
        model.zero_grad(set_to_none=True)
        det_s, dl_s, tb_s, acc_s, ht_s, _ = run(s_np, 0, g)
        _, dl_t, tb_t, acc_t, ht_t, _ = run(t_np, 1, g)
        dom = dl_s + dl_t
        g_det = flat(torch.autograd.grad(det_s, up, retain_graph=True, allow_unused=True), up)
        g_dom = flat(torch.autograd.grad(dom, up, retain_graph=True, allow_unused=True), up)
        g_disc = flat(torch.autograd.grad(dom, dp, retain_graph=True, allow_unused=True), dp)
        g_det_disc = torch.autograd.grad(det_s, dp, allow_unused=True)
        out[name] = dict(det=float(det_s), dl_s=float(dl_s), dl_t=float(dl_t), tb_s=tb_s, tb_t=tb_t, acc_s=acc_s, acc_t=acc_t,
                         ht_s=ht_s, ht_t=ht_t, g_det=g_det, g_dom=g_dom, g_disc=g_disc,
                         det_reaches_disc=any(x is not None and float(x.abs().sum()) > 0 for x in g_det_disc))
    a, b = out['grl'], out['bypass']
    print(f'  source det loss {a["det"]:.3f}; discriminator loss src {a["dl_s"]:.5f} ({", ".join("%.4f" % v for k, v in sorted(a["tb_s"].items()))}), '
          f'tar {a["dl_t"]:.5f} ({", ".join("%.4f" % v for k, v in sorted(a["tb_t"].items()))}); chance 0.25 per class')
    print(f'  discriminator accuracy per class: src {[round(x, 3) for x in a["acc_s"]]}, tar {[round(x, 3) for x in a["acc_t"]]}')
    check(all(math.isfinite(v) for v in (a['det'], a['dl_s'], a['dl_t'])), 'losses finite')
    check(a['ht_s'] and not a['ht_t'], 'CenterHead assigned targets for the source batch and none for the target batch (domain 1)')
    nd, nm = float(a['g_det'].norm()), float(a['g_dom'].norm())
    print(f'  upstream: |g_src_det| {nd:.4e}, |g_dom| (GRL {lam:g}) {nm:.4e}, ratio |g_src_det| / |g_dom| {nd / max(nm, 1e-30):.4g}')
    check(nm > 0 and math.isfinite(nm), 'domain gradient into the upstream parameters non-zero and finite (not the 09-26 zero)')
    cos = float(torch.nn.functional.cosine_similarity(a['g_dom'], b['g_dom'], dim=0))
    ratio = float(a['g_dom'].norm() / b['g_dom'].norm())
    check(cos < -0.999 and abs(ratio - 1) < 1e-2, f'GRL sign: cosine with the bypassed gradient {cos:+.6f}, norm ratio {ratio:.4f}')
    for part in ('backbone_3d.', 'backbone_2d.', 'dense_head.'):
        idx = []
        off = 0
        for k, p in zip(up_keys, up):
            if k.startswith(part):
                idx.append((off, off + p.numel()))
            off += p.numel()
        ga = torch.cat([a['g_dom'][i:j] for i, j in idx])
        gb = torch.cat([b['g_dom'][i:j] for i, j in idx])
        c = float(torch.nn.functional.cosine_similarity(ga, gb, dim=0))
        check(ga.norm() > 0 and c < -0.999, f'  {part[:-1]}: |g_dom| {float(ga.norm()):.3e}, cosine with bypassed {c:+.6f}')
    cd = float(torch.nn.functional.cosine_similarity(a['g_disc'], b['g_disc'], dim=0))
    check(a['g_disc'].norm() > 0 and cd > 0.999, f'discriminator gradient |g| {float(a["g_disc"].norm()):.3e}, NOT reversed (cosine {cd:+.6f})')
    check(not a['det_reaches_disc'], 'the detection loss sends the discriminator nothing')
    torch.cuda.empty_cache()

print('checks:', 'ALL PASS' if not fails else f'{len(fails)} FAIL')
for f in fails:
    print('   ', f)
sys.exit(0 if not fails else 1)
