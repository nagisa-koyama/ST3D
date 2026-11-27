"""Item 65 (experiments_md 20261003_05 §5 item 65, design fixed 2026-11-27 13:02 JST, before any number): TENT-style
adaptation of P_K's BatchNorm gamma / beta by entropy minimization on unlabelled KITTI testing clouds. ANALYSIS.

    python analysis/item65_tent.py --ckpt <P_K ckpt> --seed 666|667 --out_dir DIR [--epochs 10] [--save 1 10] [--max_steps N]

- only BN gamma / beta train; BN layers in train mode (batch statistics; their running statistics update as usual), every other
  module frozen and in eval mode (TENT, Wang et al. ICLR 2021);
- clouds: 59b's KITTI testing split (7,518 frames, no labels), eval-mode view (no augmentation), batch 6, shuffled by the seed;
- loss: mean binary entropy of the sigmoid heatmap over each frame's 500 highest cells (all classes), so the empty background
  does not dominate; Adam, LR 0.001, no weight decay;
- checkpoints DIR/ckpt/checkpoint_epoch_<e>.pth ({'model_state', 'epoch', 'it'}) at the epochs in --save; each is then rescored
  under global AdaBN statistics (scripts/_item65_tent_20261127.sh).
UDA-legal by construction (unlabelled target clouds; settings a priori). Tests: tests/test_item65_tent.py.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402  (also installs the allocator guard)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

B_BASE = 'cfgs/da-ieee-access-analysis/item59/centerpoint-item59-b-base.yaml'   # MODEL = P_K's; DATA_CONFIG_TAR = KITTI testing
TOPK = 500


def bn_modules(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def tent_params(model):
    """Freeze everything, then let every BN layer's weight / bias train. Returns those parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    out = []
    for m in bn_modules(model):
        for p in (m.weight, m.bias):
            p.requires_grad_(True)
            out.append(p)
    return out


def set_modes(model):
    model.eval()
    for m in bn_modules(model):
        m.train()


def topk_entropy(logits, k=TOPK):
    """Mean over frames of the mean binary entropy of each frame's k highest heatmap logits (B, C, H, W)."""
    z = logits.reshape(logits.shape[0], -1)
    top = z.topk(min(k, z.shape[1]), dim=1).values
    p = torch.sigmoid(top)
    h = p * F.softplus(-top) + (1 - p) * F.softplus(top)          # -p log p - (1 - p) log(1 - p), from logits
    return h.mean()


def testing_loader(cfg, batch_size, workers, seed, logger):
    from pcdet.datasets import __all__ as DATASETS
    block = copy.deepcopy(cfg.DATA_CONFIG_TAR)
    block.DATA_SPLIT['test'] = block.DATA_SPLIT['train']
    block.INFO_PATH['test'] = block.INFO_PATH['train']
    ds = DATASETS[block.DATASET](dataset_cfg=block, class_names=cfg.CLASS_NAMES, root_path=None, training=False,
                                 logger=logger, model_ontology=cfg.get('ONTOLOGY', None))
    g = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, collate_fn=ds.collate_batch,
                                     num_workers=workers, pin_memory=True, drop_last=False,
                                     persistent_workers=workers > 0)
    return ds, dl


# ---------------------------------------------------------------- after training: check, manifest, readings
OUT = Path('/home/koyama/data/item65')
EVAL_CFG = 'cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml'
PK_CKPT = {666: '/home/koyama/data/wandb/run-20261108_020021-0i5ai4hn/files/ckpt/checkpoint_epoch_152.pth',
           667: '/home/koyama/data/wandb/run-20261108_020023-1ya9lozy/files/ckpt/checkpoint_epoch_152.pth'}
EPOCHS = (1, 10)
SEEDS = (666, 667)
REF_57 = '/home/koyama/data/item57/item57b_manifest.json'
REF_58 = '/home/koyama/data/item58/item58_manifest.json'


def ckpt_path(seed, epoch):
    return OUT / ('s%d' % seed) / 'ckpt' / ('checkpoint_epoch_%d.pth' % epoch)


def tag(seed, epoch):
    return 'i65_e%d_s%d' % (epoch, seed)


def row_name(epoch):
    return 'P_TENT_E%d' % epoch


def result_dir(seed, epoch):
    t = tag(seed, epoch)
    return TOOLS.parent / 'output' / 'da-ieee-access-analysis' / 'factorswap-s2' / Path(EVAL_CFG).stem / t / 'eval' / (
        'epoch_%d' % epoch) / t


def cmd_check(seed):
    import item58_bn_bound as m58
    pre = torch.load(PK_CKPT[seed], map_location='cpu')['model_state']
    ok = True
    for e in EPOCHS:
        c = m58.compare(pre, torch.load(ckpt_path(seed, e), map_location='cpu')['model_state'])
        good = not c['non_bn_differ'] and not c['missing']
        ok &= good
        print('P_TENT_E%d s%d: non-BN tensors differing %d, BN tensors changed %d, key mismatch %d -> %s' % (
            e, seed, len(c['non_bn_differ']), c['bn_changed'], len(c['missing']), 'ok' if good else 'FAILED'))
    sys.exit(0 if ok else 1)


def manifest():
    sys.path.insert(0, str(TOOLS / 'analysis'))
    from item42_collect import read_log
    rows = []
    for s in SEEDS:
        for e in EPOCHS:
            d = result_dir(s, e)
            got = read_log((d, 'log_rsadabn_*.txt', None))
            rows.append(dict(group='item65', row=row_name(e), seed=s, job=None, result=str(d / 'result.pkl'),
                             logged_bev=got['bev07'] if got else None,
                             status='ok' if (d / 'result.pkl').exists() and got else 'missing'))
    for path, keep in ((REF_57, ('P_RS1', 'L_ada', 'O_ada')), (REF_58, ('P_BN_G',))):
        for r in json.load(open(path)):
            if r['row'] in keep:
                rows.append(dict(r, group='item65'))
    return rows


def reading(rows, epoch):
    sys.path.insert(0, str(TOOLS / 'analysis'))
    import item58_bn_bound as m58
    seeds = [m58.per_seed(rows, s, row_name(epoch)) for s in SEEDS]
    word = m58.reading(seeds)
    if word == 'it cannot' and all(x['delta'] <= -1.0 for x in seeds):
        word = 'it widens the far residual'
    return seeds, word


def cmd_read(path):
    rows = json.load(open(path))['rows']
    for r in rows:
        print('%-11s s%d pooled %6.2f (3D %6.2f) logged %s | %s%s' % (
            r['row'], r['seed'], r['pooled']['bev'], r['pooled']['3d'], r['logged_bev'],
            ' '.join('%6.2f' % g['bev'] for g in r['rings']), '' if r.get('reproduced') else '  NOT REPRODUCED'))
    for e in EPOCHS:
        seeds, word = reading(rows, e)
        print('\n== %s: %s' % (row_name(e), word))
        for s, d in zip(SEEDS, seeds):
            print('  seed %d: %s' % (s, 'not read' if d is None else 's_x far %.2f / %.2f, delta %+.2f, R %.2f' % (
                d['s_x'][2], d['s_x'][3], d['delta'], d['R'])))


def main():
    a = sys.argv[1:]
    if a[:1] == ['check'] and len(a) == 2:
        return cmd_check(int(a[1]))
    if a[:1] == ['manifest'] and len(a) == 2:
        rows = manifest()
        json.dump(rows, open(a[1], 'w'), indent=1)
        for r in rows:
            print(r['row'], r['seed'], r['status'], r.get('logged_bev'))
        sys.exit(0 if all(r['status'] == 'ok' for r in rows) else 1)
    if a[:1] == ['read'] and len(a) == 2:
        return cmd_read(a[1])
    if a[:1] == ['ckpt'] and len(a) == 3:
        print(ckpt_path(int(a[1]), int(a[2])))
        return
    if a[:1] == ['pk'] and len(a) == 2:
        print(PK_CKPT[int(a[1])])
        return
    return train()


def train():
    import wandb
    from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--save', type=int, nargs='+', default=[1, 10])
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max_steps', type=int, default=0, help='smoke only')
    p.add_argument('--run_name', default=None)
    args = p.parse_args()
    cfg_from_yaml_file(B_BASE, cfg)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out_dir); (out / 'ckpt').mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(out / 'log_tent.txt', rank=0)
    wandb.init(project='st3d', name=args.run_name, notes=common_utils.wandb_notes_with_job_id(),
               config=dict(item65=vars(args)), tags=['item65_tent'])
    for k, v in vars(args).items():
        logger.info(f'{k:12} {v}')
    log_config_to_file(cfg, logger=logger)
    ds, dl = testing_loader(cfg, args.batch_size, args.workers, args.seed, logger)
    logger.info('[tent] %d %s frames (split %s), no labels read' % (len(ds), ds.__class__.__name__, ds.split))
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    params = tent_params(model)
    n_all = sum(t.numel() for t in model.parameters())
    logger.info('[tent] %d parameter tensors train (BN gamma / beta, %d values of %d)' % (
        len(params), sum(t.numel() for t in params), n_all))
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=0.0)
    it, hist = 0, []
    for epoch in range(1, args.epochs + 1):
        set_modes(model)
        for batch in dl:
            load_data_to_gpu(batch)
            model(batch)
            if it == 0:
                logger.info('[tent] heatmap logits %s' % (tuple(batch['batch_cls_preds'].shape),))
            loss = topk_entropy(batch['batch_cls_preds'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            it += 1
            if it % 50 == 0:
                with torch.no_grad():
                    z = batch['batch_cls_preds'].reshape(batch['batch_size'], -1).topk(TOPK, dim=1).values
                    top = float(torch.sigmoid(z).mean())
                hist.append(dict(it=it, epoch=epoch, entropy=float(loss), top500_score=top))
                wandb.log(dict(it=it, entropy=float(loss), top500_score=top))
                logger.info('[tent] epoch %d it %d entropy %.4f top-500 mean score %.4f' % (epoch, it, float(loss), top))
            if args.max_steps and it >= args.max_steps:
                break
        if epoch in args.save or (args.max_steps and it >= args.max_steps):
            path = out / 'ckpt' / ('checkpoint_epoch_%d.pth' % epoch)
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'it': it}, path)
            logger.info('[tent] saved %s (it %d)' % (path, it))
        if args.max_steps and it >= args.max_steps:
            break
    json.dump(hist, open(out / 'tent_history.json', 'w'), indent=1)
    logger.info('[tent] done: %d steps' % it)


if __name__ == '__main__':
    main()
