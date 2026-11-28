"""GPU smoke for item 57's RS-AdaBN on the real model (experiments_md 20261003_05 §5 item 57), run first by
scripts/_item57_rsadabn_20261127.sh; exit 1 stops the job before the long runs.

1. RS-1 (one knot) through adabn_range equals adabn_utils.reestimate_bn on the same 16 KITTI train frames, layer by layer
   (max |difference| / scale < 1e-3 in running mean and variance);
2. every BatchNorm1d input lines up with a recorded sparse convolution output (no RuntimeError) and RS-5 (5 knots) runs a
   source pass, a target pass and one evaluation batch with finite outputs; the per-knot element shares are printed.

    python3 analysis/rs_adabn_smoke.py --ckpt <ckpt> [--cfg_file <eval cfg> --src_cfg <cfg> --src_block DATA_CONFIG]

The defaults are item 57's forward pair; item 70 passes the real pair's control config with `--src_block DATA_CONFIG`.
"""
import argparse
import copy
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import _init_path  # noqa: F401,E402

import torch  # noqa: E402
from easydict import EasyDict  # noqa: E402

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import adabn_range, adabn_utils, common_utils  # noqa: E402

EVAL_CFG = 'cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml'
SRC_CFG = 'cfgs/da-ieee-access-analysis/item47/centerpoint-item47-kitti-p-on-nuscenes-lattice.yaml'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--cfg_file', default=EVAL_CFG)
    p.add_argument('--src_cfg', default=SRC_CFG)
    p.add_argument('--src_block', choices=['DATA_CONFIG_TAR', 'DATA_CONFIG'], default='DATA_CONFIG_TAR')
    args = p.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    block = cfg.DATA_CONFIG_TAR
    src_block = cfg_from_yaml_file(args.src_cfg, EasyDict())[args.src_block]
    ontology = cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None)
    print('smoke: eval %s, source %s %s (%s)' % (args.cfg_file, args.src_cfg, args.src_block, src_block.DATASET))
    test_set, test_loader, _ = build_dataloader(dataset_cfg=block, class_names=cfg.CLASS_NAMES, batch_size=2, dist=False,
                                                workers=2, logger=logger, training=False, model_ontology=ontology)
    a = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    a.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    a.cuda()
    b = copy.deepcopy(a)
    tgt = adabn_utils.train_split_stats_loader(block, cfg.CLASS_NAMES, ontology, 16, 4, 2, logger)
    adabn_utils.reestimate_bn(a, tgt, logger)
    rs = adabn_range.RangeStratifiedBN(b, list(block.POINT_CLOUD_RANGE))
    rs.estimate(tgt, (0.0,), logger)
    worst = 0.0
    for (n, ma), (_, mb) in zip(adabn_utils.bn_layers(a), adabn_utils.bn_layers(b)):
        for x, y in ((ma.running_mean, mb.running_mean), (ma.running_var, mb.running_var)):
            d = float((x - y).abs().max() / (x.abs().max() + 1e-6))
            worst = max(worst, d)
            if d > 1e-3:
                print('MISMATCH %s %.2e' % (n, d))
    print('check 1: RS-1 vs reestimate_bn over %d layers, worst relative difference %.2e' % (len(rs.layers), worst))
    ok = worst <= 1e-3
    rs.restore()
    src = adabn_utils.train_split_stats_loader(src_block, cfg.CLASS_NAMES, ontology, 16, 4, 2, logger)
    s = rs.collect(src, adabn_range.KNOTS_ITEM57, logger)
    rs.estimate(tgt, adabn_range.KNOTS_ITEM57, logger)
    rs.apply()
    with torch.no_grad():
        batch = next(iter(test_loader))
        load_data_to_gpu(batch)
        pred, _ = b(batch)
    finite = all(bool(torch.isfinite(d['pred_boxes']).all() and torch.isfinite(d['pred_scores']).all()) for d in pred)
    print('check 2: RS-5 forward finite %s, %s boxes in the first frame' % (finite, len(pred[0]['pred_boxes'])))
    for n in (rs.layers[0][0], rs.layers[len(rs.layers) // 2][0], rs.layers[-1][0]):
        print('  source share per knot %-40s %s' % (n, [round(float(x), 3) for x in s[n]['share']]))
    ok = ok and finite
    print('SMOKE %s' % ('PASSED' if ok else 'FAILED'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
