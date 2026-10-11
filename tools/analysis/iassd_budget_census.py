"""How many DISTINCT points a fixed-budget point detector sees under each evaluation-time manipulation (CPU, no model,
no label): experiments_md 20261011_07 part B. IA-SSD's `sample_points` pads a frame with fewer points than its budget by
DUPLICATING points, so the number of distinct rows in the 16,384 it receives is the information it gets. A manipulation
whose distinct count stays at the budget cannot change what the network sees in COUNT (only in arrangement); one below it
reduces the count.

For each config, the evaluated block is built in evaluation mode (as test.py does) and the first point-processing output
of N strided frames is read: distinct points per frame (median, p10, p90) and the share of frames at the full budget.

    python analysis/iassd_budget_census.py <frames> <cfg> [<cfg> ...]
"""
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(TOOLS.parent))
import os  # noqa: E402
os.chdir(TOOLS)
import _init_path  # noqa: E402,F401
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

n = int(sys.argv[1])
logger = common_utils.create_logger('/dev/null', rank=0)
for path in sys.argv[2:]:
    cfg = cfg_from_yaml_file(path, EasyDict())
    blk = cfg.DATA_CONFIG_TAR if cfg.get('DATA_CONFIG_TAR', None) else cfg.DATA_CONFIG
    budget = [p for p in blk.DATA_PROCESSOR if p.NAME == 'sample_points' and p.NUM_POINTS['test'] != -1]
    ds, _, _ = build_dataloader(dataset_cfg=blk, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0,
                                logger=logger, training=False, model_ontology=cfg.get('ONTOLOGY', None))
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    np.random.seed(0)
    distinct = np.array([len(np.unique(ds[i]['points'], axis=0)) for i in idx])
    full = budget[0].NUM_POINTS['test'] if budget else None
    share = float(np.mean(distinct >= full)) if full else float('nan')
    print(f'{Path(path).stem}: distinct points median {np.median(distinct):.0f} (p10 {np.percentile(distinct, 10):.0f}, '
          f'p90 {np.percentile(distinct, 90):.0f}); at the full budget {full}: {share:.2f} of {n} frames', flush=True)
