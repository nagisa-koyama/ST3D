"""Build each config's EVALUATION dataset and exercise it, without a GPU or a training run.

Why this exists: a config that loads is not evidence that its eval path works, and the two take
different branches. A 2026-08-23 assert on cross-dataset eval of single-head models made all six
of Phase 0's C1-C6 evaluations unreachable for a month and cost 16 GPU-hours - two of those jobs
trained for 8 hours and then died with no AP. Constructing the eval dataset is a seconds-long CPU
check that would have caught it before the first submission.

Usage, from ST3D/tools inside the container:
    python3 analysis/preflight_eval.py [glob ...]
Default glob is the da-ieee-access source-only family.

Note the sys.path handling below rather than `import _init_path`: Python puts the *script's own*
directory on sys.path, which is analysis/, not tools/ - so the usual first-line import is not
available here. Getting this wrong silently resolves `pcdet` to the stale editable install baked
into the .sif at /code/ST3D instead of this checkout.
"""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent))

import glob  # noqa: E402
import traceback  # noqa: E402

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402
from pcdet.utils.ontology_mapping import get_ontology_mapping  # noqa: E402

DEFAULT_GLOB = 'cfgs/da-ieee-access/centerpoint-sourceonly-*.yaml'


def preflight(path, logger):
    name = Path(path).stem
    cfg_from_yaml_file(path, cfg)

    # Same selection test.py's get_eval_configs() makes.
    if cfg.get('DATA_CONFIG_TAR', None):
        eval_cfg = cfg.DATA_CONFIG_TAR
    else:
        eval_cfg = cfg.DATA_CONFIG

    # A resolved dataset block is not automatic - _BASE_CONFIG_ chains have silently lost their
    # second hop before. Assert the keys the eval path actually reads.
    for key in ('DATASET', 'DATA_PATH', 'POINT_CLOUD_RANGE', 'DATA_PROCESSOR',
                'POINT_FEATURE_ENCODING'):
        if eval_cfg.get(key, None) is None:
            return 'FAIL %-42s eval config lost %s along its _BASE_CONFIG_ chain' % (name, key)

    test_set, _, _ = build_dataloader(
        dataset_cfg=eval_cfg, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=0, logger=logger, training=False,
        model_ontology=cfg.get('ONTOLOGY', None),
    )
    sample = test_set[0]

    # The mapping that scores a model trained on one dataset's labels against another's GT.
    mapping = get_ontology_mapping(cfg.ONTOLOGY, eval_cfg.ONTOLOGY) if cfg.get('ONTOLOGY', None) else None
    gt_names = sorted(set(sample['gt_names'].tolist())) if 'gt_names' in sample else []

    return ('OK   %-42s eval=%-16s frames=%-6d points=%-7d gt_boxes=%-4d gt_names=%s map=%s'
            % (name, eval_cfg.DATASET, len(test_set), len(sample['points']),
               len(sample.get('gt_boxes', [])), gt_names, mapping))


def main():
    patterns = sys.argv[1:] or [DEFAULT_GLOB]
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        print('no configs matched %s' % patterns)
        return 1

    logger = common_utils.create_logger('/dev/null', rank=0)
    failures = 0
    for path in paths:
        try:
            line = preflight(path, logger)
        except Exception as e:  # noqa: BLE001 - report every config, do not stop at the first
            line = 'FAIL %-42s %s: %s' % (Path(path).stem, type(e).__name__, e)
            traceback.print_exc()
        if line.startswith('FAIL'):
            failures += 1
        print(line, flush=True)

    print('\n%d/%d configs pre-flighted their eval path' % (len(paths) - failures, len(paths)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
