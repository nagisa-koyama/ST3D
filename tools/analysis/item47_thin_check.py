"""Procedure check 1 of experiments_md 20261003_05 §5 item 47 (design fixed 2026-11-21 19:30 JST): before any GPU row runs, show
on real frames that the lattice cut acts on BOTH loaders adabn_eval.py builds from an item 47 eval config, the val loader and the
AdaBN statistics loader (adabn_utils.train_split_stats_loader: TRAIN split, eval-mode view), and that every arm of a target sees
the same clouds.

For each part (K KITTI, P Pandar64, W Waymo) and each distinct eval config of its A rows, on N frames of each loader (the
statistics loader's own strided frames, subsampled; val frames at a fixed stride), the point count per frame after the data
processor (what the network sees) is compared with the unthinned target oracle's config on the SAME frames:
  - thinned < reference on every frame, and mean(thinned) / mean(reference) < 0.5, on both loaders;
  - within a part every config gives the same count per frame (K, P: frame-seeded phases); Waymo has one config.
Exit 1 if any check fails. CPU only, run as a Slurm CPU job (never on the master):

    python3 analysis/item47_thin_check.py --part K --frames 20 --out /home/koyama/data/item47/thin_check_K.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'analysis'))

ORACLE = {'K': 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml',
          'W': 'cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml',
          'P': 'cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml'}  # as tests/test_item42_configs.py ORACLE


def loaders(cfg_file, logger):
    """(statistics dataset, its frame indices, val dataset) exactly as adabn_eval.py --stats target builds them."""
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import __all__ as DATASETS
    from pcdet.utils import adabn_utils
    cfg = cfg_from_yaml_file(cfg_file, EasyDict())
    ontology = cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None)
    block = cfg.DATA_CONFIG_TAR
    sub = adabn_utils.train_split_stats_loader(block, cfg.CLASS_NAMES, ontology, 1000, 1, 0, logger).dataset
    val = DATASETS[block.DATASET](dataset_cfg=block, class_names=cfg.CLASS_NAMES, root_path=None, training=False,
                                  logger=logger, model_ontology=ontology)
    return sub.dataset, list(sub.indices), val


def pick(idx, n):
    return idx[::max(1, len(idx) // n)][:n]


def counts(ds, idx):
    return [int(ds[i]['points'].shape[0]) for i in idx]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--part', choices=['K', 'P', 'W'], required=True)
    p.add_argument('--frames', type=int, default=20)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    import _init_path  # noqa: F401
    import make_item47_configs as m47
    from pcdet.utils import common_utils
    os.chdir(TOOLS)
    logger = common_utils.create_logger(rank=0)
    cfgs = sorted({r[6] for r in m47.ROWS if r[0] == args.part})
    res = {'part': args.part, 'frames': args.frames, 'configs': {}, 'failures': []}
    s_ds, s_idx, v_ds = loaders(ORACLE[args.part], logger)
    s_pick, v_pick = pick(s_idx, args.frames), pick(list(range(len(v_ds))), args.frames)
    ref = {'stats': counts(s_ds, s_pick), 'val': counts(v_ds, v_pick)}
    res['reference'] = dict(cfg=ORACLE[args.part], stats_frames=s_pick, val_frames=v_pick, **ref)
    print('reference %s: stats mean %.0f, val mean %.0f points' % (ORACLE[args.part], sum(ref['stats']) / len(ref['stats']),
                                                                  sum(ref['val']) / len(ref['val'])), flush=True)
    for c in cfgs:
        s_ds, s_idx2, v_ds = loaders(c, logger)
        if s_idx2 != s_idx:
            res['failures'].append('%s: statistics frames differ from the reference' % c)
        got = {'stats': counts(s_ds, s_pick), 'val': counts(v_ds, v_pick)}
        row = {}
        for split in ('stats', 'val'):
            ratio = sum(got[split]) / max(1, sum(ref[split]))
            fewer = all(g < r for g, r in zip(got[split], ref[split]))
            row[split] = dict(counts=got[split], ratio=ratio, fewer_on_every_frame=fewer)
            if not (fewer and ratio < 0.5):
                res['failures'].append('%s %s: ratio %.3f, fewer on every frame %s' % (c, split, ratio, fewer))
            print('%-100s %-5s ratio %.3f fewer-on-every-frame %s' % (c, split, ratio, fewer), flush=True)
        res['configs'][c] = row
    first = res['configs'][cfgs[0]]
    for c in cfgs[1:]:
        for split in ('stats', 'val'):
            if res['configs'][c][split]['counts'] != first[split]['counts']:
                res['failures'].append('%s %s: counts differ from %s' % (c, split, cfgs[0]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1)
    print('failures: %d %s' % (len(res['failures']), res['failures']))
    print('wrote', args.out)
    sys.exit(1 if res['failures'] else 0)


if __name__ == '__main__':
    main()
