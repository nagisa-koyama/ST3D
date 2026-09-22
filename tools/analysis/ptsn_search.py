"""PTSN scale search: find the input scale that puts pseudo-label sizes on the target's.

DALI (IEEE T-RO 2024) observes that a source-trained detector predicts target boxes whose mean
size is inherited from the source, and fixes it at inference: scale the input cloud by `s`,
divide the predicted boxes by `s`, and the resulting mean size moves as roughly (source mean)/s.
Sweep `s`, keep the one whose unscaled mean size is closest to an estimated target mean size.

The upstream release does NOT contain this search - only its result, as the config constant
CAD3D_CONFIG.SCALE: [1.20, 1.10, 1.10]. This script is ours. See experiments_md/20260922_04
section 2.2 for what else is missing from that release, and section 2.4 for the tier plan this
belongs to.

It is INFERENCE ONLY - one forward pass over `--frames` target frames per candidate scale, no
training. That is what makes it the cheapest new number in the DALI plan, and it is meant to be
run before any of Tier D2 is written: if the sweep moves the mean size onto the estimate but
target AP does not follow, the distribution-level story does not hold here and D2 is not
justified.

UDA legality is decided by the flag you pass, not by the method:

  --target_size L W H   an absolute estimate. If it came from SN it IS the target statistic, and
                        the resulting row is NOT target-free. Say so in the table caption.
  --ros_factor F        multiply the SOURCE mean size (measured here, from source labels) by F.
                        ROS only ever perturbs the source, so this keeps the row target-free and
                        comparable with our own rows. experiments_md/20260922_03 derives the
                        interval; its matched Car centre is 0.84-0.88 across every source.

Usage, from ST3D/tools inside the container (needs a GPU):

    singularity exec --nv --bind /home/koyama/data/:/storage <image>.sif \
        python3 analysis/ptsn_search.py \
            --cfg_file cfgs/da-ieee-access/centerpoint-sourceonly-lyft.yaml \
            --ckpt /storage/wandb/run-.../files/ckpt/checkpoint_epoch_30.pth \
            --ros_factor 0.86 --frames 500 --yaml

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

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils, ptsn_utils  # noqa: E402

DEFAULT_SCALES = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
MIN_BOXES = 50


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='the config whose DATA_CONFIG_TAR is the adaptation target')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='source-trained checkpoint to search with')
    parser.add_argument('--scales', type=float, nargs='+', default=DEFAULT_SCALES,
                        help='candidate input scales to sweep')
    parser.add_argument('--target_size', type=float, nargs=3, default=None,
                        metavar=('L', 'W', 'H'),
                        help='absolute E_est[Size] in metres; NOT target-free if it came from SN')
    parser.add_argument('--ros_factor', type=float, default=None,
                        help='target-free alternative: E_est[Size] = source mean size * this')
    parser.add_argument('--class_name', type=str, default='car',
                        help='substring selecting the class the scale is chosen on (case-insensitive)')
    parser.add_argument('--frames', type=int, default=500,
                        help='target frames per candidate scale')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--score_thresh', type=float, default=None,
                        help='override the per-class SELF_TRAIN.SCORE_THRESH used to select boxes')
    parser.add_argument('--yaml', action='store_true', help='emit a pasteable config block')
    parser.add_argument('--dry_run', action='store_true',
                        help='preflight only: resolve everything and load the checkpoint, but run '
                             'no forward pass. Needs no GPU. Run this before spending the real hour')
    parser.add_argument('--probe_frames', type=int, default=5,
                        help='--dry_run: frames used to measure requested-vs-achieved scale')
    args = parser.parse_args()
    if (args.target_size is None) == (args.ros_factor is None):
        parser.error('give exactly one of --target_size or --ros_factor; which one you pick is '
                     'what decides whether the resulting row is target-free (see the docstring)')
    return args


def target_config(config):
    """The adaptation target, the same selection test.py's get_eval_configs() makes."""
    if config.get('DATA_CONFIG_TAR', None):
        return config.DATA_CONFIG_TAR
    sources = source_configs(config)
    return sources[0]


def source_configs(config):
    """Every labelled source, the same selection train.py makes.

    Two shapes exist in this repo and only one of them is `DATA_CONFIG`: the da-MIRU2025 family
    is multi-source and carries `DATA_CONFIGS: {<NAME>: {...}}` instead. Reading only
    `DATA_CONFIG` raises AttributeError on exactly the configs a DALI row would be built from.
    """
    if config.get('DATA_CONFIG', None):
        return [config.DATA_CONFIG]
    if config.get('DATA_CONFIGS', None):
        return list(config.DATA_CONFIGS.values())
    raise AssertionError('config defines neither DATA_CONFIG nor DATA_CONFIGS')


def class_indices(class_names, needle):
    """1-based label ids whose class name contains `needle`. Names may be head-prefixed."""
    needle = needle.lower()
    return [i + 1 for i, n in enumerate(class_names) if needle in n.lower()]


def score_thresholds(config, override, num_classes):
    """Per-class acceptance thresholds, matching what pseudo-labelling will actually keep.

    The mean size PTSN matches is over the boxes that become pseudo-labels, so the selection
    here has to be the same one save_pseudo_label_batch makes - otherwise the scale is tuned on
    a population the student never sees.
    """
    if override is not None:
        return np.full(num_classes, float(override))
    st = config.get('SELF_TRAIN', None)
    if st is not None and st.get('SCORE_THRESH', None) is not None:
        thresh = np.array(st.SCORE_THRESH, dtype=np.float64).reshape(-1)
        if len(thresh) == num_classes:
            return thresh
        return np.full(num_classes, float(thresh[0]))
    return np.full(num_classes, 0.1)


def mean_predicted_size(model, loader, scale, wanted_labels, thresholds, frames,
                        to_device=load_data_to_gpu):
    """Mean unscaled [dx, dy, dz] of the boxes that would be kept as pseudo-labels, at `scale`.

    The loader's dataset scales its points in the worker; `unscale_boxes` puts the predictions
    back into real target metres, so every row of the sweep is directly comparable.

    `to_device` is injectable only so that tests can drive this loop with a stub model on a
    machine with no GPU. This is the function that spends the search's GPU hour, and its
    plumbing - per-class thresholds, class selection, the unscaling, the frame cap - is exactly
    the kind of thing that is cheap to get wrong and expensive to discover in a job log.
    """
    dims, seen = [], 0
    with torch.no_grad():
        for batch in loader:
            to_device(batch)
            pred_dicts, _ = model(batch)
            for pred in pred_dicts:
                if 'pred_boxes' not in pred:
                    continue
                boxes = ptsn_utils.unscale_boxes(pred['pred_boxes'].detach().cpu().numpy(), scale)
                labels = pred['pred_labels'].detach().cpu().numpy()
                scores = pred['pred_scores'].detach().cpu().numpy()
                if len(labels) == 0:
                    continue
                keep = (scores >= thresholds[labels - 1]) & np.isin(labels, wanted_labels)
                if keep.any():
                    dims.append(boxes[keep, 3:6])
            seen += len(pred_dicts)
            if seen >= frames:
                break
    if not dims:
        return np.full(3, np.nan), 0, seen
    dims = np.concatenate(dims, axis=0)
    return dims.mean(axis=0), len(dims), seen


def achieved_scale(dataset_cfg, config, scale, logger, num_frames, batch_size, ontology):
    """What a requested scale actually costs, measured on real frames without a model.

    POINT_CLOUD_RANGE is fixed while the scene grows, so at s > 1 the far field is cropped
    harder and the effective scaling is under the requested one. That is correct behaviour - it
    is the grid the network is trained under - but it means the sweep's `s` is a request, not an
    achievement, and a target with more far-field returns loses more of it. Measuring this costs
    seconds of CPU and tells you before the GPU hour whether the sweep's endpoints are real.

    Returns (achieved, kept_fraction), both medians over `num_frames` frames.
    """
    base, _, _ = build_dataloader(
        dataset_cfg=dataset_cfg, class_names=config.CLASS_NAMES, batch_size=batch_size,
        dist=False, workers=0, logger=logger, training=False, model_ontology=ontology)
    scaled, _, _ = build_dataloader(
        dataset_cfg=dataset_cfg, class_names=config.CLASS_NAMES, batch_size=batch_size,
        dist=False, workers=0, logger=logger, training=False, model_ontology=ontology)
    scaled.set_ptsn_scale(scale)

    ratios, kept = [], []
    for idx in range(min(num_frames, len(base))):
        a, b = base[idx]['points'], scaled[idx]['points']
        if len(a) == 0 or len(b) == 0:
            continue
        ratios.append(np.abs(b[:, 0:3]).mean() / np.abs(a[:, 0:3]).mean())
        kept.append(len(b) / len(a))
    if not ratios:
        return float('nan'), float('nan')
    return float(np.median(ratios)), float(np.median(kept))


def source_mean_size(config, wanted_names, args, logger):
    """Mean GT [dx, dy, dz] of the SOURCE, for the --ros_factor path.

    Built in eval mode on purpose: augmentation (random_object_scaling in particular) would
    perturb exactly the statistic being measured. Pooled over every source a multi-source config
    declares, since the detector's size prior is inherited from all of them.

    Returns (mean, n_boxes, per_source) where per_source is one (name, mean, n) row per source -
    printed rather than averaged silently, because the two Lyft platforms and the two nuScenes
    cities have genuinely different mean car sizes (dataset_sensor_and_platform_facts.md).
    """
    dims, per_source = [], []
    for source_cfg in source_configs(config):
        source_set, _, _ = build_dataloader(
            dataset_cfg=source_cfg, class_names=config.CLASS_NAMES, batch_size=1,
            dist=False, workers=0, logger=logger, training=False,
            model_ontology=config.get('ONTOLOGY', None))
        source_dims = []
        for idx in range(min(args.frames, len(source_set))):
            sample = source_set[idx]
            if 'gt_boxes' not in sample or len(sample['gt_boxes']) == 0:
                continue
            names = np.array(sample.get('gt_names', []))
            boxes = np.asarray(sample['gt_boxes'])
            if len(names) == len(boxes):
                keep = np.array([any(w in n.lower() for w in wanted_names) for n in names])
                boxes = boxes[keep]
            if len(boxes):
                source_dims.append(boxes[:, 3:6])
        if source_dims:
            source_dims = np.concatenate(source_dims, axis=0)
            per_source.append((source_cfg.get('DATASET', '?'), source_dims.mean(axis=0),
                               len(source_dims)))
            dims.append(source_dims)
    assert dims, 'no source GT boxes matched --class_name; cannot derive E_est[Size] from ROS'
    dims = np.concatenate(dims, axis=0)
    return dims.mean(axis=0), len(dims), per_source


def dry_run_report(args, config, dataset_cfg, test_set, target_size, logger):
    """Everything the real search does except the forward passes. Needs no GPU.

    This exists because of what this repo has already paid for: a cross-dataset eval assert made
    six configurations unreachable for a month and cost 16 GPU-hours, two of which trained for
    eight hours before dying with no AP (20260921_04). Constructing the pieces is seconds of CPU.
    Getting a clean line here does not promise a good scale - it promises the hour will produce
    one rather than a traceback.
    """
    print('DRY RUN - no forward pass. Checks below are what the real sweep would use.\n')
    print('  config            %s' % args.cfg_file)
    print('  checkpoint        %s' % args.ckpt)
    print('  target frames     %d of %d available' % (min(args.frames, len(test_set)), len(test_set)))
    print('  candidate scales  %s' % [round(x, 3) for x in args.scales])
    print('  E_est[Size]       %s' % np.round(target_size, 3).tolist())

    print('\n  checkpoint:')
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print('    MISSING - %s does not exist. The real run would die here.' % ckpt_path)
    else:
        blob = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        state = blob.get('model_state', blob)
        print('    loads on CPU, epoch %s, %d parameter tensors'
              % (blob.get('epoch', '?'), len(state)))
        # The classic silent mismatch is a checkpoint trained under a different class vocabulary.
        # Its head output channels encode that, and reading them needs no network.
        for key in sorted(k for k in state if 'conv_cls' in k and k.endswith('weight')):
            print('    %-52s %s   (config declares %d classes)'
                  % (key, tuple(state[key].shape), len(config.CLASS_NAMES)))

    # Building the network is what would catch a true architecture mismatch, but SECOND's anchor
    # generator calls .cuda() at construction, so it cannot run on the login node at all. Say so
    # rather than printing a check that did not happen.
    print('\n  network build + state-dict match:')
    if torch.cuda.is_available():
        model = build_network(model_cfg=config.MODEL, num_class=len(config.CLASS_NAMES),
                              dataset=test_set)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        print('    OK - the state dict matched the network built from this config')
    else:
        print('    SKIPPED - no GPU on this node, and the anchor generator allocates on CUDA at')
        print('    construction. Re-run this same command on a GPU node to close that check.')

    # The endpoints bound the whole sweep, so probing them bounds how much of it the range crop eats.
    ontology = config.get('EVAL_ONTOLOGY', None) or config.get('ONTOLOGY', None)
    print('\n  requested vs achieved scale (POINT_CLOUD_RANGE is fixed while the scene grows):')
    print('  %10s %12s %12s' % ('requested', 'achieved', 'points kept'))
    for scale in sorted({args.scales[0], args.scales[-1], 1.0}):
        got, kept = achieved_scale(dataset_cfg, config, scale, logger, args.probe_frames,
                                   args.batch_size, ontology)
        note = '' if not np.isfinite(kept) or kept > 0.98 else '   <- the crop is eating this end'
        print('  %10.3f %12.4f %11.2f%%%s' % (scale, got, 100 * kept, note))

    print('\n  Next: drop --dry_run and run it for real. One forward pass over %d frames per '
          'candidate,\n  %d candidates, inference only.' % (args.frames, len(args.scales)))
    return 0


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    tgt_cfg = target_config(cfg)
    wanted_labels = np.array(class_indices(cfg.CLASS_NAMES, args.class_name))
    assert len(wanted_labels), ('--class_name %r matches none of %s'
                                % (args.class_name, list(cfg.CLASS_NAMES)))
    thresholds = score_thresholds(cfg, args.score_thresh, len(cfg.CLASS_NAMES))

    # Whichever way it is estimated, E_est[Size] is the whole input to the decision - print its
    # provenance next to it so a reader of the log never has to guess the supervision level.
    if args.target_size is not None:
        target_size = np.array(args.target_size, dtype=np.float64)
        provenance = ('given directly (--target_size). NOT target-free if it came from SN.')
    else:
        src_mean, n_src, per_source = source_mean_size(
            cfg, [args.class_name.lower()], args, logger)
        target_size = src_mean * args.ros_factor
        provenance = ('source mean %s over %d boxes x ROS factor %.3f - target-free'
                      % (np.round(src_mean, 3).tolist(), n_src, args.ros_factor))
        for name, mean, n in per_source:
            print('  source %-18s mean %s over %d boxes'
                  % (name, np.round(mean, 3).tolist(), n))

    test_set, _, _ = build_dataloader(
        dataset_cfg=tgt_cfg, class_names=cfg.CLASS_NAMES, batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False,
        model_ontology=cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None))

    print('\n' + '=' * 94)
    print('PTSN scale search   cfg=%s' % Path(args.cfg_file).stem)
    print('  target dataset   %s   frames/scale %d   classes %s   score_thresh %s'
          % (tgt_cfg.DATASET, args.frames,
             [cfg.CLASS_NAMES[i - 1] for i in wanted_labels], np.round(thresholds, 4).tolist()))
    print('  E_est[Size]      %s' % np.round(target_size, 3).tolist())
    print('  provenance       %s' % provenance)
    print('=' * 94)

    if args.dry_run:
        return dry_run_report(args, cfg, tgt_cfg, test_set, target_size, logger)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    print('%8s %10s %10s %10s %10s %8s' % ('scale', 'dx', 'dy', 'dz', 'gap(m)', 'boxes'))

    mean_sizes, counts = [], []
    for scale in args.scales:
        # A fresh loader per candidate: its workers fork holding this scale and never see a
        # later mutation, so the scale a frame was processed at is unambiguous. Cheaper than it
        # looks - the fork cost is seconds against a full inference pass.
        scale_set, loader, _ = build_dataloader(
            dataset_cfg=tgt_cfg, class_names=cfg.CLASS_NAMES, batch_size=args.batch_size,
            dist=False, workers=args.workers, logger=logger, training=False,
            model_ontology=cfg.get('EVAL_ONTOLOGY', None) or cfg.get('ONTOLOGY', None))
        scale_set.set_ptsn_scale(scale)
        mean_size, n_boxes, n_frames = mean_predicted_size(
            model, loader, scale, wanted_labels, thresholds, args.frames)
        mean_sizes.append(mean_size)
        counts.append(n_boxes)
        gap = float(np.abs(mean_size - target_size).mean())
        flag = '' if n_boxes >= MIN_BOXES else '   <- too few boxes to trust'
        print('%8.3f %10.3f %10.3f %10.3f %10.4f %8d%s'
              % (scale, mean_size[0], mean_size[1], mean_size[2], gap, n_boxes, flag))

    best_scale, rows = ptsn_utils.select_scale(args.scales, mean_sizes, target_size)
    best = min(rows, key=lambda row: row[2] if np.isfinite(row[2]) else np.inf)
    print('-' * 94)
    print('best scale %.3f   mean size %s   gap %.4f m'
          % (best_scale, np.round(best[1], 3).tolist(), best[2]))

    if best_scale in (args.scales[0], args.scales[-1]) and len(args.scales) > 1:
        print('WARNING: the optimum is at an endpoint of the sweep - widen --scales, the true '
              'minimum is probably outside it.')
    if min(counts) < MIN_BOXES:
        print('WARNING: at least one candidate produced under %d boxes. A mean size over a handful '
              'of boxes is noise; raise --frames or lower --score_thresh.' % MIN_BOXES)

    # The stopping rule this search exists to serve (20260922_04 section 2.4): the sweep can only
    # justify Tier D2 if it first shows the size distribution is reachable at all.
    print('\nStopping rule: if this moves the pseudo-label mean size to within ~5 cm of '
          'E_est[Size]\n  (gap above) but target AP does not move, the distribution-level claim '
          'does not hold\n  in this setting - record the negative result and do not write Tier D2.')

    if args.yaml:
        print('\n' + '=' * 94)
        print('Config block - paste under SELF_TRAIN in %s' % Path(args.cfg_file).name)
        print('=' * 94)
        print('    PTSN:')
        print('        ENABLED: True')
        print('        SCALE: %.3f   # %s' % (best_scale, provenance.split(' - ')[0]))

    return 0


if __name__ == '__main__':
    sys.exit(main())
