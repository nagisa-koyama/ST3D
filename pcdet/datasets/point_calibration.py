"""Measure the radial point-density histograms the density correction needs, at load time.

`sample_points_hist_based` compares a source histogram against a target one and drops source points
until the two match. Those histograms were shipped as `hist_dist_<dataset>_<tag>.npy`, which caused
three problems:

  * they are RAW COUNTS over however many frames that analysis run happened to process (1,020-1,160
    for the files in use), so the target/source ratio silently inherits any frame-count difference;
  * their shape disagrees with a direct measurement of the same data by up to 1.8x per bin;
  * they go stale the moment anything upstream changes - a different MAX_SWEEPS, a platform subset,
    a different POINT_CLOUD_RANGE - with nothing to detect that they have.

Measuring them from the datasets actually being trained on removes all three, and is normalised
per frame by construction.

It runs ONCE per training run, before the dataloaders are first iterated, so the default sample of
1000 frames per domain is affordable against a multi-hour job. There is no incremental or
per-epoch update: a histogram measured mid-training would not reach the forked DataLoader workers
anyway (see 20260921_02).

See experiments_md/20260922_02_dataset_and_platform_domain_gap_analysis.md, defect 3.
"""
import numpy as np

MAX_DIST = 75.0
DEFAULT_BINS = 50
DEFAULT_FRAMES = 1000


def compute_range_histogram(dataset, num_frames=DEFAULT_FRAMES, num_bins=DEFAULT_BINS,
                            max_dist=MAX_DIST, logger=None):
    """Mean points per frame per radial bin, measured through the dataset's own pipeline.

    Sampling goes through `dataset[i]`, so the points counted are exactly the ones the correction
    will later see: range-masked, in the dataset's own coordinate frame, after whatever loader
    transforms that dataset applies. The correction itself no-ops while either histogram is None,
    which is the state during this call, so the measurement is not circular.

    Frames are taken on a stride rather than at random so the estimate is deterministic - two runs
    of the same config produce the same histogram.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError('cannot measure a histogram from an empty dataset')
    step = max(1, n // num_frames)
    edges = np.linspace(0, max_dist, num_bins + 1)
    total = np.zeros(num_bins, dtype=np.float64)
    used = 0
    for idx in range(0, n, step):
        if used >= num_frames:
            break
        points = dataset[idx].get('points', None)
        if points is None or not len(points):
            continue
        dist = np.linalg.norm(points[:, 0:2], axis=1)
        total += np.histogram(np.clip(dist, 0, max_dist - 1e-4), bins=edges)[0]
        used += 1
    if used == 0:
        raise ValueError('no usable frames while measuring a histogram')
    hist = total / used
    if logger is not None:
        logger.info('point calibration: measured %s over %d frames, %.0f pts/frame in range'
                    % (getattr(dataset, 'dataset_ontology', '?'), used, hist.sum()))
    return hist


def link_point_calibration(source_set, target_set, num_frames=DEFAULT_FRAMES,
                           num_bins=DEFAULT_BINS, logger=None):
    """Measure both domains and install the pair into the SOURCE dataset's processor.

    Only the source is corrected: the target's own calibration target is itself, which makes its
    rate identically 1. Installing on the source alone keeps that explicit.

    Must run before the dataloaders are first iterated, since DataLoader workers fork a copy of the
    dataset and never see later mutations - the same hazard as
    experiments_md/20260921_02_persistent_workers_stale_dataset_state.md.
    """
    src = compute_range_histogram(source_set, num_frames, num_bins, logger=logger)
    tgt = compute_range_histogram(target_set, num_frames, num_bins, logger=logger)
    source_set.data_processor.set_hist_dist(src, tgt)
    if logger is not None:
        rate = source_set.data_processor.per_bin_sample_rate()
        guarded = int((src <= 0.01 * src.mean()).sum())
        logger.info('point calibration: sample rate %.2f..%.2f, below 1 in %d of %d bins, '
                    '%d bin(s) left uncorrected as under-populated'
                    % (rate.min(), rate.max(), int((rate < 1).sum()), num_bins, guarded))
        if (rate >= 1).all():
            logger.warning('point calibration: rate >= 1 in every bin - the correction is a no-op. '
                           'The source is sparser than the target everywhere; accumulate sweeps '
                           'instead (MAX_SWEEPS).')
    return src, tgt


def compute_foreground_histograms(dataset, num_frames=DEFAULT_FRAMES, num_bins=DEFAULT_BINS,
                                  max_dist=MAX_DIST, logger=None, label=''):
    """Mean points per frame per radial bin, split into inside-box and outside-box channels.

    Boxes come from `dataset[idx]['gt_boxes']`, which is real annotation for a source domain and
    PSEUDO-LABELS for a self-training target - the target dataset fills them from `PSEUDO_LABELS`
    in train mode. That is the whole point: it makes a foreground-aware correction computable
    without target annotation, and so UDA-legal. It also means this must run AFTER the first
    pseudo-label generation pass, or the target's boxes do not exist yet.

    Ignored pseudo-labels (negative class index, the ones memory voting has demoted) are excluded,
    so the target foreground channel counts only boxes the teacher currently stands behind.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError('cannot measure a histogram from an empty dataset')
    step = max(1, n // num_frames)
    edges = np.linspace(0, max_dist, num_bins + 1)
    fg = np.zeros(num_bins, dtype=np.float64)
    bg = np.zeros(num_bins, dtype=np.float64)
    used, with_boxes = 0, 0
    for idx in range(0, n, step):
        if used >= num_frames:
            break
        data_dict = dataset[idx]
        points = data_dict.get('points', None)
        if points is None or not len(points):
            continue
        boxes = data_dict.get('gt_boxes', None)
        if boxes is not None and len(boxes):
            boxes = np.asarray(boxes)
            if boxes.shape[1] > 7:                       # drop ignored pseudo-labels
                boxes = boxes[boxes[:, 7] > 0]
            with_boxes += 1 if len(boxes) else 0
        mask = dataset.data_processor.points_in_any_box(points, boxes)
        dist = np.clip(np.linalg.norm(points[:, 0:2], axis=1), 0, max_dist - 1e-4)
        fg += np.histogram(dist[mask], bins=edges)[0]
        bg += np.histogram(dist[~mask], bins=edges)[0]
        used += 1
    if used == 0:
        raise ValueError('no usable frames while measuring a histogram')
    fg, bg = fg / used, bg / used
    if logger is not None:
        share = fg.sum() / max(fg.sum() + bg.sum(), 1e-9)
        logger.info('point calibration [%s]: %d frames (%d with boxes), %.0f fg + %.0f bg '
                    'pts/frame, foreground share %.2f%%'
                    % (label, used, with_boxes, fg.sum(), bg.sum(), 100 * share))
        if with_boxes == 0:
            logger.warning('point calibration [%s]: NO boxes in any sampled frame - the foreground '
                           'channel is empty and its rate will be 1 everywhere. For a target '
                           'domain this means pseudo-labels had not been generated yet.' % label)
    return fg, bg


def link_foreground_calibration(source_set, target_set, num_frames=DEFAULT_FRAMES,
                                num_bins=DEFAULT_BINS, logger=None, source_hist=None):
    """Foreground-aware calibration: correct inside-box and outside-box points separately.

    A single per-bin rate cannot change a bin's foreground SHARE - it scales the points on objects
    and the points on everything else by the same factor - so matching the global radial profile
    leaves source objects at exactly sigma_src/sigma_tgt of the target's object density, measured
    at 0.42-0.76x across the source datasets. Splitting the correction into two channels,
    `min(F_t/F_s, 1)` inside boxes and `min(B_t/B_s, 1)` outside, gives it the degree of freedom it
    structurally lacked, and derives the direction per bin from data rather than assuming one.

    Replaces `link_point_calibration` rather than supplementing it: the whole-cloud pair is
    installed too, as the exact sum of the two channels, because the correction needs it for the
    bin count and for the no-op guard.

    Called again after each pseudo-label update, since the labels define the target foreground
    channel and the two have to move together. Pass the source pair back in as `source_hist` on
    every call after the first - see the note in the body for why that is required, not merely
    faster.

    Ordering is load-bearing twice over. The measurement must precede any installation, or the
    source would be measured through a correction that is already running - the histograms are
    taken while every rate is still absent, so the measurement is not circular. And it must precede
    the first iteration of any loader over either dataset, since workers fork a copy.
    """
    if source_hist is None:
        fg_s, bg_s = compute_foreground_histograms(source_set, num_frames, num_bins, logger=logger,
                                                   label='source')
    else:
        # Re-measuring the source on a refresh would read points the correction installed last time
        # has ALREADY thinned, compounding the rate on every pass. The source distribution does not
        # change anyway, so it is measured once and passed back in.
        fg_s, bg_s = source_hist
    fg_t, bg_t = compute_foreground_histograms(target_set, num_frames, num_bins, logger=logger,
                                               label='target (pseudo-labels)')
    source_set.data_processor.set_hist_dist(fg_s + bg_s, fg_t + bg_t)
    source_set.data_processor.set_foreground_hist(fg_s, bg_s, fg_t, bg_t)
    if logger is not None:
        proc = source_set.data_processor
        r_fg, r_bg = proc.per_bin_sample_rate(None, 'fg'), proc.per_bin_sample_rate(None, 'bg')
        s_src = fg_s.sum() / max(fg_s.sum() + bg_s.sum(), 1e-9)
        s_tgt = fg_t.sum() / max(fg_t.sum() + bg_t.sum(), 1e-9)
        logger.info('point calibration: foreground-aware. fg rate %.2f..%.2f, bg rate %.2f..%.2f'
                    % (r_fg.min(), r_fg.max(), r_bg.min(), r_bg.max()))
        logger.info('point calibration: foreground share src %.2f%% vs tgt %.2f%% (ratio %.2f) - '
                    'below 1 means the uniform correction would have starved source objects'
                    % (100 * s_src, 100 * s_tgt, s_src / max(s_tgt, 1e-9)))
        logger.warning('point calibration: the target foreground channel is measured from '
                       'PSEUDO-LABELS, so any teacher recall below 1 UNDER-estimates it and biases '
                       'the foreground rate DOWNWARD - the same direction as the defect this is '
                       'meant to fix. Generate pseudo-labels at a high-recall SCORE_THRESH.')
    return (fg_s, bg_s), (fg_t, bg_t)
