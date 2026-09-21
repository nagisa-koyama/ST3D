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
