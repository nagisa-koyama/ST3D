"""BEV feature imitation loss for the LiDAR Distillation baseline (Wei et al., ECCV 2022).

The student sees a beam-downsampled cloud, the teacher the full one, and the student is asked to
reproduce the teacher's BEV feature map where the objects are. Ported from
`tools/train_utils/train_mimic_utils.py::cal_mimic_loss` in https://github.com/weiyithu/LiDAR-Distillation,
with three changes forced by this repo:

1. **The BEV stride is measured, not configured.** The reference reads
   `model_cfg.ROI_HEAD.ROI_GRID_POOL.DOWNSAMPLE_RATIO`, which exists only for its two-stage
   detectors. The `da-ieee-access` family is CenterPoint, which has no ROI head at all, so the
   stride is derived from the feature map's own width against the voxel grid - correct for any
   backbone and impossible to set inconsistently with the network that produced the map.

2. **Only `gt` and `all` modes.** The reference's default `roi` mode masks around the TEACHER's
   proposals, which a single-stage detector does not produce during training. `gt` is not a
   fallback here: this stage trains on the LABELLED SOURCE, so ground-truth boxes are legitimately
   available and are a cleaner mask than proposals - no dependence on how well the teacher's
   proposal head happens to be doing.

3. **The `gt` normalisation is applied once.** The reference computes the masked mean and then, for
   `gt` only, multiplies that scalar by the mask and sums again:

       mimic_loss = (mimic_loss * mask).sum() / batch_size / roi_size      # already a scalar
       if mimic_mode == 'gt':
           mimic_loss = (mimic_loss * mask).sum() / (rois[:,:,-1] > 0).sum()

   Because each valid box's mask sums to 1, the second line multiplies by
   `n_valid_boxes / n_valid_boxes` and is close to a no-op in expectation - but only by accident,
   and it makes the loss scale with the mask's total rather than with the per-cell error. Here the
   masked mean is formed once, over valid boxes.
"""

import torch


def forward_to_bev(model, batch_dict):
    """Run `model` only as far as its BEV feature map, in eval mode and without gradients.

    Two things this avoids that a plain `model(batch_dict)` would not:

    * **BatchNorm statistics.** A train-mode forward updates `running_mean`/`running_var` even under
      `no_grad`, so passing the teacher's own data through it would keep moving the frozen teacher.
    * **The detection head and post-processing.** In eval mode a full forward runs NMS/top-k to
      build `final_box_dicts`, which the imitation loss never looks at. The reference implementation
      pays for this on every iteration.

    Stopping at the first module whose output carries 'spatial_features_2d' also means this works
    for any backbone in the repo without naming one.
    """
    core = model.module if hasattr(model, 'module') else model
    was_training = core.training
    core.eval()
    try:
        with torch.no_grad():
            for cur_module in core.module_list:
                batch_dict = cur_module(batch_dict)
                if 'spatial_features_2d' in batch_dict:
                    return batch_dict
    finally:
        core.train(was_training)
    raise RuntimeError(
        "no module produced 'spatial_features_2d', so there is no BEV feature map to imitate. The "
        "LiDAR Distillation baseline needs a backbone_2d (BaseBEVBackbone or "
        "DomainAttentionBEVBackbone); a point-only detector cannot serve as its teacher.")


def bev_feature_stride(spatial_features_2d, grid_size):
    """Cells of the voxel grid per cell of the BEV feature map.

    Args:
        spatial_features_2d: (B, C, H, W) BEV feature map.
        grid_size: (3,) voxel grid size as [X, Y, Z], i.e. `dataset.grid_size`.
    """
    width = spatial_features_2d.shape[3]
    stride = float(grid_size[0]) / float(width)
    # A backbone that does not divide the grid evenly would make every box's cell extent wrong by a
    # sub-cell amount that grows with distance from the origin; better to refuse than to blur.
    assert abs(stride - round(stride)) < 1e-6, (
        'BEV feature map width {} does not divide the voxel grid width {} evenly, so the feature '
        'stride is not an integer and box extents cannot be mapped onto feature cells.'.format(
            width, grid_size[0]))
    return int(round(stride))


def boxes_to_bev_mask(boxes, spatial_features_2d, point_cloud_range, voxel_size, grid_size):
    """Per-box BEV occupancy masks, each normalised to sum 1.

    Args:
        boxes: (B, N, 7+C) with the class id in the LAST column; a zero there marks padding, which
            is how `collate_batch` pads `gt_boxes` to the batch maximum.
        spatial_features_2d: (B, C, H, W).
        point_cloud_range / voxel_size: as held on the dataset.
        grid_size: (3,) [X, Y, Z].
    Returns:
        mask: (B, H, W) - the per-box masks summed, so a cell covered by two boxes counts twice.
        num_valid: scalar tensor, the number of boxes with a non-empty footprint.
    """
    stride = bev_feature_stride(spatial_features_2d, grid_size)
    batch_size, _, height, width = spatial_features_2d.shape

    min_x, min_y = float(point_cloud_range[0]), float(point_cloud_range[1])
    cell_x = float(voxel_size[0]) * stride
    cell_y = float(voxel_size[1]) * stride

    # Axis-aligned extent, heading ignored - as in the reference. The mask is a region of interest
    # around the object, not its exact footprint.
    x1 = (boxes[:, :, 0] - boxes[:, :, 3] / 2 - min_x) / cell_x
    x2 = (boxes[:, :, 0] + boxes[:, :, 3] / 2 - min_x) / cell_x
    y1 = (boxes[:, :, 1] - boxes[:, :, 4] / 2 - min_y) / cell_y
    y2 = (boxes[:, :, 1] + boxes[:, :, 4] / 2 - min_y) / cell_y

    device = spatial_features_2d.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    grid_y = grid_y[None, None]
    grid_x = grid_x[None, None]

    in_y = (grid_y >= y1[:, :, None, None]) & (grid_y <= y2[:, :, None, None])
    in_x = (grid_x >= x1[:, :, None, None]) & (grid_x <= x2[:, :, None, None])
    per_box = (in_y & in_x).float()

    # Drop padded boxes before normalising, or they would contribute a footprint at the origin.
    valid = (boxes[:, :, -1] != 0).float()
    per_box = per_box * valid[:, :, None, None]

    area = per_box.sum(-1).sum(-1)
    # Each box gets equal total weight, so a distant (small) object is worth as much as a near one.
    # A box whose footprint falls entirely outside the feature map has area 0; the clamp keeps the
    # division finite and the mask stays all-zero, so it contributes nothing.
    per_box = per_box / torch.clamp(area, min=1.0)[:, :, None, None]

    num_valid = (area > 0).float().sum()
    return per_box.sum(1), num_valid


def bev_imitation_loss(student_batch, teacher_batch, mode, point_cloud_range, voxel_size, grid_size,
                       boxes_key='gt_boxes', normalization='reference'):
    """L2 distance between student and teacher BEV features, masked to the objects.

    normalization (mode 'gt' only):
        'reference' - the released code's denominator, `batch_size * roi_size`, where roi_size is the
            PADDED second dimension of the gt_boxes tensor (the largest box count in the batch). This
            is what `--mimic_weight 1` in the paper multiplies, so WEIGHT 1.0 here is the published
            setting. (The reference's apparent second normalisation, line 56 of train_mimic_utils.py,
            is a no-op: it multiplies an already-scalar loss by the mask, whose sum is the number of
            valid boxes, and divides by that same count.) The quirk comes with it: the effective
            weight drifts with batch composition, by the padding factor B*R / n_valid.
        'valid' - divide by the number of boxes with a footprint. Larger than 'reference' by that
            padding factor (typically 2-4x on KITTI), which is why WEIGHT 1.0 under it put the
            imitation term at 42% of the objective (experiments_md/20260927_02 section 7.2).

    Args:
        student_batch / teacher_batch: batch dicts AFTER the forward pass, so both carry
            'spatial_features_2d'. The teacher's is detached here.
        mode: 'gt' masks to ground-truth boxes; 'all' averages over every cell.
    Returns:
        scalar loss tensor.
    """
    student_features = student_batch['spatial_features_2d']
    teacher_features = teacher_batch['spatial_features_2d'].detach()

    assert student_features.shape == teacher_features.shape, (
        'student BEV feature map {} does not match the teacher\'s {}. The two streams must share '
        'POINT_CLOUD_RANGE, VOXEL_SIZE and backbone; a per-cell loss between different grids is '
        'meaningless.'.format(tuple(student_features.shape), tuple(teacher_features.shape)))

    # Channel-wise L2 per BEV cell -> (B, H, W).
    per_cell = torch.norm(teacher_features - student_features, p=2, dim=1)

    if mode == 'all':
        return per_cell.mean()
    if mode != 'gt':
        raise NotImplementedError(
            "mimic mode {!r} is not supported; use 'gt' (mask to ground-truth boxes) or 'all'. The "
            "reference's 'roi' mode needs a two-stage detector's proposals, which CenterPoint does "
            "not produce during training.".format(mode))

    boxes = student_batch[boxes_key]
    mask, num_valid = boxes_to_bev_mask(
        boxes, student_features, point_cloud_range, voxel_size, grid_size)
    if num_valid == 0:
        # Every box out of range, or a frame with no labels at all. Returning a real zero that is
        # still attached to the graph keeps DDP's gradient buckets consistent across ranks.
        return (per_cell * 0.0).sum()
    if normalization == 'reference':
        batch_size, roi_size = boxes.shape[0], boxes.shape[1]
        return (per_cell * mask).sum() / (batch_size * roi_size)
    if normalization == 'valid':
        return (per_cell * mask).sum() / num_valid
    raise NotImplementedError('normalization {!r}: use "reference" or "valid"'.format(normalization))
