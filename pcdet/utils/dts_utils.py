"""DTS (Hu, Liu, Hu, CVPR 2023, "Density-insensitive Unsupervised Domain Adaption on 3D Object Detection") ported to
CenterPoint in this fork: experiments_md 20261003_05 §5 item 51 (DTS and ST3D++ re-run under our protocol).

Reference: the released code WoodwindHu/DTS @ 4aad196 (read-only clone /home/koyama/data/released_repos/DTS), cited
below as DTS/<path>:<line>. Every departure from it is named where it is made.

`ring_upsample` is RBRS's up-sampling half (DTS `random_beam_upsample`, pcdet/datasets/augmentor/data_augmentor.py:141-166),
used on the labelled source during source pretraining (the D0 model). Departure: the ring of each point is the sensor's
stored ring index (nuScenes column 4, label-free), and angles are taken about the sensor origin. DTS labels rings by a
per-frame KMeans of elevation about the SHIFT_COOR-shifted origin (DTS data_augmentor.py:92-111 after
nuscenes_dataset.py:125-126), which mixes rings near the sensor.
"""
import numpy as np


def polar(points):
    """DTS downsample_utils.compute_angles + get_polar_image: azimuth phi in [0, 2 pi), elevation theta = atan(z / rho),
    3D range r. The [0, 2 pi) convention fixes where the seam lies (+x), which decides which pairs DTS never forms."""
    x, y, z = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64), points[:, 2].astype(np.float64)
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    phi[phi >= 2.0 * np.pi] = 0.0
    theta = np.arctan2(z, np.hypot(x, y))
    r = np.sqrt(x * x + y * y + z * z)
    return phi, theta, r


def _nearest(sorted_vals, queries):
    """Index into sorted_vals of the nearest value to each query (absolute difference, no wrap) and that difference."""
    j = np.searchsorted(sorted_vals, queries)
    lo = np.clip(j - 1, 0, len(sorted_vals) - 1)
    hi = np.clip(j, 0, len(sorted_vals) - 1)
    d_lo, d_hi = np.abs(queries - sorted_vals[lo]), np.abs(queries - sorted_vals[hi])
    take_hi = d_hi < d_lo
    return np.where(take_hi, hi, lo), np.where(take_hi, d_hi, d_lo)


def ring_upsample(points, rings, beam_prob, phi_threshold, n_rings=None, rand=np.random.rand):
    """New points between adjacent rings, as DTS `random_beam_upsample`.

    For each ring pair (i, i + 1), i = 0 .. n_rings - 2, drawn with probability beam_prob[i] (one `rand()` per pair, in
    order, as DTS): every point of ring i takes the point of ring i + 1 nearest in azimuth; if that azimuth difference is
    below phi_threshold (rad, no wrap), a new point is added at the midpoint of the pair in (phi, theta, r), with every
    further column (intensity, ring, ...) averaged too. The input points are kept unchanged, first, in order.

    points: (N, C) with x, y, z in the sensor frame first. rings: (N,) integer ring index, adjacent indices adjacent in
    elevation (nuScenes: index 0 lowest, checked 2026-11-22 on three train frames). beam_prob: scalar or one value per
    pair. Returns (N + M, C) of points.dtype. An empty ring forms no pair (DTS's argmin would raise there).
    """
    rings = np.asarray(rings).astype(np.int64)
    if n_rings is None:
        n_rings = int(rings.max()) + 1 if len(rings) else 0
    probs = np.broadcast_to(np.asarray(beam_prob, dtype=np.float64), (max(n_rings - 1, 0),))
    if len(points) == 0 or n_rings < 2:
        return points
    phi, theta, r = polar(points)
    pol = points.astype(np.float64).copy()
    pol[:, 0], pol[:, 1], pol[:, 2] = phi, theta, r
    new = []
    for i in range(n_rings - 1):
        if not rand() < probs[i]:
            continue
        cur, nxt = np.flatnonzero(rings == i), np.flatnonzero(rings == i + 1)
        if len(cur) == 0 or len(nxt) == 0:
            continue
        order = nxt[np.argsort(phi[nxt], kind='stable')]
        k, d = _nearest(phi[order], phi[cur])
        ok = d < phi_threshold
        if not ok.any():
            continue
        mid = (pol[cur[ok]] + pol[order[k[ok]]]) / 2.0
        out = mid.copy()
        out[:, 0] = np.cos(mid[:, 1]) * np.cos(mid[:, 0]) * mid[:, 2]
        out[:, 1] = np.cos(mid[:, 1]) * np.sin(mid[:, 0]) * mid[:, 2]
        out[:, 2] = np.sin(mid[:, 1]) * mid[:, 2]
        new.append(out)
    if not new:
        return points
    return np.concatenate([points] + [n.astype(points.dtype) for n in new], axis=0)


# ----------------------------------------------------------------------------------------------------------- adaptation
# DTS's adaptation stage on CenterPoint (row D of item 51). Released reference: DTS tools/train_utils/train_mt_utils.py
# (EMA teacher), pcdet/datasets/kitti/kitti_dataset_mt.py (two views), detector3d_template.get_graph_loss (OGA).
# Mapping (item 51, user Q2 (a)): graph nodes are CenterHead's decoded boxes (student 512 after NMS 0.8, teacher 100
# after NMS 0.7: the released proposal settings); node features are a rotated 7 x 7 pool of the BEV map at each box,
# flattened, the student's NOT detached (gradient reaches the backbone; DTS's single-stage PointPillars release detaches
# and so trains nothing through OGA). The two views share one augmentation draw (dataset.py DTS_VIEWS), so no box is
# reversed: the released code matched student proposals in the augmented frame against teacher proposals in the raw
# frame (reverse_augmentation was applied to another tensor). The released object loss kept only the last sample of the
# batch (`=` for `+=`); it is accumulated here. Both corrections are disclosed.


def beam_keep_mask(ring, num_beams, keep_prob, rand=np.random.rand):
    """DTS `random_beam_downsample` (data_augmentor.py:168-179): each of num_beams beams kept with probability keep_prob,
    one draw per beam, no global gate. Points with ring < 0 (Waymo's side lidars, which have no TOP row) are kept."""
    ring = np.asarray(ring).astype(np.int64)
    assert ring.max(initial=-1) < num_beams, 'ring index %d beyond NUM_BEAMS %d' % (ring.max(), num_beams)
    keep_beam = rand(num_beams) < keep_prob
    return np.where(ring >= 0, keep_beam[np.clip(ring, 0, None)], True)


def decode_nodes(head, pred_dicts, topk, nms_thresh, post_max, hm_is_prob):
    """Graph nodes from CenterHead predictions: per sample (boxes (N, 7), raw heatmap logit (N,)), all classes, after
    class-agnostic NMS, no score threshold (DTS's proposals carry none). Decoded without gradient (DTS's proposal layer
    runs under no_grad). hm_is_prob: the heatmap already went through CenterHead.sigmoid (get_loss does it in place)."""
    import torch
    from easydict import EasyDict
    from ..models.model_utils import centernet_utils, model_nms_utils
    post = head.model_cfg.POST_PROCESSING
    limit = torch.tensor(post.POST_CENTER_LIMIT_RANGE, device=pred_dicts[0]['hm'].device).float()
    nms_cfg = EasyDict(NMS_TYPE='nms_gpu', NMS_THRESH=nms_thresh, NMS_PRE_MAXSIZE=topk, NMS_POST_MAXSIZE=post_max)
    with torch.no_grad():
        per_head = []
        for pd in pred_dicts:
            hm = pd['hm'] if hm_is_prob else torch.clamp(pd['hm'].sigmoid(), min=1e-4, max=1 - 1e-4)
            per_head.append(centernet_utils.decode_bbox_from_heatmap(
                heatmap=hm, rot_cos=pd['rot'][:, 0:1], rot_sin=pd['rot'][:, 1:2], center=pd['center'],
                center_z=pd['center_z'], dim=pd['dim'].exp(), point_cloud_range=head.point_cloud_range,
                voxel_size=head.voxel_size, feature_map_stride=head.feature_map_stride, K=topk, score_thresh=None,
                post_center_limit_range=limit))
        out = []
        for k in range(len(per_head[0])):
            boxes = torch.cat([h[k]['pred_boxes'][:, :7] for h in per_head], 0)
            prob = torch.cat([h[k]['pred_scores'] for h in per_head], 0)
            sel, _ = model_nms_utils.class_agnostic_nms(box_scores=prob, box_preds=boxes, nms_config=nms_cfg)
            p = prob[sel].clamp(1e-4, 1 - 1e-4)
            out.append((boxes[sel].contiguous(), torch.log(p / (1 - p))))
    return out


def bev_box_pool(features, boxes, point_cloud_range, voxel_size, stride, grid_size=7):
    """Rotated grid_size x grid_size bilinear pool of a BEV map (C, H, W) over each box's footprint, flattened to
    (N, C * grid_size**2). The released SECONDHead / SECONDHeadMT roi_grid_pool's affine grid (same theta), on the map as
    given: pass it undetached for the student, so the gradient reaches the backbone."""
    import torch
    import torch.nn.functional as F
    c, height, width = features.shape
    if len(boxes) == 0:
        return features.new_zeros((0, c * grid_size * grid_size))
    sx, sy = voxel_size[0] * stride, voxel_size[1] * stride
    x1 = (boxes[:, 0] - boxes[:, 3] / 2 - point_cloud_range[0]) / sx
    x2 = (boxes[:, 0] + boxes[:, 3] / 2 - point_cloud_range[0]) / sx
    y1 = (boxes[:, 1] - boxes[:, 4] / 2 - point_cloud_range[1]) / sy
    y2 = (boxes[:, 1] + boxes[:, 4] / 2 - point_cloud_range[1]) / sy
    cosa, sina = torch.cos(boxes[:, 6]), torch.sin(boxes[:, 6])
    theta = torch.stack((
        (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
        (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
    ), dim=1).view(-1, 2, 3).to(features.dtype)
    n = len(boxes)
    grid = F.affine_grid(theta, torch.Size((n, c, grid_size, grid_size)), align_corners=True)  # (N, g, g, 2)
    # ONE grid_sample over all boxes' grids: the released expand(N, C, H, W) form is fine for a detached map, but its
    # backward materialises N copies of the map's gradient (512 x 256 x 376 x 376 floats on our grid). Same values.
    pooled = F.grid_sample(features.unsqueeze(0), grid.reshape(1, n * grid_size, grid_size, 2), align_corners=True)
    return pooled.view(c, n, grid_size, grid_size).permute(1, 0, 2, 3).reshape(n, -1)


GRAPH_BOX_WEIGHTS = (1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 20.0)  # DTS detector3d_template.py:483, over x y z l w h heading


def oga_loss(student, teacher, cfg):
    """DTS's object graph alignment, per sample: each student node takes its best teacher node by 3D IoU, pairs at
    IoU >= cfg.INTER_GRAPH_THRESHOLD are kept, features are L2-normalised and (GRAPH_MATRIX_REWEIGHT, default True)
    multiplied by the raw logit; A = exp(-sum_k w_k (b_i - b_j)_k^2 / T^2) on the (constant) boxes.
    inter = ||A_s - A_t||_F (no gradient, as released), GLR = |tr(F_s^T A_s F_s) - tr(F_t^T A_t F_t)| / N^2, object =
    mean exp(-<f_s, f_t>), each averaged over the batch; loss = W_inter (gamma inter + (1 - gamma) GLR) + W_obj object,
    gamma = 1 - GLR_loss_weight. student / teacher: per sample (boxes (N, 7), logits (N,), features (N, D)).
    Returns (loss, {inter, GLR, object, pairs}); loss is 0 (a float) when no sample has a pair, as released."""
    import torch
    import torch.nn.functional as F
    from ..ops.iou3d_nms import iou3d_nms_utils
    w = cfg.CONSISTENCY_LOSS
    temp = float(cfg.get('TEMPERATURE', 13.0))
    inter = glr = obj = 0.0
    pairs = 0
    batch = len(student)
    for (ba, ca, fa), (bb, cb, fb) in zip(student, teacher):
        if len(ba) == 0 or len(bb) == 0:
            continue
        ious, idx = iou3d_nms_utils.boxes_iou3d_gpu(ba[:, :7], bb[:, :7]).max(dim=1)
        keep = ious >= float(cfg.get('INTER_GRAPH_THRESHOLD', 0.7))
        if not bool(keep.any()):
            continue
        a, b = keep.nonzero().view(-1), idx[keep]
        fa_, fb_ = F.normalize(fa[a], dim=1), F.normalize(fb[b], dim=1)
        if cfg.get('GRAPH_MATRIX_REWEIGHT', True):
            fa_, fb_ = fa_ * ca[a].unsqueeze(1), fb_ * cb[b].unsqueeze(1)
        wk = torch.tensor(GRAPH_BOX_WEIGHTS, device=ba.device, dtype=ba.dtype).view(1, 1, 7)
        box_a, box_b = ba[a].detach(), bb[b].detach()
        adj_a = torch.exp(-(((box_a.unsqueeze(0) - box_a.unsqueeze(1)) ** 2) * wk).sum(2) / temp ** 2)
        adj_b = torch.exp(-(((box_b.unsqueeze(0) - box_b.unsqueeze(1)) ** 2) * wk).sum(2) / temp ** 2)
        inter = inter + torch.norm(adj_a - adj_b)
        n = len(a)
        glr = glr + torch.abs(torch.einsum('nc,nm,mc->', fa_, adj_a, fa_) - torch.einsum('nc,nm,mc->', fb_, adj_b, fb_)) / n ** 2
        obj = obj + torch.exp(-(fa_ * fb_).sum(1)).mean()
        pairs += n
    if pairs == 0:
        return 0.0, {'oga_inter': 0.0, 'oga_GLR': 0.0, 'oga_object': 0.0, 'oga_pairs': 0}
    inter, glr, obj = inter / batch, glr / batch, obj / batch
    gamma = 1.0 - float(w.get('GLR_loss_weight', 0.0))
    loss = 0.0
    if float(w['inter_graph_loss_weight']) > 0:
        loss = loss + (inter * gamma + glr * (1 - gamma)) * float(w['inter_graph_loss_weight'])
    if float(w.get('object_loss_weight', 0)) > 0:
        loss = loss + obj * float(w['object_loss_weight'])
    return loss, {'oga_inter': float(inter), 'oga_GLR': float(glr), 'oga_object': float(obj), 'oga_pairs': pairs}


def make_ema_teacher(model):
    """DTS's EMA teacher: a copy of the student as it starts (= the pretrained source model), in eval mode for the whole
    run (released: the epoch-0 pseudo-label pass sets eval() and nothing switches it back), no gradients."""
    import copy
    src = model.module if hasattr(model, 'module') else model
    ema = copy.deepcopy(src)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema.eval()


def oga_from_batches(model, ema_model, student_batch, teacher_batch, cfg):
    """One step's OGA term. The student's CenterHead predictions are those of the forward that just computed its
    detection loss (get_loss has put the heatmap through the sigmoid in place); the teacher runs its modules on the
    teacher view without gradient, in eval mode."""
    import torch
    from ..models import load_data_to_gpu
    head = (model.module if hasattr(model, 'module') else model).dense_head
    sn, tn, g = cfg.STUDENT_NODES, cfg.TEACHER_NODES, int(cfg.get('GRID_SIZE', 7))
    s_nodes = decode_nodes(head, head.forward_ret_dict['pred_dicts'], sn.TOPK, sn.NMS_THRESH, sn.POST_MAX, True)
    with torch.no_grad():
        load_data_to_gpu(teacher_batch)
        tb = teacher_batch
        for module in ema_model.module_list:
            tb = module(tb)
        t_head = ema_model.dense_head
        t_nodes = decode_nodes(t_head, t_head.forward_ret_dict['pred_dicts'], tn.TOPK, tn.NMS_THRESH, tn.POST_MAX, False)
        t_feat = tb['spatial_features_2d']
    s_feat = student_batch['spatial_features_2d']
    args = (head.point_cloud_range, head.voxel_size, head.feature_map_stride, g)
    student = [(b, l, bev_box_pool(s_feat[k], b, *args)) for k, (b, l) in enumerate(s_nodes)]
    teacher = [(b, l, bev_box_pool(t_feat[k], b, *args)) for k, (b, l) in enumerate(t_nodes)]
    return oga_loss(student, teacher, cfg)


def ema_update(ema_model, model, alpha):
    """DTS train_mt_utils.update_ema_variables: ema = alpha * ema + (1 - alpha) * student over parameters() only, every
    optimizer step; buffers (BatchNorm statistics) are not averaged, so the eval-mode teacher keeps the statistics it
    started with (the source model's), as released."""
    import torch
    src = model.module if hasattr(model, 'module') else model
    with torch.no_grad():
        for e, p in zip(ema_model.parameters(), src.parameters()):
            e.mul_(alpha).add_(p.detach(), alpha=1 - alpha)
