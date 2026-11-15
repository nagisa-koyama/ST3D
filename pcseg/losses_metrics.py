"""Cross-entropy with IGNORE and a confusion-matrix mIoU. Both mask `target >= 0` BEFORE using a label (never index or
bincount with IGNORE = -1)."""
import numpy as np
import torch
import torch.nn.functional as F


def ce_loss(logits, target):
    """Mean cross-entropy over non-ignored points; 0 (finite, zero gradient) when every point is ignored, where
    F.cross_entropy(ignore_index=-1) returns nan."""
    valid = target >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], target[valid])


class ConfusionMatrix:
    """rows = ground truth, columns = prediction, accumulated over a split."""

    def __init__(self, num_classes):
        self.n = int(num_classes)
        self.mat = np.zeros((self.n, self.n), dtype=np.int64)

    def update(self, pred, target):
        pred = np.asarray(pred).reshape(-1).astype(np.int64)
        target = np.asarray(target).reshape(-1).astype(np.int64)
        assert pred.shape == target.shape
        valid = target >= 0
        if np.any(target[valid] >= self.n):
            raise ValueError('target class above num_classes')
        if np.any((pred[valid] < 0) | (pred[valid] >= self.n)):
            raise ValueError('prediction outside [0, num_classes)')
        self.mat += np.bincount(target[valid] * self.n + pred[valid], minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self):
        """Per-class IoU; nan for a class with no ground-truth point AND no prediction."""
        tp = np.diag(self.mat).astype(np.float64)
        union = self.mat.sum(0) + self.mat.sum(1) - tp
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(union > 0, tp / union, np.nan)

    def miou(self, require_gt=True):
        """Mean IoU over classes present in the ground truth (require_gt=True; a class that is only predicted is then
        left out: report it separately) or over every class with a non-empty union."""
        iou = self.iou()
        keep = self.mat.sum(1) > 0 if require_gt else ~np.isnan(iou)
        return float(np.nanmean(np.where(keep, iou, np.nan))) if keep.any() else float('nan')
