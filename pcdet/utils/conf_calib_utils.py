import numpy as np
import torch
import wandb
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import box_utils

# complete this function to calculate ece


def ece_calculation_binary(prob_true, prob_pred, bin_sizes):
    # YOUR CODE HERE
    ece = 0
    for m in np.arange(len(bin_sizes)):
        ece = ece + (bin_sizes[m] / sum(bin_sizes)) * np.abs(prob_true[m] - prob_pred[m])
    return ece


def plot_reliability_diagram(prob_true, prob_pred, dataset_name, class_name, ax=None):
    # Plot the calibration curve for ResNet in comparison with what a perfectly calibrated model would look like
    if ax == None:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
    else:
        plt.sca(ax)

    plt.plot([0, 1], [0, 1], color="#FE4A49", linestyle=":", label="Perfectly calibrated model")
    plt.plot(prob_pred, prob_true, "s-", label="{} {} vs all".format(dataset_name, class_name), color="#162B37")

    plt.ylabel("Fraction of positives", fontsize=16)
    plt.xlabel("Mean predicted value", fontsize=16,)

    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, color="#B2C7D9")

    plt.tight_layout()
    filename = "reliability_diagram_{}_{}.png".format(dataset_name, class_name)
    plt.savefig(filename)
    wandb.save(filename)
    wandb.log({'val/{}/reliability_diagram_{}'.format(dataset_name, class_name): wandb.Image(filename)})


def compute_expected_calibration_error(pred_names, pred_scores, gt_names, class_names, dataset_name, num_bins=10):
    assert len(pred_names) == len(gt_names)
    assert len(pred_names) == len(pred_scores)

    ece_bin = []
    for class_name in class_names:
        eval_name_mask = pred_names == class_name
        eval_pred_scores = pred_scores[eval_name_mask]
        eval_gt_true = (gt_names[eval_name_mask] == class_name).astype(np.int32)
        print("pred_scores: ", pred_scores)
        print("pred_names: ", pred_names)
        print("gt_names: ", gt_names)
        print("max(pred_scores): ", np.max(pred_scores))
        print("min(pred_scores): ", np.min(pred_scores))
        print("max(eval_name_mask): ", np.max(eval_name_mask))
        print("max(eval_pred_scores): ", np.max(eval_pred_scores))
        print("min(eval_pred_scores): ", np.min(eval_pred_scores))
        print("max(eval_gt_true): ", np.max(eval_gt_true))
        print("min(eval_gt_true): ", np.min(eval_gt_true))
        print("eval_gt_true: ", eval_gt_true)
        print("eval_pred_scores: ", eval_pred_scores)

        prob_true, prob_pred = calibration_curve(eval_gt_true, eval_pred_scores, n_bins=num_bins)
        plot_reliability_diagram(prob_true, prob_pred, dataset_name, class_name)
        bin_sizes = np.histogram(a=eval_pred_scores, range=(0, 1), bins=len(prob_true))[0]
        ece_bin.append(ece_calculation_binary(prob_true, prob_pred, bin_sizes))

    return np.mean(ece_bin)


def match_pred_and_gt(pred_boxes, gt_boxes, match_iou_thresh=0.1, match_height=True):
    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
        pred_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, 0:7], gt_boxes[:, 0:7]) \
            if match_height else box_utils.boxes3d_nearest_bev_iou(pred_boxes[:, 0:7], gt_boxes[:, 0:7])

        pred_to_gt_argmax = torch.from_numpy(pred_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
        pred_to_gt_max = pred_by_gt_overlap[
            # np.arange(len(pred_boxes)), pred_to_gt_argmax
            torch.arange(len(pred_boxes), device=pred_boxes.device), pred_to_gt_argmax
        ]

        # gt_to_pred_argmax = pred_by_gt_overlap.cpu().numpy().argmax(axis=0)
        # gt_to_pred_max = pred_by_gt_overlap[gt_to_pred_argmax, np.arange(len(gt_boxes))]
        # empty_gt_mask = gt_to_pred_max == 0
        # gt_to_pred_max[empty_gt_mask] = -1

        # pred_with_max_overlap = (pred_by_gt_overlap == gt_to_pred_max).nonzero()[:, 0]
        # gt_inds_force = pred_to_gt_argmax[pred_with_max_overlap]
        # names[pred_with_max_overlap] = gt_classes[gt_inds_force]
        # gt_ids[pred_with_max_overlap] = gt_inds_force.int()

        pos_inds = pred_to_gt_max >= match_iou_thresh
        gt_inds_over_thresh = pred_to_gt_argmax[pos_inds]

        return pos_inds.cpu().numpy(), gt_inds_over_thresh.cpu().numpy()
    else:
        return None, None


def generate_calibration_curve(pred_annos, gt_annos, class_names, match_iou_thresh=0.1, match_height=False, dataset_name=""):
    assert len(pred_annos) == len(gt_annos)
    aggregated_pred_names = []
    aggregated_pred_scores = []
    aggregated_gt_names = []
    for pred_anno, gt_anno in zip(pred_annos, gt_annos):
        pred_boxes = torch.from_numpy(pred_anno['boxes_lidar']).cuda()
        gt_boxes = torch.from_numpy(gt_anno['gt_boxes_lidar']).cuda()
        pos_inds, gt_inds = match_pred_and_gt(pred_boxes, gt_boxes, match_iou_thresh, match_height)
        if pos_inds is None or gt_inds is None:
            continue

        pred_names = pred_anno['name']
        pred_scores = pred_anno['score']
        # gt_names = torch.from_numpy(gt_anno['index']).cuda()
        gt_names = gt_anno['name']

        matched_pred_names = pred_names[pos_inds]
        matched_pred_scores = pred_scores[pos_inds]
        matched_gt_names = gt_names[gt_inds]
        not_matched_pred_names = pred_names[~pos_inds]
        not_matched_pred_scores = pred_scores[~pos_inds]
        not_matched_gt_names = np.array(len(not_matched_pred_names) * ["DontCare"])

        aggregated_pred_names.append(matched_pred_names)
        aggregated_pred_names.append(not_matched_pred_names)
        aggregated_pred_scores.append(matched_pred_scores)
        aggregated_pred_scores.append(not_matched_pred_scores)
        aggregated_gt_names.append(matched_gt_names)
        aggregated_gt_names.append(not_matched_gt_names)

    aggregated_pred_names = np.concatenate(aggregated_pred_names, axis=0)
    aggregated_pred_scores = np.concatenate(aggregated_pred_scores, axis=0)
    aggregated_gt_names = np.concatenate(aggregated_gt_names, axis=0)

    print("ece: ", compute_expected_calibration_error(aggregated_pred_names,
          aggregated_pred_scores, aggregated_gt_names, class_names, dataset_name))
