import copy
import os
import pickle
import time

import mayavi.mlab as mlab
import numpy as np
import torch
import tqdm
import wandb

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.model_utils.dsnorm import set_ds_target


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, args=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
        model.apply(set_ds_target)

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}
        num_pred = 0
        for pred in pred_dicts:
            num_pred += len(pred['pred_boxes'])
        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            if i == 0:
                mlab.options.offscreen = True
                first_elem_index = 0
                first_elem_mask = batch_dict['points'][:, 0] == first_elem_index
                dataset.__vis__(
                    points=batch_dict['points'][first_elem_mask, 1:], gt_boxes=batch_dict['gt_boxes'][first_elem_index],
                    ref_boxes=annos[first_elem_index]['boxes_lidar'],
                    scores=annos[first_elem_index]['score']
                )
                filename = "scene_val_epoch{}_{}.png".format(epoch_id, dataset.dataset_ontology)
                mlab.savefig(filename=filename)
                wandb.save(filename)
                wandb.log({'val/{}/scene'.format(dataset.dataset_ontology): wandb.Image(filename)})

            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    logger.info('gt_num_cnt: %.d).' % gt_num_cnt)
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    logger.info('len(det_annots): %.d).' % len(det_annos))
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    ret_dict['num_of_avg_predictions'] = total_pred_objects / max(1, len(det_annos))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    class_names_for_evaluation = class_names
    eval_det_annos = copy.deepcopy(det_annos)

    # Multi-dataset setup.
    if ":" in class_names[0]:
        class_names_for_evaluation = []
        for cls in class_names:
            ontology, label = cls.split(":")
            if ontology == dataset.dataset_ontology:
                class_names_for_evaluation.append(label)
        for index in range(len(eval_det_annos)):
            for index2 in range(len(eval_det_annos[index]['name'])):
                # Remap only dataset-specific inferences to be evaluated. Oher inferences will be ignored for metrics computation.
                if dataset.dataset_ontology in eval_det_annos[index]['name'][index2]:
                    eval_det_annos[index]['name'][index2] = eval_det_annos[index]['name'][index2].split(":")[-1]
            assert len(eval_det_annos[index]['boxes_lidar']) == len(eval_det_annos[index]['score'])

    logger.info('datset.dataset_ontology in eval_utils is %s' % dataset.dataset_ontology)
    logger.info('class_names in eval_utils is %s' % class_names)
    logger.info('class_names_for_evaluation in eval_utils is %s' % class_names_for_evaluation)

    result_str, result_dict = dataset.evaluation(
        eval_det_annos, class_names_for_evaluation,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    # add avg predicted number of objects to tensorboard log
    ret_dict['eval_avg_pred_bboxes'] = total_pred_objects / max(1, len(det_annos))

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    for item in ret_dict.items():
        wandb.log({'val/' + dataset.dataset_ontology + '/' + item[0] : item[1]})

    return ret_dict


if __name__ == '__main__':
    pass
