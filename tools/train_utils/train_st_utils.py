import torch
import torch.distributed as dist
from torch.nn.functional import cosine_similarity
import os
import glob
import numpy as np
import tqdm.auto as tqdm
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils
from pcdet.utils import self_training_utils
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from pcdet.models.model_utils.dsnorm import set_ds_source, set_ds_target
from pcdet.datasets import (build_inference_dataloader, restart_persistent_workers,
                            link_foreground_calibration)

import wandb
import torchjd
from torchjd.aggregation import UPGrad, PCGrad

from .train_utils import save_checkpoint, checkpoint_state


def train_one_epoch_st(model, optimizer, source_readers, target_loader, model_func, lr_scheduler,
                       accumulated_iter, optim_cfg, rank, tbar, total_it_each_epoch,
                       dataloader_iter, tb_log=None, leave_pbar=False, ema_model=None, cur_epoch=None, logger=None):
    if total_it_each_epoch == len(target_loader):
        dataloader_iter = iter(target_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    ps_bbox_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    ign_ps_bbox_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    loss_total_meter = common_utils.AverageMeter()
    loss_meter = common_utils.AverageMeter()
    st_loss_meter = common_utils.AverageMeter()
    dann_loss_meter = common_utils.AverageMeter()

    disp_dict = {}

    draw_scene = True
    for cur_it in range(total_it_each_epoch):
        lr_scheduler.step(accumulated_iter)
        try:
            cur_lr = float(optimizer.param_groups[0]['lr'])
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        model.train()

        backward_together_src = cfg.SELF_TRAIN.SRC.get('BACKWARD_TOGETHER', None)
        backward_together_tar = cfg.SELF_TRAIN.TAR.get('BACKWARD_TOGETHER', None)
        assert backward_together_src, "backward_together_src is False, cfg.SELF_TRAIN.SRC.BACKWARD_TOGETHER: {}".format(cfg.SELF_TRAIN.SRC.BACKWARD_TOGETHER)
        assert backward_together_tar, "backward_together_tar is False, cfg.SELF_TRAIN.TAR.BACKWARD_TOGETHER: {}".format(cfg.SELF_TRAIN.TAR.BACKWARD_TOGETHER)

        loss_src_list = []
        dann_loss_list = []
        st_loss_list = []
        domain_preds_accuracy = None
        if cfg.SELF_TRAIN.SRC.USE_DATA:
            # Equal dataset-weight sampling.
            # source_index = cur_it % len(source_readers)

            # Equal sample-weight sampling.
            random_0to1 = np.random.rand()
            accum_rate = 0.0
            source_index = None
            total_it_aggregated = sum([reader.dataloader.dataset.__len__() for reader in source_readers])
            total_it_per_dataset = [reader.dataloader.dataset.__len__() for reader in source_readers]
            for index in range(len(total_it_per_dataset)):
                accum_rate += total_it_per_dataset[index] / total_it_aggregated
                if random_0to1 <= accum_rate:
                    source_index = index
                    break
            assert source_index is not None, "source_index is None, random_0to1: {}".format(random_0to1)

            source_ontology = source_readers[source_index].dataloader.dataset.dataset.dataset_ontology

            # forward source data with labels
            source_batch = source_readers[source_index].read_data()
            source_batch['domain_label'] = 0

            if cfg.SELF_TRAIN.get('DSNORM', None):
                model.apply(set_ds_source)

            if cfg.SELF_TRAIN.SRC.get('SEP_LOSS_WEIGHTS', None):
                source_batch['SEP_LOSS_WEIGHTS'] = cfg.SELF_TRAIN.SRC.SEP_LOSS_WEIGHTS

            loss, tb_dict, disp_dict, dann_loss = model_func(model, source_batch)
            loss = cfg.SELF_TRAIN.SRC.get('LOSS_WEIGHT', 1.0) * loss
            loss_meter.update(loss.item())
            loss_src_list.append(loss)

            # dann_loss is summed with tar later.
            if dann_loss is not None:
                dann_loss *= 0.5 # 0.5 is the weight for source domain
                dann_loss_list.append(dann_loss)

            if rank == 0:
                wandb.log({'train/' + source_ontology + '/loss': loss})
                wandb.log({'train/' + source_ontology + '/learning_rate': cur_lr})
                for key, val in tb_dict.items():
                    wandb.log({'train/' + source_ontology + '/' + key: val})
                    if key == 'domain_preds_accuracy':
                        domain_preds_accuracy = val * 0.5 # 0.5 is the weight for source domain

        if cfg.SELF_TRAIN.TAR.USE_DATA:
            try:
                target_batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(target_loader)
                target_batch = next(dataloader_iter)
                print('new iters')

            target_batch['domain_label'] = 1

            if cfg.SELF_TRAIN.get('DSNORM', None):
                model.apply(set_ds_target)

            if cfg.SELF_TRAIN.TAR.get('SEP_LOSS_WEIGHTS', None):
                target_batch['SEP_LOSS_WEIGHTS'] = cfg.SELF_TRAIN.TAR.SEP_LOSS_WEIGHTS

            # parameters for save pseudo label on the fly
            st_loss, st_tb_dict, st_disp_dict, st_dann_loss = model_func(model, target_batch)
            st_loss = cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0) * st_loss
            st_loss_meter.update(st_loss.item())
            st_loss_list.append(st_loss)

            if st_dann_loss:
                st_dann_loss *= 0.5  # 0.5 is the weight for target domain
                dann_loss_list.append(st_dann_loss)

            # count number of used ps bboxes in this batch
            pos_pseudo_bbox = target_batch['pos_ps_bbox'].mean(dim=0).cpu().numpy()
            ign_pseudo_bbox = target_batch['ign_ps_bbox'].mean(dim=0).cpu().numpy()
            ps_bbox_nmeter.update(pos_pseudo_bbox.tolist())
            ign_ps_bbox_nmeter.update(ign_pseudo_bbox.tolist())

            st_tb_dict = common_utils.add_prefix_to_dict(st_tb_dict, 'st_')
            disp_dict.update(common_utils.add_prefix_to_dict(st_disp_dict, 'st_'))

            if rank == 0:
                for key, val in st_tb_dict.items():
                    wandb.log({'train/' + key: val})
                    if key == 'st_domain_preds_accuracy':
                        # assert domain_preds_accuracy is not None
                        if domain_preds_accuracy is None:
                            print("Warning: domain_preds_accuracy is None when logging target domain preds accuracy.")
                            continue
                        domain_preds_accuracy += val * 0.5  # 0.5 is the weight for target domain

        # Control backward and optimization.
        # Gradient projection is opt-in per config rather than a hardcoded literal: whether it
        # was active is an ablation variable (the loss-gradient conflict it addresses is itself
        # a reported result), so it must be recorded in the run's config and W&B record instead
        # of requiring a source edit between runs.
        use_torchjd = cfg.SELF_TRAIN.get('USE_TORCHJD', False)
        # aggregator = UPGrad()
        aggregator = PCGrad()
        optimizer.zero_grad()

        loss_src_sum = sum(loss_src_list) if len(loss_src_list) > 0 else None
        st_loss_sum = sum(st_loss_list) if len(st_loss_list) > 0 else None
        dann_loss_sum = sum(dann_loss_list) if len(dann_loss_list) > 0 else None

        if not backward_together_src:
            # Here, we do backward for each source separately.
            for loss in loss_src_list:
                loss.backward()
                if not cfg.SELF_TRAIN.SRC.get('USE_GRAD', None):
                    optimizer.zero_grad()
            # Adds dann_loss here since it is paired with loss_src.
            if dann_loss_sum is not None:
                # dann_loss should be summed.
                dann_loss_sum.backward()
                if not cfg.SELF_TRAIN.SRC.get('USE_GRAD', None):
                    optimizer.zero_grad()

        if not backward_together_tar:
            # Here, we do backward for target separately.
            for st_loss in st_loss_list:
                st_loss.backward()
                if not cfg.SELF_TRAIN.TAR.get('USE_GRAD', None):
                    optimizer.zero_grad()

        if backward_together_src and not backward_together_tar:
            loss_sum = 0
            loss_sum += loss_src_sum if loss_src_sum is not None else 0
            # Adds dann_loss here since it is paired with loss_src.
            loss_sum += dann_loss_sum if dann_loss_sum is not None else 0
            loss_sum.backward()
            if not cfg.SELF_TRAIN.SRC.get('USE_GRAD', None):
                optimizer.zero_grad()

        if not backward_together_src and backward_together_tar:
            if st_loss_sum is not None:
                st_loss_sum.backward()
                if not cfg.SELF_TRAIN.TAR.get('USE_GRAD', None):
                    optimizer.zero_grad()

        if backward_together_src and backward_together_tar:
            if use_torchjd is False:
                loss_sum = 0
                loss_sum += loss_src_sum if loss_src_sum is not None else 0
                loss_sum += st_loss_sum if st_loss_sum is not None else 0
                loss_sum += dann_loss_sum if dann_loss_sum is not None else 0
                loss_sum.backward()
            else:
                assert loss_src_sum is not None
                assert st_loss_sum is not None
                assert dann_loss_sum is not None

                def print_weights(_, __, weights: torch.Tensor) -> None:
                    """Prints the extracted weights."""
                    print(f"Weights: {weights}")

                def print_gd_similarity(_, inputs: tuple[torch.Tensor, ...], aggregation: torch.Tensor) -> None:
                    """Prints the cosine similarity between the aggregation and the average gradient."""
                    matrix = inputs[0]
                    gd_output = matrix.mean(dim=0)
                    similarity_pcgrad_mean = cosine_similarity(gd_output, aggregation, dim=0)
                    similarity_pcgrad_src = cosine_similarity(matrix[0], aggregation, dim=0)
                    similarity_prgrad_dann = cosine_similarity(matrix[1], aggregation, dim=0)
                    similairty_pcgrad_target = cosine_similarity(matrix[2], aggregation, dim=0)
                    similarity_src_dann = cosine_similarity(matrix[0], matrix[1], dim=0)
                    similarity_src_target = cosine_similarity(matrix[0], matrix[2], dim=0)
                    similarity_dann_target = cosine_similarity(matrix[1], matrix[2], dim=0)
                    wandb.log({'train/grad_similarity_pcgrad_mean': similarity_pcgrad_mean.item()})
                    wandb.log({'train/grad_similarity_pcgrad_src': similarity_pcgrad_src.item()})
                    wandb.log({'train/grad_similarity_pcgrad_dann': similarity_prgrad_dann.item()})
                    wandb.log({'train/grad_similarity_pcgrad_target': similairty_pcgrad_target.item()})
                    wandb.log({'train/grad_similarity_src_dann': similarity_src_dann.item()})
                    wandb.log({'train/grad_similarity_src_target': similarity_src_target.item()})
                    wandb.log({'train/grad_similarity_dann_target': similarity_dann_target.item()})
                    # for i in range(matrix.shape[0]):
                    #     print(f"Cosine sim grad {i} and grad {(i+1)%matrix.shape[0]}: {cosine_similarity(matrix[i], matrix[(i+1)%matrix.shape[0]], dim=0).item():.4f}")
                    # for i in range(matrix.shape[0]):
                    #     print(f"Cosine sim grad {i} and PCGrad: {cosine_similarity(matrix[i], aggregation, dim=0).item():.4f}")
                    # print(f"gg_output: {gd_output}")
                    # print(f"Cosine sim grad mean and PCGrad: {similarity.item():.4f}")

                # aggregator.weighting.register_forward_hook(print_weights)
                aggregator.register_forward_hook(print_gd_similarity)

                # parallel_chunk_size = 1 is required to avoid "cannot access data pointer of Tensor" issue
                # https://torchjd.org/stable/docs/autojac/backward/
                torchjd.backward([loss_src_sum, dann_loss_sum, st_loss_sum], aggregator, parallel_chunk_size=1)

        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        accumulated_iter += 1

        # log to wandb and console
        if rank == 0:
            loss_src_total = sum(loss_src_list) if len(loss_src_list) > 0 else None
            st_loss_total = sum(st_loss_list) if len(st_loss_list) > 0 else None
            dann_loss_total = sum(dann_loss_list) if len(dann_loss_list) > 0 else None
            loss_total = 0
            if loss_src_total:
                loss_total += loss_src_total
                wandb.log({'train/src_loss_total': loss_src_total})
                loss_meter.update(loss_src_total.item())
                disp_dict.update({'src_loss': "{:.2f}({:.2f})".format(loss_meter.val, loss_meter.avg)})
            if st_loss_total:
                loss_total += st_loss_total
                wandb.log({'train/st_loss': st_loss_total})
                st_loss_meter.update(st_loss_total.item())
                disp_dict.update({'st_loss': "{:.2f}({:.2f})".format(st_loss_meter.val, st_loss_meter.avg)})
            if dann_loss_total:
                loss_total += dann_loss_total
                wandb.log({'train/dann_loss': dann_loss_total})
                dann_loss_meter.update(dann_loss_total.item())
                disp_dict.update({'dann_loss': "{:.2f}({:.2f})".format(dann_loss_meter.val, dann_loss_meter.avg)})
            
            wandb.log({'train/loss_total': loss_total})
            wandb.log({'train/learning_rate': cur_lr})
            loss_total_meter.update(loss_total.item())
            disp_dict.update({'total_loss': "{:.2f}({:.2f})".format(loss_total_meter.val, loss_total_meter.avg)})

            if cur_epoch is not None:
                wandb.log({'train/epoch': cur_epoch})

            if domain_preds_accuracy:
                wandb.log({'train/domain_preds_weighted_accuracy': domain_preds_accuracy})

            pbar.update()
            tbar.set_postfix(disp_dict)
            tbar.refresh()

        loss_src_list.clear()
        st_loss_list.clear()
        dann_loss_list.clear()
        torch.cuda.empty_cache()

        # Visualize one scene from target domain per epoch
        if rank == 0 and draw_scene == False:
            with torch.no_grad():
                model.eval()
                # load_data_to_gpu(target_batch)
                pred_dicts, _ = model.forward(target_batch)

                import mayavi.mlab as mlab
                mlab.options.offscreen = True
                first_elem_index = 0
                first_elem_mask = target_batch['points'][:, 0] == first_elem_index
                gt_scores = None
                if target_batch.keys().__contains__('gt_scores'):
                    gt_scores = target_batch['gt_scores'][first_elem_index]
                target_loader.dataset.dataset.__vis__(
                    points=target_batch['points'][first_elem_mask,
                                                    1:], gt_boxes=target_batch['gt_boxes'][first_elem_index],
                    ref_boxes=pred_dicts[0]['pred_boxes'], gt_scores=gt_scores, ref_scores=pred_dicts[0]['pred_scores']
                )
                filename = "scene_self_train_epoch{}_{}.png".format(
                    cur_epoch, target_loader.dataset.dataset.dataset_ontology)
                mlab.savefig(filename=filename)
                wandb.save(filename)
                wandb.log({'train/{}/self_train_scene'.format(target_loader.dataset.dataset.dataset_ontology): wandb.Image(filename)})
                model.train()
                draw_scene = True

    if rank == 0:
        pbar.close()
        for i, class_names in enumerate(target_loader.dataset.dataset.class_names):
            wandb.log({'ps_box/pos_' + class_names: ps_bbox_nmeter.meters[i].avg})
            wandb.log({'ps_box/ign_' + class_names: ign_ps_bbox_nmeter.meters[i].avg})

    return accumulated_iter


def train_model_st(model, model_teacher, optimizer, source_loaders, target_loader, model_func, lr_scheduler, optim_cfg,
                   start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ps_label_dir,
                   source_samplers=None, target_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                   max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None):
    accumulated_iter = start_iter

    if model_teacher is None:
        model_teacher = model  # Sharrow copy to share the memory.
    # A separately loaded teacher is never written to anywhere in this function - it is only read
    # by save_pseudo_label_epoch - so it stays exactly as it was pretrained on the source. When no
    # teacher was passed, the line above aliases it to the student, which trains.
    teacher_is_frozen = model_teacher is not model

    # Trying to support self training with muliple sources data.
    # Re-measured after EVERY pseudo-label pass - the labels define the target foreground channel,
    # so the two have to move together. With a frozen teacher that is one pass (see
    # FROZEN_TEACHER_SINGLE_PASS below); with the student acting as its own teacher it tracks.
    ps_label_fg_calibration = bool(
        cfg.DATA_CONFIG.get('HIST_DIST_ON_THE_FLY', False)
        and cfg.DATA_CONFIG.get('HIST_DIST_FOREGROUND_FROM_PSEUDO_LABELS', False))
    # The SOURCE half is measured once and reused. That is not an optimisation: after the first
    # install the source dataset is being corrected, so re-measuring it would read points that the
    # correction has already thinned and compound the rate on every refresh. The target carries no
    # correction, so it is safe - and necessary - to re-measure.
    ps_label_fg_source_hist = None

    # Regenerating pseudo-labels from a frozen teacher is deterministic: eval mode, no
    # augmentation, no weight updates, and memory voting only re-confirms boxes that matched
    # themselves at IoU 1. Every pass after the first therefore reproduces the same labels at the
    # cost of a full inference sweep over the target - 14 of the 15 passes an
    # UPDATE_PSEUDO_LABEL_INTERVAL of 2 makes over 30 epochs. Opt in to skip them.
    frozen_single_pass = bool(cfg.SELF_TRAIN.get('FROZEN_TEACHER_SINGLE_PASS', False))
    ps_labels_generated = False
    source_readers = [common_utils.DataReader(source_loader, source_sampler)
                      for source_loader, source_sampler in zip(source_loaders, source_samplers)]
    [source_reader.construct_iter() for source_reader in source_readers]

    # for continue training.
    # if already exist generated pseudo label result
    ps_pkl = self_training_utils.check_already_exsit_pseudo_label(ps_label_dir, start_epoch)
    if ps_pkl is not None:
        logger.info('==> Loading pseudo labels from {}'.format(ps_pkl))

    # for continue training
    if cfg.SELF_TRAIN.get('PROG_AUG', None) and cfg.SELF_TRAIN.PROG_AUG.ENABLED and \
            start_epoch > 0:
        for cur_epoch in range(start_epoch):
            if cur_epoch in cfg.SELF_TRAIN.PROG_AUG.UPDATE_AUG:
                target_loader.dataset.dataset.data_augmentor.re_prepare(
                    augmentor_configs=None, intensity=cfg.SELF_TRAIN.PROG_AUG.SCALE)

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True,
                     leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(target_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(target_loader.dataset.dataset, 'merge_all_iters_to_one_epoch')
            target_loader.dataset.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(target_loader) // max(total_epochs, 1)

        # Pseudo-label generation is an INFERENCE pass: it needs the target dataset in eval
        # mode, while training needs it in train mode. Persistent workers freeze whichever mode
        # they forked with, so one loader cannot serve both - give generation its own loader
        # over the same dataset object. Its workers fork on the first iter() inside
        # save_pseudo_label_epoch, i.e. after dataset.eval() below, and stay in eval mode.
        ps_gen_loader = build_inference_dataloader(target_loader, sampler=target_sampler)

        # DALI PTSN (IEEE T-RO 2024). One constant for the whole run, so it is installed here -
        # before either loader has been iterated - and never touched again: no mid-run mutation,
        # hence no re-fork needed. The scaling is gated on eval mode inside the DataProcessor, so
        # it reaches the generation pass only; the student still trains on unscaled target points
        # against pseudo-labels that were already divided by the same factor.
        #
        # SCALE comes from tools/analysis/ptsn_search.py. Its UDA legality is inherited from
        # whatever estimated the target mean size it was matched against - ROS keeps the row
        # target-free, SN does not. State which, wherever the row is reported.
        ptsn_cfg = cfg.SELF_TRAIN.get('PTSN', None)
        if ptsn_cfg is not None and ptsn_cfg.get('ENABLED', False):
            target_loader.dataset.dataset.set_ptsn_scale(ptsn_cfg.SCALE)
            if logger is not None:
                logger.info('self-training: PTSN enabled, pseudo labels will be generated at '
                            'input scale %.4f' % float(ptsn_cfg.SCALE))

        # Deliberately lazy: the training loader must NOT be iterated before the first
        # generation pass, or its workers would fork while the dataset is still in train mode
        # and then be reused for generation. Set back to None whenever the dataset is mutated
        # in a way the training workers have to observe.
        dataloader_iter = None
        for cur_epoch in tbar:
            if target_sampler is not None:
                target_sampler.set_epoch(cur_epoch)
                [source_reader.set_cur_epoch(cur_epoch) for source_reader in source_readers]

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # update pseudo label
            update_ps_label = (cur_epoch in cfg.SELF_TRAIN.UPDATE_PSEUDO_LABEL) or \
                    ((cur_epoch % cfg.SELF_TRAIN.UPDATE_PSEUDO_LABEL_INTERVAL == 0)
                     and cur_epoch != 0)
            if update_ps_label and ps_labels_generated and frozen_single_pass:
                if teacher_is_frozen:
                    update_ps_label = False
                    if logger is not None and cur_epoch == 1:
                        logger.info('self-training: FROZEN_TEACHER_SINGLE_PASS - the teacher is a '
                                    'separate frozen model, so regeneration is deterministic and '
                                    'every later pass is skipped.')
                elif logger is not None and cur_epoch == 1:
                    logger.warning('self-training: FROZEN_TEACHER_SINGLE_PASS is set but no '
                                   'separate teacher was loaded, so the student IS the teacher and '
                                   'its labels change as it trains. Regenerating as normal.')
            if update_ps_label:
                target_loader.dataset.dataset.eval()
                self_training_utils.save_pseudo_label_epoch(
                    model_teacher, ps_gen_loader, rank,
                    leave_pbar=True, ps_label_dir=ps_label_dir, cur_epoch=cur_epoch
                )
                target_loader.dataset.dataset.train()
                ps_labels_generated = True

                # save_pseudo_label_epoch rewrites the module-level PSEUDO_LABELS dict in THIS
                # process (clear() + update()), and fill_pseudo_labels reads that same global.
                # Training workers forked at the first iter() hold a copy-on-write snapshot of it,
                # so without a re-fork every update after the first is invisible to them: they keep
                # serving the epoch-0 labels for the rest of the run and
                # UPDATE_PSEUDO_LABEL_INTERVAL is silently dead. Same family as 20260921_02.
                restart_persistent_workers(target_loader)
                dataloader_iter = None

                # Foreground-aware density calibration. It needs the TARGET's boxes, and under UDA
                # those are pseudo-labels, so it cannot run before this pass - and it re-runs after
                # every pass, because the labels define the channel it measures. A uniform per-bin
                # rate cannot change a bin's foreground share, so the plain correction leaves source
                # objects starved; two channels fix that without target annotation. See
                # pcdet/datasets/point_calibration.py.
                if ps_label_fg_calibration:
                    for reader in source_readers:
                        ps_label_fg_source_hist, _ = link_foreground_calibration(
                            reader.dataloader.dataset.dataset, target_loader.dataset.dataset,
                            num_frames=cfg.DATA_CONFIG.get('HIST_DIST_FRAMES', 1000),
                            num_bins=cfg.DATA_CONFIG.get('HIST_DIST_BINS', 50),
                            max_dist=cfg.DATA_CONFIG.get('HIST_DIST_MAX_DIST', 75.0),
                            logger=logger, source_hist=ps_label_fg_source_hist)
                        # The source workers forked at construct_iter() holding the dataset as it
                        # was before this, and a forked worker never sees a later mutation. Re-fork
                        # them or the correction silently never runs.
                        restart_persistent_workers(reader.dataloader)
                        reader.construct_iter()

            # curriculum data augmentation
            if cfg.SELF_TRAIN.get('PROG_AUG', None) and cfg.SELF_TRAIN.PROG_AUG.ENABLED and \
                    (cur_epoch in cfg.SELF_TRAIN.PROG_AUG.UPDATE_AUG):
                target_loader.dataset.dataset.data_augmentor.re_prepare(
                    augmentor_configs=None, intensity=cfg.SELF_TRAIN.PROG_AUG.SCALE)
                # The training workers hold a frozen copy of the augmentor, so re_prepare()
                # alone would be a no-op for them. Force a re-fork. This is bounded to the few
                # epochs in PROG_AUG.UPDATE_AUG rather than every epoch, which is what 77b1baa
                # was avoiding when it enabled persistent_workers.
                restart_persistent_workers(target_loader)
                dataloader_iter = None

            if dataloader_iter is None:
                # Forks the training workers in TRAIN mode, after any generation pass and any
                # augmentor update for this epoch.
                dataloader_iter = iter(target_loader)

            accumulated_iter = train_one_epoch_st(
                model, optimizer, source_readers, target_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, ema_model=ema_model, cur_epoch=cur_epoch, logger=logger
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0:
                if dist.is_initialized():
                    dist.barrier()  # wait for all ranks before checkpoint saving
                if rank == 0:
                    ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                    ckpt_list.sort(key=os.path.getmtime)

                    if ckpt_list.__len__() >= max_ckpt_save_num:
                        for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                            os.remove(ckpt_list[cur_file_idx])

                    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                    state = checkpoint_state(model, optimizer, trained_epoch, accumulated_iter)

                    save_checkpoint(state, filename=ckpt_name)
                if dist.is_initialized():
                    dist.barrier()  # wait for rank 0 to finish saving
