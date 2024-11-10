import torch
import os
import glob
import mayavi.mlab as mlab
import tqdm.auto as tqdm
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils
from pcdet.utils import self_training_utils
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from pcdet.models.model_utils.dsnorm import set_ds_source, set_ds_target

import wandb

from .train_utils import save_checkpoint, checkpoint_state


def train_one_epoch_st(model, optimizer, source_readers, target_loader, model_func, lr_scheduler,
                       accumulated_iter, optim_cfg, rank, tbar, total_it_each_epoch,
                       dataloader_iter, tb_log=None, leave_pbar=False, ema_model=None, cur_epoch=None):
    if total_it_each_epoch == len(target_loader):
        dataloader_iter = iter(target_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    ps_bbox_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    ign_ps_bbox_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    loss_meter = common_utils.AverageMeter()
    st_loss_meter = common_utils.AverageMeter()

    disp_dict = {}

    draw_scene = True
    for cur_it in range(total_it_each_epoch):
        lr_scheduler.step(accumulated_iter)
        try:
            cur_lr = float(optimizer.param_groups[0]['lr'])
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        model.train()
        optimizer.zero_grad()
        backward_together_src = cfg.SELF_TRAIN.SRC.get('BACKWARD_TOGETHER', None)
        backward_together_tar = cfg.SELF_TRAIN.TAR.get('BACKWARD_TOGETHER', None)

        loss_total = None
        dann_loss_total = None
        domain_preds_accuracy = None
        if cfg.SELF_TRAIN.SRC.USE_DATA:
            for source_index in range(len(source_readers)):
                source_index = cur_it % len(source_readers)
                source_ontology = source_readers[source_index].dataloader.dataset.dataset_ontology

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
                loss_total = loss if loss_total is None else loss_total + loss

                # dann_loss is summed with tar later.
                if dann_loss is not None:
                    dann_loss = cfg.SELF_TRAIN.SRC.get('LOSS_WEIGHT', 1.0) * dann_loss
                    dann_loss_total = dann_loss if dann_loss_total is None else dann_loss_total + dann_loss

                # If backward together, postpone backward to the end of the loop
                if not backward_together_src:
                    # Here, we do backward for each source separately.
                    loss.backward()
                    disp_dict.update({'src_loss_' + source_ontology: "{:.2f}".format(loss.item())})
                    if not cfg.SELF_TRAIN.SRC.get('USE_GRAD', None):
                        optimizer.zero_grad()

                if rank == 0:
                    wandb.log({'train/' + source_ontology + '/loss': loss})
                    wandb.log({'train/' + source_ontology + '/learning_rate': cur_lr})
                    for key, val in tb_dict.items():
                        wandb.log({'train/' + source_ontology + '/' + key: val})
                        if key == 'domain_preds_accuracy':
                            weight = 0.5 / len(source_readers)
                            weighted_domain_preds_accuracy = val * weight
                            if domain_preds_accuracy:
                                domain_preds_accuracy += weighted_domain_preds_accuracy
                            else:
                                domain_preds_accuracy = weighted_domain_preds_accuracy

            assert loss_total is not None
            disp_dict.update({'src_loss': "{:.2f}".format(loss_total.item())})
            wandb.log({'train/' + 'src_loss_total': loss_total})
            wandb.log({'train/' + 'src_loss_total_learning_rate': cur_lr})

            # If backward together with sources is true, do backward here, but postpones if backward together with target.
            if backward_together_src and not backward_together_tar:
                loss_total.backward()
                if not cfg.SELF_TRAIN.SRC.get('USE_GRAD', None):
                    optimizer.zero_grad()

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

            loss_total = st_loss if loss_total is None else loss_total + st_loss

            if st_dann_loss:
                st_dann_loss = cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0) * st_dann_loss
                assert (dann_loss_total, "dann_loss should be summed in both SELF_TRAIN.SRC and TAR")
                dann_loss_total += st_dann_loss
                # Reflects dann_loss into loss_total here.
                loss_total += dann_loss_total
            else:
                assert (dann_loss_total is None, "dann_loss should not be summed only in SELF_TRAIN.SRC")

            # If backward together with target is true, do backward with loss_total here.
            if backward_together_tar:
                loss_total.backward()
            else:
                st_loss.backward()
                if dann_loss_total:
                    dann_loss_total.backward()

            # count number of used ps bboxes in this batch
            pos_pseudo_bbox = target_batch['pos_ps_bbox'].mean(dim=0).cpu().numpy()
            ign_pseudo_bbox = target_batch['ign_ps_bbox'].mean(dim=0).cpu().numpy()
            ps_bbox_nmeter.update(pos_pseudo_bbox.tolist())
            ign_ps_bbox_nmeter.update(ign_pseudo_bbox.tolist())
            pos_ps_result = ps_bbox_nmeter.aggregate_result()
            ign_ps_result = ign_ps_bbox_nmeter.aggregate_result()

            st_tb_dict = common_utils.add_prefix_to_dict(st_tb_dict, 'st_')
            disp_dict.update(common_utils.add_prefix_to_dict(st_disp_dict, 'st_'))
            disp_dict.update({'st_loss': "{:.2f}({:.2f})".format(st_loss_meter.val, st_loss_meter.avg)})

            if dann_loss_total:
                disp_dict.update({'dann_loss': "{:.2f}".format(dann_loss_total.item())})

            assert loss_total is not None
            disp_dict.update({'loss_total': "{:.2f}".format(loss_total.item())})

            if rank == 0 and draw_scene == False:
                with torch.no_grad():
                    model.eval()
                    # load_data_to_gpu(target_batch)
                    pred_dicts, _ = model.forward(target_batch)
                    # print("pred_dicts in train_st_utils:", pred_dicts)

                    mlab.options.offscreen = True
                    first_elem_index = 0
                    first_elem_mask = target_batch['points'][:, 0] == first_elem_index
                    gt_scores = None
                    if target_batch.keys().__contains__('gt_scores'):
                        gt_scores = target_batch['gt_scores'][first_elem_index]
                    target_loader.dataset.__vis__(
                        points=target_batch['points'][first_elem_mask,
                                                      1:], gt_boxes=target_batch['gt_boxes'][first_elem_index],
                        ref_boxes=pred_dicts[0]['pred_boxes'], gt_scores=gt_scores, ref_scores=pred_dicts[0]['pred_scores']
                    )
                    filename = "scene_self_train_epoch{}_{}.png".format(
                        cur_epoch, target_loader.dataset.dataset_ontology)
                    mlab.savefig(filename=filename)
                    wandb.save(filename)
                    wandb.log({'train/{}/self_train_scene'.format(target_loader.dataset.dataset_ontology): wandb.Image(filename)})
                    model.train()
                    draw_scene = True

        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        accumulated_iter += 1

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            if cfg.SELF_TRAIN.TAR.USE_DATA:
                pbar.set_postfix(dict(total_it=accumulated_iter, pos_ps_box=pos_ps_result,
                                      ign_ps_box=ign_ps_result))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            wandb.log({'train/learning_rate': cur_lr})

            if cfg.SELF_TRAIN.TAR.USE_DATA:
                wandb.log({'train/st_loss': st_loss})
                for key, val in st_tb_dict.items():
                    wandb.log({'train/' + key: val})
                    if key == 'st_domain_preds_accuracy':
                        weight = 0.5
                        weighted_domain_preds_accuracy = val * weight
                        if domain_preds_accuracy:
                            domain_preds_accuracy += weighted_domain_preds_accuracy
                        else:
                            domain_preds_accuracy = weighted_domain_preds_accuracy

            if dann_loss_total:
                wandb.log({'train/dann_loss': dann_loss_total})
            if domain_preds_accuracy:
                wandb.log({'train/domain_preds_weighted_accuracy': domain_preds_accuracy})

            assert loss_total is not None
            wandb.log({'train/' + 'loss_total': loss_total})
            wandb.log({'train/' + 'loss_total_learning_rate': cur_lr})

    if rank == 0:
        pbar.close()
        for i, class_names in enumerate(target_loader.dataset.class_names):
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

    # Trying to support self training with muliple sources data.
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
                target_loader.dataset.data_augmentor.re_prepare(
                    augmentor_configs=None, intensity=cfg.SELF_TRAIN.PROG_AUG.SCALE)

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True,
                     leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(target_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(target_loader.dataset, 'merge_all_iters_to_one_epoch')
            target_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(target_loader) // max(total_epochs, 1)

        dataloader_iter = iter(target_loader)
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
            if (cur_epoch in cfg.SELF_TRAIN.UPDATE_PSEUDO_LABEL) or \
                    ((cur_epoch % cfg.SELF_TRAIN.UPDATE_PSEUDO_LABEL_INTERVAL == 0)
                     and cur_epoch != 0):
                target_loader.dataset.eval()
                self_training_utils.save_pseudo_label_epoch(
                    model_teacher, target_loader, rank,
                    leave_pbar=True, ps_label_dir=ps_label_dir, cur_epoch=cur_epoch
                )
                target_loader.dataset.train()

            # curriculum data augmentation
            if cfg.SELF_TRAIN.get('PROG_AUG', None) and cfg.SELF_TRAIN.PROG_AUG.ENABLED and \
                    (cur_epoch in cfg.SELF_TRAIN.PROG_AUG.UPDATE_AUG):
                target_loader.dataset.data_augmentor.re_prepare(
                    augmentor_configs=None, intensity=cfg.SELF_TRAIN.PROG_AUG.SCALE)

            accumulated_iter = train_one_epoch_st(
                model, optimizer, source_readers, target_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, ema_model=ema_model, cur_epoch=cur_epoch
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                state = checkpoint_state(model, optimizer, trained_epoch, accumulated_iter)

                save_checkpoint(state, filename=ckpt_name)
