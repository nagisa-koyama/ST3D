import glob
import os

import mayavi.mlab as mlab
import torch
import tqdm
from torch.nn.utils import clip_grad_norm_

import wandb

def train_one_epoch(model, optimizer, train_loaders, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epochs, dataloader_iters, tb_log=None, leave_pbar=False, epoch_id=None):
    # dataloader_iter = dataloader_iters[0]
    assert(len(dataloader_iters) == len(train_loaders))
    assert(len(total_it_each_epochs) == len(train_loaders))
    for index in range(len(total_it_each_epochs)):
        if total_it_each_epochs[index] == len(train_loaders[index]):
            dataloader_iters[index] = iter(train_loaders[index])

    total_it_each_epochs_aggregated = max(total_it_each_epochs)*len(total_it_each_epochs)
    print("total_it_each_epochs:", total_it_each_epochs)
    print("total_it_each_epochs_aggregated:", total_it_each_epochs_aggregated)
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epochs_aggregated, leave=leave_pbar, desc='train', dynamic_ncols=True)

    loss_total = None
    draw_scene = True
    for cur_it in range(total_it_each_epochs_aggregated):
        dataset_index = cur_it % len(dataloader_iters)
        dataset_ontology = train_loaders[dataset_index].dataset.dataset_ontology
        try:
            batch = next(dataloader_iters[dataset_index])
        except StopIteration:
            dataloader_iters[dataset_index] = iter(train_loaders[dataset_index])
            batch = next(dataloader_iters[dataset_index])
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()

        loss, tb_dict, disp_dict = model_func(model, batch)

        backward_together = optim_cfg.get('BACKWORD_TOGETHER', None)
        if backward_together:
            if dataset_index == 0:
                # Reset grad only when first dataset is pulled in.
                # This will behave as grad accumulation.
                optimizer.zero_grad()
                loss_total = loss
            else:
                loss_total += loss
            if dataset_index == len(dataloader_iters) - 1:
                # Update params only when final dataset is pulled in.
                loss_total.backward() # Fixed on Nov 2nd, 2024.
                clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                optimizer.step()
            disp_dict.update({'loss total': loss_total.item(), 'lr': cur_lr})
        else:
            optimizer.zero_grad()
            loss_total = loss
            loss_total.backward()
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()
            disp_dict.update({'loss ' + dataset_ontology: loss.item(), 'lr': cur_lr})

        accumulated_iter += 1

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                wandb.log({'train/' +  dataset_ontology + '/loss': loss})
                wandb.log({'train/' + dataset_ontology + '/learning_rate': cur_lr})
                if backward_together == True:
                    if dataset_index ==  len(dataloader_iters) - 1:
                        wandb.log({'train/loss': loss_total})
                else:
                    wandb.log({'train/loss': loss})

                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + dataset_ontology + '/' + key, val, accumulated_iter)
                    wandb.log({'train/' + dataset_ontology + '/' + key: val})

            if draw_scene == False:
                mlab.options.offscreen = True
                first_elem_index = 0
                first_elem_mask = batch['points'][:, 0] == first_elem_index
                train_loaders[dataset_index].dataset.__vis__(
                    points=batch['points'][first_elem_mask, 1:], gt_boxes=batch['gt_boxes'][first_elem_index],
                    ref_boxes=tb_dict[first_elem_index]['pred_boxes'],
                    ref_scores=tb_dict[first_elem_index]['pred_scores']
                )
                filename = "scene_epoch{}_{}.png".format(epoch_id, dataset_ontology)
                mlab.savefig(filename=filename)
                wandb.save(filename)
                wandb.log({'train/{}/scene'.format(dataset_ontology): wandb.Image(filename)})

    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, model_teacher, optimizer, train_loaders, target_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ps_label_dir,
                source_samplers=None, target_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epochs = list()
        for train_loader in train_loaders:
            total_it_each_epoch = len(train_loader)
            if merge_all_iters_to_one_epoch:
                assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
                train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
                total_it_each_epoch = len(train_loader) // max(total_epochs, 1)
            total_it_each_epochs.append(total_it_each_epoch)

        dataloader_iters = [iter(train_loader) for train_loader in train_loaders]
        for cur_epoch in tbar:
            if source_samplers is not None:
                for index in range(len(source_samplers)):
                    if source_samplers[index] is not None:
                        source_samplers[index].set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loaders, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epochs=total_it_each_epochs,
                dataloader_iters=dataloader_iters
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
                print("ckpt_name:", ckpt_name)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )
                


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
    # Save to local storage to reduce wandb strorage usage.
    wandb.save(filename)
