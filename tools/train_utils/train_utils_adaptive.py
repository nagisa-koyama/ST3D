import glob
import os

import torch
import torch.distributed as dist
import tqdm
import wandb
from torch.nn.utils import clip_grad_norm_

from .train_utils import checkpoint_state, save_checkpoint


def train_one_epoch_adaptive(model, optimizer, source_loader, target_loader, model_func, lr_scheduler,
                              grl_scheduler, accumulated_iter, optim_cfg, rank, tbar, total_it_each_epoch,
                              source_dataloader_iter, target_dataloader_iter, leave_pbar=False, epoch_id=None):
    """Dual-dataloader adversarial training loop (UADA3D `adaptive_train.py::train_one_epoch` port).

    Each iteration pulls one source-domain batch (domain=0, gets detection loss + discriminator
    loss) and one target-domain batch (domain=1, discriminator loss only), matching UADA3D's
    second-rospm-C / centerpoint-rospm-C training procedure.
    """
    if total_it_each_epoch == len(source_loader):
        source_dataloader_iter = iter(source_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            source_batch = next(source_dataloader_iter)
        except StopIteration:
            source_dataloader_iter = iter(source_loader)
            source_batch = next(source_dataloader_iter)
        try:
            target_batch = next(target_dataloader_iter)
        except StopIteration:
            target_dataloader_iter = iter(target_loader)
            target_batch = next(target_dataloader_iter)

        grl_coeff = grl_scheduler.step(accumulated_iter)
        lr_scheduler.step(accumulated_iter)
        try:
            cur_lr = float(optimizer.lr)
        except AttributeError:
            cur_lr = optimizer.param_groups[0]['lr']

        model.train()
        optimizer.zero_grad()

        source_batch['domain'] = 0
        target_batch['domain'] = 1
        source_batch['grl_coeff'] = grl_coeff
        target_batch['grl_coeff'] = grl_coeff

        source_ret = model_func(model, source_batch)
        target_ret = model_func(model, target_batch)

        tb_dict = {}
        tb_dict.update({'source/' + k: v for k, v in source_ret.tb_dict.items()})
        tb_dict.update({'target/' + k: v for k, v in target_ret.tb_dict.items()})

        loss = source_ret.loss + target_ret.loss
        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        accumulated_iter += 1

        disp_dict = {'loss': loss.item(), 'lr': cur_lr, 'grl': grl_coeff}

        if rank == 0:
            pbar.update()
            pbar.set_postfix(disp_dict)
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if epoch_id is not None:
                wandb.log({'train/epoch': epoch_id})
            wandb.log({'train/loss': loss, 'train/learning_rate': cur_lr, 'train/grl_coeff': grl_coeff})
            for key, val in tb_dict.items():
                wandb.log({'train/' + key: val})

    if rank == 0:
        pbar.close()
    return accumulated_iter, source_dataloader_iter, target_dataloader_iter


def train_model_adaptive(model, optimizer, source_loader, target_loader, model_func, lr_scheduler,
                         grl_scheduler, optim_cfg, start_epoch, total_epochs, start_iter, rank, tb_log,
                         ckpt_save_dir, source_sampler=None, target_sampler=None, lr_warmup_scheduler=None,
                         ckpt_save_interval=1, max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(source_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(source_loader.dataset, 'merge_all_iters_to_one_epoch')
            source_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(source_loader) // max(total_epochs, 1)

        source_dataloader_iter = iter(source_loader)
        target_dataloader_iter = iter(target_loader)
        for cur_epoch in tbar:
            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)
            if target_sampler is not None:
                target_sampler.set_epoch(cur_epoch)

            cur_scheduler = lr_scheduler
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.get('WARMUP_EPOCH', -1):
                cur_scheduler = lr_warmup_scheduler

            accumulated_iter, source_dataloader_iter, target_dataloader_iter = train_one_epoch_adaptive(
                model, optimizer, source_loader, target_loader, model_func,
                lr_scheduler=cur_scheduler, grl_scheduler=grl_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, total_it_each_epoch=total_it_each_epoch,
                source_dataloader_iter=source_dataloader_iter, target_dataloader_iter=target_dataloader_iter,
                leave_pbar=(cur_epoch + 1 == total_epochs), epoch_id=cur_epoch
            )

            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0:
                if dist.is_initialized():
                    dist.barrier()
                if rank == 0:
                    ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                    ckpt_list.sort(key=os.path.getmtime)

                    if ckpt_list.__len__() >= max_ckpt_save_num:
                        for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                            os.remove(ckpt_list[cur_file_idx])

                    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                    )
                if dist.is_initialized():
                    dist.barrier()
