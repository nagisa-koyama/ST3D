#!/usr/bin/env bash

python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_secondnetiou_base_bev_lyft_car_ped_21epochs.yaml
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_st3d_naive_secondnetiou_base_bev_lyft_to_kitti_car_ped.yaml --pretrained_model /storage/wandb/run-20240429_165129-bx33byw7/files/checkpoint_epoch_21.pth
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_st3d_naive_secondnetiou_base_bev_lyft_to_nuscenes_car_ped.yaml --pretrained_model /storage/wandb/run-20240429_165129-bx33byw7/files/checkpoint_epoch_21.pth
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_secondnetiou_base_bev_lyft_car_ped_21epochs.yaml --ckpt /storage/wandb/run-20240429_165129-bx33byw7/files/checkpoint_epoch_21.pth