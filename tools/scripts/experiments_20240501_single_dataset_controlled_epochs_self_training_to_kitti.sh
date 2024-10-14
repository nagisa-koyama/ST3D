#!/usr/bin/env bash
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_st3d_base_bev_lyft_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_st3d_base_bev_pandaset_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240430_002527-1ujfqgpz/files/checkpoint_epoch_81.pth
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_st3d_base_bev_nuscenes_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_203450-yitkd51y/files/checkpoint_epoch_14.pth
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_st3d_base_bev_waymo_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240430_142542-4oyn47qk/files/checkpoint_epoch_5.pth

