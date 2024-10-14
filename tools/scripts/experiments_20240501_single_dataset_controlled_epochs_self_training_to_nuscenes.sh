#!/usr/bin/env bash

# Added above models as pretrained_model_teacher.
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_st3d_base_bev_lyft_to_nuscenes_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_st3d_base_bev_pandaset_to_nuscenes_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240430_002527-1ujfqgpz/files/checkpoint_epoch_81.pth
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_st3d_base_bev_waymo_to_nuscenes_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240430_142542-4oyn47qk/files/checkpoint_epoch_5.pth
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_st3d_base_bev_kitti_to_nuscenes_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240501_025631-trex14j2/files/checkpoint_epoch_106.pth

