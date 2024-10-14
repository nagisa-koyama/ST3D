#!/usr/bin/env bash

python demo.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs.yaml --ckpt /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth
python demo.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_base_bev_pandaset_car_ped_81epochs.yaml --ckpt /storage/wandb/run-20240430_002527-1ujfqgpz/files/checkpoint_epoch_81.pth
python demo.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_base_bev_waymo_car_ped_5epochs.yaml --ckpt /storage/wandb/run-20240430_142542-4oyn47qk/files/checkpoint_epoch_5.pth
python demo.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_base_bev_kitti_car_ped_106epochs.yaml --ckpt /storage/wandb/run-20240501_025631-trex14j2/files/checkpoint_epoch_106.pth
python demo.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_base_bev_nuscenes_car_ped_14epochs.yaml --ckpt /storage/wandb/run-20240429_203450-yitkd51y/files/checkpoint_epoch_14.pth
