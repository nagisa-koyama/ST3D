#!/usr/bin/env bash

python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs.yaml
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_nuscenes_car_ped_14epochs.yaml
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_pandaset_car_ped_81epochs.yaml
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_waymo_car_ped_5epochs.yaml
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_kitti_car_ped_106epochs.yaml

