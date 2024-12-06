#!/usr/bin/env bash
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention_head_per_dataset/second_old_anchor_basebev_waymo_lyft_pandaset_nuscenes_car_ped.yaml --batch_size 16 --epochs 5 -platform offscreen
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention_head_per_dataset/second_old_anchor_basebev_waymo_lyft_pandaset_kitti_car_ped.yaml --batch_size 16 --epochs 5 -platform offscreen



