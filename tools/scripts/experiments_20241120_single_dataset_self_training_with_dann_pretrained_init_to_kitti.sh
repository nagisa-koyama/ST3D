#!/usr/bin/env bash

# Dann-pretrained model as init
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --pretrained_model /storage/wandb/run-20241110_145136-5zep34i6/files/checkpoint_epoch_100.pth --batch_size 8 --epochs 100 -platform offscreen
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_target_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --pretrained_model /storage/wandb/run-20241110_145136-5zep34i6/files/checkpoint_epoch_100.pth --batch_size 8 --epochs 100 -platform offscreen

# Test source only with dann-pretrained model as init
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241119_161933-8kkp9cs5/files --eval_all -platform offscreen

# Test target only with dann-pretrained model as init
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241120_011812-zpmxlrbg/files --eval_all -platform offscreen

# Train target only with dann-and-source-pretrained model as init
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_target_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --pretrained_model /storage/wandb/run-20241111_160513-x3i6rtqe/files/checkpoint_epoch_100.pth --batch_size 8 --epochs 100 -platform offscreen

# Test target only with dann-and-source-pretrained model as init
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241120_145054-nz3t1ot8/files --eval_all -platform offscreen

# Train target only with source-pretrained model as init
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_target_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --pretrained_model /storage/wandb/run-20241104_161410-e354coth/files/checkpoint_epoch_100.pth --batch_size 8 --epochs 100 -platform offscreen

# Test target only with source-pretrained model as inits
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241125_154310-q1x79h3m/files --eval_all -platform offscreen

# Train all
python train.py --cfg_file cfgs/da-waymo-lyfexit-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --batch_size 8 --epochs 100 -platform offscreen
