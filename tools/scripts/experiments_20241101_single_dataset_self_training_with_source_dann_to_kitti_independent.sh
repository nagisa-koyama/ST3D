#!/usr/bin/env bash

# Train with source only
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --batch_size 8 --epochs 100 -platform offscreen
# Test
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_only_car_ped.yaml --ckpt_dir /storage/wandb/run-20241104_161410-e354coth/files/ --eval_all -platform offscreen

# Train with target only
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_target_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --batch_size 8 --epochs 100 -platform offscreen

# Train with dann only
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --batch_size 8 --epochs 100 -platform offscreen

# Train with dann and source only
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_and_source_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --batch_size 8 --epochs 100 -platform offscreen
# Test with dann simple and source only 
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241201_161453-lgt39ghv/files --eval_all -platform offscreen

# Train with dann and target only
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_and_target_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --batch_size 8 --epochs 100 -platform offscreen
# Test with dann simple128 and target only
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241202_040354-pav37qbo/files --eval_all -platform offscreen

# Test with dann and source only
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_and_source_only_car_ped.yaml --ckpt_dir /storage/wandb/run-20241111_160513-x3i6rtqe/files --eval_all -platform offscreen

# Test with dann only
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_only_car_ped.yaml --ckpt_dir /storage/wandb/run-20241110_145136-5zep34i6/files --eval_all -platform offscreen

# Test with dann and target only
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_and_target_only_car_ped.yaml --ckpt_dir /storage/wandb/run-20241110_103232-g4qujukf/files --eval_all -platform offscreen

# python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --pretraned_model /storage/wandb/run-20241110_145136-5zep34i6/files/checkpoint_epoch_100.pth --batch_size 8 --epochs 100 -platform offscreen
# python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_target_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --pretraned_model /storage/wandb/run-20241110_145136-5zep34i6/files/checkpoint_epoch_100.pth --batch_size 8 --epochs 100 -platform offscreen
