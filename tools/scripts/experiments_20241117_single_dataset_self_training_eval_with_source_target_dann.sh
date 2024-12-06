#!/usr/bin/env bash

# Source only eval
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241104_161410-e354coth/files/ --eval_all -platform offscreen
# Target only eval
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241031_161514-nfzq3i87/files --eval_all -platform offscreen
# Dann and Source eval
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241111_160513-x3i6rtqe/files --eval_all -platform offscreen
# Dann only eval
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241110_145136-5zep34i6/files --eval_all -platform offscreen
# Dann and Target eval
python test.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241110_103232-g4qujukf/files --eval_all -platform offscreen

