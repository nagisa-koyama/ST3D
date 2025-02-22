#!/usr/bin/env bash

common_setting=fov_only_off_common_aug_wtcwgdxj

# Train with dann and source only https://wandb.ai/nagisa/st3d/runs/oxk9u979?nw=nwusernagisa https://wandb.ai/nagisa/st3d/runs/dtjs9zj0?nw=nwusernagisa
# python train.py --run_name self_trian_lyft2kitti_dann_source_${common_setting} --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_and_source_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --batch_size 8 --epochs 100 -platform offscreen
# Test https://wandb.ai/nagisa/st3d/runs/oxk9u979?nw=nwusernagisa
#python test.py --run_name test_self_trian_lyft2kitti_dann_source_${common_setting}_oxk9u979 --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241222_163525-oxk9u979/files --eval_all -platform offscreen
# TSNE https://wandb.ai/nagisa/st3d/runs/oxk9u979?nw=nwusernagisa
#python tsne.py --run_name tsne_self_trian_lyft2kitti_dann_source_${common_setting}_oxk9u979 --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241222_163525-oxk9u979/files/checkpoint_epoch_100.pth --out_filename tsne_oxk9u979.png -platform offscreen
# Test https://wandb.ai/nagisa/st3d/runs/dtjs9zj0?nw=nwusernagisa
python test.py --run_name test_self_trian_lyft2kitti_dann_source_${common_setting}_dtjs9zj0 --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241230_140227-dtjs9zj0/files --eval_all -platform offscreen
# TSNE https://wandb.ai/nagisa/st3d/runs/dtjs9zj0?nw=nwusernagisa
python tsne.py --run_name tsne_self_trian_lyft2kitti_dann_source_${common_setting}_dtjs9zj0 --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241230_140227-dtjs9zj0/files/checkpoint_epoch_100.pth -platform offscreen

# Test fov only off https://wandb.ai/nagisa/st3d/runs/s7zhcax4?nw=nwusernagisa
#python test.py --run_name test_self_trian_lyft2kitti_dann_source_fov_only_off_wtcwgdxj_s7zhcax4 --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241222_052751-s7zhcax4/files --eval_all -platform offscreen
# TSNE
#python tsne.py --run_name tsne_self_trian_lyft2kitti_dann_source_fov_only_off_wtcwgdxj_s7zhcax4 --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241222_052751-s7zhcax4/files/checkpoint_epoch_100.pth --out_filename tsne_7zhcax4.png -platform offscreen

# Train with dann and target only
#python train.py --run_name self_train_lyft2kitti_dann_and_target_no_fov_only_common_aug_wtcwgdxj --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_and_target_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --batch_size 8 --epochs 100 -platform offscreen
# Test https://wandb.ai/nagisa/st3d/runs/x7erwwgj?nw=nwusernagisa
#python test.py --run_name test_self_train_lyft2kitti_dann_target_${common_setting}_x7erwwgj  --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241223_082008-x7erwwgj/files/ --eval_all -platform offscreen
# TSNE
#python tsne.py --run_name tsne_self_train_lyft2kitti_dann_target_${common_setting}_x7erwwgj  --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241223_082008-x7erwwgj/files/checkpoint_epoch_100.pth -platform offscreen

# Train with source only
# python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --batch_size 8 --epochs 100 -platform offscreen
# Test
#python test.py --run_name test_self_train_lyft2kitti_source_${common_setting}_e354coth --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241104_161410-e354coth/files/ --eval_all -platform offscreen
# TSNE
#python tsne.py --run_name tsne_self_train_lyft2kitti_source_${common_setting}_e354coth --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241104_161410-e354coth/files/checkpoint_epoch_100.pth -platform offscreen

# Train with target only https://wandb.ai/nagisa/st3d/runs/xmtjvfvb?nw=nwusernagisa
#python train.py --run_name self_train_lyft2kitti_target_${common_setting} --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_target_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --batch_size 8 --epochs 100 -platform offscreen
# Test 
# python test.py --run_name test_self_train_lyft2kitti_target_${common_setting}_xmtjvfvb --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt_dir /storage/wandb/run-20241225_002306-xmtjvfvb/files/ --eval_all -platform offscreen
# TSNE
#python tsne.py --run_name tsne_self_train_lyft2kitti_target_${common_setting}_xmtjvfvb --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241225_002306-xmtjvfvb/files/checkpoint_epoch_100.pth -platform offscreen

# Train with dann only https://wandb.ai/nagisa/st3d/runs/ys0v1q2s?nw=nwusernagisa
#python train.py --run_name self_train_lyft2kitti_dann_${common_setting} --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_only_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240429_154545-wtcwgdxj/files/checkpoint_epoch_21.pth --batch_size 8 --epochs 100 -platform offscreen
# TSNE
#python tsne.py --run_name tsne_self_train_lyft2kitti_dann_${common_setting}_ys0v1q2s --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_target_eval_car_ped.yaml --ckpt /storage/wandb/run-20241225_094941-ys0v1q2s/files/checkpoint_epoch_100.pth -platform offscreen

