#!/usr/bin/env bash
python train.py --cfg_file cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_waymo_lyft_pandaset_nuscenes_to_kitti_car_ped.yaml --pretrained_model_teacher /storage/wandb/run-20240311_160001-ki04bs16/files/checkpoint_epoch_5.pth --batch_size 2 --epochs 1 -platform offscreen
