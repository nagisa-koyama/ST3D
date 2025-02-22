import os
import subprocess

# teacher_ckpt = "/storage/wandb/run-20241225_094941-ys0v1q2s/files/checkpoint_epoch_21.pth"
teacher_ckpt = "/storage/wandb/run-20250111_151219-o0ilt19d/files/checkpoint_epoch_21.pth"
common_prefix = "self_train_lyft2kitti"
common_suffix = "point_peak_offset_pred_shift_fov_only_off_common_aug_ys0v1q2s"
common_args = "--batch_size 8 --epochs 100"
common_platform = "-platform offscreen"

dict_cfg_dann_target = {
    "name": f"{common_prefix}_dann_target_car_ped_{common_suffix}",
    "cfg_file": "cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_and_target_only_car_ped_point_peak_offset.yaml",
    "teacher_ckpt": teacher_ckpt,
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_dann_source = {
    "name": f"{common_prefix}_dann_source_car_ped_{common_suffix}",
    "cfg_file": "cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_and_source_only_car_ped_point_peak_offset.yaml",
    "teacher_ckpt": teacher_ckpt,
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_target = {
    "name": f"{common_prefix}_target_car_ped_{common_suffix}",
    "cfg_file": "cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_target_only_car_ped_point_peak_offset.yaml",
    "teacher_ckpt": teacher_ckpt,
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_source = {
    "name": f"{common_prefix}_source_car_ped_{common_suffix}",
    "cfg_file": "cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_source_only_car_ped_point_peak_offset.yaml",
    "teacher_ckpt": teacher_ckpt,
    "args": common_args,
    "platform": common_platform,
}

# for dict_cfg in [dict_cfg_dann_target, dict_cfg_dann_source]:
for dict_cfg in [dict_cfg_target, dict_cfg_source]:
    cmd = "python train.py --run_name " + dict_cfg["name"] + " --cfg_file " + dict_cfg["cfg_file"] + \
        " --pretrained_model_teacher " + dict_cfg["teacher_ckpt"] + " " + dict_cfg["args"] + " " + dict_cfg["platform"]

    print(cmd)
    subprocess.call(cmd.split())
