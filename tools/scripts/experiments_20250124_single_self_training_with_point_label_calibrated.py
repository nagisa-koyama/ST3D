import os
import subprocess

common_prefix = "self_train_lyft2kitti"
common_prefix2 = "test_lyft"
common_suffix = "point_label_calibrated_v4_fov_only_off_common_augv2_100epochs"
common_suffix2 = "fov_only_off_common_augv2_100epochs"
common_args = "--batch_size 8 --epochs 100"
common_platform = "-platform offscreen"
teacher_ckpt = "/storage/wandb/run-20250122_temp-b4tufpds/ckpt/checkpoint_epoch_87.pth"
teacher_ckpt_shift_coor = "/storage/wandb/run-20250119_081031-lnqisgyg/files/ckpt/checkpoint_epoch_100.pth"

dict_cfg_dann_target = {
    "name": f"{common_prefix}_dann_target_{common_suffix}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_and_target_only_car_ped_point_label_calibrated_v4.yaml',
    "teacher_ckpt": teacher_ckpt,
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_target = {
    "name": f"{common_prefix}_target_{common_suffix}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_target_only_car_ped_point_label_calibrated_v4.yaml',
    "teacher_ckpt": teacher_ckpt,
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_target_shift_coor = {
    "name": f"{common_prefix}_target_shift_coor_{common_suffix2}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_target_only_car_ped_shift_coor.yaml',
    "teacher_ckpt": teacher_ckpt_shift_coor,
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_target_calibrated_v8 = {
    "name": f"{common_prefix}_target_point_label_calibrated_v8_{common_suffix2}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_basebev_lyft_to_kitti_target_only_car_ped_point_label_calibrated_v8.yaml',
    "teacher_ckpt": "/storage/wandb/run-20250215_114724-q5o3iw5k/files/ckpt/checkpoint_epoch_50.pth", # lyft calibrated v8
    "pretrained_ckpt": "/storage/wandb/run-20250215_114724-q5o3iw5k/files/ckpt/checkpoint_epoch_50.pth", # lyft calibrated v8
    "ckpt_dir": "",
    "args": "--batch_size 16 --epochs 100",
    "platform": common_platform,
}

dict_cfg_dann_target_calibrated_v8 = {
    "name": f"{common_prefix}_dann_target_point_label_calibrated_v8_{common_suffix2}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_and_target_only_car_ped_point_label_calibrated_v8.yaml',
    "teacher_ckpt": "/storage/wandb/run-20250215_114724-q5o3iw5k/files/ckpt/checkpoint_epoch_50.pth", # lyft calibrated v8
    "pretrained_ckpt": "/storage/wandb/run-20250215_114724-q5o3iw5k/files/ckpt/checkpoint_epoch_50.pth", # lyft calibrated v8
    "ckpt_dir": "",
    "args": "--batch_size 16 --epochs 100",
    "platform": common_platform,
}

dict_cfg_dann_source_target_calibrated_v8 = {
    "name": f"{common_prefix}_dann_source_target_point_label_calibrated_v8_{common_suffix2}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/second_old_anchor_st3d_dann_basebev_lyft_to_kitti_dann_source_target_car_ped_point_label_calibrated_v8.yaml',
    "teacher_ckpt": "/storage/wandb/run-20250215_114724-q5o3iw5k/files/ckpt/checkpoint_epoch_50.pth", # lyft calibrated v8
    "pretrained_ckpt": "/storage/wandb/run-20250215_114724-q5o3iw5k/files/ckpt/checkpoint_epoch_50.pth", # lyft calibrated v8
    "ckpt_dir": "",
    "args": "--batch_size 16 --epochs 100",
    "platform": common_platform,
}

# dict_cfgs = [dict_cfg_dann_target]
# dict_cfgs = [dict_cfg_target]
# dict_cfgs = [dict_cfg_target_shift_coor]
# dict_cfgs = [dict_cfg_target_calibrated_v8]
# dict_cfgs = [dict_cfg_dann_target_calibrated_v8]
dict_cfgs = [dict_cfg_dann_source_target_calibrated_v8]

for dict_cfg in dict_cfgs:
    cmd = "python"
    if dict_cfg["script"]:
        cmd += " " + dict_cfg["script"]
    if dict_cfg["cfg_file"]:
        cmd += " --cfg_file " + dict_cfg["cfg_file"]
    if dict_cfg["name"]:
        cmd += " --run_name " + dict_cfg["name"]
    if dict_cfg["args"]:
        cmd += " " + dict_cfg["args"]
    if dict_cfg["platform"]:
        cmd += " " + dict_cfg["platform"]
    if dict_cfg["teacher_ckpt"]:
        cmd += " --pretrained_model_teacher " + dict_cfg["teacher_ckpt"]
    if dict_cfg["pretrained_ckpt"]:
        cmd += " --pretrained_model " + dict_cfg["pretrained_ckpt"]
    if dict_cfg["ckpt_dir"]:
        cmd += " --ckpt_dir " + dict_cfg["ckpt_dir"] + " --eval_all"

    print(cmd)
    subprocess.call(cmd.split())
