import os
import subprocess

common_prefix = "train_lyft2nuscenes"
common_prefix2 = "test_lyft2nuscenes"
common_suffix = "fov_only_off_common_augv2_100epochs"
common_suffix2 = "fov_only_off_common_augv2_50epochs"
common_suffix3 = "fov_only_off_common_augv2_25epochs"
common_platform = "-platform offscreen"

dict_cfg_shift_coor = {
    "name": f"{common_prefix2}_shift_coor_lyft_nuscenes_{common_suffix}",
    "script": 'test.py',
    "cfg_file": "cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_lyft_nuscenes_coor.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250119_081031-lnqisgyg/files/ckpt/", # shift coor lyft model
    "args": "--batch_size 16",
    "platform": common_platform,
}

dict_cfg_point_label_v9 = {
    "name": f"{common_prefix}_point_label_calibrated_v9_{common_suffix2}",
    "script": 'train.py',
    "cfg_file": "cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_label_calibrated_v9.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": "--batch_size 16 --epochs 50",
    "platform": common_platform,
}

dict_cfg_no_shift_coor = {
    "name": f"{common_prefix2}_no_shift_coor_lyft_nuscenes_{common_suffix}_pnevxgo9",
    "script": 'test.py',
    "cfg_file": "cfgs/da-waymo-lyft-pandaset-kitti-to-nuscenes_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_lyft_nuscenes_no_shift_coor.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250119_080940-pnevxgo9/files/ckpt/", # no shift coor lyft model
    "args": "--batch_size 16",
    "platform": common_platform,
}

# dict_cfgs = [dict_cfg_shift_coor, dict_cfg_point_label_v9]
dict_cfgs = [dict_cfg_no_shift_coor]
# dict_cfgs = [dict_cfg_point_label_v9]

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
    if dict_cfg["ckpt_dir"]:
        cmd += " --ckpt_dir " + dict_cfg["ckpt_dir"] + " --eval_all"

    print(cmd)
    subprocess.call(cmd.split())
