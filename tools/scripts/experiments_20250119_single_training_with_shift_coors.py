import os
import subprocess

common_prefix = "train_lyft"
common_prefix2 = "test_lyft"
common_suffix = "fov_only_off_common_augv2"
common_args = "--batch_size 8 --epochs 100"
common_platform = "-platform offscreen"

dict_cfg_no_shift = {
    "name": f"{common_prefix}_no_shift_coor_{common_suffix}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_no_shift_coor.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_lyft_shift = {
    "name": f"{common_prefix}_shift_coor_lyft_{common_suffix}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_lyft_shift_coor.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_kitti_shift = {
    "name": f"{common_prefix2}_shift_coor_kitti_{common_suffix}",
    "script": 'test.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_kitti_shift_coor.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250119_080940-pnevxgo9/files/ckpt/", # no shift coor model
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_lyft_kitti_shift = {
    "name": f"{common_prefix2}_shift_coor_lyft_kitti_{common_suffix}",
    "script": 'test.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_lyft_kitti_shift_coor.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250119_081031-lnqisgyg/files/ckpt/", # shift coor lyft model
    "args": common_args,
    "platform": common_platform,
}

# dict_cfgs = [dict_cfg_no_shift]
# dict_cfgs = [dict_cfg_lyft_shift]
# dict_cfgs = [dict_cfg_kitti_shift, dict_cfg_lyft_kitti_shift]
dict_cfgs = [dict_cfg_lyft_kitti_shift]

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
