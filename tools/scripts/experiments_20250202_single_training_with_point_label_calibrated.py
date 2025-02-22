import os
import subprocess

common_prefix = "train_lyft"
common_prefix2 = "test_lyft"
common_suffix = "fov_only_off_common_augv2_100epochs"
common_suffix2 = "fov_only_off_common_augv2_50epochs"
common_suffix3 = "fov_only_off_common_augv2_25epochs"
common_args = "--batch_size 16 --epochs 100"
common_platform = "-platform offscreen"

dict_cfg_shift_point_label_v5 = {
    "name": f"{common_prefix}_shift_coor_point_label_calibrated_v5_num25_{common_suffix}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_label_calibrated_v5.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_shift_point_label_v5_test = {
    "name": f"{common_prefix2}_shift_coor_point_label_calibrated_v5_{common_suffix}",
    "script": 'test.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_label_calibrated_v5.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250125_121148-t1tvvc36/files/ckpt",
    "args": "--batch_size 8",
    "platform": common_platform,
}

dict_cfg_shift_point_label_v6 = {
    "name": f"{common_prefix}_shift_coor_point_label_calibrated_v6_num25_{common_suffix2}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_label_calibrated_v6.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": "--batch_size 16 --epochs 50",
    "platform": common_platform,
}

dict_cfg_shift_point_label_v8 = {
    "name": f"{common_prefix}_shift_coor_point_label_calibrated_v8_{common_suffix2}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_label_calibrated_v8.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": "--batch_size 16 --epochs 50",
    "platform": common_platform,
}

dict_cfg_shift_point_label_v9 = {
    "name": f"{common_prefix}_shift_coor_point_label_calibrated_v9_{common_suffix2}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_label_calibrated_v9.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": "--batch_size 16 --epochs 50",
    "platform": common_platform,
}

dict_cfg_no_shift_coor_num_points10 = {
    "name": f"{common_prefix}_no_shift_coor_{common_suffix2}",
    "script": 'train.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_no_shift_coor_num_points10.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": "--batch_size 16 --epochs 50",
    "platform": common_platform,
}

# dict_cfgs = [dict_cfg_no_shift_num_point, dict_cfg_no_shift_label]
# dict_cfgs = [dict_cfg_shift_point_label]
# dict_cfgs = [dict_cfg_shift_point_label_v5]
# dict_cfgs = [dict_cfg_shift_point_label_v5_test]
# dict_cfgs = [dict_cfg_shift_point_label_v9]
dict_cfgs = [dict_cfg_no_shift_coor_num_points10]

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
