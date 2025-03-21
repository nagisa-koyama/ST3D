import os
import subprocess

common_suffix = "post_MIRU2025_hist_517oe15r"
common_platform = "-platform offscreen"
common_args = "--batch_size 20"

dict_cfg_waymo2kitti_default = {
    "name": f"train_waymo2kitti_default_{common_suffix}",
    # "name": "",
    "script": 'train.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_basebev_multi_waymo2kitti_car_ped_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_waymo2kitti_point_calibrated = {
    "name": f"train_waymo2kitti_point_calibrated_{common_suffix}",
    # "name": "",
    "script": 'train.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_basebev_multi_waymo2kitti_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_waymo2kitti_point_label_calibrated = {
    "name": f"train_waymo2kitti_point_label_calibrated_{common_suffix}",
    # "name": "",
    "script": 'train.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_basebev_multi_waymo2kitti_car_ped_point_label_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_waymo2nuscenes_default = {
    "name": f"train_waymo2nuscenes_default_{common_suffix}",
    # "name": "",
    "script": 'train.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_basebev_multi_waymo2nuscenes_car_ped_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_waymo2nuscenes_point_calibrated = {
    "name": f"train_waymo2nuscenes_point_calibrated_{common_suffix}",
    # "name": "",
    "script": 'train.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_basebev_multi_waymo2nuscenes_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_waymo2nuscenes_point_label_calibrated = {
    "name": f"train_waymo2nuscenes_point_label_calibrated_{common_suffix}",
    # "name": "",
    "script": 'train.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_basebev_multi_waymo2nuscenes_car_ped_point_label_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfgs = [
    dict_cfg_waymo2kitti_default,
    dict_cfg_waymo2kitti_point_calibrated,
    dict_cfg_waymo2kitti_point_label_calibrated,
    dict_cfg_waymo2nuscenes_default,
    dict_cfg_waymo2nuscenes_point_calibrated,
    dict_cfg_waymo2nuscenes_point_label_calibrated,
]

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
