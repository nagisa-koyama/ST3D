import os
import subprocess

import os
import subprocess

common_prefix = "check_point_calibrated_old_hists"
common_suffix = "100samples"
common_args = "--is_train"
common_platform = "-platform offscreen"

dict_cfg_kitti2nuscenes = {
    "name": f"{common_prefix}_kitti2nuscenes_{common_suffix}",
    "script": 'dataset_analysis.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_kitti2nuscenes_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_lyft2nuscenes = {
    "name": f"{common_prefix}_lyft2nuscenes_{common_suffix}",
    "script": 'dataset_analysis.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2nuscenes_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_waymo2nuscenes = {
    "name": f"{common_prefix}_waymo2nuscenes_{common_suffix}",
    "script": 'dataset_analysis.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_waymo2nuscenes_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_pandaset2nuscenes = {
    "name": f"{common_prefix}_pandaset2nuscenes_{common_suffix}",
    "script": 'dataset_analysis.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_pandaset2nuscenes_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_lyft2kitti = {
    "name": f"{common_prefix}_lyft2kitti_{common_suffix}",
    "script": 'dataset_analysis.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2kitti_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_nuscenes2kitti = {
    "name": f"{common_prefix}_nuscenes2kitti_{common_suffix}",
    "script": 'dataset_analysis.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_nuscenes2kitti_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_waymo2kitti = {
    "name": f"{common_prefix}_waymo2kitti_{common_suffix}",
    "script": 'dataset_analysis.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_waymo2kitti_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_pandaset2kitti = {
    "name": f"{common_prefix}_pandaset2kitti_{common_suffix}",
    "script": 'dataset_analysis.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_pandaset2kitti_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args,
    "platform": common_platform,
}


dict_cfgs = [
    dict_cfg_kitti2nuscenes,
    dict_cfg_lyft2nuscenes,
    dict_cfg_waymo2nuscenes,
    dict_cfg_pandaset2nuscenes,
    dict_cfg_lyft2kitti,
    dict_cfg_nuscenes2kitti,
    dict_cfg_waymo2kitti,
    dict_cfg_pandaset2kitti,
]

for dict_cfg in dict_cfgs:
    cmd = "xvfb-run -a python"
    if dict_cfg["script"]:
        cmd += " " + dict_cfg["script"]
    if dict_cfg["cfg_file"]:
        cmd += " --cfg_file " + dict_cfg["cfg_file"]
    if dict_cfg["ckpt"]:
        cmd += " --ckpt " + dict_cfg["ckpt"]
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

