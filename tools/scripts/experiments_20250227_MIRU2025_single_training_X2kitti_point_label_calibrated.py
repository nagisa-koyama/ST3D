import os
import subprocess

common_suffix = "MIRU2025_point_label_calibrated"
common_platform = "-platform offscreen"
common_args = "--batch_size 20"

dict_cfg_lyft2kitti = {
    "name": f"train_lyft2kitti_{common_suffix}",
    "script": 'train.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2kitti_car_ped_point_label_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_nuscenes2kitti = {
    "name": f"train_nuscenes2kitti_{common_suffix}",
    "script": 'train.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_nuscenes2kitti_car_ped_point_label_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_pandaset2kitti = {
    "name": f"train_pandaset2kitti_{common_suffix}",
    "script": 'train.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_pandaset2kitti_car_ped_point_label_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_waymo2kitti = {
    "name": f"train_waymo2kitti_{common_suffix}",
    "script": 'train.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_waymo2kitti_car_ped_point_label_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

# dict_cfg_kitti2nuscenes = {
#     "name": f"train_2nuscenes_{common_suffix}",
#     "script": 'train.py',
#     "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_kitti2nuscenes_car_ped_point_calibrated.yaml",
#     "teacher_ckpt": "",
#     "ckpt_dir": "",
#     "args": common_args,
#     "platform": common_platform,
# }


dict_cfgs = [dict_cfg_lyft2kitti, dict_cfg_nuscenes2kitti, dict_cfg_waymo2kitti, dict_cfg_pandaset2kitti] 
# dict_cfgs = [dict_cfg_lyft2kitti, dict_cfg_nuscenes2kitti]
# dict_cfgs = [dict_cfg_pandaset2kitti, dict_cfg_waymo2kitti]

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
