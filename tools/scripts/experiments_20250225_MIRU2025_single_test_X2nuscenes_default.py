import os
import subprocess

common_suffix = "MIRU2024_default"
common_platform = "-platform offscreen"
common_args = "--batch_size 20"

dict_cfg_lyft2nuscenes = {
    "name": f"test_lyft2nuscenes_{common_suffix}_2p12ncu2",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2nuscenes_car_ped_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250222_231738-2p12ncu2/files/ckpt/",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_kitti2nuscenes = {
    "name": f"test_kitti2nuscenes_{common_suffix}_vk724186",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_kitti2nuscenes_car_ped_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250222_162224-vk724186/files/ckpt/",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_pandaset2nuscenes = {
    "name": f"test_pandaset2nuscenes_{common_suffix}_29mwwr9p",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_pandaset2kitti_car_ped_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250222_231807-29mwwr9p/files/ckpt/",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_waymo2nuscenes = {
    "name": f"test_waymo2nuscenes_{common_suffix}_ijkkukae",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_waymo2kitti_car_ped_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250223_193159-ijkkukae//files/ckpt/",
    "args": common_args,
    "platform": common_platform,
}

dict_cfgs = [dict_cfg_lyft2nuscenes, dict_cfg_kitti2nuscenes, dict_cfg_waymo2nuscenes, dict_cfg_pandaset2nuscenes]

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
