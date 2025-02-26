import os
import subprocess

common_suffix = "MIRU2024_default"
common_platform = "-platform offscreen"
common_args = "--batch_size 20"

dict_cfg_lyft2kitti = {
    # "name": f"test_lyft2kitti_{common_suffix}_2p12ncu2",
    "name": "",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2kitti_car_ped_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250222_231738-2p12ncu2/files/ckpt/",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_nuscenes2kitti = {
    "name": f"test_nuscenes2kitti_{common_suffix}_5tno72se",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_nuscenes2kitti_car_ped_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250223_095354-5tno72se/files/ckpt/",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_pandaset2kitti = {
    # "name": f"test_pandaset2kitti_{common_suffix}_29mwwr9p",
    "name": "",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_pandaset2kitti_car_ped_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250222_231807-29mwwr9p/files/ckpt/",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_waymo2kitti = {
    # "name": f"test_waymo2kitti_{common_suffix}_ijkkukae",
    "name": "",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_waymo2kitti_car_ped_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250223_193159-ijkkukae//files/ckpt/",
    "args": common_args,
    "platform": common_platform,
}

# dict_cfgs = [dict_cfg_lyft2kitti, dict_cfg_nuscenes2kitti, dict_cfg_pandaset2kitti, dict_cfg_waymo2kitti]
# dict_cfgs = [dict_cfg_lyft2kitti, dict_cfg_nuscenes2kitti]
# dict_cfgs = [dict_cfg_lyft2kitti]
dict_cfgs = [dict_cfg_pandaset2kitti, dict_cfg_waymo2kitti]
# dict_cfgs = [dict_cfg_waymo2kitti]

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
