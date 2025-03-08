import os
import subprocess

common_suffix = "MIRU2025_default"
common_platform = "-platform offscreen"
common_args = "--batch_size 20"

dict_cfg_nuscenes2kitti_ckpt = {
    "name": f"test_nuscenes2kitti_{common_suffix}_5tno72se",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_nuscenes2kitti_car_ped_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "/storage/wandb/run-20250223_095354-5tno72se/files/ckpt/checkpoint_epoch_30.pth",
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

dict_cfgs = [dict_cfg_nuscenes2kitti_ckpt]

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
    if dict_cfg["ckpt"]:
        cmd += " --ckpt " + dict_cfg["ckpt"]

    print(cmd)
    subprocess.call(cmd.split())
