import os
import subprocess

common_suffix = "MIRU2025_point_calibrated_hist_517oe15r"
common_platform = "-platform offscreen"
common_args = "--batch_size 20"


# https://wandb.ai/nagisa/st3d/runs/z1ordcdb/overview
dict_cfg_kitti2nuscenes = {
    "name": f"test_kitti2nuscenes_{common_suffix}_z1ordcdb",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_kitti2nuscenes_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "/storage/wandb/run-20250305_014703-z1ordcdb/files/ckpt/checkpoint_epoch_100.pth",
    "args": common_args,
    "platform": common_platform,
}

dict_cfgs = [dict_cfg_kitti2nuscenes]

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
