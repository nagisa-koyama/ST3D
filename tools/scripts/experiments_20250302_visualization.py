import os
import subprocess

common_prefix = "visualize"
common_suffix = "fov_only_off_common_augv2_100epochs"
common_args = "--is_train"
common_platform = "-platform offscreen"
# teacher_ckpt = "/storage/wandb/run-20250122_temp-b4tufpds/ckpt/checkpoint_epoch_87.pth"
# teacher_ckpt_shift_coor = "/storage/wandb/run-20250119_081031-lnqisgyg/files/ckpt/checkpoint_epoch_100.pth"

# https://wandb.ai/nagisa/st3d/runs/pnevxgo9?nw=nwusernagisa
dict_cfg = {
    # "name": f"{common_prefix}_no_shift_coor_{common_suffix}_pnevxgo9",
    "name": "",
    "script": 'demo.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_kitti2nuscenes_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    # "ckpt": "/storage/wandb/run-20250119_080940-pnevxgo9/files/ckpt/checkpoint_epoch_100.pth",
    "args": common_args,
    "platform": common_platform,
}

# dict_cfgs = [dict_cfg_no_shift, dict_cfg_no_shift_num_point, dict_cfg_no_shift_label, dict_cfg_shift_point_label]
# dict_cfgs = [dict_cfg_with_fov]
dict_cfgs = [dict_cfg]

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
