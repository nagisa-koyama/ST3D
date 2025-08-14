import os
import subprocess

common_prefix = "post-MIRU2025"
common_suffix = "point_label_calibrated_equal_sample_weight"
common_args = "--batch_size 20"
# common_platform = "-platform offscreen"
common_platform = ""


dict_cfg_lyft_kitti2nuscenes_default = {
    "name": f"{common_prefix}_lyft_kitti2nuscenes_{common_suffix}",
    # "name": "",
    "script": 'train.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_basebev_multi_lyft_kitti2nuscenes_car_ped_point_label_calibrated_backward_together_off.yaml",
    "teacher_ckpt": "",
    "pretrained_ckpt": "",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfgs = [
    dict_cfg_lyft_kitti2nuscenes_default
]

for dict_cfg in dict_cfgs:
    cmd = "python3"
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
    if dict_cfg["pretrained_ckpt"]:
        cmd += " --pretrained_model " + dict_cfg["pretrained_ckpt"]
    if dict_cfg["ckpt_dir"]:
        cmd += " --ckpt_dir " + dict_cfg["ckpt_dir"] + " --eval_all"

    print(cmd)
    subprocess.call(cmd.split())
