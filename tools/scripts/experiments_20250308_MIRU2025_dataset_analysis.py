import os
import subprocess

import os
import subprocess

common_prefix = "MIRU2025_dataset_analysis"
common_suffix = "100samples"
common_args = "--is_train"
common_platform = "-platform offscreen"

dict_cfg_default = {
    "name": f"{common_prefix}_default_{common_suffix}",
    # "name": "",
    "script": 'dataset_analysis.py',
    "cfg_file": "cfgs/da-MIRU2025/dataset_analysis_default.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args + " --figure_suffix default",
    "platform": common_platform,
}

dict_cfg_point_label_calibrated = {
    "name": f"{common_prefix}_point_label_calibrated_517oe15r_to_nuscenes_{common_suffix}",
    "script": 'dataset_analysis.py',
    "cfg_file": "cfgs/da-MIRU2025/dataset_analysis_point_label_calibrated_517oe15r_to_nuscenes.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args + " --figure_suffix point_label_calibrated_517oe15r_to_nuscenes",
    "platform": common_platform,
}

dict_cfgs = [
    dict_cfg_default, 
    dict_cfg_point_label_calibrated
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

