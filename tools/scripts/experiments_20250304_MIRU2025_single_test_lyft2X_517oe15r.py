import os
import subprocess

common_suffix = "MIRU2025_hist_517oe15r"
common_platform = "-platform offscreen"
common_args = "--batch_size 20"


dict_cfg_lyft2kitti_point = {
    "name": f"test_lyft2kitti_point_calibrated_{common_suffix}_k9wrh8b0",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2kitti_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250303_152921-k9wrh8b0/files/ckpt",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_lyft2kitti_point_label = {
    "name": f"test_lyft2kitti_point_label_calibrated_{common_suffix}_io1npo3n",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2kitti_car_ped_point_label_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250303_153637-io1npo3n/files/ckpt",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_lyft2nuscenes_point_label = {
    "name": f"test_lyft2nuscenes_point_label_calibrated_{common_suffix}_ggpm88cg",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2nuscenes_car_ped_point_label_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250303_153658-ggpm88cg/files/ckpt",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_lyft2nuscenes_point = {
    "name": f"test_lyft2nuscenes_point_calibrated_{common_suffix}_g9tp422c",
    "script": 'test.py',
    "cfg_file": "cfgs/da-MIRU2025/second_old_anchor_basebev_multi_lyft2nuscenes_car_ped_point_calibrated.yaml",
    "teacher_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250303_152923-g9tp422c/files/ckpt",
    "args": common_args,
    "platform": common_platform,
}

dict_cfgs = [dict_cfg_lyft2kitti_point, dict_cfg_lyft2kitti_point_label, dict_cfg_lyft2nuscenes_point_label, dict_cfg_lyft2nuscenes_point]

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
