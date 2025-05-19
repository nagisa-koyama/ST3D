import os
import subprocess

common_prefix = "post_MIRU2025_st3d_dann_source_target_student_xyzi"
common_suffix = "point_label_calibrated_hist_517oe15r"
common_args = "--batch_size 20"
common_platform = "-platform offscreen"

dict_cfg_lyft2kitti = {
    "name": f"{common_prefix}_lyft2kitti_io1npo3n_{common_suffix}",
    # "name": "",
    "script": 'train.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_st3d_basebev_multi_lyft2kitti_dann_source_target_car_ped_point_label_calibrated_xyzi.yaml",
    "teacher_ckpt": "/storage/wandb/run-20250303_153637-io1npo3n/files/ckpt/checkpoint_epoch_50.pth",
    "pretrained_ckpt": "/storage/wandb/run-20250303_153637-io1npo3n/files/ckpt/checkpoint_epoch_50.pth",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_lyft2nuscenes = {
    "name": f"{common_prefix}_lyft2nuscenes_ggpm88cg_{common_suffix}",
    "script": 'train.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_st3d_basebev_multi_lyft2nuscenes_dann_source_target_car_ped_point_label_calibrated_xyzi.yaml",
    "teacher_ckpt": "/storage/wandb/run-20250303_153658-ggpm88cg/files/ckpt/checkpoint_epoch_50.pth",
    "pretrained_ckpt": "/storage/wandb/run-20250303_153658-ggpm88cg/files/ckpt/checkpoint_epoch_50.pth",
    "ckpt_dir": "",
    "args": common_args,
    "platform": common_platform,
}

# dict_cfgs = [dict_cfg_lyft2kitti]
dict_cfgs = [dict_cfg_lyft2nuscenes]

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
    if dict_cfg["pretrained_ckpt"]:
        cmd += " --pretrained_model " + dict_cfg["pretrained_ckpt"]
    if dict_cfg["ckpt_dir"]:
        cmd += " --ckpt_dir " + dict_cfg["ckpt_dir"] + " --eval_all"

    print(cmd)
    subprocess.call(cmd.split())
