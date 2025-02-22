import os
import subprocess

common_prefix = "visualize"
common_suffix = "fov_only_off_common_augv2_100epochs"
common_args = "--is_train"
common_platform = "-platform offscreen"
# teacher_ckpt = "/storage/wandb/run-20250122_temp-b4tufpds/ckpt/checkpoint_epoch_87.pth"
# teacher_ckpt_shift_coor = "/storage/wandb/run-20250119_081031-lnqisgyg/files/ckpt/checkpoint_epoch_100.pth"

# https://wandb.ai/nagisa/st3d/runs/pnevxgo9?nw=nwusernagisa
dict_cfg_no_shift = {
    "name": f"{common_prefix}_no_shift_coor_{common_suffix}_pnevxgo9",
    "script": 'demo.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_no_shift_coor.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    # "ckpt": "/storage/wandb/run-20250119_080940-pnevxgo9/files/ckpt/checkpoint_epoch_100.pth",
    "args": common_args,
    "platform": common_platform,
}

# https://wandb.ai/nagisa/st3d/runs/kr5zgest?nw=nwusernagisa
dict_cfg_no_shift_num_point = {
    "name": f"{common_prefix}_no_shift_coor_num_point_calibrated_{common_suffix}_kr5zgest",
    "script": 'demo.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_no_shift_coor_num_point_calibrated.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    # "ckpt": "/storage/wandb/run-20250120_131033-kr5zgest/files/ckpt/checkpoint_epoch_100.pth",
    "args": common_args,
    "platform": common_platform,
}

# https://wandb.ai/nagisa/st3d/runs/aree4p2z?nw=nwusernagisa
dict_cfg_no_shift_label = {
    "name": f"{common_prefix}_no_shift_coor_label_calibrated_{common_suffix}_aree4p2z",
    "script": 'demo.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_no_shift_coor_label_calibrated.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    # "ckpt": "/storage/wandb/run-20250122_100135-aree4p2z/files/ckpt/checkpoint_epoch_100.pth",
    "args": common_args,
    "platform": common_platform,
}

# https://wandb.ai/nagisa/st3d/runs/b4tufpds?nw=nwusernagisa
dict_cfg_shift_point_label = {
    "name": f"{common_prefix}_shift_coor_point_label_calibrated_v3_{common_suffix}_b4tufpds",
    "script": 'demo.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_label_calibrated_v3.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    # "ckpt": "/storage/wandb/run-20250121_172940-b4tufpds/files/ckpt/checkpoint_epoch_100.pth",
    "args": common_args,
    "platform": common_platform,
}

# https://wandb.ai/nagisa/st3d/runs/wtcwgdxj?nw=nwusernagisa
dict_cfg_with_fov = {
    "name": f"{common_prefix}_fov_only_wtcwgdxj",
    "script": 'demo.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    # "ckpt": "/storage/wandb/run-20250121_172940-b4tufpds/files/ckpt/checkpoint_epoch_100.pth",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_shift_point_label_v5 = {
    "name": f"{common_prefix}_shift_coor_point_label_calibrated_v5/",
    "script": 'demo.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_label_calibrated_v5.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args,
    "platform": common_platform,
}

dict_cfg_shift_point_label_v9 = {
    "name": f"{common_prefix}_shift_coor_point_label_calibrated_v9/",
    "script": 'demo.py',
    "cfg_file": 'cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_label_calibrated_v9.yaml',
    "teacher_ckpt": "",
    "ckpt_dir": "",
    "ckpt": "",
    "args": common_args,
    "platform": common_platform,
}

# dict_cfgs = [dict_cfg_no_shift, dict_cfg_no_shift_num_point, dict_cfg_no_shift_label, dict_cfg_shift_point_label]
# dict_cfgs = [dict_cfg_with_fov]
dict_cfgs = [dict_cfg_shift_point_label_v9]

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
