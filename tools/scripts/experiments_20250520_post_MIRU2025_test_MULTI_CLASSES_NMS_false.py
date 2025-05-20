import os
import subprocess


common_args = "--batch_size 20"
common_platform = "-platform offscreen"

# Skipped
# ommon_prefix = "post-MIRU2025"
# common_suffix = "point_label_calibrated_wd_weight"
# dict_cfg_lyft_kitti2nuscenes_default = {
#     "name": f"{common_prefix}_lyft_kitti2nuscens_{common_suffix}",
#     # "name": "",
#     "script": 'train.py',
#     "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_basebev_multi_lyft_kitti2nuscenes_car_ped_point_label_calibrated.yaml",
#     "teacher_ckpt": "",
#     "pretrained_ckpt": "",
#     "ckpt_dir": "",
#     "args": common_args,
#     "platform": common_platform,
# }

# Original training with muiti_classes_nms = true https://wandb.ai/nagisa/st3d/runs/t4is3dtv/overview
common_prefix = "post-MIRU2025"
common_suffix = "point_label_calibrated_equal_sample_weight"
dict_cfg_lyft_kitti2nuscenes_default = {
    "name": f"test_{common_prefix}_lyft_kitti2nuscens_{common_suffix}_multi_classes_nms_false",
    # "name": "",
    "script": 'test.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_basebev_multi_lyft_kitti2nuscenes_car_ped_point_label_calibrated_backward_together_off.yaml",
    "teacher_ckpt": "",
    "pretrained_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250518_150144-t4is3dtv/files/ckpt/",
    "args": common_args,
    "platform": common_platform,
}

# Skipped
# common_prefix = "post-MIRU2025"
# common_suffix = "point_label_calibrated_wd_weight"
# dict_cfg_lyft_nuscenes2kitti_default = {
#     "name": f"{common_prefix}_lyft_nuscenes_2kitti_{common_suffix}",
#     # "name": "",
#     "script": 'train.py',
#     "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_basebev_multi_lyft_nuscenes2kitti_car_ped_point_label_calibrated.yaml",
#     "teacher_ckpt": "",
#     "pretrained_ckpt": "",
#     "ckpt_dir": "",
#     "args": common_args,
#     "platform": common_platform,
# }


# Original training with muiti_classes_nms = true https://wandb.ai/nagisa/st3d/runs/yu16yll1/overview
ommon_prefix = "post-MIRU2025"
common_suffix = "point_label_calibrated_equal_sample_weight"

dict_cfg_lyft_nuscenes2kitti_default = {
    "name": f"test_{common_prefix}_lyft_nuscenes_2kitti_{common_suffix}_multi_classes_nms_false",
    # "name": "",
    "script": 'test.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_basebev_multi_lyft_nuscenes2kitti_car_ped_point_label_calibrated_backward_together_off.yaml",
    "teacher_ckpt": "",
    "pretrained_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250510_162112-yu16yll1/files/ckpt/",
    "args": common_args,
    "platform": common_platform,
}

# Original training with muiti_classes_nms = true https://wandb.ai/nagisa/st3d/runs/stt9n6o1/overview
# Also missing 
common_prefix = "post_MIRU2025_st3d_target_equal_sampling"
common_suffix = "point_label_calibrated_hist_517oe15r"
dict_cfg_lyft_nuscenes2kitti_target = {
    "name": f"test_{common_prefix}_lyft_nuscenes2kitti_yu16yll1_{common_suffix}_multi_classes_nms_false",
    # "name": "",
    "script": 'test.py',
    "cfg_file": "cfgs/da-post-MIRU2025/second_old_anchor_st3d_basebev_multi_lyft_nuscenes2kitti_target_car_ped_point_label_calibrated.yaml",
    "teacher_ckpt": "",
    "pretrained_ckpt": "",
    "ckpt_dir": "/storage/wandb/run-20250512_035713-stt9n6o1/files/ckpt/",
    "args": common_args,
    "platform": common_platform,
}

dict_cfgs = [
    dict_cfg_lyft_kitti2nuscenes_default,
    # dict_cfg_lyft_nuscenes2kitti_default,
    # dict_cfg_lyft_nuscenes2kitti_target,
]

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
