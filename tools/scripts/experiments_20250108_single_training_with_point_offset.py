import os
import subprocess

cfg_file = "cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_peak_offset.yaml"
# cfg_file = "cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_label_calibrated.yaml"
# name = "train_lyft_point_label_calibrated"
name = "test"

platform_option = "-platform offscreen"

# cmd = "python train.py --run_name " + name + " --cfg_file " + cfg_file + " " + platform_option + " --epochs 1"
# print(cmd)
# subprocess.call(cmd.split())

# Test model trained with intensity
# ckpt_files = "/storage/wandb/run-20241225_094941-ys0v1q2s/files/"
# name = "test_lyft_point_peak_offset_ys0v1q2s"
# cmd = "python test.py --run_name " + name + " --cfg_file " + cfg_file + " --ckpt_dir " + ckpt_files + " --eval_all " + platform_option
# print(cmd)
# subprocess.call(cmd.split())

# Test model trained without intensity
ckpt = "/storage/wandb/run-20250111_151219-o0ilt19d/files/checkpoint_epoch_21.pth"
name = "test"
cmd = "python test.py --run_name " + name + " --cfg_file " + cfg_file + " --ckpt " + ckpt + " " + platform_option
print(cmd)
subprocess.call(cmd.split())
