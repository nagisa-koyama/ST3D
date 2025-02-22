import os
import subprocess

# lyft point offset cfg
cfg_file = "cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_label_calibrated.yaml"
# lyft point offset ckpt
ckpt_file = "/storage/wandb/run-20250116_181207-az9fl86g/files/checkpoint_epoch_21.pth"
platform_option = "-platform offscreen"
name = "test"

cmd = "python dataset_analysis.py --run_name " + name + " --cfg_file " + cfg_file + " --ckpt " + ckpt_file + " " + platform_option
print(cmd)
subprocess.call(cmd.split())
