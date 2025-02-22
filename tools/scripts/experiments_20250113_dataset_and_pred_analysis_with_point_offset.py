import os
import subprocess

# lyft point offset cfg
cfg_file = "cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/domain_attention/second_old_anchor_base_bev_lyft_car_ped_21epochs_point_peak_offset.yaml"
# lyft point offset ckpt
ckpt_file = "/storage/wandb/run-20250111_151219-o0ilt19d/files/checkpoint_epoch_21.pth"
platform_option = "-platform offscreen"
name = "dataset_and_pred_analysis_with_point_offset_o0ilt19d"

cmd = "python dataset_analysis.py --run_name " + name + " --cfg_file " + cfg_file + " --ckpt " + ckpt_file + " " + platform_option
print(cmd)
subprocess.call(cmd.split())
