import os
import subprocess

cfg_file = "cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/dataset_analysis.yaml"
platform_option = "-platform offscreen"
name = "dataset_analysis_no_shift_bin400"
name = ""

cmd = "python dataset_analysis.py --cfg_file " + cfg_file + " " + platform_option
if name:
    cmd += " --run_name " + name
print(cmd)
subprocess.call(cmd.split())
