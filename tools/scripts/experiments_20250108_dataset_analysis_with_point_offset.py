import os
import subprocess

cfg_file = "cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/dataset_analysis_point_peak_offset.yaml"
platform_option = "-platform offscreen"

cmd = "python dataset_analysis.py --cfg_file " + cfg_file + " " + platform_option
print(cmd)
subprocess.call(cmd.split())
