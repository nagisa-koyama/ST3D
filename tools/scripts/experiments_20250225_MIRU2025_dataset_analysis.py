import os
import subprocess

# lyft point offset cfg
cfg_file = "cfgs/da-waymo-lyft-pandaset-nuscenes-to-kitti_models/head_per_dataset_with_source/dataset_analysis_point_and_label_calibrated_v7_before.yaml"

platform_option = "-platform offscreen --batch_size 1"
# name = "dataset_analysis_point_num_with_sweep_within_75m"
# name = "dataset_analysis_point_and_label_calibrated_75m_v2"
# name = "dataset_analysis_point_and_label_calibrated_75m_v4_dist_hist"
# name = "dataset_analysis_point_and_label_calibrated_75m_v5_dist_hist"
# name = "dataset_analysis_point_and_label_calibrated_75m_v8"
# name =  "dataset_analysis_point_and_label_calibrated_75m_v7"
name =  "dataset_analysis_MIRU2025_point_and_label_calibrated_75m_v7_before"

cmd = "python dataset_analysis.py --cfg_file " + cfg_file + " --is_train " + platform_option
if name:
    cmd += " --run_name " + name 
print(cmd)
subprocess.call(cmd.split())
