#!/usr/bin/env bash
# CPU validation of the ray-drop translation (experiments_md 20261017_01 §5): completeness of sim_nuscenes_rd, the SAME acceptance
# check as the calibration but on the files actually cast with the drop, lattice statistics (analysis), zero-point recount,
# emptied_boxes and target_dump on the rd config. Any failure fails the job, which (afterok) keeps the row from training.
set -e
ROOT=/home/koyama/data/lit/data
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
NR=/home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval
cd /home/koyama/code/ST3D/tools
CFG=cfgs/da-ieee-access/centerpoint-lit-rd-waymo2nuscenes.yaml
EX="singularity exec --bind /home/koyama/data/:/storage $SIF"
ARGS="$ROOT $ROOT/plan.json /home/koyama/data/waymo_top_calib/top_calib.pkl $NR $NR/nuscenes_infos_10sweeps_train.pkl"
$EX python3 -I lit_port/validate_translation.py $ROOT $ROOT/plan.json $ROOT/validation_rd.json sim_nuscenes_rd simrd
$EX python3 -I lit_port/calibrate_ray_drop.py check $ROOT/sim_nuscenes_rd 0 $ROOT/ray_drop_v1.json $ARGS $ROOT/ray_drop_check_cast.json
SC=$($EX python3 -I -c "import json;s=[x['name'] for x in json.load(open('$ROOT/plan.json'))['sequences']];print(','.join(s[i] for i in range(0,len(s),20)))")
$EX python3 -I lit_port/smoke_stats.py $ROOT/sim_nuscenes_rd /home/koyama/code/ST3D/data/waymo/waymo_processed_data \
    /home/koyama/data/waymo_top_calib/top_calib.pkl $NR $NR/nuscenes_infos_10sweeps_train.pkl $SC $ROOT/lattice_stats_rd.json 200
$EX python3 lit_port/smoke_recount.py $CFG 1000
$EX python3 analysis/emptied_boxes.py $CFG DATA_CONFIG 100
$EX python3 analysis/target_dump.py $CFG DATA_CONFIG 24
echo ALL VALIDATION STEPS PASSED
