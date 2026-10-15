#!/usr/bin/env bash
# CPU validation job of the full LiT translation (experiments_md 20261015_02 §7): completeness, lattice statistics against real
# nuScenes TRAIN, zero-point recount, emptied_boxes.py and target_dump.py on the REAL translated data. Any failure fails the job,
# which (afterok) keeps the row from training.
set -e
ROOT=/home/koyama/data/lit/data
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
cd /home/koyama/code/ST3D/tools
CFG=cfgs/da-ieee-access/centerpoint-lit-waymo2nuscenes.yaml
EX="singularity exec --bind /home/koyama/data/:/storage $SIF"
$EX python3 -I lit_port/validate_translation.py $ROOT $ROOT/plan.json $ROOT/validation.json
# 40 evenly spaced sequences for the lattice statistics (3-70 m, rows, car boxes), real nuScenes TRAIN keyframes beside them
SC=$($EX python3 -I -c "import json;s=[x['name'] for x in json.load(open('$ROOT/plan.json'))['sequences']];print(','.join(s[i] for i in range(0,len(s),20)))")
$EX python3 -I lit_port/smoke_stats.py $ROOT/sim_nuscenes /home/koyama/code/ST3D/data/waymo/waymo_processed_data \
    /home/koyama/data/waymo_top_calib/top_calib.pkl /home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval \
    /home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl $SC $ROOT/lattice_stats.json 200
$EX python3 lit_port/smoke_recount.py $CFG 1000
$EX python3 analysis/emptied_boxes.py $CFG DATA_CONFIG 100
$EX python3 analysis/target_dump.py $CFG DATA_CONFIG 24
echo ALL VALIDATION STEPS PASSED
