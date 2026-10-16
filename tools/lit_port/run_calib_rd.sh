#!/usr/bin/env bash
# CPU job: one-shot calibration of the label-free ray-drop, its acceptance check on disjoint sequences / keyframes, and the
# equivalence of drop-during-cast and drop-as-filter (experiments_md 20261017_01 §3). Any failure fails the job; the full cast and
# the row are Slurm afterok dependents of it.
set -e
ROOT=/home/koyama/data/lit/data
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
NR=/home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval
cd /home/koyama/code/ST3D/tools
EX="singularity exec --bind /home/koyama/data/:/storage $SIF"
ARGS="$ROOT $ROOT/plan.json /home/koyama/data/waymo_top_calib/top_calib.pkl $NR $NR/nuscenes_infos_10sweeps_train.pkl"
$EX python3 -I lit_port/calibrate_ray_drop.py calibrate $ARGS $ROOT/ray_drop_v1.json
$EX python3 -I lit_port/calibrate_ray_drop.py check $ROOT/sim_nuscenes 1 $ROOT/ray_drop_v1.json $ARGS $ROOT/ray_drop_check_prefilter.json
lit_port/lit_exec.sh python -I lit_port/check_ray_drop_equiv.py $ROOT $ROOT/plan.json $ROOT/ray_drop_v1.json /home/koyama/data/waymo_top_calib/top_calib.pkl 2 6
echo CALIBRATION CHECKS PASSED
