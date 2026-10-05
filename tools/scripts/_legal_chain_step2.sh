#!/usr/bin/env bash
# Fully legal S2 chain, STEP 2 (experiments_md/20261004_01 §5d-5e, pre-declared): pseudo-label passes of
# the 27468 teacher AS TRAINED (no AdaBN), score floor 0.0001, no W&B, then the two pre-declared rules.
#   - nuScenes val, per platform at its own depth (n008 at 15 sweeps, n015 at 10) -> rule B (source
#     precision, all classes) -> SCORE_THRESH
#   - KITTI train -> rule A (background plateau's upper half-maximum) -> NEG_THRESH
# Usage (inside an sbatch allocation, 1 GPU):  bash scripts/_legal_chain_step2.sh <27468 checkpoint_epoch_20.pth>
# Do not edit while a job runs it.
set -u
CKPT=${1:?usage: _legal_chain_step2.sh <teacher checkpoint>}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
OUT=/storage/pseudo_labels/legaldepth27468
gen() {  # cfg out
  singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
    python $P/generate_pseudo_labels.py --cfg_file "$1" --teacher_ckpt "$CKPT" --out_dir "$2" --thresh 0.0001
  echo "gen $2 exit $?"
}
gen $P/cfgs/teacher_s2_on_nuscenes_val_n008_15sweeps.yaml ${OUT}_nuscenes_val_n008_15sweeps
gen $P/cfgs/teacher_s2_on_nuscenes_val_n015_10sweeps.yaml ${OUT}_nuscenes_val_n015_10sweeps
gen cfgs/da-ieee-access/centerpoint-accum-st3d-nuscenes2kitti.yaml ${OUT}_kitti_train
echo "=== rule B: SCORE_THRESH (source-val precision, all classes) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python $P/source_precision_cut.py \
  ${OUT}_nuscenes_val_n008_15sweeps/ps_label_e0.pkl --extra_ps ${OUT}_nuscenes_val_n015_10sweeps/ps_label_e0.pkl \
  --classes Car Pedestrian Cyclist
echo "=== rule A: NEG_THRESH (background plateau, KITTI train) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python $P/plateau_neg_thresh.py ${OUT}_kitti_train/ps_label_e0.pkl
echo "=== done ==="
