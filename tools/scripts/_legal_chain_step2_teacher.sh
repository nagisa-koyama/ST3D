#!/usr/bin/env bash
# Legal S2 chain STEP 2 for ANY teacher (generalises _legal_chain_step2.sh, which is pinned to 27468; same rules,
# experiments_md/20261004_01 §5d-5e, pre-declared): pseudo-label passes of the teacher AS TRAINED (no AdaBN),
# score floor 0.0001, no W&B, then the two pre-declared rules.
#   - nuScenes val, per platform at its own depth (n008 cfg, n015 cfg) -> rule B (source precision, all classes)
#     -> SCORE_THRESH
#   - KITTI train, through the pseudo-labelling config's own target block -> rule A (background plateau's upper
#     half-maximum) -> NEG_THRESH
# The teacher is built from each config's SELF_TRAIN.MODEL_TEACHER, so its encoder must match the checkpoint;
# check generate.log in each output dir for "Not updated" weights.
# Usage (inside an sbatch allocation, 1 GPU):
#   bash scripts/_legal_chain_step2_teacher.sh <ckpt> <st_cfg> <n008_cfg> <n015_cfg> <out_prefix under /storage>
# Do not edit while a job runs it.
set -u
CKPT=${1:?ckpt}; ST_CFG=${2:?st_cfg}; N008=${3:?n008_cfg}; N015=${4:?n015_cfg}; OUT=${5:?out_prefix}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
gen() {  # cfg out
  singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
    python $P/generate_pseudo_labels.py --cfg_file "$1" --teacher_ckpt "$CKPT" --out_dir "$2" --thresh 0.0001
  echo "gen $2 exit $?"
}
gen "$N008" ${OUT}_nuscenes_val_n008
gen "$N015" ${OUT}_nuscenes_val_n015
gen "$ST_CFG" ${OUT}_kitti_train
echo "=== rule B: SCORE_THRESH (source-val precision, all classes) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python $P/source_precision_cut.py \
  ${OUT}_nuscenes_val_n008/ps_label_e0.pkl --extra_ps ${OUT}_nuscenes_val_n015/ps_label_e0.pkl \
  --classes Car Pedestrian Cyclist
echo "=== rule A: NEG_THRESH (background plateau, KITTI train) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python $P/plateau_neg_thresh.py ${OUT}_kitti_train/ps_label_e0.pkl
echo "=== done ==="
