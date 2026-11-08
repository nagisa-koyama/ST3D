#!/usr/bin/env bash
# experiments_md 20261108_03 §2 C (T05 §5 item 39): the label-free cuts re-derived for one S2 source-only control teacher
# (centerpoint-sourceonly-nuscenes epoch 20, single sweep), by the SAME pre-declared rules and inputs as
# _genuine_st3d_cuts.sh (job 28641 for 25743), with the output tag as an argument:
#   - nuScenes val (1 sweep, its training input) -> rule B (source-val precision, all classes) -> SCORE_THRESH
#   - KITTI train                                  -> rule A (background plateau upper half-maximum) -> NEG_THRESH
# No target label is read. Usage (inside an sbatch allocation, 1 GPU):
#   bash scripts/_item39_cuts.sh <checkpoint host path> <tag>
# Do not edit while a job runs it.
set -u
CKPT=${1:?usage: _item39_cuts.sh <checkpoint host path> <tag>}
TAG=${2:?usage: _item39_cuts.sh <checkpoint host path> <tag>}
CKPT=${CKPT/#\/home\/koyama\/data\//\/storage\/}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
OUT=/storage/pseudo_labels/item39_${TAG}
CFG=cfgs/da-ieee-access/centerpoint-st3d-genuine-legal-nuscenes2kitti.yaml
gen() {  # cfg out
  singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
    python $P/generate_pseudo_labels.py --cfg_file "$1" --teacher_ckpt "$CKPT" --out_dir "$2" --thresh 0.0001
  echo "gen $2 exit $?"
}
gen $P/cfgs/teacher_25743_on_nuscenes_val_1sweep.yaml ${OUT}_nuscenes_val_1sweep
gen $CFG ${OUT}_kitti_train
echo "=== rule B: SCORE_THRESH (source-val precision, all classes) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python $P/source_precision_cut.py \
  ${OUT}_nuscenes_val_1sweep/ps_label_e0.pkl --classes Car Pedestrian Cyclist
echo "=== rule A: NEG_THRESH (background plateau, KITTI train) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python $P/plateau_neg_thresh.py ${OUT}_kitti_train/ps_label_e0.pkl
echo "=== done ==="
