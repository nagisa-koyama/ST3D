#!/usr/bin/env bash
# Item 51 (experiments_md 20261003_05 §5): the label-free cuts re-derived for a teacher by the SAME pre-declared rules
# and generation configs as _genuine_st3d_cuts.sh (job 28641 for 25743), for nuScenes -> KITTI:
#   - nuScenes val (1 sweep, the teacher's training input) -> rule B (source-val precision, all classes) -> SCORE_THRESH
#   - KITTI train                                            -> rule A (background plateau upper half-maximum) -> NEG_THRESH
# No target label is read. Usage (1 GPU): bash scripts/_item51_cuts_n2k.sh <ckpt host path | job:<id>:<epoch>> <tag>
# Do not edit while a job runs it.
set -u
SPEC=${1:?ckpt or job:<id>:<epoch>}; TAG=${2:?tag}
cd /home/koyama/code/ST3D/tools
case "$SPEC" in
  job:*) J=$(echo "$SPEC" | cut -d: -f2); E=$(echo "$SPEC" | cut -d: -f3); CKPT=$(bash analysis/ckpt_of_job.sh "$J" "$E") || exit 2 ;;
  *) CKPT=$SPEC ;;
esac
echo "teacher $CKPT"
CKPT=${CKPT/#\/home\/koyama\/data\//\/storage\/}
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
OUT=/storage/pseudo_labels/item51_${TAG}
CFG=cfgs/da-ieee-access/centerpoint-st3d-genuine-legal-nuscenes2kitti.yaml   # its MODEL_TEACHER = the same CenterPoint
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
