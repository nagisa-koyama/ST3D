#!/usr/bin/env bash
# SCENE, stage 2 of the PL S->B row (experiments_md/20261022_06 §8; design T05 §5 "Check PL S->B", fixed 2026-10-23 14:22).
# The steps of _scene_after_teacher.sh bb (4)-(7), job 30190, with the cities swapped; rules of 20261004_01 §5e.
#
#   bash scripts/_scene_pl_s2b_cuts.sh <S->S training job id>      (inside a 1-GPU allocation, from ST3D/tools)
#
#   (1) teacher AS TRAINED (saved BN statistics) on the labelled SOURCE val = Singapore val (n015), 1 sweep, floor 0.0001
#       (flat config analysis/pseudo_label_threshold/cfgs/teacher_scene_ss_on_nuscenes_val_n015_1sweep.yaml)
#   (2) teacher on Boston TRAIN (n008) through the PL S->B row's own target block, floor 0.0001
#   (3) rule B (source-val precision >= 0.50, all classes) -> SCORE; rule A (plateau upper half-maximum) -> NEG; a SCORE below
#       its NEG raised to it (analysis/a4_final_cuts.py) -> /home/koyama/data/pseudo_labels/scene_ss_ep20_cuts.json
#   (4) the cut labels: per-frame positives / ignore per class (the row's expected first pass), points per box on the row's
#       own target loader (analysis/a4_cut_check.py, 200 frames), CenterHead target dump on 20 frames (analysis/target_dump.py;
#       bug_prevention_checklist rule 4)
# The row itself is submitted by the session after reading this job's log and writing the cuts into its config.
# A bash script, never a --wrap (dash rejects ${var/#..}). Runs from the LIVE checkout. Do not edit while a job runs it.
set -u
JOB=${1:?usage: _scene_pl_s2b_cuts.sh <S->S training job id>}
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git -C /home/koyama/code/ST3D rev-parse --short HEAD) on $(hostname) at $(date '+%F %T') ==="
CK=$(bash analysis/ckpt_of_job.sh "$JOB" 20) || { echo "no epoch-20 checkpoint for job $JOB" >&2; exit 2; }
CKS=${CK/#\/home\/koyama\/data//storage}
echo "teacher: $CK"
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
PL=cfgs/da-ieee-access/centerpoint-st3d-nuscenes-singapore2boston.yaml
OUT=/storage/pseudo_labels/scene_ss_ep20            # container path
OUTH=/home/koyama/data/pseudo_labels/scene_ss_ep20  # the same, host path
X="singularity exec --bind /home/koyama/data/:/storage $SIF"
gen() {  # cfg out
  singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
    python $P/generate_pseudo_labels.py --cfg_file "$1" --teacher_ckpt "$CKS" --out_dir "$2" --thresh 0.0001
  echo "gen $2 exit $?"
}
echo "##### (1)-(2) pseudo-label passes"
gen $P/cfgs/teacher_scene_ss_on_nuscenes_val_n015_1sweep.yaml ${OUT}_nuscenes_val_n015_1sweep
gen "$PL" ${OUT}_boston_train
echo "=== rule B: SCORE_THRESH (Singapore val precision, all classes) ==="
$X python $P/source_precision_cut.py ${OUT}_nuscenes_val_n015_1sweep/ps_label_e0.pkl --classes Car Pedestrian Cyclist | tee ${OUTH}_ruleB.txt
echo "=== rule A: NEG_THRESH (background plateau, Boston train) ==="
$X python $P/plateau_neg_thresh.py ${OUT}_boston_train/ps_label_e0.pkl | tee ${OUTH}_ruleA.txt
echo "=== final cuts (rule A step 6: SCORE below NEG raised to it) ==="
$X python3 analysis/a4_final_cuts.py ${OUT}_ruleB.txt ${OUT}_ruleA.txt ${OUT}_cuts.json
echo "final cuts exit $?"
echo "##### (4) cut labels: expected first pass, points per box, target dump"
$X python3 analysis/a4_cut_check.py "$PL" ${OUT}_boston_train/ps_label_e0.pkl \
  ${OUT}_boston_train/ps_label_e0_cut.pkl --cuts_json ${OUT}_cuts.json --frames 200
echo "cut check exit $?"
$X python3 analysis/target_dump.py "$PL" TAR 20 ${OUT}_boston_train/ps_label_e0_cut.pkl
echo "target dump exit $?"
echo "##### all done (PL S->B stage 2) $(date '+%F %T')"
