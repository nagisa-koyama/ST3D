#!/usr/bin/env bash
# SCENE, the chained successor of an in-city oracle (experiments_md/20261022_06 §4; design T05 §5 item 1, fixed 2026-10-22 22:36).
# Runs on ONE GPU, `afterany` on the training row, resolving its epoch-20 checkpoint at run time (analysis/ckpt_of_job.sh).
#
#   bash scripts/_scene_after_teacher.sh <training job id> <bb|ss>
#
#   bb = the Boston in-city oracle (centerpoint-sourceonly-nuscenes-boston2boston); ss = the Singapore one.
#   (1) the SAME model rescored on the OTHER city's val with the cross-city config (B->S or S->B): the cross-city reading.
#       A cross-city source-only row is the in-city model with a different eval set (same source, recipe, seed), so it is
#       not trained twice (reuse rule). `none` = the saved BN statistics.
#   (2) the cross-city reading at EQUAL STATISTICS: test-time AdaBN (analysis/adabn_eval.sh, `target` arm, 1,000 frames of
#       the target city's TRAIN clouds, all BN layers, mix 1.0; 29306's procedure), no label.
#   (3) the in-city reading at its own city's train statistics (`target` arm on the in-city config), for the same table.
#   bb only, stage 2 of the PL B->S row (20261004_01 §5e rules; 20261017_09 §4's procedure):
#   (4) teacher AS TRAINED on the labelled SOURCE val = Boston val (n008), 1 sweep, floor 0.0001
#       (flat config analysis/pseudo_label_threshold/cfgs/teacher_scene_bb_on_nuscenes_val_n008_1sweep.yaml)
#   (5) teacher on Singapore TRAIN (n015) through the PL row's own target block, floor 0.0001
#   (6) rule B (source-val precision >= 0.50, all classes) -> SCORE; rule A (plateau upper half-maximum) -> NEG; a SCORE
#       below its NEG raised to it (analysis/a4_final_cuts.py) -> /home/koyama/data/pseudo_labels/scene_bb_ep20_cuts.json
#   (7) the cut labels: per-frame positives / ignore per class (the PL row's expected first pass), points per box on the PL
#       row's own target loader (analysis/a4_cut_check.py), CenterHead target dump on 20 frames (analysis/target_dump.py;
#       bug_prevention_checklist rule 4)
# The PL row itself is submitted by the session after reading this job's log and writing the cuts into its config.
# Runs from the LIVE checkout. Do not edit while a job runs it.
set -u
JOB=${1:?usage: _scene_after_teacher.sh <training job id> <bb|ss>}
WHICH=${2:?usage: _scene_after_teacher.sh <training job id> <bb|ss>}
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git rev-parse --short HEAD) on $(hostname) at $(date '+%F %T') ==="
CK=$(bash analysis/ckpt_of_job.sh "$JOB" 20) || { echo "no epoch-20 checkpoint for job $JOB" >&2; exit 2; }
CKS=${CK/#\/home\/koyama\/data//storage}
echo "teacher: $CK"
D=cfgs/da-ieee-access
case "$WHICH" in
  bb) IN=$D/centerpoint-sourceonly-nuscenes-boston2boston.yaml;       CROSS=$D/centerpoint-sourceonly-nuscenes-boston2singapore.yaml
      CROSS_NAME="B->S"; IN_NAME="B->B"; TGT_CITY=Singapore ;;
  ss) IN=$D/centerpoint-sourceonly-nuscenes-singapore2singapore.yaml; CROSS=$D/centerpoint-sourceonly-nuscenes-singapore2boston.yaml
      CROSS_NAME="S->B"; IN_NAME="S->S"; TGT_CITY=Boston ;;
  *)  echo "second argument must be bb or ss" >&2; exit 2 ;;
esac
BASE="SCENE analysis: scene term, nuScenes Boston/Singapore, T05 §5 item 1 (experiments_md 20261022_06). Teacher = job $JOB epoch 20."
echo "##### (1) $CROSS_NAME: the in-city model rescored on $TGT_CITY val, saved statistics"
WANDB_NOTES="$BASE Row $CROSS_NAME: rescored on $TGT_CITY val with the cross-city config, saved BN statistics (the cross-city source-only reading)." \
  bash analysis/adabn_eval.sh "$CROSS" "$CKS" none 1.0 "scene_${WHICH}_cross_none_$JOB"
echo "eval $CROSS_NAME none exit $?"
echo "##### (2) $CROSS_NAME at equal statistics: test-time AdaBN on $TGT_CITY TRAIN clouds"
WANDB_NOTES="$BASE Row $CROSS_NAME at equal statistics: test-time AdaBN, BN statistics from 1,000 $TGT_CITY TRAIN frames (no label), all layers, mix 1.0." \
  bash analysis/adabn_eval.sh "$CROSS" "$CKS" target 1.0 "scene_${WHICH}_cross_adabn_$JOB"
echo "eval $CROSS_NAME target exit $?"
echo "##### (3) $IN_NAME at its own city's train statistics"
WANDB_NOTES="$BASE Row $IN_NAME at equal statistics: test-time AdaBN on its own city's TRAIN frames (1,000, all layers, mix 1.0); the in-run eval gives the saved-statistics value." \
  bash analysis/adabn_eval.sh "$IN" "$CKS" target 1.0 "scene_${WHICH}_in_adabn_$JOB"
echo "eval $IN_NAME target exit $?"
[ "$WHICH" = bb ] || { echo "##### all done ($WHICH)"; exit 0; }

SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
PL=$D/centerpoint-st3d-nuscenes-boston2singapore.yaml
OUT=/storage/pseudo_labels/scene_bb_ep20            # container path
OUTH=/home/koyama/data/pseudo_labels/scene_bb_ep20  # the same, host path
X="singularity exec --bind /home/koyama/data/:/storage $SIF"
gen() {  # cfg out
  singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
    python $P/generate_pseudo_labels.py --cfg_file "$1" --teacher_ckpt "$CKS" --out_dir "$2" --thresh 0.0001
  echo "gen $2 exit $?"
}
echo "##### (4)-(5) pseudo-label passes"
gen $P/cfgs/teacher_scene_bb_on_nuscenes_val_n008_1sweep.yaml ${OUT}_nuscenes_val_n008_1sweep
gen "$PL" ${OUT}_singapore_train
echo "=== rule B: SCORE_THRESH (Boston val precision, all classes) ==="
$X python $P/source_precision_cut.py ${OUT}_nuscenes_val_n008_1sweep/ps_label_e0.pkl --classes Car Pedestrian Cyclist | tee ${OUTH}_ruleB.txt
echo "=== rule A: NEG_THRESH (background plateau, Singapore train) ==="
$X python $P/plateau_neg_thresh.py ${OUT}_singapore_train/ps_label_e0.pkl | tee ${OUTH}_ruleA.txt
echo "=== final cuts (rule A step 6: SCORE below NEG raised to it) ==="
$X python3 analysis/a4_final_cuts.py ${OUT}_ruleB.txt ${OUT}_ruleA.txt ${OUT}_cuts.json
echo "final cuts exit $?"
echo "##### (7) cut labels: expected first pass, points per box, target dump"
$X python3 analysis/a4_cut_check.py "$PL" ${OUT}_singapore_train/ps_label_e0.pkl \
  ${OUT}_singapore_train/ps_label_e0_cut.pkl --cuts_json ${OUT}_cuts.json --frames 200
echo "cut check exit $?"
$X python3 analysis/target_dump.py "$PL" TAR 20 ${OUT}_singapore_train/ps_label_e0_cut.pkl
echo "target dump exit $?"
echo "##### all done (bb)"
