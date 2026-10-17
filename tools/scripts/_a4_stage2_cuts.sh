#!/usr/bin/env bash
# A4 STAGE 2 (experiments_md/20261017_09 §4; rules pre-declared in 20261004_01 §5e, procedure 20261017_03 §3): the cuts of the
# procedure's own nuScenes -> Waymo row, derived for ITS teacher (stage 1, centerpoint-accum-procdepth-nuscenes2waymo, MeanVFE,
# Boston 22 / Singapore 17), the checks of the cut labels, and the teacher's matched reference on Waymo val.
#   (1)-(2) teacher AS TRAINED (no AdaBN) on the labelled SOURCE val, per platform at its own training depth (n008 at 22 sweeps,
#           n015 at 17; 200-sweep infos), score floor 0.0001
#   (3)     teacher on WAYMO TRAIN through the stage-3 config's own target block (SAMPLED_INTERVAL 10, rule-G SHIFT_COOR -0.044)
#   (4)     rule B (source-val precision >= 0.50, all classes) -> SCORE; rule A (plateau upper half-maximum) -> NEG; a SCORE below
#           its NEG raised to it (analysis/a4_final_cuts.py) -> <OUT>_cuts.json
#   (5)     the cut labels: per-frame positives / ignore per class (the expected first pass), points per box on the row's own
#           target loader (analysis/a4_cut_check.py), CenterHead target dump on 20 frames (analysis/target_dump.py)
#   (6)     the teacher scored on Waymo val at the rule-G SHIFT_COOR (-0.044), the stage-3 row's input: T of the stage-3 reading
# The same python tools and rules as scripts/_a4_cuts_27185.sh (29315), _legal_chain_step2_teacher.sh (27608), _legal_chain_step2_waymo.sh (27699).
# Usage (inside a 1-GPU sbatch allocation, from ST3D/tools; the checkpoint as a HOST path, e.g. from analysis/ckpt_of_job.sh):
#   bash scripts/_a4_stage2_cuts.sh /home/koyama/data/wandb/run-.../files/ckpt/checkpoint_epoch_20.pth
# Do not edit while a job runs it.
set -u
CKPT_H=${1:?usage: _a4_stage2_cuts.sh <teacher checkpoint, host path>}
[ -f "$CKPT_H" ] || { echo "no such checkpoint: $CKPT_H" >&2; exit 2; }
CKPT_S=${CKPT_H/#\/home\/koyama\/data//storage}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
S1=cfgs/da-ieee-access/centerpoint-accum-procdepth-nuscenes2waymo.yaml
S3=cfgs/da-ieee-access/centerpoint-accum-procdepth-st3d-nuscenes2waymo.yaml
OUT=/storage/pseudo_labels/a4procdepth_ep20          # container path
OUTH=/home/koyama/data/pseudo_labels/a4procdepth_ep20  # the same, host path
X="singularity exec --bind /home/koyama/data/:/storage $SIF"
echo "teacher: $CKPT_H"
gen() {  # cfg out
  singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
    python $P/generate_pseudo_labels.py --cfg_file "$1" --teacher_ckpt "$CKPT_S" --out_dir "$2" --thresh 0.0001
  echo "gen $2 exit $?"
}
echo "##### (1)-(3) pseudo-label passes"
gen $P/cfgs/teacher_a4procdepth_on_nuscenes_val_n008_22sweeps.yaml ${OUT}_nuscenes_val_n008_22sweeps
gen $P/cfgs/teacher_a4procdepth_on_nuscenes_val_n015_17sweeps.yaml ${OUT}_nuscenes_val_n015_17sweeps
gen "$S3" ${OUT}_waymo_train_int10_ruleG
echo "=== rule B: SCORE_THRESH (source-val precision, all classes) ==="
$X python $P/source_precision_cut.py ${OUT}_nuscenes_val_n008_22sweeps/ps_label_e0.pkl \
  --extra_ps ${OUT}_nuscenes_val_n015_17sweeps/ps_label_e0.pkl --classes Car Pedestrian Cyclist | tee ${OUTH}_ruleB.txt
echo "=== rule A: NEG_THRESH (background plateau, Waymo train) ==="
$X python $P/plateau_neg_thresh.py ${OUT}_waymo_train_int10_ruleG/ps_label_e0.pkl | tee ${OUTH}_ruleA.txt
echo "=== final cuts (rule A step 6: SCORE below NEG raised to it) ==="
$X python3 analysis/a4_final_cuts.py ${OUT}_ruleB.txt ${OUT}_ruleA.txt ${OUT}_cuts.json
echo "final cuts exit $?"
echo "##### (5) cut labels: expected first pass, points per box, target dump"
$X python3 analysis/a4_cut_check.py "$S3" ${OUT}_waymo_train_int10_ruleG/ps_label_e0.pkl \
  ${OUT}_waymo_train_int10_ruleG/ps_label_e0_cut.pkl --cuts_json ${OUT}_cuts.json --frames 200
echo "cut check exit $?"
$X python3 analysis/target_dump.py "$S3" TAR 20 ${OUT}_waymo_train_int10_ruleG/ps_label_e0_cut.pkl
echo "target dump exit $?"
echo "##### (6) the teacher on Waymo val at rule G (z = -0.044)"
WANDB_NOTES="A4 stage 2 (experiments_md 20261017_09 §4): the A4 stage-1 teacher (MeanVFE, Boston 22 / Singapore 17, $CKPT_H) scored on Waymo val at the rule-G target SHIFT_COOR z = -0.044, the stage-3 row's input: T of the stage-3 reading. Its in-run score at z = 0 is the stage-1 job's." \
  bash analysis/eval_checkpoint.sh "$S1" "$CKPT_H" a4_teacher_ruleG_m0044 20261017_a4_stage2 \
  --set DATA_CONFIG_TAR.SHIFT_COOR "[0.0,0.0,-0.044]"
echo "eval a4_teacher_ruleG_m0044 exit $?"
echo "##### all done"
