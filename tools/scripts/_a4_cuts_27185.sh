#!/usr/bin/env bash
# A4 step 2 (experiments_md/20261017_09 §4; rules pre-declared in 20261004_01 §5e, procedure in 20261017_03 §3):
# the cuts of the procedure's own nuScenes -> Waymo row, re-derived for ITS teacher 27185 (MeanVFE, Boston 30 /
# Singapore 20, training run p1a5erux, epoch 20), plus the teacher's matched reference on Waymo val.
#   (1)-(2) teacher AS TRAINED (no AdaBN) on the labelled SOURCE val, per platform at its own training depth
#           (n008 at 30 sweeps, n015 at 20; 200-sweep infos, as MAX_SWEEPS > 10 needs), score floor 0.0001
#           -> rule B (source-val precision >= 0.50, all classes) -> SCORE_THRESH
#   (3)     teacher on WAYMO TRAIN through the A4 config's own target block (SAMPLED_INTERVAL 10, rule-G SHIFT_COOR),
#           score floor 0.0001 -> rule A (background plateau's upper half-maximum) -> NEG_THRESH
#   (4)     27185 epoch 20 scored on Waymo val at the rule-G SHIFT_COOR (-0.044), the A4 row's input: the teacher
#           reference of the pre-declared reading (27185 as scored at 0: 50.67 / 26.00).
# The same python tools and rules as scripts/_legal_chain_step2_teacher.sh / _legal_chain_step2_waymo.sh (27608, 27699).
# No W&B for (1)-(3). Usage (inside a 1-GPU sbatch allocation, from ST3D/tools):  bash scripts/_a4_cuts_27185.sh
# Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
CKPT_S=/storage/wandb/run-20261003_105441-p1a5erux/files/ckpt/checkpoint_epoch_20.pth   # container path
CKPT_H=/home/koyama/data/wandb/run-20261003_105441-p1a5erux/files/ckpt/checkpoint_epoch_20.pth   # host path
A4=cfgs/da-ieee-access/centerpoint-accum-depthwaymo-st3d-nuscenes2waymo.yaml
OUT=/storage/pseudo_labels/a4_27185_ep20
gen() {  # cfg out
  singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
    python $P/generate_pseudo_labels.py --cfg_file "$1" --teacher_ckpt "$CKPT_S" --out_dir "$2" --thresh 0.0001
  echo "gen $2 exit $?"
}
echo "##### (1)-(3) pseudo-label passes, teacher 27185 ep20"
gen $P/cfgs/teacher_a4_27185_on_nuscenes_val_n008_30sweeps.yaml ${OUT}_nuscenes_val_n008_30sweeps
gen $P/cfgs/teacher_a4_27185_on_nuscenes_val_n015_20sweeps.yaml ${OUT}_nuscenes_val_n015_20sweeps
gen "$A4" ${OUT}_waymo_train_int10_ruleG
echo "=== rule B: SCORE_THRESH (source-val precision, all classes) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python $P/source_precision_cut.py \
  ${OUT}_nuscenes_val_n008_30sweeps/ps_label_e0.pkl --extra_ps ${OUT}_nuscenes_val_n015_20sweeps/ps_label_e0.pkl \
  --classes Car Pedestrian Cyclist
echo "=== rule A: NEG_THRESH (background plateau, Waymo train) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python $P/plateau_neg_thresh.py ${OUT}_waymo_train_int10_ruleG/ps_label_e0.pkl
echo "##### (4) teacher 27185 ep20 on Waymo val at rule G (z = -0.044)"
WANDB_NOTES="A4 (experiments_md 20261017_09 §4): teacher 27185 (p1a5erux ep20, MeanVFE, Boston 30 / Singapore 20) scored on Waymo val at the rule-G target SHIFT_COOR z = -0.044, the A4 row's input; the matched teacher reference of its pre-declared reading. As scored at z = 0: 50.67 / 26.00 (0ru2bxor)." \
  bash analysis/eval_checkpoint.sh cfgs/da-ieee-access/centerpoint-accum-depthwaymo-nuscenes2waymo.yaml "$CKPT_H" \
  a4_teacher_ruleG_m0044 20261017_a4 --set DATA_CONFIG_TAR.SHIFT_COOR "[0.0,0.0,-0.044]"
echo "eval a4_teacher_ruleG_m0044 exit $?"
echo "##### all done"
