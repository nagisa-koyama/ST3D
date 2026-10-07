#!/usr/bin/env bash
# One 1-GPU allocation for two of the user's 2026-10-07 requests (experiments_md/20261007_01 §7):
#  (1) step 2 of the legal chain for the GBlobs + accumulation teacher 27471 (cuts by rules A / B);
#  (2) the UDA-legal height-alignment test on nuScenes -> Waymo, eval-only. Offsets are COMPUTED by
#      analysis/ground_height_offset.py from TRAIN clouds + the published extrinsic (no sweep, no selection):
#      rule G (ground alignment, method rule) target SHIFT_COOR z = -0.044; rule S (sensor alignment, analysis)
#      -0.434. Each scored once on 27471 (GBlobs + accumulation) and 27468 (MeanVFE accumulation).
# Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
GB=/home/koyama/data/wandb/run-20261006_024401-jguwly5u/files/ckpt/checkpoint_epoch_20.pth
MV=/home/koyama/data/wandb/run-20261006_004144-o3l96a3p/files/ckpt/checkpoint_epoch_20.pth
P=analysis/pseudo_label_threshold/cfgs
echo "##### (1) step 2, teacher 27471"
bash scripts/_legal_chain_step2_teacher.sh "$GB" cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2kitti.yaml \
  $P/teacher_s2gblobs_on_nuscenes_val_n008_15sweeps.yaml $P/teacher_s2gblobs_on_nuscenes_val_n015_10sweeps.yaml \
  /storage/pseudo_labels/legaldepth27471gb
BASE_NOTE="Height alignment on nuScenes -> Waymo, eval-only, UDA-legal offset (20261007_01 §7): target SHIFT_COOR z computed by analysis/ground_height_offset.py from TRAIN clouds + published TOP extrinsic, scored once, never swept."
ev() {  # label cfg ckpt tag z
  echo "##### (2) $1 z=$5"
  WANDB_NOTES="$BASE_NOTE $1 at z = $5. Compare the same model as scored (27596 GBlobs 51.58 / 28.35; 27590 MeanVFE 48.30 / 23.47)." \
    bash analysis/eval_checkpoint.sh "$2" "$3" "$4" 20260922_sourceonly --set DATA_CONFIG_TAR.SHIFT_COOR "[0.0,0.0,$5]"
  echo "eval $4 exit $?"
}
ev "rule S (sensor alignment, analysis), GBlobs+accum 27471" cfgs/da-ieee-access/centerpoint-gblobs-sourceonly-nuscenes2waymo.yaml "$GB" height_ruleS_m0434 -0.434
ev "rule S (sensor alignment, analysis), MeanVFE accum 27468" cfgs/da-ieee-access/centerpoint-accum-legaldepth-nuscenes2waymo.yaml "$MV" height_ruleS_m0434 -0.434
ev "rule G (ground alignment, method rule), GBlobs+accum 27471" cfgs/da-ieee-access/centerpoint-gblobs-sourceonly-nuscenes2waymo.yaml "$GB" height_ruleG_m0044 -0.044
ev "rule G (ground alignment, method rule), MeanVFE accum 27468" cfgs/da-ieee-access/centerpoint-accum-legaldepth-nuscenes2waymo.yaml "$MV" height_ruleG_m0044 -0.044
echo "##### all done"
