#!/bin/bash
# experiments_md 20261003_05 §5 item 27 (design fixed 2026-11-02 before any number): additivity on nuScenes -> KITTI (S2)
# of accumulation x GBlobs x the target height alignment (rule G), a 2 x 2 x 2 design. EVAL-ONLY: the four source-only
# models train on nuScenes only, so the KITTI SHIFT_COOR acts only at evaluation and each height cell is a rescoring of
# the existing final checkpoint (epoch 20). ANALYSIS (KITTI val labels only score).
#   AS  z = 1.70   as scored in-run (reproduction check at the current commit; must land within +-0.3 BEV)
#   G   z = 1.706  rule G, ground alignment (g_src -1.815 + S_src 1.75 - g_tgt -1.771; Slurm CPU job 31453)
#   A   z = 0      no height handling, KITTI in its raw sensor frame (ground 1.77 m below the trained ground; points
#                  below z -2 are cropped by POINT_CLOUD_RANGE, disclosed)
# Rule S (sensor-origin alignment) gives 1.750 on S2, 4.4 cm from rule G: not a distinct arm, not scored.
# Each offset is computed, scored once, never swept.
cd /home/koyama/code/ST3D/tools
W=/home/koyama/data/wandb
C_CFG=cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml          # control, MeanVFE single sweep
A_CFG=cfgs/da-ieee-access/centerpoint-accum-legaldepth-nuscenes2kitti.yaml    # MeanVFE accumulation 15 / 10
G_CFG=cfgs/da-ieee-access/centerpoint-gblobs-sourceonly-nuscenes2kitti.yaml   # GBlobs single sweep
GA_CFG=cfgs/da-ieee-access/centerpoint-gblobs-accum-legaldepth-nuscenes2kitti.yaml  # GBlobs + accumulation 15 / 10
BASE_NOTE="Item 27 (20261003_05 §5), S2 accumulation x GBlobs x height alignment, eval-only rescoring of the final checkpoint with the KITTI SHIFT_COOR z set by rule, never swept; analysis."
ev() {  # label cfg ckpt tag z
  echo "##### $1 z=$5"
  WANDB_NOTES="$BASE_NOTE $1 at z = $5." \
    bash analysis/eval_checkpoint.sh "$2" "$3" "$4" 20260922_sourceonly --set DATA_CONFIG_TAR.SHIFT_COOR "[0.0,0.0,$5]"
  echo "eval $4 exit $?"
}
for arm in AS:1.70 G:1.706 A:0.0; do
  a=${arm%%:*}; z=${arm#*:}
  ev "control MeanVFE single sweep (26265, ckpt wfdcq75s), arm $a" $C_CFG $W/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth item27_${a}_26265 $z
  ev "MeanVFE accumulation seed 666 (27468), arm $a" $A_CFG $W/run-20261006_004144-o3l96a3p/files/ckpt/checkpoint_epoch_20.pth item27_${a}_27468 $z
  ev "GBlobs single sweep (27263), arm $a" $G_CFG $W/run-20261004_080341-id1m6yxi/files/ckpt/checkpoint_epoch_20.pth item27_${a}_27263 $z
  ev "GBlobs + accumulation (27471), arm $a" $GA_CFG $W/run-20261006_024401-jguwly5u/files/ckpt/checkpoint_epoch_20.pth item27_${a}_27471 $z
  ev "MeanVFE accumulation seed 667 (27804, noise check), arm $a" $A_CFG $W/run-20261008_202000-udtv9tvw/files/ckpt/checkpoint_epoch_20.pth item27_${a}_27804 $z
done
echo "##### all done"
