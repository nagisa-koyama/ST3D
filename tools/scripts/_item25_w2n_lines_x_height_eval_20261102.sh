#!/bin/bash
# experiments_md 20261003_05 §5 item 25 (design fixed 2026-11-02 before any number): additivity on Waymo -> nuScenes of
# the scan-pattern lines (27807 / 27840) against the density twin C1 (27850 / 28278), crossed with the target height
# alignment. EVAL-ONLY: the four models train on Waymo only, so the nuScenes SHIFT_COOR acts only at evaluation and each
# cell is a rescoring of the existing epoch-7 checkpoint. ANALYSIS (nuScenes val labels only score).
#   G   z = 1.794  rule G, ground alignment (-0.021 + 0 + 1.815; the method rule, 20261007_01 §7.1)
#   A   z = 0      no height handling, nuScenes in its native sensor frame (ground 1.8 m below the trained ground)
#   B   z = 2.184  rule S, sensor-origin alignment (Waymo TOP 2.184 m, published extrinsic)
#   AS  z = 1.75   as scored in-run (reproduction check at the current commit)
# Each offset is computed, scored once, never swept.
cd /home/koyama/code/ST3D/tools
W=/home/koyama/data/wandb
P_CFG=cfgs/da-ieee-access/centerpoint-ringpattern-waymo2nuscenes.yaml
C_CFG=cfgs/da-ieee-access/centerpoint-global-toponly-recount-waymo2nuscenes.yaml
BASE_NOTE="Item 25 (20261003_05 §5), W->N lines x height alignment, eval-only rescoring of the epoch-7 checkpoint with the nuScenes SHIFT_COOR z set by rule, never swept; analysis."
ev() {  # label cfg ckpt tag z
  echo "##### $1 z=$5"
  WANDB_NOTES="$BASE_NOTE $1 at z = $5." \
    bash analysis/eval_checkpoint.sh "$2" "$3" "$4" 20260922_sourceonly --set DATA_CONFIG_TAR.SHIFT_COOR "[0.0,0.0,$5]"
  echo "eval $4 exit $?"
}
for arm in AS:1.75 G:1.794 A:0.0 B:2.184; do
  a=${arm%%:*}; z=${arm#*:}
  ev "lines P seed 666 (27807), arm $a" $P_CFG $W/run-20261008_212958-87h61vfd/files/ckpt/checkpoint_epoch_7.pth item25_${a}_27807 $z
  ev "lines P seed 667 (27840), arm $a" $P_CFG $W/run-20261009_074254-bpertm50/files/ckpt/checkpoint_epoch_7.pth item25_${a}_27840 $z
  ev "density twin C1 seed 666 (27850), arm $a" $C_CFG $W/run-20261009_075900-pybgphs0/files/ckpt/checkpoint_epoch_7.pth item25_${a}_27850 $z
  ev "density twin C1 seed 667 (28278), arm $a" $C_CFG $W/run-20261013_020040-n6acyx3p/files/ckpt/checkpoint_epoch_7.pth item25_${a}_28278 $z
done
echo "##### all done"
