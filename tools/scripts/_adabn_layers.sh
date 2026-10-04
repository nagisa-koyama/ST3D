#!/usr/bin/env bash
# First-layer-only vs deep-only AdaBN (experiments_md/20261005_01 §6, decisive test; 20261004_01).
# Eval-only, one GPU. Run inside an sbatch allocation:  --wrap="bash scripts/_adabn_layers.sh"
# FIRST = only the first BatchNorm (backbone_3d.conv_input.1, after the input sparse conv) re-estimated
#         on target train clouds; every other layer keeps its source statistics.
# DEEP  = every BatchNorm EXCEPT that one re-estimated.
# Do not edit while a job runs it (bash reads scripts incrementally).
set -u
cd /home/koyama/code/ST3D/tools
S=/storage/wandb
FIRST='^backbone_3d\.conv_input\.1$'
DEEP='^(?!backbone_3d\.conv_input\.1$)'
C=cfgs/da-ieee-access
MV2=$S/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth    # S2 MeanVFE single sweep (25743)
GB2=$S/run-20261004_080341-id1m6yxi/files/ckpt/checkpoint_epoch_20.pth    # S2 GBlobs single sweep (27263)
MV1=$S/run-20260924_111041-ldb35c2o/files/ckpt/checkpoint_epoch_30.pth    # S1 MeanVFE seed 1 (25835)
MV1b=$S/run-20261003_005227-n6giwlc8/files/ckpt/checkpoint_epoch_30.pth   # S1 MeanVFE seed 2 (27172)
GB1=$S/run-20260922_131445-qyerdcln/files/ckpt/checkpoint_epoch_30.pth    # S1 GBlobs (25723)
arm() {  # cfg ckpt layers-regex layers-name run-name
  bash analysis/adabn_eval.sh "$1" "$2" target 1.0 "$5" --layers "$3" --layers_name "$4"
  echo "arm $5 exit $?"
}
arm $C/centerpoint-sourceonly-nuscenes2kitti.yaml        "$MV2"  "$FIRST" first adabnL_S2mv_first
arm $C/centerpoint-sourceonly-nuscenes2kitti.yaml        "$MV2"  "$DEEP"  deep  adabnL_S2mv_deep
arm $C/centerpoint-gblobs-sourceonly-nuscenes2kitti.yaml "$GB2"  "$FIRST" first adabnL_S2gb_first
arm $C/centerpoint-gblobs-sourceonly-nuscenes2kitti.yaml "$GB2"  "$DEEP"  deep  adabnL_S2gb_deep
arm $C/centerpoint-sourceonly-lyft.yaml                  "$MV1"  "$FIRST" first adabnL_S1mv_first
arm $C/centerpoint-sourceonly-lyft.yaml                  "$MV1"  "$DEEP"  deep  adabnL_S1mv_deep
arm $C/centerpoint-sourceonly-lyft.yaml                  "$MV1b" "$FIRST" first adabnL_S1mv2_first
arm $C/centerpoint-gblobs-sourceonly-lyft.yaml           "$GB1"  "$DEEP"  deep  adabnL_S1gb_deep
arm $C/centerpoint-sourceonly-nuscenes2waymo.yaml        "$MV2"  "$FIRST" first adabnL_NWmv_first
echo "=== done ==="
