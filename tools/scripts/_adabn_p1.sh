#!/usr/bin/env bash
# P1 (experiments_md 20261010_10 §8, pre-declared): first-BN-only vs full AdaBN on two pairs never used to choose the
# layer. Eval-only, one GPU, three arms per pair: none (N = 1 rescore control), full AdaBN, first BN only. Statistics
# from 1,000 strided nuScenes TRAIN frames (the tool's default), no labels, no weight changed.
# Run inside an sbatch allocation:  --wrap="bash scripts/_adabn_p1.sh <pandaset|kitti>"
# Do not edit while a job runs it (bash reads scripts incrementally).
set -u
cd /home/koyama/code/ST3D/tools
S=/storage/wandb
FIRST='^backbone_3d\.conv_input\.1$'
C=cfgs/da-ieee-access
case "${1:?pair: pandaset|kitti}" in
  pandaset) CFG=$C/centerpoint-sourceonly-pandaset.yaml; CKPT=$S/run-20260929_043502-7jdabg72/files/ckpt/checkpoint_epoch_115.pth ;;  # 26533
  kitti)    CFG=$C/centerpoint-sourceonly-kitti.yaml;    CKPT=$S/run-20260923_211459-omw8p3nu/files/ckpt/checkpoint_epoch_152.pth ;;  # 25817
  *) echo "unknown pair $1" >&2; exit 2 ;;
esac
P=$1
bash analysis/adabn_eval.sh "$CFG" "$CKPT" none   1.0 adabnP1_${P}_none;  echo "arm none exit $?"
bash analysis/adabn_eval.sh "$CFG" "$CKPT" target 1.0 adabnP1_${P}_full;  echo "arm full exit $?"
bash analysis/adabn_eval.sh "$CFG" "$CKPT" target 1.0 adabnP1_${P}_first --layers "$FIRST" --layers_name first
echo "arm first exit $?"
echo "=== done ==="
