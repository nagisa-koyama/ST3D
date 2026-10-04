#!/usr/bin/env bash
# DIAGNOSTIC (experiments_md/20261003_03 s7): does each with-intensity in-domain oracle USE intensity?
# Scores every oracle on its own eval set twice, eval-only, with intensity shuffled among each frame's points
# and with intensity set to the dataset median (cfgs/da-ieee-access-intensity/ablate/). Read each against the
# oracle's banked raw score. One GPU, sequential; submit with
#   WANDB_NOTES=... scripts/submit.sh --partition=a6000_ada,a6000,rtx8000 --gres=gpu:1 --cpus-per-task=8 \
#     --mem=64G --time=03:00:00 --job-name=xiAblate --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
#     --wrap="bash analysis/intensity_ablation_evals.sh"
set -u
cd /home/koyama/code/ST3D/tools
W=/home/koyama/data/wandb
A=cfgs/da-ieee-access-intensity/ablate
ROWS=(
  "kitti    $W/run-20261003_105442-8kseeo7z/files/ckpt/checkpoint_epoch_152.pth"
  "nuscenes $W/run-20261003_173748-xhj87sz8/files/ckpt/checkpoint_epoch_20.pth"
  "spin     $W/run-20261003_230622-ukuzfkp3/files/ckpt/checkpoint_epoch_15.pth"
  "flash    $W/run-20261003_182510-z6of5mlr/files/ckpt/checkpoint_epoch_15.pth"
  "waymo    $W/run-20261003_173053-buxwi54n/files/ckpt/checkpoint_epoch_7.pth"
)
status=0
for row in "${ROWS[@]}"; do
  read -r name ckpt <<< "$row"
  for mode in shuffle constant; do
    echo "=== ablate_intensity $name $mode ($(date '+%F %T')) ==="
    bash analysis/eval_checkpoint.sh "$A/centerpoint-xi-ablate-$name-$mode.yaml" "$ckpt" "ablate_${mode}" 20261004_xi_ablate \
      || { echo "=== FAILED: $name $mode ==="; status=1; }
  done
done
exit $status
