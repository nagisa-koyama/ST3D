#!/usr/bin/env bash
# Submit a 2-GPU smoke test of SELF-TRAINING under DDP (experiments_md/20260928_01). Not an sbatch
# script itself: it goes through scripts/submit.sh + scripts/run_sourceonly_2gpu.sh, so the launch
# is the SAME torch.distributed.launch path, frozen-code snapshot and auto eval-recovery every real
# 2-GPU row uses - a smoke test of a different launcher would prove nothing about that one.
#
#   scripts/_smoke_st_ddp.sh global     # ST3D + global correction: the plain summed backward
#   scripts/_smoke_st_ddp.sh pcgrad     # ST3D + global + DANN + PCGrad: the torchjd manual-sync path
#   scripts/_smoke_st_ddp.sh fgv2       # ST3D + foreground v2: pseudo-labelled target re-spawned
#                                       #   after the calibration (the spawn-pickling path, twice)
#
# What green means: two ranks run the pseudo-label pass (shards gathered), the target workers are
# spawned with the labels on board (no "Cannot find pseudo label"), training iterates, one
# checkpoint is scored on the full nuScenes val, and for pcgrad `train/grad_similarity_*` is logged
# from rank 0 only. --use_subset cuts every loader to 16 frames, so nothing here is a measurement.
set -euo pipefail
cd /home/koyama/code/ST3D/tools
ARM=${1:?global | pcgrad | fgv2}
case "$ARM" in
  global) CFG=cfgs/da-ieee-access-tier1/centerpoint-st3d-global-lyft2nuscenes-4ep.yaml ;;
  pcgrad) CFG=cfgs/da-ieee-access-tier1/centerpoint-st3d-global-dann-pcgrad-lyft2nuscenes-4ep.yaml ;;
  fgv2)   CFG=cfgs/da-ieee-access-tier1/centerpoint-foreground-v2-lyft2nuscenes-4ep.yaml ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
T=/storage/wandb/run-20260923_093504-iwg6l5v1/files/ckpt/checkpoint_epoch_30.pth
WANDB_NOTES="${WANDB_NOTES:-Smoke test, 2 GPUs (DDP), self-training arm '$ARM' on a 16-frame subset, 2 epochs: does train_model_st run under DDP after 20260928_01 (pseudo-labels carried on the dataset, torchjd manual grad sync)? Not a measurement.}" \
  scripts/submit.sh --gres=gpu:2 --cpus-per-task=10 --mem=96G --time=00:45:00 \
    --job-name="smk_ddp_$ARM" \
    scripts/run_sourceonly_2gpu.sh "$CFG" "smoke_ddp_$ARM" "smoke_ddp_st_${ARM}_lyft2nuscenes" \
      --pretrained_model "$T" --pretrained_model_teacher "$T" \
      --epochs 2 --use_subset --num_epochs_to_eval 0 --workers 2 \
      --set DATA_CONFIGS.LYFT_40BEAM.HIST_DIST_FRAMES 100 DATA_CONFIGS.LYFT_64BEAM.HIST_DIST_FRAMES 100
