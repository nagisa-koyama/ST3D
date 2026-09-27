#!/usr/bin/env bash
# Smoke test for the CenterPoint in-head DANN rows (experiments_md/20260927_06), ~15 min on 1 GPU.
#
#   scripts/submit.sh --gres=gpu:1 --cpus-per-task=10 --mem=64G --time=00:30:00 \
#       --job-name=smk_dann --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
#       scripts/_smoke_dann.sh dann        # ST3D + global + DANN
#   ... scripts/_smoke_dann.sh pcgrad      # ST3D + global + DANN + PCGrad
#
# What it proves, and what it does not. --use_subset cuts EVERY loader to 16 frames (source, target,
# and the pseudo-label pass, which wraps the target loader's dataset), --epochs 2 makes the
# PROG_AUG ramp [1, 2, 3] fire once, HIST_DIST_FRAMES 100 cuts the calibration from ~3.5 min to
# seconds, and --num_epochs_to_eval 0 scores exactly one checkpoint on the FULL nuScenes val
# (~5 min; not subsetted, and worth keeping: it is the eval path with a discriminator in the model).
# So a green run means: iwg6l5v1 loads with the discriminator's keys missing (expect
# "Not updated weight ... domain_discriminator ..."), dann_loss appears in the progress bar and W&B,
# the torchjd branch runs and logs train/grad_similarity_* (pcgrad arm), and one GPU holds it.
# It says NOTHING about AP, and its s/it is from 6 iterations, warm-up included: read it as an
# upper bound. The 4-epoch tier-1 proxies are the measurement.
set -euo pipefail
cd /home/koyama/code/ST3D/tools

ARM=${1:?dann | pcgrad}
case "$ARM" in
  dann)   CFG=cfgs/da-ieee-access-tier1/centerpoint-st3d-global-dann-lyft2nuscenes-4ep.yaml ;;
  pcgrad) CFG=cfgs/da-ieee-access-tier1/centerpoint-st3d-global-dann-pcgrad-lyft2nuscenes-4ep.yaml ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

# Teacher and student init: the global-correction model (job 25781, 28.22 BEV), as for every ST arm.
T=/storage/wandb/run-20260923_093504-iwg6l5v1/files/ckpt/checkpoint_epoch_30.pth

singularity exec --nv --bind /home/koyama/data/:/storage \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  python3 train.py \
    --cfg_file "$CFG" \
    --pretrained_model "$T" --pretrained_model_teacher "$T" \
    --batch_size 6 --epochs 2 --use_subset --num_epochs_to_eval 0 --workers 2 \
    --fix_random_seed \
    --extra_tag "smoke_$ARM" --run_name "smoke_st3d_global_${ARM}_lyft2nuscenes" \
    --set DATA_CONFIGS.LYFT_40BEAM.HIST_DIST_FRAMES 100 DATA_CONFIGS.LYFT_64BEAM.HIST_DIST_FRAMES 100
