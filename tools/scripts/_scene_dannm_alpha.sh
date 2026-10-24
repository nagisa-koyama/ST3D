#!/usr/bin/env bash
# SCENE, alpha_M of the matched-weight DANN row (experiments_md 20261022_06 §10; T05 §5 design fixed 2026-10-24 10:13 JST).
# 200 training iterations of 30306's setup (alpha = 1) from the init checkpoint (S->S 30189 ep 20), with 30306's own first-pass
# pseudo-labels; r = |g_src_det| / |g_dom| on the parameters upstream of the discriminator input; alpha_M = median, 2 s.f.
#   WANDB_NOTES="..." scripts/submit.sh --partition=a6000_ada,a6000,rtx8000 --gres=gpu:1 --cpus-per-task=8 --mem=64G \
#       --time=01:00:00 --job-name=dannmAlpha --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt scripts/_scene_dannm_alpha.sh
# A bash script job, never a --wrap. Runs from the LIVE checkout. Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git -C /home/koyama/code/ST3D rev-parse --short HEAD) on $(hostname) at $(date '+%F %T') ==="
singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  python3 analysis/dann_grad_ratio.py cfgs/da-ieee-access/centerpoint-st3d-dann-nuscenes-singapore2boston.yaml \
  /storage/wandb/run-20261022_225659-6iu7h6p5/files/ckpt/checkpoint_epoch_20.pth \
  /storage/wandb/run-20261024_030135-oswx20xc/files/ps_label/ps_label_e0.pkl 200 /storage/scene_dann/alpha_M_30306setup.json
R=$?; echo "alpha measurement exit $R at $(date '+%F %T')"; exit $R
