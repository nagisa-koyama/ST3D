#!/usr/bin/env bash
# SCENE, pre-launch smoke of one PL S->B + domain-loss row (experiments_md 20261022_06 §10). One GPU, ~10 min.
#
#   scripts/_scene_row_smoke.sh <cfg> <short tag> [grad]      (submit as a script job via scripts/submit.sh; never --wrap)
#
#   (1) CPU unit tests of the DANN / GRL / PCGrad-DDP paths
#   (2) [if 'grad'] analysis/dann_grad_smoke.py: the row's real student on real Singapore / pseudo-labelled Boston batches -
#       domain loss on both halves, both upstream gradient norms, GRL sign (cosine -1, norm ratio = alpha), discriminator
#       accuracy, pseudo-labels untouched, plain BN
#   (3) analysis/target_dump.py on the row's config, TAR, 20 frames, the stage-2 cut labels of 30241 (as 30243's §8.4)
#   (4) train.py through the real self-training loop, 1 GPU, --use_subset (16 frames per loader), 1 epoch, then one checkpoint
#       scored on the full Boston val
# Runs from the LIVE checkout. Do not edit while a job runs it.
set -u
CFG=${1:?usage: _scene_row_smoke.sh <cfg> <short tag> [grad]}
TAG=${2:?usage: _scene_row_smoke.sh <cfg> <short tag> [grad]}
GRAD=${3:-}
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git -C /home/koyama/code/ST3D rev-parse --short HEAD) on $(hostname) at $(date '+%F %T') ==="
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
CK=/storage/wandb/run-20261022_225659-6iu7h6p5/files/ckpt/checkpoint_epoch_20.pth   # S->S 30189 epoch 20 (teacher + init)
PS=/storage/pseudo_labels/scene_ss_ep20_boston_train/ps_label_e0_cut.pkl            # stage-2 cut labels of 30241
X="singularity exec --nv --bind /home/koyama/data/:/storage $SIF"
RC=0
echo "##### (1) unit tests"
(cd /home/koyama/code/ST3D && $X python3 -m pytest -q -p no:warnings tests/test_centerpoint_dann.py tests/test_dann_ddp_grl.py \
  tests/test_dann_grl_coeff.py tests/test_dann_pcgrad_ddp.py tests/test_self_training_ddp_grad_sync.py tests/test_conditional_grl.py)
R=$?; echo "unit tests exit $R"; [ "$R" -eq 0 ] || RC=$R
if [ "$GRAD" = "grad" ]; then
  echo "##### (2) gradient smoke on the real model"
  $X python3 analysis/dann_grad_smoke.py "$CFG" "$CK" "$PS" 3
  R=$?; echo "grad smoke exit $R"; [ "$R" -eq 0 ] || RC=$R
fi
echo "##### (3) target dump (the row's config, TAR, 20 frames)"
$X python3 analysis/target_dump.py "$CFG" TAR 20 "$PS"
R=$?; echo "target dump exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "##### (4) real loop, 1 GPU, 16-frame subsets, 1 epoch, full Boston val eval"
$X python3 train.py --cfg_file "$CFG" --pretrained_model "$CK" --pretrained_model_teacher "$CK" \
  --batch_size 3 --epochs 1 --use_subset --num_epochs_to_eval 0 --workers 2 --fix_random_seed \
  --extra_tag "smoke_${TAG}" --run_name "smoke_${TAG}_nuscenes_singapore2boston"
R=$?; echo "train smoke exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "=== done $(date '+%F %T') rc=$RC ==="
exit "$RC"
