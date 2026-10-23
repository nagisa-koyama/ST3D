#!/usr/bin/env bash
# SCENE, pre-launch smoke of PL S->B + DANN (experiments_md/20261022_06 §9; design T05 §5 "User 10-24 ~02:40" item 2).
# One GPU, ~30 min. Submit as a script job, never a --wrap (dash):
#
#   WANDB_NOTES="..." scripts/submit.sh --partition=a6000_ada,a6000,rtx8000 --gres=gpu:1 --cpus-per-task=8 --mem=64G \
#       --time=01:00:00 --job-name=smkPLdann --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
#       scripts/_scene_pl_s2b_dann_smoke.sh
#
#   (1) CPU unit tests of the DANN path: tests/test_centerpoint_dann.py, tests/test_dann_ddp_grl.py (DDP over gloo)
#   (2) analysis/dann_grad_smoke.py: the row's real student on real Singapore / pseudo-labelled Boston batches - domain loss
#       on both halves, non-zero and sign-reversed gradient through the GRL into backbone_3d / backbone_2d parameters,
#       discriminator not reversed, pseudo-labels untouched, plain BN
#   (3) analysis/target_dump.py on the row's config, TAR, 20 frames, the stage-2 cut labels of 30241 (as 30243's §8.4)
#   (4) train.py through the real self-training loop, 1 GPU, --use_subset (16 frames per loader), 1 epoch: the domain loss
#       in the progress bar and W&B, then one checkpoint scored on the FULL Boston val (the eval path with a discriminator)
# Runs from the LIVE checkout. Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git -C /home/koyama/code/ST3D rev-parse --short HEAD) on $(hostname) at $(date '+%F %T') ==="
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
CFG=cfgs/da-ieee-access/centerpoint-st3d-dann-nuscenes-singapore2boston.yaml
CK=/storage/wandb/run-20261022_225659-6iu7h6p5/files/ckpt/checkpoint_epoch_20.pth   # S->S 30189 epoch 20 (teacher + init)
PS=/storage/pseudo_labels/scene_ss_ep20_boston_train/ps_label_e0_cut.pkl            # stage-2 cut labels of 30241
X="singularity exec --nv --bind /home/koyama/data/:/storage $SIF"
RC=0
echo "##### (1) unit tests"
(cd /home/koyama/code/ST3D && $X python3 -m pytest -q -p no:warnings tests/test_centerpoint_dann.py tests/test_dann_ddp_grl.py)
R=$?; echo "unit tests exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "##### (2) gradient smoke on the real model"
$X python3 analysis/dann_grad_smoke.py "$CFG" "$CK" "$PS" 3
R=$?; echo "grad smoke exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "##### (3) target dump (the row's config, TAR, 20 frames)"
$X python3 analysis/target_dump.py "$CFG" TAR 20 "$PS"
R=$?; echo "target dump exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "##### (4) real loop, 1 GPU, 16-frame subsets, 1 epoch, full Boston val eval"
$X python3 train.py --cfg_file "$CFG" --pretrained_model "$CK" --pretrained_model_teacher "$CK" \
  --batch_size 3 --epochs 1 --use_subset --num_epochs_to_eval 0 --workers 2 --fix_random_seed \
  --extra_tag smoke_pls2b_dann --run_name smoke_st3d_dann_nuscenes_singapore2boston
R=$?; echo "train smoke exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "=== done $(date '+%F %T') rc=$RC ==="
exit "$RC"
