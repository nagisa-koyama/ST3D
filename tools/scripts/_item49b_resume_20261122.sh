#!/usr/bin/env bash
# Item 49b amendment (experiments_md 20261003_05 §5 item 49b, 2026-11-22 03:20 JST, procedure only, before any number):
# E2 (34828) runs at ~1.2-1.7 s/it on node02 (28283, its seed-667 twin, ran 0.44 s/it on node06) and is projected past its
# 30 h limit at about epoch 17-18. By the standing rule a row that hits its limit is RESUMED, not retrained (T14; precedent
# 32630): this script, chained afterany:34828, resumes it from its last checkpoint with the same command plus --ckpt.
# train.py --ckpt restores model, optimizer, epoch and iteration; the frozen teacher's epoch-0 pseudo-labels are reloaded
# from the row's ps_label dir (same extra tag); only augmentation RNG draws differ from an uninterrupted run.
# If 34828 reached epoch 20, it exits without training and the eval step reads 34828.
set -eu
cd /home/koyama/code/ST3D/tools
J=34828
if bash analysis/ckpt_of_job.sh "$J" 20 >/dev/null 2>&1; then
  echo "item49b resume: $J saved checkpoint_epoch_20; nothing to resume"
  exit 0
fi
FIRST=$(bash analysis/ckpt_of_job.sh "$J" 1)              # exits 2 if the row never saved an epoch
CKD=$(dirname "$FIRST")
LAST=$(ls "$CKD"/checkpoint_epoch_*.pth | sort -V | tail -n 1)
echo "item49b resume: $J stopped after $(basename "$LAST"); resuming"
exec bash scripts/run_sourceonly_2gpu.sh cfgs/da-ieee-access/centerpoint-accum-legaldepth-st3d-nuscenes2kitti.yaml \
  i46E2_s2 i46E2_s2_resume \
  --pretrained_model /storage/wandb/run-20261006_004144-o3l96a3p/files/ckpt/checkpoint_epoch_20.pth \
  --pretrained_model_teacher /storage/wandb/run-20261006_004144-o3l96a3p/files/ckpt/checkpoint_epoch_20.pth \
  --seed 667 --ckpt "${LAST/#\/home\/koyama\/data//storage}" \
  --set SELF_TRAIN.TAR.LOSS_WEIGHT 0.0
