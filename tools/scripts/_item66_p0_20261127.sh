#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 66 job P0 (design fixed 2026-11-27 20:55 JST, before any number): the frozen teacher
# T = 27468 epoch 20 + global AdaBN (SELF_TRAIN.TEACHER_ADABN of the item-66 base config, statistics from KITTI train clouds)
# labels every KITTI train frame at the score floor 0.0001 -> /home/koyama/data/item66/p0/ps_label_e0.pkl. No target label is
# read. The cuts follow in a CPU job (analysis/item66_realpair.py cuts). Usage (from ST3D/tools, 1 GPU):
#   bash scripts/_item66_p0_20261127.sh
# Do not edit while a job runs it.
set -eu
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
T=/home/koyama/data/wandb/run-20261006_004144-o3l96a3p/files/ckpt/checkpoint_epoch_20.pth
test -f "$T"
echo "=== item66 P0 ($(date '+%F %T')) ==="
singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
  python3 analysis/pseudo_label_threshold/generate_pseudo_labels.py \
    --cfg_file cfgs/da-ieee-access-analysis/item66/centerpoint-item66-base.yaml --teacher_ckpt "$T" \
    --out_dir /home/koyama/data/item66/p0 --thresh 0.0001 < /dev/null
test -s /home/koyama/data/item66/p0/ps_label_e0.pkl
echo "=== item66 P0 done ($(date '+%F %T')) ==="
