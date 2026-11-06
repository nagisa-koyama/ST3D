#!/bin/bash
# experiments_md 20261003_05 §5 item 33c (user 2026-11-06 ~22:30 JST: 「回す」, design fixed before any number of its own):
# third seed (668) of the S2 factor-swap rows R1 (lines) and R3 (density), whose two seeds moved 21.5 and 7.95 AP.
# Each is its seed-667 job's submit line (31946 / 31948, item 33a) with --seed 668: config, 1 GPU x 6 CPUs, frame-seeded
# swapped set unchanged. ANALYSIS (KITTI TRAIN labels). Submitted one at a time as slots of the 12-job cap free.
#
# Run once per row from anywhere: bash scripts/_item33c_launch_20261106.sh <r1|r3>
set -euo pipefail
cd /home/koyama/code/ST3D/tools
case ${1:?usage: _item33c_launch_20261106.sh <r1|r3>} in
  r1) r=r1-lines;   TERM="lines = mean(R2 31410, 31947) 76.36 - mean(R1 31409, 31946, this) (two seeds +5.39 / +26.79)";;
  r3) r=r3-density; TERM="density = mean(O 26843, 31817) 79.56 - mean(R3 31414, 31948, this) (two seeds +22.12 / +12.63)";;
  *) echo "unknown row $1" >&2; exit 2;;
esac
n=${r%%-*}
D=cfgs/da-ieee-access-analysis/factorswap-s2
[ -f $D/centerpoint-factorswap-s2-$r.yaml ] || { echo "missing config $r" >&2; exit 2; }
WANDB_NOTES="Item 33c (T05 §5, user 11-06 「回す」), labelled ANALYSIS (KITTI TRAIN labels): FACTOR-SWAP S2 ${n^^} seed 668, config and layout of the seed-667 row unchanged (1 GPU x 6), same frame-seeded swapped set. Term over three seeds of this row: $TERM. Read if the mean is >= 2 and the term has the same sign at every seed." \
  scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 --exclude=node03 --gres=gpu:1 --cpus-per-task=6 --mem=96G \
  --time=1-06:00:00 --job-name=fs${n}s3 \
  scripts/run_sourceonly_2gpu.sh $D/centerpoint-factorswap-s2-$r.yaml fsS2_$r-seed668 factorswap-s2-$r-seed668 --seed 668 < /dev/null
