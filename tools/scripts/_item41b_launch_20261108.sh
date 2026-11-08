#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 41b (design fixed 2026-11-08 19:32 JST; approved by the user 2026-11-08 21:14 JST): one
# seed of L_K, the KITTI count-matched lattice control, trained at 2 GPUs. The recipe of 32289 / 32290
# (centerpoint-factorswap-s2-l-latcount.yaml, 152 epochs, total batch 6, in-run eval on unperturbed KITTI val) with only the
# layout changed (3 per GPU x 2 GPUs), WITHOUT --ddp_sampler_seed, the condition of L_W and Pandar64 L. The config is
# unchanged since 9361e73 (the commit 32289 / 32290 froze) and its count gate passed in item 37c.
# ANALYSIS: trained on KITTI TRAIN labels, never a method.
#   bash scripts/_item41b_launch_20261108.sh <seed>      # 666 or 667, one call per free slot of the 12-job cap
set -euo pipefail
cd /home/koyama/code/ST3D/tools
SEED=${1:?seed}
C=cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-l-latcount.yaml
[ -f $C ] || { echo "missing $C" >&2; exit 2; }
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
H="Item 41b (T05 §5, design fixed 2026-11-08 19:32 JST, approved 21:14 JST), labelled ANALYSIS (KITTI TRAIN labels)"
T=$(EST_H=13 WANDB_NOTES="$H: L_K at 2 GPUs, seed $SEED, the recipe of 32289 / 32290 with only the layout changed (3 per GPU x 2 GPUs), without --ddp_sampler_seed. Layout cross-check of the 1-GPU L_K spread 16.23." \
  scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:2 --cpus-per-task=12 --mem=96G --time=1-06:00:00 \
  --exclude=node03 --job-name=i41LK2g${SEED: -1} \
  scripts/run_sourceonly_2gpu.sh $C 20261108_i41_2gpu_seed$SEED factorswap-s2-l-latcount-2gpu-seed$SEED --seed $SEED \
  < /dev/null | jid)
[ -n "$T" ] || { echo "submission of seed $SEED failed" >&2; exit 3; }
echo "item41b seed $SEED: $T"
