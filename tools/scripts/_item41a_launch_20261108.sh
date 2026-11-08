#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 41a (design fixed 2026-11-08 19:32 JST): one seed of the S2 source-only control at 1 GPU,
# the recipe of 31818 / 32207 (centerpoint-sourceonly-nuscenes.yaml, 20 epochs, total batch 6) with only the layout changed
# (6 per GPU x 1 GPU), plus its exempt 1-GPU evaluation successor (afterany). UDA-LEGAL: nuScenes labels only.
#   bash scripts/_item41a_launch_20261108.sh <seed>      # 666, 667 or 668, one call per free slot of the 12-job cap
set -euo pipefail
cd /home/koyama/code/ST3D/tools
SEED=${1:?seed}
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
H="Item 41a (T05 §5, design fixed 2026-11-08 19:32 JST), UDA-LEGAL (nuScenes labels only)"
T=$(EST_H=16 WANDB_NOTES="$H: S2 source-only control at 1 GPU, seed $SEED, the recipe of 31818 / 32207 with only the layout changed (6 per GPU x 1 GPU). Layout cross-check of the 2-GPU seeds 25743 / 31818 / 32207 (KITTI spread 9.81)." \
  scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:1 --mem=150G --exclude=node03 --job-name=i41c1g${SEED: -1} \
  scripts/run_sourceonly_2gpu.sh cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml 20261108_i41_1gpu_seed$SEED \
  sourceonly-nuscenes-1gpu-seed$SEED --seed $SEED < /dev/null | jid)
[ -n "$T" ] || { echo "submission of seed $SEED failed" >&2; exit 3; }
echo "item41a seed $SEED: $T"
E=$(WANDB_NOTES="$H: evaluations of $T (seed $SEED) on KITTI val (epochs 18-20), AdaBN with KITTI TRAIN statistics and the three own-input twins of item 39 B2" \
  scripts/submit.sh --dependency=afterany:$T --kill-on-invalid-dep=yes --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:1 \
  --cpus-per-task=8 --mem=96G --time=02:30:00 --job-name=i41ev${SEED: -1} --output=logs/output_%j_%x.txt \
  --error=logs/error_%j_%x.txt --wrap="bash scripts/_item41_evals_20261108.sh $T $SEED" < /dev/null | jid)
[ -n "$E" ] || { echo "evaluation successor of $T failed; submit it by hand" >&2; exit 3; }
echo "item41a seed $SEED evals: $E"
