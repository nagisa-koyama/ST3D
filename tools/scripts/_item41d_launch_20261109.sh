#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 41d (design fixed 2026-11-09 02:05 JST; user: 「sync batch norm 2 GPUと1GPUが同等の結果を示すことを
# 確認してほしい」): one seed of the S2 source-only control at 2 GPUs x 3 with --sync_bn (BN over the global batch of 6) and
# --ddp_sampler_seed (per-seed frame order), the recipe of 31818 / 32207 / item 41a otherwise, plus its exempt 1-GPU evaluation
# successor (afterany). UDA-LEGAL: nuScenes labels only. Read against the 41a 1-GPU seeds (T05 item 41d reading).
#   bash scripts/_item41d_launch_20261109.sh <seed>      # 666, 667 or 668, one call per free slot of the 12-job cap
set -euo pipefail
cd /home/koyama/code/ST3D/tools
SEED=${1:?seed}
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
H="Item 41d (T05 §5, design fixed 2026-11-09 02:05 JST), UDA-LEGAL (nuScenes labels only)"
T=$(EST_H=26 WANDB_NOTES="$H: S2 source-only control at 2 GPUs x 3 with --sync_bn and --ddp_sampler_seed, seed $SEED; the 1-GPU x 6 computation of item 41a split over two ranks. Equivalence check against the 41a seeds (nuScenes val and KITTI val, means and spreads)." \
  scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:2 --cpus-per-task=20 --mem=150G --time=24:00:00 --exclude=node03 \
  --job-name=i41dS${SEED: -1} scripts/run_sourceonly_2gpu.sh cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml \
  20261109_i41d_2gpusync_seed$SEED sourceonly-nuscenes-2gpusync-seed$SEED --seed $SEED --sync_bn --ddp_sampler_seed < /dev/null | jid)
[ -n "$T" ] || { echo "submission of seed $SEED failed" >&2; exit 3; }
echo "item41d seed $SEED: $T"
E=$(WANDB_NOTES="$H: evaluations of $T (seed $SEED) on KITTI val (epochs 18-20), AdaBN with KITTI TRAIN statistics and the three own-input twins of item 39 B2" \
  scripts/submit.sh --dependency=afterany:$T --kill-on-invalid-dep=yes --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:1 \
  --cpus-per-task=8 --mem=96G --time=02:30:00 --job-name=i41dev${SEED: -1} --output=logs/output_%j_%x.txt \
  --error=logs/error_%j_%x.txt --wrap="bash scripts/_item41d_evals_20261109.sh $T $SEED" < /dev/null | jid)
[ -n "$E" ] || { echo "evaluation successor of $T failed; submit it by hand" >&2; exit 3; }
echo "item41d seed $SEED evals: $E"
