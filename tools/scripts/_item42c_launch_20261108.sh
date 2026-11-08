#!/bin/bash
# experiments_md 20261003_05 §5 item 42c (design fixed 2026-11-08 22:40 JST, before any number; report 20261108_07): R7r,
# the random label-drop twin of the label-range swap R7, on the targets where R7 is read "+". One row per call:
#     bash scripts/_item42c_launch_20261108.sh <target> <seed>        target: waymo | pandaset | lyft64; seed: 666 | 667
# Each row is its target's R7 row's recipe (config, layout, epochs): the R7r config is R7's with GT_IGNORE_BY_RANGE
# replaced by GT_IGNORE_RANDOM {CLASS: Car, KEEP: R7's kept share} (checked against both R7 jobs' logged configs by
# analysis/item42c_random_drop_check.py). Seed 666 passes no --seed (the default, as R7's seed-666 rows); 667 passes
# --seed 667 (as R7's seed-667 rows). User policy 2026-11-09 ~01:50 JST (「2GPUで複数実行する場合はseedを変えてシャッフルすること」):
# a 2-GPU row at a second seed also passes --ddp_sampler_seed (train.py, ST3D 4366f05), so its DDP frame order follows the
# seed: Waymo / Pandar64 seed 667 pass `--seed 667 --ddp_sampler_seed`. Seed-666 rows and the 1-GPU Lyft-64 rows are
# unchanged (1 GPU: --seed already sets the frame order). R7's 2-GPU seed-667 rows (31812, 32125) ran WITHOUT the flag
# (seed 0 + epoch order at both seeds), so a 2-GPU R7r seed-667 row differs from its R7 twin in frame order too.
# The IGNORE set is frame-seeded (frame key + box index + arm), so both seeds train on the same set.
# Rows (R7 seed 666 / 667 jobs in brackets):
#   waymo    (31621 / 31812)  KEEP 0.5647, 2 GPUs x 3, 7 epochs
#   pandaset (31981 / 32125)  KEEP 0.4484, 2 GPUs x 3, 115 epochs, rule-A CPU successor (spin val, epoch 115)
#   lyft64   (32083 / 32127)  KEEP 0.6400, 1 GPU x 6, 30 epochs
# Readings (T05 item 42c): count part = O - R7r; range selectivity = R7r - R7; one seed >= 3; two seeds mean >= 2 with
# the same sign; per-seed spread > 3 -> sign only. ANALYSIS (each target's TRAIN labels).
# The coordinating session launches these into free slots of the 12-job cap, in the order of T05 item 42c.
set -euo pipefail
cd /home/koyama/code/ST3D/tools
TGT=${1:?target}; SEED=${2:?seed}
A=cfgs/da-ieee-access-analysis
RS=/home/koyama/data/factorswap_pandaset/rescore
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
case $SEED in 666|667) ;; *) echo "seed must be 666 or 667" >&2; exit 2 ;; esac
case $TGT in
  waymo)    C=$A/factorswap-n2w/centerpoint-factorswap-n2w-r7r-labelrandom.yaml;           R=w2;  K=0.5647
            N="Waymo R7r random label drop"; R7J=$([ "$SEED" = 666 ] && echo 31621 || echo 31812) ;;
  pandaset) C=$A/factorswap-pandaset/centerpoint-factorswap-pandaset-r7r-labelrandom.yaml; R=p2;  K=0.4484
            N="Pandar64 R7r random label drop"; R7J=$([ "$SEED" = 666 ] && echo 31981 || echo 32125) ;;
  lyft64)   C=$A/factorswap-lyft64/centerpoint-factorswap-lyft64-r7r-labelrandom.yaml;     R=l64; K=0.6400
            N="Lyft-64 R7r random label drop"; R7J=$([ "$SEED" = 666 ] && echo 32083 || echo 32127) ;;
  *) echo "unknown target $TGT (waymo | pandaset | lyft64)" >&2; exit 2 ;;
esac
[ -f $C ] || { echo "missing $C" >&2; exit 2; }
case $R in
  w2)  RES="--gres=gpu:2 --cpus-per-task=20 --mem=150G --time=24:00:00"; EST=17; NEED=1 ;;
  p2)  RES="--gres=gpu:2 --cpus-per-task=20 --mem=150G --time=24:00:00"; EST=25; NEED=2 ;;
  l64) RES="--gres=gpu:1 --cpus-per-task=6 --mem=96G --time=12:00:00"; EST=6; NEED=1 ;;
esac
# Slurm MaxSubmit is 20 per user
Q=$(squeue -u koyama -h | wc -l)
[ $((Q + NEED)) -le 20 ] || { echo "only $((20 - Q)) submissions left, need $NEED" >&2; exit 2; }
STEM=$(basename $C .yaml | sed 's/^centerpoint-//')
EXTRA=""; [ "$SEED" = 666 ] || EXTRA="--seed $SEED"
# user policy 2026-11-09: a second-seed 2-GPU row shuffles by its seed
[ "$SEED" = 666 ] || [ $R = l64 ] || EXTRA="$EXTRA --ddp_sampler_seed"
SAMPLER=""; [ "$SEED" = 666 ] || [ $R = l64 ] || SAMPLER=" With --ddp_sampler_seed (user policy 2026-11-09; the R7 twin ran without it)."
TAG=20261108_fs42c_${STEM}-seed$SEED
H="Item 42c (T05 §5, design fixed 2026-11-08 22:40 JST; report 20261108_07), labelled ANALYSIS"
T=$(EST_H=$EST WANDB_NOTES="$H: $N, seed $SEED (config $STEM): each Car label IGNORE (-99) with probability 1 - $K independent of range, $K = R7's kept share; frame-seeded set, same at both seeds. Twin of R7 job $R7J (same recipe and seed).$SAMPLER Count part = O - R7r, range selectivity = R7r - R7, paired by seed." \
  scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 $RES --exclude=node03 --job-name=f42c${TGT:0:1}s${SEED: -1} \
  scripts/run_sourceonly_2gpu.sh $C $TAG ${STEM}-seed$SEED $EXTRA < /dev/null | jid)
[ -n "$T" ] || { echo "submission of $TGT seed $SEED failed" >&2; exit 3; }
echo "item42c $TGT seed $SEED: $T"
if [ $R = p2 ]; then
  mkdir -p $RS
  S=$(WANDB_NOTES="CPU only, no W&B: item 42c (T05 §5) rule-A / as-scored rescoring of $N seed $SEED ($T, epoch 115, spin val)" \
    scripts/submit.sh --dependency=afterany:$T --kill-on-invalid-dep=yes --partition=a100,a6000_ada --cpus-per-task=4 \
    --mem=24G --time=02:00:00 --job-name=i42cPrs --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
    --wrap="bash analysis/pandaset_rule_a_of_jobs.sh $RS/rescore_42c_p_s$SEED.txt '$N seed $SEED=$T:115:0:false'" \
    < /dev/null | jid)
  [ -n "$S" ] || { echo "rule-A successor of $T failed; submit it by hand" >&2; exit 3; }
  echo "item42c $TGT seed $SEED rule A: $S"
fi
