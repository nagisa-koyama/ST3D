#!/bin/bash
# experiments_md 20261003_05 §5 items 37d / 37f (design fixed 2026-11-08 01:40 JST, before any number). One row per call:
#     bash scripts/_item37df_launch_20261108.sh <row> <seed>
# 37d: the Lyft pitch rows R5 (l64-r5, l40-r5) at seed 666, then 667 under 37f.
# 37f: a second seed (667) of every one-seed factor-swap row of the G64 table, launched as slots of the 12-job cap free
#      (lowest priority; R11 rows only after their seed-666 runs end; PandaSet rows get a CPU rule-A successor).
# Each row is its seed-666 job's recipe (config, layout, epochs) with --seed and a -seed<seed> tag; the factor-swap cuts
# are frame-seeded, so both seeds train on the same swapped data. Reading (T05 item 37): two-seed terms read if the mean is
# >= 2 with the same sign; one-seed rows read if >= 3. ANALYSIS (each target's TRAIN labels).
# Rows (seed-666 job in brackets):
#   k-r5 (31415) k-r9 (31413) k-r11 (32141) k-r11r (32143)             KITTI, 1 GPU x 6, 152 epochs
#   w-r5 (31619) w-r6 (31620) w-r11 (32142) w-r11r (32144)             Waymo N->W, 2 GPUs x 3, 7 epochs
#   p-r5 (31979) p-r6 (31980) p-r9 (32084) p-r11 (32171) p-r11r (32176) Pandar64, 2 GPUs x 3, 115 epochs, rule A after
#   l64-d (31977) l64-r6 (32075) l64-r5 (37d)                          Lyft-64, 1 GPU x 6, 30 epochs
#   l40-r6 (32010) l40-r5 (37d)                                        Lyft-40, 1 GPU x 6, 30 epochs
set -euo pipefail
cd /home/koyama/code/ST3D/tools
ROW=${1:?row}; SEED=${2:?seed}
A=cfgs/da-ieee-access-analysis
RS=/home/koyama/data/factorswap_pandaset/rescore
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
case $ROW in
  k-r5)   C=$A/factorswap-s2/centerpoint-factorswap-s2-r5-pitch.yaml;               R=k1; N="KITTI R5 pitch" ;;
  k-r9)   C=$A/factorswap-s2/centerpoint-factorswap-s2-r9-bg050.yaml;               R=k1; N="KITTI R9 background 1/2" ;;
  k-r11)  C=$A/factorswap-s2/centerpoint-factorswap-s2-r11-reflection.yaml;         R=k1; N="KITTI R11 reflection threshold" ;;
  k-r11r) C=$A/factorswap-s2/centerpoint-factorswap-s2-r11r-reflection-random.yaml; R=k1; N="KITTI R11r reflection random twin" ;;
  w-r5)   C=$A/factorswap-n2w/centerpoint-factorswap-n2w-r5-pitch.yaml;               R=w2; N="Waymo R5 pitch" ;;
  w-r6)   C=$A/factorswap-n2w/centerpoint-factorswap-n2w-r6-labelsize.yaml;           R=w2; N="Waymo R6 label size" ;;
  w-r11)  C=$A/factorswap-n2w/centerpoint-factorswap-n2w-r11-reflection.yaml;         R=w2; N="Waymo R11 reflection threshold" ;;
  w-r11r) C=$A/factorswap-n2w/centerpoint-factorswap-n2w-r11r-reflection-random.yaml; R=w2; N="Waymo R11r reflection random twin" ;;
  p-r5)   C=$A/factorswap-pandaset/centerpoint-factorswap-pandaset-r5-pitch.yaml;               R=p2; N="Pandar64 R5 pitch" ;;
  p-r6)   C=$A/factorswap-pandaset/centerpoint-factorswap-pandaset-r6-labelsize.yaml;           R=p2; N="Pandar64 R6 label size" ;;
  p-r9)   C=$A/factorswap-pandaset/centerpoint-factorswap-pandaset-r9-bg050.yaml;               R=p2; N="Pandar64 R9 background 1/2" ;;
  p-r11)  C=$A/factorswap-pandaset/centerpoint-factorswap-pandaset-r11-reflection.yaml;         R=p2; N="Pandar64 R11 reflection threshold" ;;
  p-r11r) C=$A/factorswap-pandaset/centerpoint-factorswap-pandaset-r11r-reflection-random.yaml; R=p2; N="Pandar64 R11r reflection random twin" ;;
  l64-d)  C=$A/factorswap-lyft64/centerpoint-factorswap-lyft64-d-density.yaml;  R=l64; N="Lyft-64 D density" ;;
  l64-r6) C=$A/factorswap-lyft64/centerpoint-factorswap-lyft64-r6-labelsize.yaml; R=l64; N="Lyft-64 R6 label size" ;;
  l64-r5) C=$A/factorswap-lyft64/centerpoint-factorswap-lyft64-r5-pitch.yaml;   R=l64; N="Lyft-64 R5 pitch (item 37d)" ;;
  l40-r6) C=$A/factorswap-lyft40/centerpoint-factorswap-lyft40-r6-labelsize.yaml; R=l40; N="Lyft-40 R6 label size" ;;
  l40-r5) C=$A/factorswap-lyft40/centerpoint-factorswap-lyft40-r5-pitch.yaml;   R=l40; N="Lyft-40 R5 pitch (item 37d)" ;;
  *) echo "unknown row $ROW" >&2; exit 2 ;;
esac
[ -f $C ] || { echo "missing $C" >&2; exit 2; }
case $R in
  k1)  RES="--gres=gpu:1 --cpus-per-task=6 --mem=96G --time=1-06:00:00"; EST=13 ;;
  w2)  RES="--gres=gpu:2 --cpus-per-task=20 --mem=150G --time=24:00:00"; EST=17 ;;
  p2)  RES="--gres=gpu:2 --cpus-per-task=20 --mem=150G --time=24:00:00"; EST=25 ;;
  l64) RES="--gres=gpu:1 --cpus-per-task=6 --mem=96G --time=12:00:00"; EST=6 ;;
  l40) RES="--gres=gpu:1 --cpus-per-task=6 --mem=96G --time=1-02:00:00"; EST=16 ;;
esac
STEM=$(basename $C .yaml | sed 's/^centerpoint-//')
EXTRA=""; [ "$SEED" = 666 ] || EXTRA="--seed $SEED"
TAG=20261108_fs37_${STEM}-seed$SEED
H="Item 37$([ "$SEED" = 666 ] && echo d || echo f) (T05 §5, design fixed 2026-11-08 01:40 JST), labelled ANALYSIS"
T=$(EST_H=$EST WANDB_NOTES="$H: $N, seed $SEED (config $STEM, the seed-666 row's recipe; frame-seeded cut). Term against the target oracle, paired by seed." \
  scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 $RES --exclude=node03 --job-name=f37${ROW//-/}s${SEED: -1} \
  scripts/run_sourceonly_2gpu.sh $C $TAG ${STEM}-seed$SEED $EXTRA < /dev/null | jid)
[ -n "$T" ] || { echo "submission of $ROW failed" >&2; exit 3; }
echo "item37 $ROW seed $SEED: $T"
if [ $R = p2 ]; then
  mkdir -p $RS
  S=$(WANDB_NOTES="CPU only, no W&B: item 37 (T05 §5) rule-A / as-scored rescoring of $N seed $SEED ($T, epoch 115, spin val)" \
    scripts/submit.sh --dependency=afterany:$T --kill-on-invalid-dep=yes --partition=a100,a6000_ada --cpus-per-task=4 \
    --mem=24G --time=02:00:00 --job-name=i37Prs --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
    --wrap="bash analysis/pandaset_rule_a_of_jobs.sh $RS/rescore_37_${ROW}_s$SEED.txt '$N seed $SEED=$T:115:0:false'" \
    < /dev/null | jid)
  [ -n "$S" ] || { echo "rule-A successor of $T failed; submit it by hand" >&2; exit 3; }
  echo "item37 $ROW seed $SEED rule A: $S"
fi
