#!/bin/bash
# experiments_md 20261003_05 §5 item 35g (user 2026-11-07: 「お願いします。gpuが空いていたら更なる追加も」; design fixed 13:11 JST,
# before any number of R11): the reflection rows R11 (threshold model) and R11r (count-matched random twin) on S2 KITTI,
# N->W Waymo TOP and PandaSet Pandar64, seed 666, each on its oracle's recipe and layout. Build and checks: report
# 20261107_01. ANALYSIS (each target's TRAIN labels train the rows; nothing feeds a method).
#   s2  / s2r  : KITTI, 1 GPU x 6 (oracle 26843; terms read against 80.33, moderate, epoch 152)
#   n2w / n2wr : Waymo, 2 GPUs x 20 as O2 31624 (65.63 on Waymo val every 5th frame, epoch 7); a 2-GPU Waymo-target row can
#                read FAILED on an NCCL timeout in its in-run eval; the launcher's recovery scores the final checkpoint
#   ps  / psr  : Pandar64, 2 GPUs x 20 as O_P 26389 / the item-34 R rows (81.67 rule A, spin val, epoch 115), with a CPU
#                successor (afterany) rescoring the in-run epoch-115 result under rule A
# Reading (35g): reflection term = O - R11, count part = O - R11r, selectivity part = R11r - R11; each read if >= 3 AP.
# Submitted one row at a time as slots of the 12-job cap free, in the order s2, n2w, s2r, n2wr, ps, psr.
#
# Run once per row from anywhere: bash scripts/_item35g_launch_20261107.sh <s2|s2r|n2w|n2wr|ps|psr>
set -euo pipefail
cd /home/koyama/code/ST3D/tools
A=cfgs/da-ieee-access-analysis
RS=/home/koyama/data/factorswap_pandaset/rescore
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
H="Item 35g (T05 §5, design fixed 2026-11-07 13:11 JST; build report 20261107_01), labelled ANALYSIS"
case ${1:?usage: _item35g_launch_20261107.sh <s2|s2r|n2w|n2wr|ps|psr>} in
  s2)   D=factorswap-s2; R=r11-reflection;          JN=fsr11;   EST=18; G=1; C=6;  M=96G;  TL=1-06:00:00
        O="oracle 26843 (80.33 moderate, epoch 152)"; WHAT="(KITTI TRAIN labels) S2 R11 reflection, threshold model";;
  s2r)  D=factorswap-s2; R=r11r-reflection-random;  JN=fsr11r;  EST=18; G=1; C=6;  M=96G;  TL=1-06:00:00
        O="oracle 26843 (80.33 moderate, epoch 152)"; WHAT="(KITTI TRAIN labels) S2 R11r reflection, count-matched random twin";;
  n2w)  D=factorswap-n2w; R=r11-reflection;         JN=fn-r11;  EST=30; G=2; C=20; M=150G; TL=24:00:00
        O="O2 31624 (65.63, Waymo val every 5th frame, epoch 7)"; WHAT="(Waymo TRAIN labels) N->W R11 reflection on TOP returns, threshold model";;
  n2wr) D=factorswap-n2w; R=r11r-reflection-random; JN=fn-r11r; EST=30; G=2; C=20; M=150G; TL=24:00:00
        O="O2 31624 (65.63, Waymo val every 5th frame, epoch 7)"; WHAT="(Waymo TRAIN labels) N->W R11r reflection on TOP returns, count-matched random twin";;
  ps)   D=factorswap-pandaset; R=r11-reflection;    JN=fsPr11;  EST=25; G=2; C=20; M=150G; TL=24:00:00
        O="O_P 26389 (81.67 rule A, spin val, epoch 115)"; WHAT="(PandaSet TRAIN labels) Pandar64 R11 reflection, threshold model";;
  psr)  D=factorswap-pandaset; R=r11r-reflection-random; JN=fsPr11r; EST=25; G=2; C=20; M=150G; TL=24:00:00
        O="O_P 26389 (81.67 rule A, spin val, epoch 115)"; WHAT="(PandaSet TRAIN labels) Pandar64 R11r reflection, count-matched random twin";;
  *) echo "unknown row $1" >&2; exit 2;;
esac
CFG=$A/$D/centerpoint-$D-$R.yaml
[ -f $CFG ] || { echo "missing $CFG" >&2; exit 2; }
N=1; case $1 in ps|psr) N=2;; esac
Q=$(squeue -u koyama -h | wc -l)
[ $((Q + N)) -le 20 ] || { echo "only $((20 - Q)) submissions left, need $N" >&2; exit 2; }
NOTE="$H: $WHAT (E, T, Q in the config, from tools/analysis/reflection_r11_stats.py). Seed 666, paired with the $O. Reflection term = O - R11, count part = O - R11r, selectivity part = R11r - R11; each read if >= 3 AP."
ID=$(EST_H=$EST WANDB_NOTES="$NOTE" scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:$G \
     --cpus-per-task=$C --mem=$M --time=$TL --exclude=node03 --job-name=$JN \
     scripts/run_sourceonly_2gpu.sh $CFG 20261107_fs_${D#factorswap-}_$R < /dev/null | tee /dev/stderr | jid)
[ -n "$ID" ] || { echo "submission of $1 failed" >&2; exit 3; }
if [ $N = 2 ]; then
  mkdir -p $RS
  RR=$(WANDB_NOTES="CPU only, no W&B: item 35g (T05 §5) rule-A / as-scored rescoring of Pandar64 $R $ID (in-run epoch-115 result on spin val)" \
    scripts/submit.sh --dependency=afterany:$ID --kill-on-invalid-dep=yes --partition=a100,a6000_ada --cpus-per-task=4 \
    --mem=24G --time=02:00:00 --job-name=i35gPrs --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
    --wrap="bash analysis/pandaset_rule_a_of_jobs.sh $RS/rescore_35g_$R.txt '$R=$ID:115:0:false'" \
    < /dev/null | tee /dev/stderr | jid)
  [ -n "$RR" ] || { echo "submission of the rule-A successor failed ($1 $ID is in; submit it by hand)" >&2; exit 3; }
  echo "item35g $1: $ID rescore $RR"
else
  echo "item35g $1: $ID"
fi
