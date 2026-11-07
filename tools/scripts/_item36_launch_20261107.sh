#!/bin/bash
# experiments_md 20261003_05 §5 item 36 (user 2026-11-07: 「gpuが空いていたら更なる追加も」; design fixed 13:11 JST, before any
# number of its own): second seed (667) of the oracles and the label-range rows R7 on Pandar64 / Lyft-64 / Lyft-40, and
# of S2 R7 (its oracle's seed 667 exists: 31817). Each row is its seed-666 job's submit line unchanged plus --seed 667 and
# a -seed667 tag; the IGNORE set of R7 is frame-seeded, so both seeds train on the same set. O_P at seed 667 trains
# centerpoint-pandaset-spin2spin.yaml, the R rows' _BASE_CONFIG_ (26389's spin2flash recipe, spin val in-run); a CPU
# successor (afterany) rescores O_P and R7 at epoch 115 under rule A. ANALYSIS (TRAIN labels).
# Reading (T05 item 36): range term = O - R7 paired by seed; READ at two seeds if the mean is >= 2 with the same sign.
#
# Run once from anywhere: bash scripts/_item36_launch_20261107.sh
set -euo pipefail
cd /home/koyama/code/ST3D/tools
A=cfgs/da-ieee-access-analysis
RS=/home/koyama/data/factorswap_pandaset/rescore
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
OPC=cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml
PR7=$A/factorswap-pandaset/centerpoint-factorswap-pandaset-r7-labelrange.yaml
L64O=$A/factorswap-lyft64/centerpoint-factorswap-lyft64-oracle.yaml
L64R7=$A/factorswap-lyft64/centerpoint-factorswap-lyft64-r7-labelrange.yaml
L40O=$A/factorswap-lyft40/centerpoint-factorswap-lyft40-oracle.yaml
L40R7=$A/factorswap-lyft40/centerpoint-factorswap-lyft40-r7-labelrange.yaml
S2R7=$A/factorswap-s2/centerpoint-factorswap-s2-r7-labelrange.yaml
for f in $OPC $PR7 $L64O $L64R7 $L40O $L40R7 $S2R7; do [ -f $f ] || { echo "missing $f" >&2; exit 2; }; done
# Slurm MaxSubmit is 20 per user: eight submissions (seven rows and the rule-A successor)
Q=$(squeue -u koyama -h | wc -l)
[ $((Q + 8)) -le 20 ] || { echo "only $((20 - Q)) submissions left, need 8" >&2; exit 2; }

H="Item 36 (T05 §5, design fixed 2026-11-07 13:11 JST), labelled ANALYSIS"
train() {  # name est_h gres cpus mem time cfg tag run_name note
  local id
  id=$(EST_H=$2 WANDB_NOTES="$H: ${10}" scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:$3 \
       --cpus-per-task=$4 --mem=$5 --time=$6 --exclude=node03 --job-name=$1 \
       scripts/run_sourceonly_2gpu.sh $7 $8 $9 --seed 667 < /dev/null | tee /dev/stderr | jid)
  [ -n "$id" ] || { echo "submission of $1 failed" >&2; exit 3; }
  echo "$id"
}
PO=$(train fsPos2 25 2 20 150G 24:00:00 $OPC 20261107_fs_pandaset_oracle-seed667 factorswap-pandaset-oracle-seed667 \
  "(PandaSet TRAIN labels) O_P seed 667: spin2spin (the R rows' base, no swap), spin val in-run, rule A by a CPU successor. Pairs with O_P 26389 (seed 666, 81.67); range term = O_P - R7 paired by seed; the seed spread of O_P is PandaSet's noise floor.")
PR=$(train fsPr7s2 25 2 20 150G 24:00:00 $PR7 20261107_fs_pandaset_r7-seed667 factorswap-pandaset-r7-labelrange-seed667 \
  "(PandaSet TRAIN labels) Pandar64 R7 label range seed 667 (kept share 0.448, same frame-seeded IGNORE set). Seed 666: R7 31981 78.50, term +3.17. Two-seed term read if mean >= 2 with the same sign.")
LO=$(train fsL64os2 6 1 6 96G 12:00:00 $L64O 20261107_fs_lyft64_oracle-seed667 factorswap-lyft64-oracle-seed667 \
  "(Lyft TRAIN labels) O_L64 seed 667. Pairs with O_L64 31976 (62.34); its seed spread is Lyft-64's noise floor.")
LR=$(train fsL64r7s2 6 1 6 96G 12:00:00 $L64R7 20261107_fs_lyft64_r7-seed667 factorswap-lyft64-r7-labelrange-seed667 \
  "(Lyft TRAIN labels) Lyft-64 R7 label range seed 667 (kept share 0.640). Seed 666: R7 32083 59.05, term +3.29. Two-seed term read if mean >= 2 with the same sign.")
FO=$(train fsL40os2 16 1 6 96G 1-02:00:00 $L40O 20261107_fs_lyft40_oracle-seed667 factorswap-lyft40-oracle-seed667 \
  "(Lyft TRAIN labels) O_L40 seed 667. Pairs with O_L40 31978 (76.60); its seed spread is Lyft-40's noise floor.")
FR=$(train fsL40r7s2 16 1 6 96G 1-02:00:00 $L40R7 20261107_fs_lyft40_r7-seed667 factorswap-lyft40-r7-labelrange-seed667 \
  "(Lyft TRAIN labels) Lyft-40 R7 label range seed 667 (kept share 0.583). Seed 666: R7 32033 74.59, term +2.01. Two-seed term read if mean >= 2 with the same sign.")
SR=$(train fsr7s2 18 1 6 96G 1-06:00:00 $S2R7 fsS2_r7-labelrange-seed667 factorswap-s2-r7-labelrange-seed667 \
  "(KITTI TRAIN labels) S2 R7 label range seed 667. Pairs with O 31817 (seed 667, 78.79); seed 666: R7 31417, term -0.60. Two-seed term read if mean >= 2 with the same sign.")

mkdir -p $RS
RR=$(WANDB_NOTES="CPU only, no W&B: item 36 (T05 §5) rule-A / as-scored rescoring of Pandar64 O_P seed 667 $PO and R7 seed 667 $PR (in-run epoch-115 results on spin val)" \
  scripts/submit.sh --dependency=afterany:$PO:$PR --kill-on-invalid-dep=yes --partition=a100,a6000_ada --cpus-per-task=4 \
  --mem=24G --time=02:00:00 --job-name=i36Prs --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="bash analysis/pandaset_rule_a_of_jobs.sh $RS/rescore_36_seed667.txt 'O_P seed 667=$PO:115:0:false' 'R7 label range seed 667=$PR:115:0:false'" \
  < /dev/null | tee /dev/stderr | jid)
[ -n "$RR" ] || { echo "submission of the rule-A successor failed (O_P $PO, R7 $PR are in; submit it by hand)" >&2; exit 3; }
echo "item36: O_P $PO PS R7 $PR O_L64 $LO L64 R7 $LR O_L40 $FO L40 R7 $FR S2 R7 $SR rescore $RR"
