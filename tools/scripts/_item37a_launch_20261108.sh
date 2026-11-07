#!/bin/bash
# experiments_md 20261003_05 §5 item 37a (design fixed 2026-11-08 01:40 JST, before any number): count-matched structure
# terms on Waymo and Pandar64 by reuse, eval-only (1 GPU, exempt from the cap). ANALYSIS (the scored rows trained on the
# targets' TRAIN data).
#   W: the W->N count-matched control L_W 31823 / 31824 (epoch 7) on Waymo val with the oracle's evaluation config, as
#      item 26 (31623) scored P and C1.
#   P: the P->N count-matched control L 31539 / 31550 (epoch 115) on Pandar64 spin val through the eval twin
#      centerpoint-factorswap-pandaset-evalonly-l-on-spin.yaml; a CPU successor rescores both under rule A.
# Reading (T05 item 37): structure beyond count = AP(L) - AP(P), count part = AP(C1) - AP(L), paired by seed; read at two
# seeds if the mean is >= 2 with the same sign.
# Run once: bash scripts/_item37a_launch_20261108.sh
set -euo pipefail
cd /home/koyama/code/ST3D/tools
W=/home/koyama/data/wandb
RS=/home/koyama/data/factorswap_pandaset/rescore
IMG=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
OUT=/home/koyama/code/ST3D/output
O_CFG=cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml
L_TWIN=cfgs/da-ieee-access-analysis/factorswap-pandaset/centerpoint-factorswap-pandaset-evalonly-l-on-spin.yaml
ET=20261108_fs37
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
LW1=$W/run-20261105_184654-06mp9qfk/files/ckpt/checkpoint_epoch_7.pth
LW2=$W/run-20261105_202725-apim2jyc/files/ckpt/checkpoint_epoch_7.pth
LP1=$W/run-20261103_163544-li7j89q6/files/ckpt/checkpoint_epoch_115.pth
LP2=$W/run-20261104_005205-e7w3kplr/files/ckpt/checkpoint_epoch_115.pth
for f in $O_CFG $L_TWIN $LW1 $LW2 $LP1 $LP2; do [ -f $f ] || { echo "missing $f" >&2; exit 2; }; done
mkdir -p $RS
H="Item 37a (T05 §5, design fixed 2026-11-08 01:40 JST), labelled ANALYSIS, eval-only"
EW=$(WANDB_NOTES="$H: W->N count-matched control L_W 31823 / 31824 at epoch 7 on unperturbed Waymo val with the oracle's eval block (as item 26 for P 27807 / 27840 and C1 27850 / 28278). Structure beyond count = L_W - P; count part = C1 - L_W, paired by seed." \
  scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=03:00:00 \
  --exclude=node03 --job-name=i37aW --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="F=0; bash analysis/eval_checkpoint.sh $O_CFG $LW1 item37a_LW_31823 20260922_sourceonly || F=\$((F+1)); bash analysis/eval_checkpoint.sh $O_CFG $LW2 item37a_LW_31824 20260922_sourceonly || F=\$((F+1)); echo item37a W failures \$F; [ \$F -eq 0 ]" \
  < /dev/null | tee /dev/stderr | jid)
[ -n "$EW" ] || { echo "submission of i37aW failed" >&2; exit 3; }
EP=$(WANDB_NOTES="$H: P->N count-matched control L 31539 / 31550 at epoch 115 on Pandar64 spin val (eval twin evalonly-l-on-spin); rule A by the CPU successor. Structure beyond count = L - P (28034 / 28279); count part = C1 (28041 / 28280) - L, paired by seed." \
  scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --exclude=node03 --job-name=i37aP --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="F=0; bash analysis/eval_checkpoint.sh $L_TWIN $LP1 fs37_L_31539 $ET || F=\$((F+1)); bash analysis/eval_checkpoint.sh $L_TWIN $LP2 fs37_L_31550 $ET || F=\$((F+1)); echo item37a P failures \$F; [ \$F -eq 0 ]" \
  < /dev/null | tee /dev/stderr | jid)
[ -n "$EP" ] || { echo "submission of i37aP failed" >&2; exit 3; }
row() {  # label cfg epoch tag
  local s d; s=$(basename "$2" .yaml); d=$(dirname "${2#cfgs/}")
  printf '["%s", "%s/%s/%s/%s/eval/epoch_%s/val/%s/result.pkl", 0, false]' "$1" "$OUT" "$d" "$s" "$ET" "$3" "$4"
}
{
  echo "["
  row "L 31539 on spin (37a $EP)" $L_TWIN 115 fs37_L_31539; echo ","
  row "L 31550 on spin" $L_TWIN 115 fs37_L_31550; echo
  echo "]"
} > $RS/rows_37a_l.json
python3 -c "import json; r = json.load(open('$RS/rows_37a_l.json')); print(len(r), 'rows')"
ER=$(WANDB_NOTES="CPU only, no W&B: item 37a (T05 §5) rule-A / as-scored rescoring of eval job $EP (L 31539 / 31550 on spin val)" \
  scripts/submit.sh --dependency=afterok:$EP --kill-on-invalid-dep=yes \
  --partition=a100,a6000_ada --cpus-per-task=4 --mem=24G --time=02:00:00 --job-name=i37aPrs \
  --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D $IMG python3 analysis/pandaset_eval_protocols.py --rows_json $RS/rows_37a_l.json --protocols as_scored rule_A > $RS/rescore_37a_l.txt" \
  < /dev/null | tee /dev/stderr | jid)
[ -n "$ER" ] || { echo "submission of i37aPrs failed (EW $EW, EP $EP are in)" >&2; exit 3; }
echo "item37a: Waymo $EW PandaSet $EP rule-A $ER"
