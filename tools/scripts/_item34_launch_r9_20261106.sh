#!/bin/bash
# experiments_md 20261003_05 §5 item 34 (design fixed 2026-11-06 12:27 JST; build ST3D cf8ef19, report 20261106_01):
# one background row R9 (background kept 1/2 outside boxes + 0.25 m) with its matched-cut successors, submitted when a
# slot of the 12-job cap frees (queue in T05 item 34d: PandaSet, then Lyft-64, then Lyft-40). ANALYSIS (TRAIN labels).
#   ps : R9 on Pandar64 (2 GPUs); a 1-GPU eval (afterany) scores R9's epoch 115 on background-1/2 spin val (O_P's
#        twin is in E1, 31983); a CPU job (afterok) rescores R9 in-run (spin val) and R9 on background-1/2 under rule A.
#   l64: R9 on Lyft-64; a 1-GPU eval (afterany) scores O_L64 (31976) and R9 at epoch 30 on background-1/2 Lyft-64 val.
#   l40: R9 on Lyft-40; as l64 with O_L40 (31978) on Lyft-40 val. The eval also waits for the oracle job while queued.
# Reading (T05 item 34): term = O_T - R9 on unperturbed val; R9_cut >= O_cut + 2 means the cut leaks box positions and
# that target's background row is dropped.
#
# Run once per target from anywhere: bash scripts/_item34_launch_r9_20261106.sh <ps|l64|l40>
set -euo pipefail
cd /home/koyama/code/ST3D/tools
T=${1:?usage: _item34_launch_r9_20261106.sh <ps|l64|l40>}
A=cfgs/da-ieee-access-analysis
ET=20261106_fs34
RS=/home/koyama/data/factorswap_pandaset/rescore
IMG=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
OUT=/home/koyama/code/ST3D/output
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
H="Item 34 (T05 §5, design fixed 2026-11-06 12:27; code ST3D cf8ef19; report 20261106_01), labelled ANALYSIS"

case $T in
  ps)  F=$A/factorswap-pandaset; P=pandaset; JN=fsPr9; EST=25; G=2; C=20; M=150G; TL=24:00:00; EP=115; N=3
       O_JOB=""; O_NOTE="O_P 26389 (81.67 rule A; its background-1/2 twin 81.66 is in E1 31983)";;
  l64) F=$A/factorswap-lyft64; P=lyft64; JN=fsL64r9; EST=6; G=1; C=6; M=96G; TL=12:00:00; EP=30; N=2
       O_JOB=31976; O_NOTE="O_L64 31976 (62.34)";;
  l40) F=$A/factorswap-lyft40; P=lyft40; JN=fsL40r9; EST=16; G=1; C=6; M=96G; TL=1-02:00:00; EP=30; N=2
       O_JOB=31978; O_NOTE="O_L40 31978";;
  *) echo "unknown target $T" >&2; exit 2;;
esac
CFG=$F/centerpoint-factorswap-$P-r9-bg050.yaml
TWIN=$F/centerpoint-factorswap-$P-evalbg050.yaml
for f in $CFG $TWIN; do [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }; done
# Slurm MaxSubmit is 20 per user: never leave a row without its successors
Q=$(squeue -u koyama -h | wc -l)
[ $((Q + N)) -le 20 ] || { echo "only $((20 - Q)) submissions left, need $N" >&2; exit 2; }

R9=$(EST_H=$EST WANDB_NOTES="$H: ($P TRAIN labels) R9 background kept 1/2 outside boxes + 0.25 m. Term = $O_NOTE - R9; read if >= 3 AP; matched-cut check with $(basename $TWIN .yaml) by the successor." \
  scripts/submit.sh --gres=gpu:$G --cpus-per-task=$C --mem=$M --time=$TL --exclude=node03 --job-name=$JN \
  scripts/run_sourceonly_2gpu.sh $CFG 20261106_fs_${P}_r9 < /dev/null | tee /dev/stderr | jid)
[ -n "$R9" ] || { echo "submission of R9 failed" >&2; exit 3; }

if [ $T = ps ]; then
  BODY="set -e; CK=\$(bash analysis/ckpt_of_job.sh $R9 $EP); bash analysis/eval_checkpoint.sh $TWIN \$CK fs34_R9_bg050 $ET"
  DEP=afterany:$R9
else
  BODY="set -e; CO=\$(bash analysis/ckpt_of_job.sh $O_JOB $EP); CK=\$(bash analysis/ckpt_of_job.sh $R9 $EP);"
  BODY="$BODY bash analysis/eval_checkpoint.sh $TWIN \$CO fs34_O_bg050 $ET; bash analysis/eval_checkpoint.sh $TWIN \$CK fs34_R9_bg050 $ET"
  DEP=afterany:$R9
  # the oracle too while it is still in the queue (a finished job may be purged and then refused as a dependency)
  squeue -h -j $O_JOB 2>/dev/null | /usr/bin/grep -q . && DEP=$DEP:$O_JOB
fi
EV=$(WANDB_NOTES="$H. Eval-only: $P R9 matched-cut check, background-1/2 val at epoch $EP ($O_NOTE and R9 $R9)." \
  scripts/submit.sh --dependency=$DEP --kill-on-invalid-dep=yes --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:1 \
  --cpus-per-task=8 --mem=64G --time=02:00:00 --job-name=ev34$T --output=logs/output_%j_%x.txt \
  --error=logs/error_%j_%x.txt --wrap="$BODY" < /dev/null | tee /dev/stderr | jid)
[ -n "$EV" ] || { echo "submission of the eval successor failed (R9 $R9 is in; submit it by hand)" >&2; exit 3; }

if [ $T = ps ]; then
  mkdir -p $RS
  printf '[\n["R9 background on bg-1/2 spin (eval %s)", "%s/%s/%s/%s/eval/epoch_%s/val/fs34_R9_bg050/result.pkl", 0, false]\n]\n' \
    "$EV" "$OUT" "${F#cfgs/}" "$(basename $TWIN .yaml)" "$ET" "$EP" > $RS/rows_34a_r9_bg050.json
  RR=$(WANDB_NOTES="CPU only, no W&B: item 34a (T05 §5) rule-A / as-scored rescoring of Pandar64 R9 $R9 (in-run epoch 115, spin val) and of R9 on background-1/2 spin val (eval $EV)" \
    scripts/submit.sh --dependency=afterok:$EV --kill-on-invalid-dep=yes --partition=a100,a6000_ada --cpus-per-task=4 \
    --mem=24G --time=02:00:00 --job-name=i34R9rs --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
    --wrap="bash analysis/pandaset_rule_a_of_jobs.sh $RS/rescore_34a_r9.txt 'R9 background=$R9:115:0:false'; singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D $IMG python3 analysis/pandaset_eval_protocols.py --rows_json $RS/rows_34a_r9_bg050.json --protocols as_scored rule_A > $RS/rescore_34a_r9_bg050.txt" \
    < /dev/null | tee /dev/stderr | jid)
  [ -n "$RR" ] || { echo "submission of the rule-A successor failed (R9 $R9, eval $EV are in)" >&2; exit 3; }
  echo "item34 R9 $T: R9 $R9 eval $EV rescore $RR"
else
  echo "item34 R9 $T: R9 $R9 eval $EV"
fi
