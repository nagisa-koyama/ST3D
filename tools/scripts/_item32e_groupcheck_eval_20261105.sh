#!/bin/bash
# experiments_md 20261003_05 §5 item 32e (amendment fixed 2026-11-05 23:09 JST, before any number): Lyft per platform,
# Pandar64 spin and the full-length PandarGT flash oracle under the item 32 arms. Submits four 1-GPU eval-only jobs
# (analysis/eval_checkpoint.sh, one config after another) and one CPU job that rescores the two PandaSet jobs under rule
# A (pandaset_eval_protocols.py, flash in its cone), afterok on both. Configs: cfgs/da-ieee-access-analysis/groupcheck/
# (analysis/make_groupcheck_configs_32e.py, pre-flight analysis/groupcheck_32e_cpu_check.py) and, for PandarGT's density
# and accumulation arms, the existing sensitivity-pandaset twins of 20261011_03. ANALYSIS (val labels only score).
#
#   oracle                         checkpoint                                                     epoch
#   Lyft 25769 (ioasskls)          run-20260923_011952-ioasskls  (LIDAR_CONFIG 64 / 40 per job)     30
#   Pandar64 spin 26389 (u2i837pw) run-20260928_054827-u2i837pw                                    115
#   PandarGT flash 29328 (tmjprck9) run-20261017_211636-tmjprck9 (scored as 29329)                  115
#
# Run once from anywhere: bash scripts/_item32e_groupcheck_eval_20261105.sh
set -euo pipefail
cd /home/koyama/code/ST3D/tools
W=/home/koyama/data/wandb
G=cfgs/da-ieee-access-analysis/groupcheck
SP=cfgs/da-ieee-access-analysis/sensitivity-pandaset
ET=20261105_item32e
GPU=(--partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:1 --cpus-per-task=8 --mem=64G)
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
OSENS="dzm030 dzp030 pitchm1 pitchp1 pitchm2 pitchp2 bgkeep050 bgkeep025 fgkeep050 fgkeep025 groundrm020"

LYFT=$W/run-20260923_011952-ioasskls/files/ckpt/checkpoint_epoch_30.pth
SPIN=$W/run-20260928_054827-u2i837pw/files/ckpt/checkpoint_epoch_115.pth
FLASH=$W/run-20261017_211636-tmjprck9/files/ckpt/checkpoint_epoch_115.pth
for c in $LYFT $SPIN $FLASH; do [ -f "$c" ] || { echo "missing $c" >&2; exit 2; }; done

# key | checkpoint | config list (as scored first) | time | note
cfgs_lyft() { echo "$G/centerpoint-sourceonly-lyft2lyft-$1-asscored.yaml"; for a in ratio075 ratio050 ratio025 $OSENS; do echo "$G/centerpoint-sourceonly-lyft2lyft-$1-$a.yaml"; done; }
cfgs_spin() { echo cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml; for a in $OSENS; do echo "$G/centerpoint-pandaset-spin2spin-$a.yaml"; done; }
cfgs_flash() { echo cfgs/da-ieee-access/centerpoint-pandaset-flash2flash.yaml; for a in ratio075 ratio050 ratio025 acc3 acc7; do echo "$SP/centerpoint-pandaset-flash2flash-$a.yaml"; done; for a in $OSENS; do echo "$G/centerpoint-pandaset-flash2flash-$a.yaml"; done; }

declare -A JOBS
submit() {  # key ckpt time note configs...
  local key=$1 ckpt=$2 t=$3 note=$4; shift 4
  for c in "$@"; do [ -f "$c" ] || { echo "missing config $c" >&2; exit 2; }; done
  local body="F=0; for c in $*; do s=\$(basename \$c .yaml); bash analysis/eval_checkpoint.sh \$c $ckpt i32e_\$s $ET || F=\$((F+1)); done; echo item32e failures \$F; [ \$F -eq 0 ]"
  JOBS[$key]=$(WANDB_NOTES="Item 32e (20261003_05 §5): within-group consistency of the oracle analyses. $note Eval-only, one config after another (as scored first). ANALYSIS; val labels only score." \
    scripts/submit.sh "${GPU[@]}" --time=$t --job-name=i32e$key --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
    --wrap="$body" < /dev/null | tee /dev/stderr | jid)
  [ -n "${JOBS[$key]}" ] || { echo "submission of $key failed" >&2; exit 3; }
}
submit L64 $LYFT 04:00:00 "Lyft oracle 25769 (ioasskls, epoch 30) on Lyft val, 64-beam platform only (LIDAR_CONFIG 64): density, height, pitch, region arms." $(cfgs_lyft l64)
submit L40 $LYFT 05:00:00 "Lyft oracle 25769 (ioasskls, epoch 30) on Lyft val, 40-beam platform only (LIDAR_CONFIG 40): density, height, pitch, region arms." $(cfgs_lyft l40)
submit PS $SPIN 03:00:00 "Pandar64 spin oracle 26389 (u2i837pw, epoch 115): the OSENS-2 arms G64 lacks for it (height, pitch, region); rule A by the CPU successor." $(cfgs_spin)
submit PF $FLASH 03:30:00 "Full-length PandarGT flash oracle (29328 tmjprck9, epoch 115, scored as 29329): density, accumulation and the OSENS-2 arms, in the cone; rule A by the CPU successor." $(cfgs_flash)

# rule-A (and as-scored) rescoring of both PandaSet jobs on CPU once both have succeeded
PS_DIR=/home/koyama/data/pandaset_sens
O=/home/koyama/code/ST3D/output
row() {  # config device cone
  local c=$1 s; s=$(basename "$1" .yaml); local d; d=$(dirname "${c#cfgs/}")
  printf '["i32e_%s", "%s/%s/%s/%s/eval/epoch_115/val/i32e_%s/result.pkl", %s, %s]' "$s" "$O" "$d" "$s" "$ET" "$s" "$2" "$3"
}
{
  echo "["
  first=1
  for c in $(cfgs_spin); do [ $first = 1 ] || echo ","; first=0; row "$c" 0 false; done
  for c in $(cfgs_flash); do echo ","; row "$c" 1 true; done
  echo "]"
} > $PS_DIR/rows_item32e.json
python3 -c "import json; r = json.load(open('$PS_DIR/rows_item32e.json')); print(len(r), 'rows')"
IMG=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
WANDB_NOTES="CPU only, no W&B: item 32e (20261003_05 §5) rule-A / as-scored rescoring of the Pandar64 spin (job ${JOBS[PS]}) and PandarGT flash (job ${JOBS[PF]}, in the cone) group-check evaluations" \
  scripts/submit.sh --dependency=afterok:${JOBS[PS]}:${JOBS[PF]} --kill-on-invalid-dep=yes \
  --partition=a100,a6000_ada --cpus-per-task=4 --mem=24G --time=02:00:00 --job-name=i32ePrs \
  --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D $IMG python3 analysis/pandaset_eval_protocols.py --rows_json $PS_DIR/rows_item32e.json --protocols as_scored rule_A > $PS_DIR/rescore_item32e.txt" < /dev/null
echo "eval jobs: L64 ${JOBS[L64]} L40 ${JOBS[L40]} PS ${JOBS[PS]} PF ${JOBS[PF]}"
