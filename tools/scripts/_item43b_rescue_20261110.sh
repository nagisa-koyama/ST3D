#!/usr/bin/env bash
# Item 43b (T05 §5), seed 667 = job 32456: rescue rescore if the row hits its 24 h TimeLimit.
#
# Why: 32456 shared node01 with the nuScenes-source Waymo row 32457 for 16 h and ran at ~1.5 h per epoch
# (seed 666, 32455, alone on node04: 31 min per epoch). At 04:22 JST on 2026-11-10 it was at epoch 8 of 20 with
# the limit at 12:26; it may not reach the in-run evaluation of epoch 20. This successor (afterany) does nothing
# when the log shows epoch 20 evaluated in-run; otherwise it rescores the highest checkpoint on the row's own
# val (single-sweep nuScenes val, the 43b reading), so the reading is taken at the last finished epoch.
#
# usage: sbatch'ed from ST3D/tools via scripts/submit.sh --dependency=afterany:32456 ... --wrap="bash scripts/_item43b_rescue_20261110.sh"
set -eu
cd /home/koyama/code/ST3D/tools
JOB=32456
CFG=cfgs/da-ieee-access-analysis/item43/centerpoint-i43b-lprime-k.yaml
LOG=$(ls logs/error_${JOB}_*.txt | head -n 1)
if tr '\r' '\n' < "$LOG" | /usr/bin/grep -q "EPOCH 20 of .* EVALUATION end"; then
  echo "item43b rescue: $JOB evaluated epoch 20 in-run; nothing to do"
  exit 0
fi
C=""
for e in 20 19 18 17 16 15 14 13 12 11 10 9 8; do
  if C=$(bash analysis/ckpt_of_job.sh $JOB $e 2>/dev/null); then E=$e; break; fi
done
[ -n "$C" ] || { echo "item43b rescue: no checkpoint of $JOB at epochs 8-20" >&2; exit 2; }
echo "item43b rescue: $JOB stopped before the epoch-20 evaluation; rescoring epoch $E: $C"
bash analysis/eval_checkpoint.sh "$CFG" "$C" "i43b_s667_rescue_e$E" 20261109_i43b
