#!/usr/bin/env bash
# Rule-A (and as-scored) rescoring of PandaSet TRAINING rows from their in-run evaluations, with each row's W&B run
# directory resolved from the row's job log WHEN THIS SCRIPT RUNS (in a dependency chain the successor is submitted
# before the row starts, so the run directory cannot be named at submit time; same lookup as ckpt_of_job.sh).
#
#   bash analysis/pandaset_rule_a_of_jobs.sh [--dry] <out.txt> <label>=<jobid>:<epoch>:<device>:<cone> ...
#     device 0 = spin (Pandar64), 1 = flash (PandarGT); cone true | false (score inside PandarGT's cone)
#
# Writes <out>.json (the rows of analysis/pandaset_eval_protocols.py --rows_json) and <out.txt> (its table). A row whose
# log, run line or result.pkl is missing is printed as missing and the others are still scored; exits 2 if none
# resolves. --dry prints the rows and stops (quick check on the master). Otherwise run it inside a Slurm CPU job
# (memory/skills/launch-job §3b), from ST3D/tools, e.g. --dependency=afterany:<row jobs>.
set -u
DRY=0; [ "${1:-}" = "--dry" ] && { DRY=1; shift; }
OUT=${1:?usage: pandaset_rule_a_of_jobs.sh [--dry] <out.txt> <label>=<jobid>:<epoch>:<device>:<cone> ...}; shift
[ $# -gt 0 ] || { echo "no rows given" >&2; exit 2; }
cd /home/koyama/code/ST3D/tools
JSON=${OUT%.txt}.json
ROWS=(); MISSING=0
for spec in "$@"; do
  label=${spec%%=*}; IFS=: read -r job epoch device cone <<< "${spec#*=}"
  LOG=$(ls logs/error_"${job}"_*.txt 2>/dev/null | head -n 1 || true)
  RUN=""
  [ -n "$LOG" ] && RUN=$(tr '\r' '\n' < "$LOG" | /usr/bin/grep -m 1 -o 'Run data is saved locally in /storage/wandb/run-[0-9_]*-[a-z0-9]*' \
                         | sed 's#^Run data is saved locally in /storage#/home/koyama/data#' || true)
  PKL="$RUN/files/eval/eval_with_train/epoch_${epoch}/val/result.pkl"
  if [ -z "$RUN" ] || [ ! -f "$PKL" ]; then
    echo "MISSING $label (job $job): log '${LOG:-none}', run '${RUN:-none}'" >&2; MISSING=$((MISSING+1)); continue
  fi
  ROWS+=("[\"$label (job $job)\", \"$PKL\", $device, $cone]")
done
[ ${#ROWS[@]} -gt 0 ] || { echo "no row resolved" >&2; exit 2; }
{ echo "["; for i in "${!ROWS[@]}"; do [ "$i" -gt 0 ] && echo ","; printf '%s' "${ROWS[$i]}"; done; echo; echo "]"; } > "$JSON"
python3 -c "import json, sys; r = json.load(open(sys.argv[1])); print(len(r), 'rows resolved,', sys.argv[2], 'missing')" "$JSON" "$MISSING"
[ $DRY -eq 1 ] && { cat "$JSON"; exit 0; }
singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  python3 analysis/pandaset_eval_protocols.py --rows_json "$JSON" --protocols as_scored rule_A > "$OUT"
