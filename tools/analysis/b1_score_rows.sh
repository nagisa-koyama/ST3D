#!/usr/bin/env bash
# CPU-only Slurm job for experiments_md 20261017_07 (B1, full-length S3 / S4 rows): rule-A rescoring of stored
# PandaSet predictions, Car at IoU 0.7 / 0.5 and Pedestrian at 0.5 / 0.25, with the existing tools unchanged:
#   - analysis/pandaset_eval_protocols.py --rows_json ... --protocols as_scored rule_A   (as_scored must reproduce
#     the AP the job logged: the validation; rule_A is the reported number)
#   - analysis/oracle_box_error_multiiou.py --dataset pandaset --arms base               (IoU 0.5 beside 0.7)
#
# usage (from ST3D/tools, through scripts/submit.sh, no --gres):
#   bash analysis/b1_score_rows.sh <out_dir> <epoch> <spec> [<spec> ...]
#   spec = <label>:<source>:<device>:<cone>
#     label   [A-Za-z0-9_.-]+, names the output files
#     source  a Slurm job id of a TRAINING row (its in-run evaluation at <epoch> is resolved from the job's own log
#             when this runs, as analysis/ckpt_of_job.sh resolves a checkpoint), or an absolute result.pkl path
#     device  target sensor: 0 Pandar64 (spin), 1 PandarGT (flash)
#     cone    1 = cut GT and predictions to PandarGT's +-30 deg cone, 0 = 360 deg
# Every row is resolved BEFORE anything is scored; a missing log, run line or result.pkl exits 2 and says which.
# Outputs on NFS only: <out_dir>/rows_<jobid>.json, protocols_<cls>_<jobid>.txt, <label>_<cls>.jsonl / .log.
# Do not edit while a job runs it.
set -euo pipefail
OUT=${1:?usage: b1_score_rows.sh <out_dir> <epoch> <label:source:device:cone> ...}
EPOCH=${2:?usage: b1_score_rows.sh <out_dir> <epoch> <label:source:device:cone> ...}
shift 2
[ "$#" -gt 0 ] || { echo "b1_score_rows: no rows given" >&2; exit 2; }
cd /home/koyama/code/ST3D/tools
mkdir -p "$OUT"
JID=${SLURM_JOB_ID:-manual$(date +%H%M%S)}
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "=== b1_score_rows on $(hostname), $(date), epoch $EPOCH, code at $(git -C /home/koyama/code/ST3D rev-parse --short HEAD)"

resolve() {  # <source> -> host path of the result.pkl
  local src=$1
  if [[ "$src" =~ ^[0-9]+$ ]]; then
    local log run
    log=$(ls logs/error_"${src}"_*.txt 2>/dev/null | head -n 1 || true)
    [ -n "$log" ] || { echo "b1_score_rows: no logs/error_${src}_*.txt" >&2; return 2; }
    run=$(tr '\r' '\n' < "$log" | /usr/bin/grep -m 1 -o 'Run data is saved locally in /storage/wandb/run-[0-9_]*-[a-z0-9]*' \
          | sed 's#^Run data is saved locally in /storage#/home/koyama/data#' || true)
    [ -n "$run" ] || { echo "b1_score_rows: no W&B run line in $log" >&2; return 2; }
    echo "$run/files/eval/eval_with_train/epoch_${EPOCH}/val/result.pkl"
  else
    echo "$src"
  fi
}

ROWS_JSON=$OUT/rows_${JID}.json
SPECS=()
echo -n '[' > "$ROWS_JSON"
first=1
for spec in "$@"; do
  IFS=: read -r label src dev cone <<< "$spec"
  [[ "$label" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "b1_score_rows: bad label '$label'" >&2; exit 2; }
  [[ "$dev" =~ ^[01]$ && "$cone" =~ ^[01]$ ]] || { echo "b1_score_rows: bad device/cone in '$spec'" >&2; exit 2; }
  pkl=$(resolve "$src") || exit 2
  [ -f "$pkl" ] || { echo "b1_score_rows: $label: $pkl does not exist" >&2; exit 2; }
  echo "row $label <- $pkl (device $dev, cone $cone)"
  cone_json=false; [ "$cone" = 1 ] && cone_json=true
  [ "$first" = 1 ] || echo -n ',' >> "$ROWS_JSON"
  printf '["%s", "%s", %s, %s]' "$label" "$pkl" "$dev" "$cone_json" >> "$ROWS_JSON"
  first=0
  SPECS+=("$label|$pkl|$dev|$cone")
done
echo ']' >> "$ROWS_JSON"

for cls in Car Pedestrian; do
  echo "=== pandaset_eval_protocols, $cls, as_scored + rule_A"
  singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/pandaset_eval_protocols.py \
    --rows_json "$ROWS_JSON" --protocols as_scored rule_A --cls "$cls" 2>&1 | tee "$OUT/protocols_${cls}_${JID}.txt"
done

for s in "${SPECS[@]}"; do
  IFS='|' read -r label pkl dev cone <<< "$s"
  for cls in Car Pedestrian; do
    th=0.7,0.5; [ "$cls" = Pedestrian ] && th=0.5,0.25
    echo "=== multi-IoU $label $cls ($th)"
    rm -f "$OUT/${label}_${cls}.jsonl"
    singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/oracle_box_error_multiiou.py \
      --dataset pandaset --cls "$cls" --thresholds "$th" --arms base --result "$pkl" --label "${label}_${cls}" \
      --out "$OUT/${label}_${cls}.jsonl" --device "$dev" --cone "$cone" > "$OUT/${label}_${cls}.log" 2>&1
    tail -n 1 "$OUT/${label}_${cls}.log"
  done
done
echo "=== ALL ROWS DONE $(date)"
