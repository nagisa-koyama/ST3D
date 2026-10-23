#!/usr/bin/env bash
# ENV E1 second seed (experiments_md 20261024_02 §9; design fixed in T05 §5, sub-entry "User 10-24 (received between 03:22
# and 03:25)" under item 4's RESULT ENV, experiments_md 8d96563): the chained successor of one seed-667 in-city row.
# ONE GPU, `afterok` on the training row (as the design fixes), resolving its epoch-20 checkpoint at run time.
#
#   bash scripts/_env_seed667_after_row.sh <training job id> <bb|ss>
#
#   (1) the in-city epoch-20 predictions: train.py's in-run evaluation writes them under the row's OWN W&B run directory
#       (<run>/files/eval/eval_with_train/epoch_20/val/result.pkl, unique per run): checked to exist and to be newer than
#       the checkpoint; the logged Car BEV AP_R40 @0.7 of that evaluation is read from the row's log.
#   (2) the cross-city rescore with analysis/eval_checkpoint.sh (bb: B->B 667 on Singapore val = B->S 667; ss: S->S 667 on
#       Boston val = S->B 667). Its predictions go to output/<cfg>/20261024_env_seed667/eval/epoch_20/val/env_e1_seed667/,
#       a directory no other job uses: eval_checkpoint.sh keys it by config, extra_tag, epoch and eval_tag, and both tags are
#       passed here (adabn_eval.py's directory, keyed by config and epoch only, is NOT used; experiments_md T13 trap).
#   (3) /home/koyama/data/env_splits/seed667/<bb|ss>.env with the paths and logged values, read by
#       analysis/env_e1_seed667_job.sh (the CPU job chained afterok on both successors).
# Runs from the LIVE checkout. Do not edit while a job runs it.
set -euo pipefail
JOB=${1:?usage: _env_seed667_after_row.sh <training job id> <bb|ss>}
WHICH=${2:?usage: _env_seed667_after_row.sh <training job id> <bb|ss>}
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git rev-parse --short HEAD) on $(hostname) at $(date '+%F %T %Z') ==="
D=cfgs/da-ieee-access
case "$WHICH" in
  bb) CROSS=$D/centerpoint-sourceonly-nuscenes-boston2singapore.yaml ;;
  ss) CROSS=$D/centerpoint-sourceonly-nuscenes-singapore2boston.yaml ;;
  *)  echo "second argument must be bb or ss" >&2; exit 2 ;;
esac
OUTDIR=/home/koyama/data/env_splits/seed667
mkdir -p "$OUTDIR"

# bev AP of the first "Car AP_R40@0.70, 0.70, 0.70" block after the first line matching $2 (the whole file when empty)
bev_r40() {
  tr '\r' '\n' < "$1" | awk -v start="$2" '
    start == "" || index($0, start) { f = 1 }
    f && /Car AP_R40@0.70, 0.70, 0.70:/ { g = 1; next }
    g && /bev  AP:/ { split($0, a, /[:,]/); gsub(/ /, "", a[2]); print a[2]; exit }'
}

# (1) in-city predictions of the row
CK=$(bash analysis/ckpt_of_job.sh "$JOB" 20) || { echo "no epoch-20 checkpoint for job $JOB" >&2; exit 2; }
FILES=$(dirname "$(dirname "$CK")")
IN_PRED=$FILES/eval/eval_with_train/epoch_20/val/result.pkl
[ -f "$IN_PRED" ] || { echo "no in-city epoch-20 predictions at $IN_PRED" >&2; exit 2; }
[ "$IN_PRED" -nt "$CK" ] || { echo "$IN_PRED is older than $CK" >&2; exit 2; }
ROW_LOG=$(ls logs/error_"${JOB}"_*.txt | head -n 1)
/usr/bin/grep -q "seed             667" <(tr '\r' '\n' < "$ROW_LOG") || { echo "row $JOB did not log seed 667" >&2; exit 2; }
IN_LOGGED=$(bev_r40 "$ROW_LOG" "checkpoint_epoch_20.pth")
[ -n "$IN_LOGGED" ] || { echo "no epoch-20 Car BEV AP_R40 in $ROW_LOG" >&2; exit 2; }
echo "checkpoint: $CK"
echo "in-city predictions: $IN_PRED (logged Car BEV AP_R40 @0.7 at epoch 20: $IN_LOGGED)"

# (2) cross-city rescore, predictions under their own tags
EXTRA=20261024_env_seed667
EVTAG=env_e1_seed667
bash analysis/eval_checkpoint.sh "$CROSS" "$CK" "$EVTAG" "$EXTRA"
XDIR=/home/koyama/code/ST3D/output/da-ieee-access/$(basename "$CROSS" .yaml)/$EXTRA/eval/epoch_20/val/$EVTAG
X_PRED=$XDIR/result.pkl
[ -f "$X_PRED" ] && [ "$X_PRED" -nt "$CK" ] || { echo "no fresh cross-city predictions at $X_PRED" >&2; exit 2; }
X_LOG=$(ls -t "$XDIR"/log_eval_*.txt | head -n 1)
X_LOGGED=$(bev_r40 "$X_LOG" "")
[ -n "$X_LOGGED" ] || { echo "no Car BEV AP_R40 in $X_LOG" >&2; exit 2; }
echo "cross-city predictions: $X_PRED (logged Car BEV AP_R40 @0.7: $X_LOGGED, $X_LOG)"

# (3) hand-over to the CPU job
cat > "$OUTDIR/$WHICH.env" <<EOF
ROW_JOB=$JOB
CKPT=$CK
IN_PRED=$IN_PRED
IN_LOGGED=$IN_LOGGED
X_PRED=$X_PRED
X_LOGGED=$X_LOGGED
X_LOG=$X_LOG
EOF
cat "$OUTDIR/$WHICH.env"
echo "=== _env_seed667_after_row $WHICH exit 0 at $(date '+%F %T %Z') ==="
