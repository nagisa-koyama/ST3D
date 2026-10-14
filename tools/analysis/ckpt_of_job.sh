#!/usr/bin/env bash
# Print the HOST path of a job's checkpoint, resolved from the job's own log when this script runs.
#
# Why: in a Slurm dependency chain (memory/skills/launch-job §5) the successor of a training row is submitted
# before the row starts, so the row's W&B run directory (run-<timestamp>-<id>) is not known at submit time.
# Resolve it inside the successor instead:
#   --wrap="set -e; C=\$(bash analysis/ckpt_of_job.sh <row jobid> <epoch>); bash analysis/eval_checkpoint.sh <cfg> \$C"
#
# usage (from ST3D/tools): bash analysis/ckpt_of_job.sh <jobid> <epoch>
# Exits 2, saying why on stderr, if the log, its W&B run line or the checkpoint is missing.
set -eu
JOB=${1:?usage: ckpt_of_job.sh <jobid> <epoch>}
EPOCH=${2:?usage: ckpt_of_job.sh <jobid> <epoch>}
LOG=$(ls logs/error_"${JOB}"_*.txt 2>/dev/null | head -n 1 || true)
[ -n "$LOG" ] || { echo "ckpt_of_job: no logs/error_${JOB}_*.txt (run from ST3D/tools)" >&2; exit 2; }
# The first "Run data is saved locally in" line is the job's own run (a teacher's run dir appears later, by path only).
RUN=$(tr '\r' '\n' < "$LOG" | /usr/bin/grep -m 1 -o 'Run data is saved locally in /storage/wandb/run-[0-9_]*-[a-z0-9]*' \
      | sed 's#^Run data is saved locally in /storage#/home/koyama/data#' || true)
[ -n "$RUN" ] || { echo "ckpt_of_job: no W&B run line in $LOG" >&2; exit 2; }
CKPT="$RUN/files/ckpt/checkpoint_epoch_${EPOCH}.pth"
[ -f "$CKPT" ] || { echo "ckpt_of_job: $CKPT does not exist" >&2; exit 2; }
echo "$CKPT"
