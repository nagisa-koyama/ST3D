#!/usr/bin/env bash
# sbatch, but refuse to submit without a WANDB_NOTES description.
#
#   WANDB_NOTES="Control for 25780: nuScenes->KITTI, no accum, no correction" \
#     scripts/submit.sh [sbatch options] <script> [script args]
#
# The check has to happen HERE, before the job queues: a job that checked its own note would only
# fail after waiting for a slot. sbatch exports the caller's environment, so the note reaches the
# job, and train.py / adaptive_train.py / test.py prefix it with "[job <id>]" (the ID does not exist
# yet at this point). Plain `sbatch` still works and bypasses the check - use this instead.
set -euo pipefail
if [ -z "${WANDB_NOTES//[[:space:]]/}" ]; then
  echo "submit.sh: WANDB_NOTES is empty. Describe the run (what it tests, what it pairs with):" >&2
  echo "  WANDB_NOTES=\"...\" $0 $*" >&2
  exit 2
fi
export WANDB_NOTES
# --comment puts the note on the Slurm job too, so the queue says what each job is for:
#   squeue -u $USER -o "%.8i %.10j %.3t %.10M %.8N %k"
JOB=$(sbatch --parsable --comment="$WANDB_NOTES" "$@")
echo "submitted job ${JOB%%;*}: [job ${JOB%%;*}] $WANDB_NOTES"
