#!/usr/bin/env bash
# Keep the Slurm queue topped up to MAXJOBS jobs, submitting from a fixed priority list.
#
# Run in the BACKGROUND from the master node (tmux/nohup), not via sbatch - it is a submitter, not
# a job. It submits one row at a time and only when a slot is free.
#
# MAXJOBS=8 is a COURTESY limit, not Slurm's: Slurm allows this account 12 running / 16 submitted
# jobs and 12 GPUs (QOS limit12), but the ~36 GPUs this container can use are shared by ~8 active
# users (experiments_md/20260927_04). The count includes PENDING jobs, which is what stops a deep
# queue from taking every GPU as it frees ahead of anyone who submits later. GPUs are deliberately
# NOT capped here - the user's decision - so 2-GPU rows can take the total past 8, up to Slurm's 12.
#
# Every row goes through scripts/submit.sh with its own EST_H (hours on ONE GPU, from the closest
# comparable run's real elapsed time), and size_job.py turns that into --gres (2 GPUs above 24 h),
# --cpus-per-task and --time. run_sourceonly_2gpu.sh reads SLURM_GPUS_ON_NODE and picks the plain or
# the DDP path accordingly, so a row's result does not depend on how many GPUs it landed on:
# --batch_size 6 is the total across ranks either way.
#
# A row already present in squeue by job name is skipped, so re-running this after an interruption
# does not double-submit.
#
# Every row carries a WANDB_NOTES description (last column), and all rows are checked before
# anything is submitted - a row without one aborts the whole queue rather than running unlabelled.
set -uo pipefail
cd /home/koyama/code/ST3D/tools
unset EST_H  # set per row below; one inherited from the caller's shell must not size every row
MAXJOBS=${MAXJOBS:-8}
POLL=${POLL:-180}
S=scripts/run_sourceonly_2gpu.sh

# Row format (9 columns, notes last):  name | mem | est_h | env | config-or-source | tag | run_name | extra | notes
#   est_h  hours on ONE GPU, evaluation included, from the closest comparable run's real elapsed time
#   env    space-separated VAR=value for this submission (NGPU=1, STAGE_PANDASET=1, ENTRY=...), or -
#   extra  arguments appended after run_name, passed to the entry point verbatim, or -
# DRY_RUN=1 prints each submission instead of sending it, and does not wait for free slots.
#
# Rows queued 2026-09-29 (experiments_md/20260928_03 section 5):
#   st3dFull  40  26220 (foreground v1, same loop, 1 GPU on node61): 40.6 h. NGPU=1: full-length
#                 self-training under DDP has only been smoke-tested (20260928_01).
#   accumRosS 42  26388 (the parent config): 21.1 h on 2 GPUs.
#   fgv2tso2   6  26386 (the same config): 4.4 h on node03. NGPU=1 like its first seed.
#   t1S2Fcc2  15  26512 (the same config): 8.5 h on 2 GPUs incl. 12 min staging + 55 min calibration.
LDB=/storage/wandb/run-20260924_111041-ldb35c2o/files/ckpt/checkpoint_epoch_30.pth
ROWS=(
  "fgv2tso2|150G|6|NGPU=1|cfgs/da-ieee-access-tier1/centerpoint-foreground-v2-lyft2nuscenes-teacher-sourceonly-4ep.yaml|tier1|fgv2-teacher-sourceonly-4ep-seed667|--pretrained_model $LDB --pretrained_model_teacher $LDB --seed 667|SECOND SEED (667) of 26386: foreground v2 4 ep with the source-only teacher ldb35c2o (23.17). 26386 scored 26.81 / 12.94, +3.6 over its teacher - the only self-training gain above noise; this checks it."
  # t1S2Fcc2 (second seed of the cone arm 26512, 15 h on 2 GPUs) is DEFERRED until 26514 lands:
  # if accumulation alone also reaches ~23.5, the correction adds nothing and the seed is moot.
  "st3dFull|150G|40|NGPU=1|cfgs/da-ieee-access/centerpoint-st3d-lyft2nuscenes.yaml|20260929_st3d_full|st3d_lyft2nuscenes|--pretrained_model $LDB --pretrained_model_teacher $LDB|ST3D baseline row, Lyft->nuScenes, FULL 30 ep, 1 GPU: plain self-training (no correction), teacher+init = source-only ldb35c2o ep30 (23.17). The paper's textbook ST3D row; compare with foreground v1 26220 (29.49) and v2 26396."
  "accumRosS|150G|42|-|cfgs/da-ieee-access/centerpoint-accum-rosshrink-nuscenes2kitti.yaml|20260929_accum_rosshrink|accum_rosshrink_nuscenes2kitti|-|nuScenes->KITTI accumulation only with a SHRINKING Car ROS interval [0.75, 1.00] (was [0.85, 1.20]); only that differs from 26388 (71.93 BEV / 39.95 3D moderate). Tests whether the 3D gap to the oracle (64.23) is box size, per 20260926_03. FOV filter on."
)

njobs() { squeue -u "$USER" -h -o '%i' 2>/dev/null | wc -l; }
queued() { squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$1"; }

for row in "${ROWS[@]}"; do
  IFS='|' read -r NAME _ _ _ _ _ _ _ NOTES <<< "$row"
  if [ -z "${NOTES//[[:space:]]/}" ]; then echo "row $NAME has no notes; refusing to submit anything" >&2; exit 2; fi
done

for row in "${ROWS[@]}"; do
  IFS='|' read -r NAME MEM EST ENVS CFG TAG RUN EXTRA NOTES <<< "$row"
  export WANDB_NOTES="$NOTES"
  if queued "$NAME"; then echo "[$(date +%H:%M)] $NAME already queued, skipping"; continue; fi
  ENVARR=(); [ "$ENVS" != "-" ] && read -ra ENVARR <<< "$ENVS"
  EXTRARR=(); [ "$EXTRA" != "-" ] && read -ra EXTRARR <<< "$EXTRA"
  CMD=(env ${ENVARR[@]+"${ENVARR[@]}"} EST_H="$EST" scripts/submit.sh --job-name="$NAME" --mem="$MEM"
       --partition=a6000_ada,a6000,rtx8000 "$S" "$CFG" "$TAG" "$RUN" ${EXTRARR[@]+"${EXTRARR[@]}"})
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "[dry run] ${CMD[*]}"; continue; fi
  while [ "$(njobs)" -ge "$MAXJOBS" ]; do sleep "$POLL"; done
  echo "[$(date +%H:%M)] submitting $NAME ($CFG, EST_H $EST, env: $ENVS)"
  "${CMD[@]}"
  sleep 20
done
echo "[$(date +%H:%M)] all rows submitted"
squeue -u "$USER" -o "%.8i %.10j %.3t %.8M %.16R"
