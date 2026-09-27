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

# est_h = hours on ONE GPU, evaluation included. Where it comes from, per row:
#   psflash 70  job 25827, same config, 1 GPU on node03: 69.2 h of training
#   soKITTI 22  job 25817, same config, 1 GPU on node61: 21.3 h
#   soLYFT  15  job 25835, same config, 1 GPU on node13: 14.2 h
#   soPANDA 38  job 25782, PandaSet spin->flash, 1 GPU on node61: 37.6 h (same source, 115 epochs)
#   soWAYMO 22  20260922_06 section 2e: 17.5 h exclusive, x1.16 for sharing as KITTI measured, + eval
#     name      mem   est_h config-or-source                                        tag                            run_name   notes
ROWS=(
  "psflash|150G|70|cfgs/da-ieee-access/centerpoint-accum-global-pandaset-flash2spin.yaml|20260923_ps_flash2spin_global|accum_global_pandaset_flash2spin|PandaSet flash->spin, MAX_SWEEPS 5 accumulation + global density correction"
  "PTSN|-|-|-|-|-|DALI Tier D1 PTSN scale sweep, nuScenes->KITTI (inference only)"
  "soKITTI|96G|22|kitti|20260923_sourceonly|-|Source-only KITTI -> nuScenes val (da-ieee-access floor)"
  "soLYFT|96G|15|lyft|20260923_sourceonly|-|Source-only Lyft -> nuScenes val (control for GBlobs and global correction rows)"
  "soPANDA|96G|38|pandaset|20260923_sourceonly|-|Source-only PandaSet -> nuScenes val (da-ieee-access floor)"
  "soWAYMO|96G|22|waymo|20260923_sourceonly|-|Source-only Waymo -> nuScenes val (da-ieee-access floor)"
)

njobs() { squeue -u "$USER" -h -o '%i' 2>/dev/null | wc -l; }
queued() { squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$1"; }

for row in "${ROWS[@]}"; do
  IFS='|' read -r NAME _ _ _ _ _ NOTES <<< "$row"
  if [ -z "${NOTES//[[:space:]]/}" ]; then echo "row $NAME has no notes; refusing to submit anything" >&2; exit 2; fi
done

for row in "${ROWS[@]}"; do
  IFS='|' read -r NAME MEM EST CFG TAG RUN NOTES <<< "$row"
  export WANDB_NOTES="$NOTES"
  if queued "$NAME"; then echo "[$(date +%H:%M)] $NAME already queued, skipping"; continue; fi
  while [ "$(njobs)" -ge "$MAXJOBS" ]; do sleep "$POLL"; done
  if [ "$NAME" = "PTSN" ]; then
    # DALI Tier D1 has its own script: inference only, ~1 GPU-hour, no train.py involved.
    echo "[$(date +%H:%M)] submitting PTSN search"
    scripts/submit.sh scripts/run_ptsn_search.sh
  else
    echo "[$(date +%H:%M)] submitting $NAME ($CFG, EST_H $EST)"
    ARGS=(--job-name="$NAME" --mem="$MEM" --partition=a6000_ada,a6000,rtx8000)
    if [ "$RUN" = "-" ]; then EST_H=$EST scripts/submit.sh "${ARGS[@]}" "$S" "$CFG" "$TAG"
    else                      EST_H=$EST scripts/submit.sh "${ARGS[@]}" "$S" "$CFG" "$TAG" "$RUN"; fi
  fi
  sleep 20
done
echo "[$(date +%H:%M)] all rows submitted"
squeue -u "$USER" -o "%.8i %.10j %.3t %.8M %.16R"
