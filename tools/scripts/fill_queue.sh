#!/usr/bin/env bash
# Keep the Slurm queue topped up to MAXJOBS, submitting from a fixed priority list.
#
# Run in the BACKGROUND from the master node (tmux/nohup), not via sbatch - it is a submitter, not
# a job. It submits one row at a time and only when a slot is free, so the per-user cap (4 running
# / 8 queued) is never the thing that decides what runs.
#
# Every row is submitted through run_sourceonly_2gpu.sh, which reads SLURM_GPUS_ON_NODE and picks
# the plain or the DDP path accordingly - so a row's result does not depend on how many GPUs it
# happened to land on. --batch_size 6 is the total across ranks either way.
#
# A row already present in squeue by job name is skipped, so re-running this after an interruption
# does not double-submit.
#
# Every row carries a WANDB_NOTES description (last column), and all rows are checked before
# anything is submitted - a row without one aborts the whole queue rather than running unlabelled.
set -uo pipefail
cd /home/koyama/code/ST3D/tools
MAXJOBS=${MAXJOBS:-4}
POLL=${POLL:-180}
S=scripts/run_sourceonly_2gpu.sh

#     name      mem   config-or-source                                            tag                            run_name   notes
ROWS=(
  "psflash|220G|cfgs/da-ieee-access/centerpoint-accum-global-pandaset-flash2spin.yaml|20260923_ps_flash2spin_global|accum_global_pandaset_flash2spin|PandaSet flash->spin, MAX_SWEEPS 5 accumulation + global density correction"
  "PTSN|-|-|-|-|DALI Tier D1 PTSN scale sweep, nuScenes->KITTI (inference only)"
  "soKITTI|96G|kitti|20260923_sourceonly|-|Source-only KITTI -> nuScenes val (da-ieee-access floor)"
  "soLYFT|96G|lyft|20260923_sourceonly|-|Source-only Lyft -> nuScenes val (control for GBlobs and global correction rows)"
  "soPANDA|96G|pandaset|20260923_sourceonly|-|Source-only PandaSet -> nuScenes val (da-ieee-access floor)"
  "soWAYMO|96G|waymo|20260923_sourceonly|-|Source-only Waymo -> nuScenes val (da-ieee-access floor)"
)

njobs() { squeue -u "$USER" -h -o '%i' 2>/dev/null | wc -l; }
queued() { squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$1"; }

for row in "${ROWS[@]}"; do
  IFS='|' read -r NAME _ _ _ _ NOTES <<< "$row"
  if [ -z "${NOTES//[[:space:]]/}" ]; then echo "row $NAME has no notes; refusing to submit anything" >&2; exit 2; fi
done

for row in "${ROWS[@]}"; do
  IFS='|' read -r NAME MEM CFG TAG RUN NOTES <<< "$row"
  export WANDB_NOTES="$NOTES"
  if queued "$NAME"; then echo "[$(date +%H:%M)] $NAME already queued, skipping"; continue; fi
  while [ "$(njobs)" -ge "$MAXJOBS" ]; do sleep "$POLL"; done
  if [ "$NAME" = "PTSN" ]; then
    # DALI Tier D1 has its own script: inference only, ~1 GPU-hour, no train.py involved.
    echo "[$(date +%H:%M)] submitting PTSN search"
    sbatch scripts/run_ptsn_search.sh
  else
    echo "[$(date +%H:%M)] submitting $NAME ($CFG)"
    ARGS=(--job-name="$NAME" --gres=gpu:1 --cpus-per-task=10 --mem="$MEM" --time=120:00:00
          --partition=a6000_ada,a6000,rtx8000)
    if [ "$RUN" = "-" ]; then sbatch "${ARGS[@]}" "$S" "$CFG" "$TAG"
    else                      sbatch "${ARGS[@]}" "$S" "$CFG" "$TAG" "$RUN"; fi
  fi
  sleep 20
done
echo "[$(date +%H:%M)] all rows submitted"
squeue -u "$USER" -o "%.8i %.10j %.3t %.8M %.16R"
